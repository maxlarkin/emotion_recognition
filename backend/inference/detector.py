"""YuNet face detector.

`FaceDetector` uses OpenCV `FaceDetectorYN` on the CPU (the pip OpenCV build has no CUDA DNN).
`CudaYuNet` runs the same ONNX model on the ONNX Runtime CUDA EP with OpenCV's post-processing
(pad to a multiple of 32, priors per stride 8/16/32, score = sqrt(cls * obj), box decode, NMS),
so both give the same boxes. `make_detector` picks one per device and falls back to OpenCV.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import numpy as np

log = logging.getLogger(__name__)

SCORE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3
TOP_K = 5000


class FaceDetector:
    def __init__(self, model_path: Path, score_threshold: float = SCORE_THRESHOLD):
        import cv2  # lazy: keeps clip_boxes importable in the model-free test env

        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, NMS_THRESHOLD, TOP_K
        )
        self._size: tuple[int, int] = (320, 320)

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """BGR image -> face boxes (x, y, w, h), clipped to the frame, non-empty."""
        h, w = image.shape[:2]
        if (w, h) != self._size:
            self._det.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._det.detect(image)
        if faces is None:
            return []
        return clip_boxes(faces[:, :4], w, h)


def clip_boxes(raw: np.ndarray, w: int, h: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for x, y, bw, bh in np.asarray(raw, dtype=np.float64).reshape(-1, 4):
        x0 = int(round(max(0.0, min(x, w))))
        y0 = int(round(max(0.0, min(y, h))))
        x1 = int(round(max(0.0, min(x + bw, w))))
        y1 = int(round(max(0.0, min(y + bh, h))))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            boxes.append((x0, y0, x1 - x0, y1 - y0))
    return boxes


STRIDES = (8, 16, 32)


MAX_SESSIONS = 4  # input sizes kept warm (the demo mix uses 3; each session holds its own arena)


class CudaYuNet:
    """YuNet ONNX on the ONNX Runtime CUDA EP; same `detect` surface and thresholds as FaceDetector.

    One ORT session per padded input size: a single session fed alternating sizes (640, 1920, 2560
    streams in one batch) re-plans on every shape change and was measured 30x slower (1407 ms vs
    46 ms for one round of the three sizes on the RTX 3050).
    """

    def __init__(self, model_path: Path, score_threshold: float = SCORE_THRESHOLD, nms_threshold: float = NMS_THRESHOLD, top_k: int = TOP_K):
        self.model_path = model_path
        self.score_threshold, self.nms_threshold, self.top_k = score_threshold, nms_threshold, top_k
        self._sessions: "OrderedDict[tuple[int, int], object]" = OrderedDict()
        self._lock = threading.Lock()  # guards the session/prior caches
        self._priors: dict[tuple[int, int], dict[int, np.ndarray]] = {}
        probe = self._new_session()  # fails here, not on the first frame, if CUDA is unusable
        self.input_name = probe.get_inputs()[0].name
        self.output_names = [o.name for o in probe.get_outputs()]
        missing = {f"{k}_{s}" for k in ("cls", "obj", "bbox") for s in STRIDES} - set(self.output_names)
        if missing:
            raise RuntimeError(f"unexpected YuNet outputs, missing {sorted(missing)}")
        del probe  # not cached: no stream uses its size
        self._cpu_fallback: Optional[FaceDetector] = None

    def _session(self, size: tuple[int, int]):
        with self._lock:
            return self._session_locked(size)

    def _new_session(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # kSameAsRequested: each session's arena grows only as needed (VRAM is the scarce resource)
        cuda = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
        sess = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=[cuda])
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError("CUDAExecutionProvider not available for YuNet")
        return sess

    def _session_locked(self, size: tuple[int, int]):
        sess = self._sessions.get(size)
        if sess is not None:
            self._sessions.move_to_end(size)
            return sess
        sess = self._new_session()
        self._sessions[size] = sess
        while len(self._sessions) > MAX_SESSIONS:
            old, _ = self._sessions.popitem(last=False)
            self._priors.pop(old, None)
        return sess

    @staticmethod
    def padded(w: int, h: int) -> tuple[int, int]:
        return ((w - 1) // 32 + 1) * 32, ((h - 1) // 32 + 1) * 32

    def _grid(self, pw: int, ph: int) -> dict[int, np.ndarray]:
        """Per stride: (A, 2) array of (col, row) cell indices, row-major like the model outputs."""
        key = (pw, ph)
        with self._lock:
            cached = self._priors.get(key)
        if cached is not None:
            return cached
        grids = {}
        for s in STRIDES:
            cols, rows = pw // s, ph // s
            cc, rr = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
            grids[s] = np.stack([cc.ravel(), rr.ravel()], axis=1)
        with self._lock:
            self._priors[key] = grids
        return grids

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """BGR image -> face boxes (x, y, w, h), clipped to the frame, non-empty."""
        import cv2

        h, w = image.shape[:2]
        pw, ph = self.padded(w, h)
        # BGR float32 NCHW, no scaling or mean, zero-padded at the bottom/right (as FaceDetectorYN does)
        padded = image if (pw, ph) == (w, h) else cv2.copyMakeBorder(image, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0)
        blob = cv2.dnn.blobFromImage(padded)
        try:
            sess = self._session((pw, ph))
        except Exception as exc:  # noqa: BLE001 - e.g. no VRAM left for a new size: detect on the CPU
            if self._cpu_fallback is None:
                log.warning("CUDA YuNet session for %dx%d failed (%s); OpenCV for that size", pw, ph, exc)
                self._cpu_fallback = FaceDetector(self.model_path)
            return self._cpu_fallback.detect(image)
        outs = dict(zip(self.output_names, sess.run(self.output_names, {self.input_name: blob}), strict=True))
        grids = self._grid(pw, ph)
        boxes, scores = [], []
        for s in STRIDES:
            cls = np.clip(outs[f"cls_{s}"].reshape(-1), 0.0, 1.0)
            obj = np.clip(outs[f"obj_{s}"].reshape(-1), 0.0, 1.0)
            score = np.sqrt(cls * obj)
            keep = score >= self.score_threshold
            if not keep.any():
                continue
            bbox = outs[f"bbox_{s}"].reshape(-1, 4)[keep]
            cell = grids[s][keep]
            cx = (cell[:, 0] + bbox[:, 0]) * s
            cy = (cell[:, 1] + bbox[:, 1]) * s
            bw = np.exp(bbox[:, 2]) * s
            bh = np.exp(bbox[:, 3]) * s
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1))
            scores.append(score[keep])
        if not boxes:
            return []
        all_boxes = np.concatenate(boxes)
        all_scores = np.concatenate(scores)
        idx = cv2.dnn.NMSBoxes(all_boxes.tolist(), all_scores.tolist(), self.score_threshold, self.nms_threshold, top_k=self.top_k)
        if len(idx) == 0:
            return []
        return clip_boxes(all_boxes[np.asarray(idx).reshape(-1)], w, h)


def make_detector(model_path: Path, device: str) -> Union[FaceDetector, CudaYuNet]:
    """CUDA YuNet on the GPU device, OpenCV otherwise or when the CUDA session cannot be created."""
    if device == "cuda":
        try:
            det = CudaYuNet(model_path)
            log.info("face detection: YuNet on the ONNX Runtime CUDA EP")
            return det
        except Exception as exc:  # noqa: BLE001 - the CPU detector always works
            log.warning("CUDA YuNet unavailable (%s); using OpenCV on the CPU", exc)
    return FaceDetector(model_path)
