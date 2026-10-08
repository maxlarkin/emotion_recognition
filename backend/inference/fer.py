"""Facial emotion recognition: EmotiEffLib enet_b0_8_va_mtl through ONNX Runtime, batched crops -> 7 keys.

The preprocessing reproduces EmotiEffLibRecognizerOnnx._preprocess (RGB crop, resize to 224,
/255, ImageNet mean/std, CHW); the model's 10 outputs are 8 emotion logits + valence/arousal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from inference.types import EMOTIONS, FER_CLASSES, fer_probs_to_emotions, softmax

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# On CUDA every run uses one fixed batch size (zero padding): with the ORT CUDA EP each change
# of input shape re-plans every Conv node (~65 ms on the RTX 3050), far more than the padding
# costs. On the CPU EP shape changes are free, so batches are run unpadded.
CUDA_BATCH = 4
CPU_MAX_BATCH = 16


def ort_providers(device: str) -> list:
    if device == "cuda":
        return [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                },
            ),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def preprocess_rgb(crop_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 crop -> float32 [3, 224, 224]."""
    import cv2

    x = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)


def crop_rgb(frame_bgr: np.ndarray, box: Sequence[int]) -> np.ndarray:
    import cv2

    x, y, w, h = box
    return cv2.cvtColor(frame_bgr[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)


class FacialEmotion:
    def __init__(self, model_path: Path, device: str = "cpu", warmup: bool = True, cuda_batch: int = CUDA_BATCH):
        import onnxruntime as ort  # lazy: CUDA_BATCH etc. stay importable without the runtime

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=ort_providers(device))
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("FER: CUDAExecutionProvider requested but not active")
        self.input_name = self.session.get_inputs()[0].name
        self.fixed_batch = cuda_batch if device == "cuda" else None
        if warmup:
            for b in (self.fixed_batch,) if self.fixed_batch else (1, 4):
                self._run(np.zeros((b, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32))

    def _run(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch})[0]

    def logits(self, batch: np.ndarray) -> np.ndarray:
        """[N, 3, 224, 224] -> [N, 8] emotion logits (valence/arousal dropped)."""
        n = batch.shape[0]
        out = np.empty((n, len(FER_CLASSES)), dtype=np.float32)
        step = self.fixed_batch or CPU_MAX_BATCH
        for start in range(0, n, step):
            part = batch[start : start + step]
            m = part.shape[0]
            if self.fixed_batch and m < self.fixed_batch:
                pad = np.zeros((self.fixed_batch - m,) + part.shape[1:], dtype=np.float32)
                part = np.concatenate([part, pad])
            out[start : start + m] = self._run(np.ascontiguousarray(part))[:m, : len(FER_CLASSES)]
        return out

    def predict(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """RGB crops -> [N, 7] probabilities in EMOTIONS order."""
        if not crops_rgb:
            return np.zeros((0, len(EMOTIONS)), dtype=np.float64)
        batch = np.stack([preprocess_rgb(c) for c in crops_rgb])
        return fer_probs_to_emotions(softmax(self.logits(batch)))
