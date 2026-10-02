"""YuNet face detector (OpenCV FaceDetectorYN). Runs on the CPU (the pip OpenCV build has no CUDA DNN)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
