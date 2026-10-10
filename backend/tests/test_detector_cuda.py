"""CUDA YuNet against OpenCV FaceDetectorYN (needs the GPU image: CUDA EP and weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.models

ort = pytest.importorskip("onnxruntime")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    pytest.skip("no CUDA EP in this environment", allow_module_level=True)

FACE = Path(__file__).resolve().parents[1] / "bench" / "data" / "face.jpg"


@pytest.fixture(scope="module")
def detectors():
    from app.config import get_settings
    from inference.detector import CudaYuNet, FaceDetector

    path = get_settings().model_path("yunet")
    return FaceDetector(path), CudaYuNet(path)


def scenes():
    import cv2

    face = cv2.imread(str(FACE))
    yield "bench 320x320", face
    canvas = np.full((480, 854, 3), 40, np.uint8)  # odd width: padding path
    canvas[80:400, 260:580] = face
    yield "854x480", canvas
    big = np.full((1440, 2560, 3), 40, np.uint8)
    big[600:680, 1200:1280] = cv2.resize(face, (80, 80))  # ~35 px face, security-sized
    yield "2560x1440 small face", big


@pytest.mark.parametrize("name", ["bench 320x320", "854x480", "2560x1440 small face"])
def test_cuda_boxes_match_opencv(detectors, name):
    cpu, gpu = detectors
    image = dict(scenes())[name]
    a, b = sorted(cpu.detect(image)), sorted(gpu.detect(image))
    assert len(a) == len(b) >= 1, (a, b)
    for x, y in zip(a, b, strict=True):
        assert max(abs(p - q) for p, q in zip(x, y, strict=True)) <= 2, (x, y)


def test_no_face_gives_no_boxes(detectors):
    _, gpu = detectors
    assert gpu.detect(np.zeros((720, 1280, 3), np.uint8)) == []
