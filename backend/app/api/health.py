"""GET /api/health: compute device and model-file presence. Never loads models."""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter

from app.config import MODEL_FILES, Settings, get_settings

log = logging.getLogger(__name__)

router = APIRouter()


@dataclass(frozen=True)
class DeviceInfo:
    device: str  # "cuda" or "cpu"
    gpu_name: Optional[str] = None
    warning: Optional[str] = None


# Runs in a short-lived child process so CUDA is never initialised in the API process
# (a later fork-based inference worker could not re-initialise it). Prints the GPU name only
# when both torch and onnxruntime can use CUDA.
_PROBE_CODE = """
import onnxruntime, torch
if (torch.cuda.is_available() and torch.cuda.device_count() > 0
        and "CUDAExecutionProvider" in onnxruntime.get_available_providers()):
    print(torch.cuda.get_device_name(0))
"""
_PROBE_TIMEOUT_S = 60


def probe_cuda() -> Optional[str]:
    """Return the GPU name if CUDA is usable (torch and onnxruntime), else None. Never raises."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
        name = proc.stdout.strip()
        if proc.returncode == 0 and name:
            return name.splitlines()[-1]
        log.warning("CUDA probe found no usable GPU (rc=%s): %s", proc.returncode, proc.stderr.strip()[-500:])
    except Exception as exc:  # noqa: BLE001 - any failure or timeout means "no usable GPU"
        log.warning("CUDA probe failed: %s", exc)
    return None


def detect_device(variant: str) -> DeviceInfo:
    if variant != "gpu":
        return DeviceInfo(device="cpu")
    gpu_name = probe_cuda()
    if gpu_name is None:
        return DeviceInfo(device="cpu", warning="cuda_unavailable")
    return DeviceInfo(device="cuda", gpu_name=gpu_name)


@lru_cache(maxsize=1)
def _cached_device(variant: str) -> DeviceInfo:
    return detect_device(variant)


def model_presence(settings: Settings) -> dict[str, bool]:
    return {key: settings.model_path(key).is_file() for key in MODEL_FILES}


def build_health(settings: Settings, device: DeviceInfo) -> dict:
    models = model_presence(settings)
    body: dict = {
        "status": "ok" if all(models.values()) else "degraded",
        "device": device.device,
        "models": models,
    }
    if device.gpu_name:
        body["gpu_name"] = device.gpu_name
    if device.warning:
        body["warning"] = device.warning
    return body


@router.get("/api/health")
def health() -> dict:
    settings = get_settings()
    return build_health(settings, _cached_device(settings.variant))
