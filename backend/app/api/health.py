"""GET /api/health: model state and compute device, as reported by the inference worker.

`models[k]` is true when the weight file is present and the worker has loaded every model.
`device` and `gpu_name` come from the worker, so they show what inference really runs on.
`warning`: models_loading | inference_failed | cuda_unavailable (GPU image fell back to the CPU).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request

from app.config import MODEL_FILES, Settings, get_settings
from app.inference_hub import FAILED, READY, InferenceHub

router = APIRouter()


def model_presence(settings: Settings) -> dict[str, bool]:
    return {key: settings.model_path(key).is_file() for key in MODEL_FILES}


def build_health(settings: Settings, hub: Optional[InferenceHub]) -> dict:
    files = model_presence(settings)
    state = hub.state if hub is not None else FAILED  # no hub: the lifespan did not start one
    ready = state == READY
    info = hub.info if hub is not None and ready else {}
    device = info.get("device") or "cpu"
    body: dict = {
        "status": "ok" if ready and all(files.values()) else "degraded",
        "device": device,
        "models": {key: ready and present for key, present in files.items()},
    }
    if info.get("gpu_name"):
        body["gpu_name"] = info["gpu_name"]
    if state == FAILED:
        body["warning"] = "inference_failed"
    elif not ready:
        body["warning"] = "models_loading"
    elif settings.variant == "gpu" and device != "cuda":
        body["warning"] = "cuda_unavailable"
    return body


@router.get("/api/health")
def health(request: Request) -> dict:
    return build_health(get_settings(), getattr(request.app.state, "hub", None))
