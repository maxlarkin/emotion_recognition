"""Runtime settings. Everything is read from environment variables with container defaults."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Model files relative to the models directory. Single source of truth for the image layout:
# used by scripts/fetch_models.py at build time and by /api/health at runtime.
MODEL_FILES: dict[str, str] = {
    "yunet": "yunet/face_detection_yunet_2026may.onnx",
    "fer": "fer/enet_b0_8_va_mtl.onnx",
    "vad": "vad/silero_vad.onnx",
    "ser": "ser/emo.ckpt",
}


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    data_dir: Path
    variant: str  # "cpu" or "gpu": which image this process runs in

    def model_path(self, key: str) -> Path:
        return self.models_dir / MODEL_FILES[key]


def get_settings() -> Settings:
    variant = os.environ.get("APP_VARIANT", "cpu").strip().lower()
    if variant not in ("cpu", "gpu"):
        log.warning("unknown APP_VARIANT=%r, using cpu", variant)
        variant = "cpu"
    return Settings(
        models_dir=Path(os.environ.get("MODELS_DIR", "/models")),
        data_dir=Path(os.environ.get("DATA_DIR", "/data")),
        variant=variant,
    )
