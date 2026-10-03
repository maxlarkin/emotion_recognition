"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import _cached_device, router as health_router
from app.config import get_settings
from app.inference_hub import InferenceHub
from app.ws.live import router as live_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    info = _cached_device(settings.variant)  # probe once at startup, not per request
    log.info("variant=%s device=%s gpu=%s warning=%s", settings.variant, info.device, info.gpu_name, info.warning)
    # One inference worker per backend process. Models load in a background thread so the API
    # (health, UI) is reachable meanwhile; /ws/live answers "loading" until the hub is ready.
    hub = InferenceHub()
    app.state.hub = hub
    hub.start()
    try:
        yield
    finally:
        await asyncio.to_thread(hub.stop)
        app.state.hub = None


app = FastAPI(title="Распознавание эмоций", lifespan=lifespan)
app.include_router(health_router)
app.include_router(live_router)
