"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.sessions import router as sessions_router
from app.api.streams import router as streams_router
from app.api.system import router as system_router
from app.config import get_settings
from app.inference_hub import InferenceHub
from app.store import SessionStore
from app.streams import StreamManager
from app.ws.live import router as live_router
from app.ws.streams import router as streams_ws_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    log.info("variant=%s", settings.variant)
    # Session store (CAP-4). A store that cannot open is logged; live analysis works without it.
    try:
        app.state.store = SessionStore(settings.data_dir / "app.db")
    except Exception:  # noqa: BLE001
        log.exception("session store unavailable")
        app.state.store = None
    # One inference worker per backend process. Models load in a background thread so the API
    # (health, UI) is reachable meanwhile; health and /ws/live report "loading" until the hub is ready.
    hub = InferenceHub()
    app.state.hub = hub
    hub.start()
    # RTSP streams (CAP-10): persisted streams start at once and wait for the hub themselves.
    streams = StreamManager(hub, app.state.store, asyncio.get_running_loop(), nvdec=settings.variant == "gpu")
    app.state.streams = streams
    try:
        await streams.start()
    except Exception:  # noqa: BLE001 - the app (webcam, sessions) works without streams
        log.exception("stream manager failed to start")
    try:
        yield
    finally:
        await streams.stop()
        app.state.streams = None
        await asyncio.to_thread(hub.stop)
        app.state.hub = None
        store, app.state.store = app.state.store, None
        if store is not None:
            await asyncio.to_thread(store.close)


app = FastAPI(title="Распознавание эмоций", lifespan=lifespan)
app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(live_router)
app.include_router(streams_router)
app.include_router(system_router)
app.include_router(streams_ws_router)
