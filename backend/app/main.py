"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.sessions import router as sessions_router
from app.config import get_settings
from app.inference_hub import InferenceHub
from app.store import SessionStore
from app.ws.live import router as live_router

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
    try:
        yield
    finally:
        await asyncio.to_thread(hub.stop)
        app.state.hub = None
        store, app.state.store = app.state.store, None
        if store is not None:
            await asyncio.to_thread(store.close)


app = FastAPI(title="Распознавание эмоций", lifespan=lifespan)
app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(live_router)
