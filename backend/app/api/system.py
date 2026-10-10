"""GET /api/system: the same payload as the `system` message of /ws/streams."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.system_load import snapshot

router = APIRouter()


@router.get("/api/system")
async def system() -> dict:
    return await asyncio.to_thread(snapshot)
