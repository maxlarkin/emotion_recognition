"""REST /api/streams* (interfaces.md, CAP-10)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response

from concurrent.futures import ThreadPoolExecutor

from app.streams import StreamDisabled, StreamManager, StreamNotFound, valid_url, validate_new, validate_patch
from sources import rtsp

router = APIRouter(prefix="/api/streams")

MSG_NOT_FOUND = "Поток не найден"
MSG_UNAVAILABLE = "Управление потоками недоступно"
MSG_DISABLED = "Анализ этой камеры выключен"
TEST_TIMEOUT_S = 6.0  # probe opens with a 5 s timeout; this bounds the whole request
# Probes get their own small pool: a hung camera must not occupy the default executor that
# JPEG encoding, store calls and system load use.
_PROBES = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stream-probe")
MSG_PROBE_TIMEOUT = "Нет ответа от камеры"
MSG_PROBE_BUSY = "Идёт другая проверка, повторите через несколько секунд"
PROBES_MAX = 4  # running + waiting probes; a hung probe holds its slot until its own 5 s timeouts end
_probes_active = 0


def _manager(request: Request) -> StreamManager:
    mgr = getattr(request.app.state, "streams", None)
    if mgr is None:
        raise HTTPException(503, MSG_UNAVAILABLE)
    return mgr


async def _body(request: Request) -> Any:
    try:
        return await request.json()
    except (ValueError, json.JSONDecodeError):
        raise HTTPException(422, "Ожидается JSON") from None


@router.get("")
async def list_streams(request: Request) -> list[dict]:
    return _manager(request).list()


@router.post("")
async def create_stream(request: Request) -> dict:
    mgr = _manager(request)
    try:
        body = validate_new(await _body(request))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"id": await mgr.create(body)}


@router.post("/test")
async def test_stream(request: Request) -> dict:
    body = await _body(request)
    url = body.get("url") if isinstance(body, dict) else None
    if not valid_url(url):
        return {"ok": False, "error": "Адрес должен начинаться с rtsp://"}
    global _probes_active
    if _probes_active >= PROBES_MAX:
        return {"ok": False, "error": MSG_PROBE_BUSY}
    _probes_active += 1
    fut = asyncio.get_running_loop().run_in_executor(_PROBES, rtsp.probe, url.strip())

    def _release(_f) -> None:
        global _probes_active
        _probes_active -= 1

    fut.add_done_callback(_release)  # released when the probe really ends, not when we stop waiting
    try:
        info = await asyncio.wait_for(asyncio.shield(fut), TEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"ok": False, "error": MSG_PROBE_TIMEOUT}
    except rtsp.SourceError as exc:
        return {"ok": False, "error": exc.message}
    return {"ok": True, "resolution": list(info.resolution), "has_audio": info.has_audio}


@router.patch("/{stream_id}")
async def update_stream(stream_id: str, request: Request) -> dict:
    mgr = _manager(request)
    try:
        patch = validate_patch(await _body(request))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    try:
        return await mgr.update(stream_id, patch)
    except StreamNotFound:
        raise HTTPException(404, MSG_NOT_FOUND) from None


@router.delete("/{stream_id}", status_code=204)
async def delete_stream(stream_id: str, request: Request) -> Response:
    try:
        await _manager(request).delete(stream_id)
    except StreamNotFound:
        raise HTTPException(404, MSG_NOT_FOUND) from None
    return Response(status_code=204)


@router.post("/{stream_id}/reconnect", status_code=204)
async def reconnect_stream(stream_id: str, request: Request) -> Response:
    try:
        await _manager(request).reconnect(stream_id)
    except StreamNotFound:
        raise HTTPException(404, MSG_NOT_FOUND) from None
    except StreamDisabled:
        raise HTTPException(409, MSG_DISABLED) from None
    return Response(status_code=204)
