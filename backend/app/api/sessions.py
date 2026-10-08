"""REST /api/sessions (interfaces.md): list, summary, export, delete."""

from __future__ import annotations

import asyncio
import concurrent.futures
import sqlite3
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response

from app.export import to_csv, to_json
from app.store import SessionRunning, SessionStore

router = APIRouter(prefix="/api/sessions")

MSG_NOT_FOUND = "Сессия не найдена"
MSG_RUNNING = "Сессия ещё идёт"
MSG_BAD_FORMAT = "Неизвестный формат выгрузки: допустимы csv и json"
MSG_NO_STORE = "Хранилище сессий недоступно, подробности в журнале сервера"

# Store closed (shutdown), delete timed out on a busy writer, or SQLite failing: 503, not an English 500.
# (A read after a slow flush is not an error: flush() gives up and the read returns what is committed.)
STORE_ERRORS = (TimeoutError, concurrent.futures.TimeoutError, RuntimeError, sqlite3.Error)


def _store(request: Request) -> SessionStore:
    store: Optional[SessionStore] = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(503, MSG_NO_STORE)
    return store


async def _fresh(store: SessionStore, fn, *args):
    """Run a read after the writer has committed everything queued so far (off the loop)."""

    def job():
        store.flush()
        return fn(*args)

    try:
        return await asyncio.to_thread(job)
    except STORE_ERRORS:
        raise HTTPException(503, MSG_NO_STORE) from None


@router.get("")
async def list_sessions(request: Request) -> list[dict]:
    store = _store(request)
    return await _fresh(store, store.list)


@router.get("/{session_id}")
async def get_session(session_id: str, request: Request) -> dict:
    store = _store(request)
    info = await _fresh(store, store.summary, session_id)
    if info is None:
        raise HTTPException(404, MSG_NOT_FOUND)
    return info


@router.get("/{session_id}/export")
async def export_session(session_id: str, request: Request, fmt: str = Query("csv", alias="format")) -> Response:
    store = _store(request)
    fmt = fmt.strip().lower()
    if fmt not in ("csv", "json"):
        # An unknown id is reported as such whatever the format.
        if await _fresh(store, store.get, session_id) is None:
            raise HTTPException(404, MSG_NOT_FOUND)
        raise HTTPException(422, MSG_BAD_FORMAT)
    rows = await _fresh(store, store.export_rows, session_id)
    if rows is None:
        raise HTTPException(404, MSG_NOT_FOUND)
    if fmt == "csv":
        body, media = await asyncio.to_thread(to_csv, rows), "text/csv; charset=utf-8"
    else:
        body, media = await asyncio.to_thread(to_json, rows), "application/json"
    headers = {"Content-Disposition": f'attachment; filename="session-{session_id}.{fmt}"'}
    return Response(content=body, media_type=media, headers=headers)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> Response:
    store = _store(request)
    try:
        found = await _fresh(store, store.delete, session_id)
    except SessionRunning:
        raise HTTPException(409, MSG_RUNNING) from None
    if not found:
        raise HTTPException(404, MSG_NOT_FOUND)
    return Response(status_code=204)
