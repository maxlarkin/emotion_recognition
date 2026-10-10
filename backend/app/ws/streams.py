"""WebSocket /ws/streams (interfaces.md, RTSP streams): one viewing socket for the whole grid.

Each client keeps only the latest analysed frame per stream (binary frame, then its JSON result);
older ones are overwritten if the client falls behind. Status, event and system messages queue
in order. `{"type":"subscribe","stream_ids":[...]}` limits frames and results to those streams.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from typing import Optional

import anyio
from fastapi import APIRouter, WebSocket

log = logging.getLogger(__name__)

router = APIRouter()

TEXT_BACKLOG = 1000  # event messages kept for a slow client (statuses and system load are coalesced)


class StreamsClient:
    """Per-client buffers: latest frame and latest status per stream, latest system load, events in order."""

    def __init__(self) -> None:
        self.subscribed: Optional[set[str]] = None
        self.frames: dict[str, tuple[bytes, str]] = {}
        self.statuses: dict[str, str] = {}
        self.system: Optional[str] = None
        self.texts: deque[str] = deque(maxlen=TEXT_BACKLOG)
        self.wake = asyncio.Event()

    def wants(self, stream_id: str) -> bool:
        return self.subscribed is None or stream_id in self.subscribed

    def push_status(self, stream_id: str, text: str) -> None:
        self.statuses[stream_id] = text
        self.wake.set()

    def push_system(self, text: str) -> None:
        self.system = text
        self.wake.set()

    def push_frame(self, stream_id: str, binary: bytes, text: str) -> None:
        if not self.wants(stream_id):
            return
        self.frames[stream_id] = (binary, text)
        self.wake.set()

    def push_text(self, text: str) -> None:
        self.texts.append(text)
        self.wake.set()

    def subscribe(self, obj: object) -> None:
        ids = obj.get("stream_ids") if isinstance(obj, dict) else None
        if isinstance(ids, list) and all(isinstance(i, str) for i in ids):
            self.subscribed = set(ids)
            for sid in [s for s in self.frames if s not in self.subscribed]:
                del self.frames[sid]


async def _send_loop(ws: WebSocket, client: StreamsClient) -> None:
    while True:
        await client.wake.wait()
        client.wake.clear()
        statuses, client.statuses = client.statuses, {}
        for text in statuses.values():
            await ws.send_text(text)
        while client.texts:
            await ws.send_text(client.texts.popleft())
        system, client.system = client.system, None
        if system is not None:
            await ws.send_text(system)
        frames, client.frames = client.frames, {}
        for binary, text in frames.values():
            await ws.send_bytes(binary)
            await ws.send_text(text)


async def _receive_loop(ws: WebSocket, client: StreamsClient) -> None:
    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            return
        text = msg.get("text")
        if text is None:
            continue
        try:
            obj = json.loads(text)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict) and obj.get("type") == "subscribe":
            client.subscribe(obj)


async def _guard(fn, ws: WebSocket, client: StreamsClient, scope: anyio.CancelScope) -> None:
    try:
        await fn(ws, client)
    except Exception as exc:  # noqa: BLE001 - client gone
        log.info("streams socket ended: %s: %s", type(exc).__name__, exc)
    finally:
        scope.cancel()


@router.websocket("/ws/streams")
async def streams_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    mgr = getattr(websocket.app.state, "streams", None)
    client = StreamsClient()
    if mgr is not None:
        mgr.add_client(client)
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_guard, _send_loop, websocket, client, tg.cancel_scope)
            tg.start_soon(_guard, _receive_loop, websocket, client, tg.cancel_scope)
    finally:
        if mgr is not None:
            mgr.remove_client(client)
