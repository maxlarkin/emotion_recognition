"""WebSocket /ws/live: webcam video analysis (interfaces.md, Live session).

One connection = at most one session at a time; a session is one worker source.
A receive task parses messages; a send task is the only writer to the socket and serialises
everything queued for this connection (own replies and results routed by the InferenceHub).
Audio chunks (0x02) and `config` are accepted and ignored until story 1.4.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

import anyio
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketState

from app.inference_hub import STARTING, HubUnavailable, InferenceHub, WorkerGone
from inference.types import AudioResult, ErrorResult, FaceResult, FrameResult
from sources.webcam import AudioPacket, BadMessage, WebcamFrame, decode_jpeg, parse_message

log = logging.getLogger(__name__)

router = APIRouter()

MSG_NOT_STARTED = "Сессия не начата"
MSG_ALREADY_STARTED = "Сессия уже начата"
MSG_LOADING = "Модели ещё загружаются, повторите попытку через несколько секунд"
MSG_UNAVAILABLE = "Модуль распознавания недоступен, подробности в журнале сервера"
MSG_WORKER_DIED = "Модуль распознавания остановился, сессия завершена"
MSG_BAD_CONTROL = "Некорректное управляющее сообщение (ожидается JSON с полем type)"
MSG_FRAME_FAILED = "Не удалось обработать кадр"

PENDING_MAX = 512  # frame ids awaiting a result per session (dropped ids are pruned by newer results)


def face_to_wire(face: FaceResult) -> dict:
    return {
        "track_id": face.track_id,
        "box": [int(v) for v in face.box],
        "face": face.face,
        "face_label": face.face_label,
        "fused": face.fused,
        "fused_label": face.fused_label,
        "confidence": face.confidence,
    }


def audio_to_wire(audio: Optional[AudioResult]) -> Optional[dict]:
    if audio is None:
        return None
    return {
        "speech": audio.speech,
        "probs": audio.probs,
        "label": audio.label,
        "window_end_ts": audio.window_end_ts * 1000.0,
    }


def frame_result_to_wire(res: FrameResult, capture_ts: float, proc_ms: float) -> dict:
    return {
        "type": "frame_result",
        "frame_id": res.frame_id,
        "capture_ts": capture_ts,
        "proc_ms": round(proc_ms, 2),
        "faces": [face_to_wire(f) for f in res.faces],
        "audio": audio_to_wire(res.audio),
    }


def error_msg(message: str) -> dict:
    return {"type": "error", "message": message}


class LiveConnection:
    def __init__(self, ws: WebSocket, hub: Optional[InferenceHub]):
        self.ws = ws
        self.hub = hub
        self.loop = asyncio.get_running_loop()
        self.out: asyncio.Queue[Any] = asyncio.Queue()
        self.source_id: Optional[str] = None
        # frame_id -> (capture_ts ms as sent, receipt time perf_counter s), in receipt order
        self.pending: OrderedDict[int, tuple[float, float]] = OrderedDict()
        self.not_started_reported = False

    # ---- main

    async def run(self) -> None:
        try:
            async with anyio.create_task_group() as tg:
                # Whichever side ends first (client gone, worker gone) ends the other one too.
                tg.start_soon(self._guard, self._send_loop, tg.cancel_scope)
                tg.start_soon(self._guard, self._receive_loop, tg.cancel_scope)
        finally:
            sid, self.source_id = self.source_id, None
            if sid is not None and self.hub is not None:
                # Off the loop (may block on the worker queue) and shielded: if this task is being
                # cancelled, a cancelled to_thread() would drop the job before it starts.
                job = asyncio.get_running_loop().run_in_executor(None, self.hub.unregister, sid)
                await asyncio.shield(job)

    @staticmethod
    async def _guard(fn, scope: anyio.CancelScope) -> None:
        try:
            await fn()
        except Exception as exc:  # noqa: BLE001 - e.g. sending to a socket the client closed
            log.info("live connection ended: %s: %s", type(exc).__name__, exc)
        finally:
            scope.cancel()

    def _reply(self, msg: dict) -> None:
        self.out.put_nowait(msg)

    # ---- receive side

    async def _receive_loop(self) -> None:
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            data = msg.get("bytes")
            if data is not None:
                await self._on_binary(data)
                continue
            text = msg.get("text")
            if text is not None:
                await self._on_text(text)

    async def _on_text(self, text: str) -> None:
        try:
            obj = json.loads(text)
            kind = obj.get("type") if isinstance(obj, dict) else None
        except (ValueError, RecursionError):  # bad or absurdly nested JSON
            kind = None
        if not isinstance(kind, str):
            self._reply(error_msg(MSG_BAD_CONTROL))
        elif kind == "start":
            await self._start()
        elif kind == "stop":
            sid, self.source_id = self.source_id, None
            self.pending.clear()
            self.not_started_reported = False
            if sid is not None and self.hub is not None:
                await asyncio.to_thread(self.hub.unregister, sid)
        elif kind == "config":
            pass  # fusion weights: story 1.4
        else:
            self._reply(error_msg(f"Неизвестный тип управляющего сообщения: {kind[:40]}"))

    async def _start(self) -> None:
        if self.source_id is not None:
            self._reply(error_msg(MSG_ALREADY_STARTED))
            return
        if self.hub is None:
            self._reply(error_msg(MSG_UNAVAILABLE))
            return
        try:
            sid = await asyncio.to_thread(self.hub.register, self.loop, self.out)
        except HubUnavailable as exc:
            self._reply(error_msg(MSG_LOADING if exc.state == STARTING else MSG_UNAVAILABLE))
            return
        self.source_id = sid
        self.not_started_reported = False
        self._reply({"type": "session", "session_id": sid})

    async def _on_binary(self, data: bytes) -> None:
        t_recv = time.perf_counter()
        try:
            parsed = parse_message(data)
        except BadMessage as exc:
            self._reply(error_msg(exc.message))
            return
        if isinstance(parsed, AudioPacket):
            return  # audio: story 1.4
        sid = self.source_id
        if sid is None or self.hub is None:
            if not self.not_started_reported:
                self.not_started_reported = True
                self._reply(error_msg(MSG_NOT_STARTED))
            return
        # Recorded before submitting: the result may be routed back before to_thread returns.
        self.pending[parsed.frame_id] = (parsed.capture_ts, t_recv)
        self.pending.move_to_end(parsed.frame_id)
        while len(self.pending) > PENDING_MAX:
            self.pending.popitem(last=False)
        try:
            accepted = await asyncio.to_thread(self._decode_and_submit, self.hub, sid, parsed)
        except BadMessage as exc:
            self.pending.pop(parsed.frame_id, None)
            self._reply(error_msg(exc.message))
            return
        except (HubUnavailable, KeyError):
            self.pending.pop(parsed.frame_id, None)
            return  # worker gone (WorkerGone follows) or the session was stopped meanwhile
        if not accepted:
            self.pending.pop(parsed.frame_id, None)  # inbound queue full: dropped, no reply

    @staticmethod
    def _decode_and_submit(hub: InferenceHub, sid: str, frame: WebcamFrame) -> bool:
        bgr = decode_jpeg(frame.jpeg)
        # The worker's ts is in seconds and drives track expiry.
        return hub.submit_frame(sid, bgr, frame.frame_id, frame.capture_ts / 1000.0)

    # ---- send side

    async def _send(self, msg: dict) -> None:
        await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    async def _send_loop(self) -> None:
        while True:
            item = await self.out.get()
            if isinstance(item, dict):
                await self._send(item)
            elif isinstance(item, FrameResult):
                if item.source_id == self.source_id:
                    await self._send(self._frame_result(item))
            elif isinstance(item, ErrorResult):
                if item.source_id == self.source_id:
                    log.warning("frame error for %s: %s %s", item.source_id, item.kind, item.message)
                    if item.kind != "unknown_source":
                        await self._send(error_msg(MSG_FRAME_FAILED))
            elif isinstance(item, WorkerGone):
                if item.source_id == self.source_id:
                    self.source_id = None  # the worker is gone: nothing to unregister
                    await self._send(error_msg(MSG_WORKER_DIED))
                    if self.ws.application_state == WebSocketState.CONNECTED:
                        await self.ws.close(code=1011)
                    return

    def _frame_result(self, res: FrameResult) -> dict:
        entry = self.pending.get(res.frame_id)
        if entry is None:  # pruned: fall back to the worker's view
            return frame_result_to_wire(res, res.ts * 1000.0, res.timings.get("total", 0.0))
        capture_ts, t_recv = entry
        # This result answers its frame and supersedes every older pending (dropped) frame.
        while self.pending:
            fid, (_, t) = next(iter(self.pending.items()))
            if t > t_recv:
                break
            del self.pending[fid]
        return frame_result_to_wire(res, capture_ts, (time.perf_counter() - t_recv) * 1000.0)


@router.websocket("/ws/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    hub: Optional[InferenceHub] = getattr(websocket.app.state, "hub", None)
    await LiveConnection(websocket, hub).run()

