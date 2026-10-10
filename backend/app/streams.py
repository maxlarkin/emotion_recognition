"""Stream manager (CAP-10): RTSP streams persisted in SQLite, each analysed as an `rtsp` session.

* Every enabled stream runs `sources.rtsp.run` in its own thread (decode, sample, scale) and
  submits frames/audio to the shared InferenceHub. A lost source is retried after 1, 2, 4 … 30 s.
* Results come back on an asyncio queue per stream; one task per stream turns them into
  `stream_result` messages (box rescaled to the tile, `decision`/`face_px`), encodes the tile JPEG
  off the loop, records the session and fans the frame out to /ws/streams clients.
* A settings, url or camera-type change restarts only that stream's thread (a new generation;
  callbacks of the old thread are ignored). The session and the hub source are kept.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
import struct
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.faces import TrackAges, face_to_wire
from app.inference_hub import HubUnavailable, InferenceHub, WorkerGone
from app.store import SessionStore
from app.ws.live import audio_to_wire
from inference.types import AudioResult, ErrorResult, FrameResult
from sources import rtsp

log = logging.getLogger(__name__)

CAMERA_TYPES = ("webcam", "sector", "security")
DETECTION_SIZE = {"webcam": 640, "sector": 1920, "security": 2560}  # camera-profiles.md
DEFAULT_SETTINGS = {"enabled": True, "analysis_fps": 5, "detection_size": "auto", "min_face_px": 48}
BACKOFF_MAX_S = 30.0
TILES_KEPT = 16  # tiles awaiting their result, per stream (a result without its tile is still recorded)
META_KEPT = 256  # decode time and scale per frame awaiting its result
BACKLOG_SKIP_WS = 4  # queued results beyond which a stream stops pushing tiles (it still records them)
FPS_WINDOW_S = 5.0
JPEG_QUALITY = 80
FRAME_HEADER = 0x11

MSG_LOADING = "Модели ещё загружаются"
MSG_INFERENCE = "Модуль распознавания недоступен"


class StreamNotFound(KeyError):
    pass


class StreamDisabled(RuntimeError):
    pass


MASK = "***"
_USERINFO_RE = re.compile(r"^(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<user>[^:@/]*):(?P<password>[^@/]*)@", re.IGNORECASE)


def mask_url(url: str) -> str:
    """rtsp://user:secret@host/x -> rtsp://user:***@host/x. Only the password span is replaced, so
    IPv6 brackets, odd ports and the path stay exactly as stored. No password: unchanged."""
    m = _USERINFO_RE.match(url)
    if not m:
        return url
    return f"{m['scheme']}{m['user']}:{MASK}@{url[m.end():]}"


def unmask_url(new: str, stored: str) -> str:
    """A submitted URL that still carries the masked password gets the stored password back."""
    m, old = _USERINFO_RE.match(new), _USERINFO_RE.match(stored)
    if m and old and m["password"] == MASK:
        return f"{m['scheme']}{m['user']}:{old['password']}@{new[m.end():]}"
    return new


def detection_size(camera_type: str, settings: dict) -> int:
    size = settings.get("detection_size", "auto")
    return DETECTION_SIZE[camera_type] if size == "auto" else int(size)


def encode_frame(stream_id: str, frame_id: int, jpeg: bytes) -> bytes:
    """Binary analysed frame of /ws/streams: 0x11 | u16 len | stream_id | u32 frame_id | JPEG."""
    sid = stream_id.encode("utf-8")
    return struct.pack("<BH", FRAME_HEADER, len(sid)) + sid + struct.pack("<I", frame_id & 0xFFFFFFFF) + jpeg


def encode_jpeg(tile: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


@dataclass
class _Runtime:
    queue: asyncio.Queue
    state: str = "stopped"
    message: Optional[str] = None
    session_id: Optional[str] = None
    hub_sid: Optional[str] = None
    gen: int = 0
    stop: threading.Event = field(default_factory=threading.Event)
    wake: threading.Event = field(default_factory=threading.Event)
    source_resolution: Optional[tuple[int, int]] = None
    decode: Optional[str] = None
    has_audio: Optional[bool] = None
    tiles: "OrderedDict[int, np.ndarray]" = field(default_factory=OrderedDict)  # for the browser, few kept
    meta: "OrderedDict[int, tuple[float, float]]" = field(default_factory=OrderedDict)  # decode_ms, tile scale
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame_ids: Any = field(default_factory=lambda: itertools.count(1))
    ages: TrackAges = field(default_factory=TrackAges)
    result_times: deque = field(default_factory=deque)
    consumer: Optional[asyncio.Task] = None

    def fps(self, now: float) -> float:
        while self.result_times and now - self.result_times[0] > FPS_WINDOW_S:
            self.result_times.popleft()
        if len(self.result_times) < 2:
            return 0.0
        span = max(now - self.result_times[0], 1e-6)
        return round((len(self.result_times) - 1) / span, 2)


class _Sink:
    """Callbacks of one decode thread (generation `gen`); stale generations are ignored."""

    def __init__(self, mgr: "StreamManager", stream_id: str, rt: _Runtime, gen: int):
        self.mgr, self.stream_id, self.rt, self.gen = mgr, stream_id, rt, gen
        self.was_open = False

    def live(self) -> bool:
        return self.rt.gen == self.gen and not self.rt.stop.is_set()

    def opened(self, info: rtsp.OpenInfo) -> None:
        if not self.live():
            return
        self.was_open = True
        self.rt.source_resolution, self.rt.decode, self.rt.has_audio = info.resolution, info.decode, info.has_audio
        log.info("stream %s opened: %sx%s, decode=%s, audio=%s", self.stream_id, *info.resolution, info.decode, info.has_audio)
        self.mgr.post_status(self.stream_id, self.gen, "running", None)

    def frame(self, frame: rtsp.DecodedFrame) -> None:
        rt = self.rt
        if not self.live() or rt.hub_sid is None:
            return
        fid = next(rt.frame_ids)
        scale = frame.tile.shape[1] / frame.image.shape[1]
        with rt.lock:
            rt.tiles[fid] = frame.tile
            rt.meta[fid] = (frame.decode_ms, scale)
            while len(rt.tiles) > TILES_KEPT:
                rt.tiles.popitem(last=False)
            while len(rt.meta) > META_KEPT:
                rt.meta.popitem(last=False)
        try:
            self.mgr.hub.submit_frame(rt.hub_sid, frame.image, fid, frame.ts)
        except (HubUnavailable, KeyError):
            pass  # worker gone (WorkerGone follows) or source being replaced

    def audio(self, pcm: bytes, ts: float) -> None:
        if not self.live() or self.rt.hub_sid is None:
            return
        try:
            self.mgr.hub.submit_audio(self.rt.hub_sid, pcm, ts)
        except (HubUnavailable, KeyError):
            pass


class StreamManager:
    def __init__(self, hub: Optional[InferenceHub], store: Optional[SessionStore], loop: asyncio.AbstractEventLoop, nvdec: bool = False):
        self.hub = hub
        self.store = store
        self.loop = loop
        self.nvdec = nvdec
        self.records: dict[str, dict] = {}  # id -> record (with the real url)
        self.runtimes: dict[str, _Runtime] = {}
        self.clients: set[Any] = set()  # StreamsClient instances of /ws/streams
        self._tasks: list[asyncio.Task] = []
        self._locks: dict[str, asyncio.Lock] = {}
        self.system: Optional[dict] = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        records = await asyncio.to_thread(self.store.load_streams) if self.store is not None else []
        if not records and self.store is not None:
            records = await self._seed()
        for rec in records:
            self.records[rec["id"]] = rec
            self.runtimes[rec["id"]] = _Runtime(asyncio.Queue())
            if rec["settings"].get("enabled", True):
                self._run(rec["id"])
        self._tasks.append(asyncio.create_task(self._system_ticker()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for sid in list(self.runtimes):
            await self._halt(sid)

    async def _seed(self) -> list[dict]:
        """DEFAULT_STREAMS seeds an empty list once (a later delete is not undone by a restart)."""
        raw = os.environ.get("DEFAULT_STREAMS", "").strip()
        if not raw or await asyncio.to_thread(self.store.get_meta, "seeded"):
            return []
        try:
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError("not a list")
        except ValueError as exc:
            log.warning("DEFAULT_STREAMS ignored: %s", exc)
            return []
        out = []
        for i, item in enumerate(items):
            try:
                rec = self._new_record(validate_new(item), created_at=time.time() * 1000.0 + i)
            except ValueError as exc:
                log.warning("DEFAULT_STREAMS entry %d ignored: %s", i, exc)
                continue
            await asyncio.to_thread(self.store.save_stream, rec)
            out.append(rec)
        if out:  # a list with only invalid entries can be fixed and seeded on the next start
            await asyncio.to_thread(self.store.set_meta, "seeded", "1")
        log.info("seeded %d stream(s) from DEFAULT_STREAMS", len(out))
        return out

    # ------------------------------------------------------------------ API

    def row(self, stream_id: str) -> dict:
        rec, rt = self.records[stream_id], self.runtimes[stream_id]
        return {
            "id": stream_id,
            "name": rec["name"],
            "site": rec["site"],
            "url": mask_url(rec["url"]),
            "camera_type": rec["camera_type"],
            "settings": dict(rec["settings"]),
            "state": rt.state,
            "fps": rt.fps(time.monotonic()),
            "has_audio": bool(rt.has_audio),
            "source_resolution": list(rt.source_resolution) if rt.source_resolution else None,
            "decode": rt.decode,
            "session_id": rt.session_id,
        }

    def list(self) -> list[dict]:
        return [self.row(sid) for sid in self.records]

    def _new_record(self, body: dict, created_at: Optional[float] = None) -> dict:
        return {
            "id": uuid.uuid4().hex[:12],
            "name": body["name"],
            "site": body.get("site", ""),
            "url": body["url"],
            "camera_type": body.get("camera_type", "sector"),
            "settings": {**DEFAULT_SETTINGS, **body.get("settings", {})},
            "created_at": created_at if created_at is not None else time.time() * 1000.0,
        }

    async def create(self, body: dict) -> str:
        rec = self._new_record(body)
        if self.store is not None:
            await asyncio.to_thread(self.store.save_stream, rec)
        self.records[rec["id"]] = rec
        self.runtimes[rec["id"]] = _Runtime(asyncio.Queue())
        if rec["settings"]["enabled"]:
            self._run(rec["id"])
        else:
            self._broadcast_status(rec["id"])
        return rec["id"]

    def _lock(self, stream_id: str) -> asyncio.Lock:
        return self._locks.setdefault(stream_id, asyncio.Lock())

    async def update(self, stream_id: str, patch: dict) -> dict:
        async with self._lock(stream_id):
            return await self._update(stream_id, patch)

    async def _update(self, stream_id: str, patch: dict) -> dict:
        if stream_id not in self.records:
            raise StreamNotFound(stream_id)
        rec = self.records[stream_id]
        old = json.dumps({k: rec[k] for k in ("url", "camera_type", "settings")}, sort_keys=True)
        new = dict(rec)
        for key in ("name", "site", "camera_type"):
            if key in patch:
                new[key] = patch[key]
        if "url" in patch:  # the masked password means "keep the stored one"
            new["url"] = unmask_url(patch["url"], rec["url"])
        if "settings" in patch:
            new["settings"] = {**rec["settings"], **patch["settings"]}
        if self.store is not None:
            await asyncio.to_thread(self.store.save_stream, new)
        if stream_id not in self.records:  # deleted meanwhile (cannot happen under the lock; kept as a guard)
            raise StreamNotFound(stream_id)
        self.records[stream_id] = new
        changed = json.dumps({k: new[k] for k in ("url", "camera_type", "settings")}, sort_keys=True) != old
        enabled = new["settings"]["enabled"]
        rt = self.runtimes[stream_id]
        if not enabled and rt.session_id is not None:
            await self._halt(stream_id)
        elif enabled and (rt.session_id is None or changed):
            self._run(stream_id)
        return self.row(stream_id)

    async def delete(self, stream_id: str) -> None:
        async with self._lock(stream_id):
            if stream_id not in self.records:
                raise StreamNotFound(stream_id)
            await self._halt(stream_id)
            if self.store is not None:
                await asyncio.to_thread(self.store.delete_stream, stream_id)
            self.records.pop(stream_id, None)
            self.runtimes.pop(stream_id, None)
        self._locks.pop(stream_id, None)

    async def reconnect(self, stream_id: str) -> None:
        async with self._lock(stream_id):
            if stream_id not in self.records:
                raise StreamNotFound(stream_id)
            if not self.records[stream_id]["settings"]["enabled"]:
                raise StreamDisabled(stream_id)
            self._run(stream_id)

    # ------------------------------------------------------------------ run / halt

    def _run(self, stream_id: str) -> None:
        """Start (or restart) the decode thread; a new session only if none is running."""
        rec, rt = self.records[stream_id], self.runtimes[stream_id]
        rt.stop.set()
        rt.wake.set()  # the old thread exits at its next check; its callbacks are ignored from now on
        rt.gen += 1
        rt.stop, rt.wake = threading.Event(), threading.Event()
        with rt.lock:
            rt.tiles.clear()
            rt.meta.clear()
        rt.ages.clear()
        if rt.session_id is None:
            rt.session_id = uuid.uuid4().hex
            if self.store is not None:
                self._store_call(self.store.create, rt.session_id, "rtsp", stream_id)
        if rt.consumer is None or rt.consumer.done():
            rt.consumer = asyncio.create_task(self._consume(stream_id, rt))
        rt.state, rt.message = "connecting", None
        self._broadcast_status(stream_id)
        args = (stream_id, rt, rt.gen, rec["url"], detection_size(rec["camera_type"], rec["settings"]), float(rec["settings"]["analysis_fps"]))
        threading.Thread(target=self._thread, args=args, name=f"rtsp-{stream_id}", daemon=True).start()

    async def _halt(self, stream_id: str) -> None:
        rt = self.runtimes.get(stream_id)
        if rt is None:
            return
        rt.stop.set()
        rt.wake.set()
        rt.gen += 1
        if rt.consumer is not None:
            rt.consumer.cancel()
            rt.consumer = None
        while not rt.queue.empty():  # results of the finished session must not reach the next one
            rt.queue.get_nowait()
        hub_sid, rt.hub_sid = rt.hub_sid, None
        if hub_sid is not None and self.hub is not None:
            await asyncio.to_thread(self.hub.unregister, hub_sid)
        if rt.session_id is not None and self.store is not None:
            self._store_call(self.store.finish, rt.session_id)
        rt.session_id = None
        rt.state, rt.message = "stopped", None
        rt.result_times.clear()
        self._broadcast_status(stream_id)

    def _thread(self, stream_id: str, rt: _Runtime, gen: int, url: str, det: int, fps: float) -> None:
        stop, wake = rt.stop, rt.wake
        attempt = 0
        while not stop.is_set():
            if rt.hub_sid is None:
                try:
                    if self.hub is None:
                        raise HubUnavailable("failed")
                    hub_sid = self.hub.register(self.loop, rt.queue)
                    if stop.is_set():  # halted while registering: do not leak the hub source
                        self.hub.unregister(hub_sid)
                        return
                    rt.hub_sid = hub_sid
                except HubUnavailable as exc:
                    self.post_status(stream_id, gen, "connecting", MSG_LOADING if exc.state == "starting" else MSG_INFERENCE)
                    wake.wait(2.0)
                    continue
            sink = _Sink(self, stream_id, rt, gen)
            message = None
            try:
                rtsp.run(url, det, fps, stop, sink, nvdec=self.nvdec)
            except rtsp.SourceError as exc:
                message = exc.message
            except Exception as exc:  # noqa: BLE001 - never let a stream thread die silently
                log.exception("stream %s: unexpected error", stream_id)
                message = f"Ошибка обработки потока: {type(exc).__name__}"
            if stop.is_set():
                return
            attempt = 1 if sink.was_open else attempt + 1
            delay = min(BACKOFF_MAX_S, 2.0 ** (attempt - 1))
            self.post_status(stream_id, gen, "reconnecting", f"{message}. Повтор через {int(delay)} с" if message else None)
            wake.wait(delay)

    # ------------------------------------------------------------------ status

    def post_status(self, stream_id: str, gen: int, state: str, message: Optional[str]) -> None:
        """Thread-safe: apply a decode thread's status on the loop (stale generations ignored)."""
        self.loop.call_soon_threadsafe(self._apply_status, stream_id, gen, state, message)

    def _apply_status(self, stream_id: str, gen: int, state: str, message: Optional[str]) -> None:
        rt = self.runtimes.get(stream_id)
        if rt is None or rt.gen != gen or rt.stop.is_set():
            return
        if (rt.state, rt.message) == (state, message):
            return
        rt.state, rt.message = state, message
        self._broadcast_status(stream_id)

    def status_msg(self, stream_id: str) -> dict:
        rt = self.runtimes[stream_id]
        msg = {"type": "stream_status", "stream_id": stream_id, "state": rt.state}
        if rt.message:
            msg["message"] = rt.message
        return msg

    def _broadcast_status(self, stream_id: str) -> None:
        text = json.dumps(self.status_msg(stream_id), ensure_ascii=False)
        for c in list(self.clients):
            c.push_status(stream_id, text)

    # ------------------------------------------------------------------ results

    async def _consume(self, stream_id: str, rt: _Runtime) -> None:
        while True:
            item = await rt.queue.get()
            try:
                if isinstance(item, FrameResult):
                    await self._on_frame(stream_id, rt, item)
                elif isinstance(item, AudioResult):
                    if not item.skipped and rt.session_id is not None and self.store is not None:
                        self._store_call(self.store.record_audio, rt.session_id, audio_to_wire(item))
                elif isinstance(item, ErrorResult):
                    log.warning("stream %s worker error: %s %s", stream_id, item.kind, item.message)
                elif isinstance(item, WorkerGone):
                    # The hub stays failed until the backend restarts: stop decoding for nothing.
                    rt.hub_sid = None
                    rt.stop.set()
                    rt.wake.set()
                    rt.gen += 1
                    if rt.session_id is not None and self.store is not None:
                        self._store_call(self.store.finish, rt.session_id)
                    rt.session_id = None
                    rt.state, rt.message = "error", MSG_INFERENCE
                    self._broadcast_status(stream_id)
                    return
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - one bad result must not stop the stream
                log.exception("stream %s: result handling failed", stream_id)

    async def _on_frame(self, stream_id: str, rt: _Runtime, res: FrameResult) -> None:
        with rt.lock:
            meta = rt.meta.pop(res.frame_id, None)
            tile = rt.tiles.pop(res.frame_id, None)
            for d in (rt.meta, rt.tiles):  # frames dropped by the worker
                for fid in [f for f in d if f < res.frame_id]:
                    del d[fid]
        if meta is None or stream_id not in self.records:
            return  # a frame of a previous generation
        decode_ms, scale = meta
        min_face = int(self.records[stream_id]["settings"].get("min_face_px", 48))
        ages = rt.ages.update([f.track_id for f in res.faces], res.ts)
        faces = [face_to_wire(f, ages[f.track_id], min_face, scale) for f in res.faces]
        audio = audio_to_wire(res.audio) if res.audio is not None else None
        now = time.monotonic()
        rt.result_times.append(now)
        if rt.session_id is not None and self.store is not None:
            self._store_call(self.store.record_frame, rt.session_id, res.frame_id, decode_ms, faces, audio)
        wanted = [c for c in self.clients if c.wants(stream_id)]
        if tile is None or not wanted or rt.queue.qsize() > BACKLOG_SKIP_WS:
            return  # nobody watches this stream, or encoding fell behind: record only
        jpeg = await asyncio.to_thread(encode_jpeg, tile)
        msg = {
            "type": "stream_result",
            "stream_id": stream_id,
            "frame_id": res.frame_id,
            "decode_ts": decode_ms,
            "emit_ts": time.time() * 1000.0,
            "faces": faces,
            "audio": audio,
            "fps": rt.fps(time.monotonic()),
        }
        binary = encode_frame(stream_id, res.frame_id, jpeg)
        text = json.dumps(msg, ensure_ascii=False)
        for c in wanted:
            c.push_frame(stream_id, binary, text)

    # ------------------------------------------------------------------ system load

    async def _system_ticker(self) -> None:
        from app.system_load import snapshot

        while True:
            try:
                self.system = await asyncio.to_thread(snapshot)
                if self.clients:
                    text = json.dumps({"type": "system", **self.system})
                    for c in list(self.clients):
                        c.push_system(text)
            except Exception:  # noqa: BLE001
                log.exception("system load failed")
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------------ clients

    def add_client(self, client: Any) -> None:
        self.clients.add(client)
        for sid in self.records:
            client.push_status(sid, json.dumps(self.status_msg(sid), ensure_ascii=False))

    def remove_client(self, client: Any) -> None:
        self.clients.discard(client)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _store_call(fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 - store failures never break analysis
            log.warning("session store write failed: %s: %s", type(exc).__name__, exc)


# ---------------------------------------------------------------------- validation

RTSP_RE = re.compile(r"^rtsps?://\S+$", re.IGNORECASE)
SETTING_VALUES: dict[str, tuple] = {
    "enabled": (True, False),
    "analysis_fps": (3, 5, 8),
    "detection_size": ("auto", 640, 1280, 1920, 2560),
    "min_face_px": (40, 48, 56, 64),
}


def _check_settings(settings: Any) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("Некорректные настройки потока")
    out = {}
    for key, value in settings.items():
        allowed = SETTING_VALUES.get(key)
        if allowed is None:
            raise ValueError(f"Неизвестная настройка «{key}»")
        numeric_ok = isinstance(value, bool) if key == "enabled" else (isinstance(value, int) and not isinstance(value, bool)) or value == "auto"
        if not numeric_ok or value not in allowed:
            raise ValueError(f"Недопустимое значение настройки «{key}»")
        out[key] = value
    return out


def _check_common(body: dict, out: dict, allow_masked: bool = False) -> None:
    if "name" in body:
        name = body["name"]
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
            raise ValueError("Укажите название потока (до 100 символов)")
        out["name"] = name.strip()
    if "site" in body:
        site = body["site"]
        if not isinstance(site, str) or len(site) > 100:
            raise ValueError("Некорректная точка (до 100 символов)")
        out["site"] = site.strip()
    if "url" in body:
        url = body["url"]
        if not isinstance(url, str) or not RTSP_RE.match(url.strip()):
            raise ValueError("Адрес должен начинаться с rtsp://")
        out["url"] = url.strip()
        m = _USERINFO_RE.match(out["url"])
        if m and m["password"] == MASK and not allow_masked:
            raise ValueError("Укажите пароль камеры вместо ***")
    if "camera_type" in body:
        if body["camera_type"] not in CAMERA_TYPES:
            raise ValueError("Тип камеры: webcam, sector или security")
        out["camera_type"] = body["camera_type"]
    if "settings" in body:
        out["settings"] = _check_settings(body["settings"])


def validate_new(body: Any) -> dict:
    if not isinstance(body, dict):
        raise ValueError("Ожидается объект JSON")
    if "name" not in body or "url" not in body:
        raise ValueError("Укажите название и адрес потока")
    out: dict = {}
    _check_common(body, out)
    return out


def valid_url(url: object) -> bool:
    return isinstance(url, str) and bool(RTSP_RE.match(url.strip()))


def validate_patch(body: Any) -> dict:
    if not isinstance(body, dict):
        raise ValueError("Ожидается объект JSON")
    unknown = set(body) - {"name", "site", "url", "camera_type", "settings"}
    if unknown:
        raise ValueError(f"Неизвестное поле «{sorted(unknown)[0]}»")
    out: dict = {}
    _check_common(body, out, allow_masked=True)  # a masked password keeps the stored one
    return out
