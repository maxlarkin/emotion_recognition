"""RTSP source (CAP-10): one blocking decode loop per stream, run in a thread by the stream manager.

PyAV opens the URL over TCP with a 5 s timeout. The GPU image first tries NVDEC (hwaccel cuda)
and falls back to CPU decode. Video is sampled to the analysis rate and scaled to the camera
profile's detection size; a tile (longest side <= 640) is kept for the browser. The audio track,
if any, is resampled to PCM16 mono 16 kHz and cut into 500 ms chunks.

Frames and audio share one media clock mapped once to server time (`t0 + (pts - pts0)`), so the
worker can match audio windows to frames. `decode_ms` is the server wall time of the decode.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger(__name__)

OPEN_TIMEOUT_S = 5.0
READ_TIMEOUT_S = 5.0
TILE_MAX = 640
AUDIO_RATE = 16000
AUDIO_CHUNK_SAMPLES = AUDIO_RATE // 2  # 500 ms
RTSP_OPTIONS = {"rtsp_transport": "tcp"}
RESYNC_S = 0.1  # audio continuity tolerance before a chunk restarts at the packet's time
CLOCK_JUMP_S = 2.0  # media time moving back (or forward) more than this re-anchors the clock


class SourceError(RuntimeError):
    """The source could not be opened or stopped delivering. `message` is Russian text for the UI."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class OpenInfo:
    resolution: tuple[int, int]  # source width, height
    has_audio: bool
    decode: str  # "nvdec" | "cpu"


@dataclass
class DecodedFrame:
    image: np.ndarray  # BGR at detection size
    tile: np.ndarray  # BGR, longest side <= TILE_MAX
    ts: float  # media clock mapped to server time, seconds
    decode_ms: float  # server wall time of the decode, ms


class Sink(Protocol):
    def opened(self, info: OpenInfo) -> None: ...
    def frame(self, frame: DecodedFrame) -> None: ...
    def audio(self, pcm: bytes, ts: float) -> None: ...


def fit(size: tuple[int, int], longest: int) -> tuple[int, int]:
    """(w, h) scaled so the longest side is `longest` (never upscaled), both even."""
    w, h = size
    k = min(1.0, longest / max(w, h))
    return max(2, int(round(w * k / 2)) * 2), max(2, int(round(h * k / 2)) * 2)


def mask_error(exc: BaseException) -> str:
    """Russian message for an open/read failure."""
    text = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text:
        return "Нет ответа от камеры"
    if "401" in text or "unauthorized" in text:
        return "Камера отклонила логин или пароль"
    if "404" in text or "not found" in text:
        return "Поток не найден по этому адресу"
    if "refused" in text:
        return "Камера отклонила подключение"
    return "Не удалось открыть поток"


def _open(url: str, nvdec: bool):
    import av

    kwargs = {"options": dict(RTSP_OPTIONS), "timeout": (OPEN_TIMEOUT_S, READ_TIMEOUT_S)}
    if nvdec:
        from av.codec.hwaccel import HWAccel

        kwargs["hwaccel"] = HWAccel(device_type="cuda", allow_software_fallback=True)
    return av.open(url, **kwargs)


def probe(url: str) -> OpenInfo:
    """Open the URL, read one video frame size and close (blocking, ~5 s max). Raises SourceError."""
    try:
        container = _open(url, nvdec=False)
    except Exception as exc:  # noqa: BLE001 - every av error becomes a Russian message
        raise SourceError(mask_error(exc)) from exc
    try:
        if not container.streams.video:
            raise SourceError("В потоке нет видео")
        vs = container.streams.video[0]
        w, h = vs.codec_context.width, vs.codec_context.height
        if not w or not h:
            for frame in container.decode(vs):
                w, h = frame.width, frame.height
                break
        if not w or not h:
            raise SourceError("Не удалось прочитать кадр")
        return OpenInfo((int(w), int(h)), bool(container.streams.audio), "cpu")
    except SourceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SourceError(mask_error(exc)) from exc
    finally:
        container.close()


class _AudioCutter:
    """Resampled PCM16 → 500 ms chunks with the media time of their first sample."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.start_ts: Optional[float] = None

    def push(self, pcm: bytes, ts: float, sink: Sink) -> None:
        if self.start_ts is not None and self.buf:
            expected = self.start_ts + len(self.buf) / 2 / AUDIO_RATE
            if abs(ts - expected) > RESYNC_S:  # lost packets or a clock jump: restart the chunk here
                self.buf.clear()
        if self.start_ts is None or not self.buf:
            self.start_ts = ts
        self.buf.extend(pcm)
        step = AUDIO_CHUNK_SAMPLES * 2
        while len(self.buf) >= step:
            sink.audio(bytes(self.buf[:step]), self.start_ts)
            del self.buf[:step]
            self.start_ts += AUDIO_CHUNK_SAMPLES / AUDIO_RATE


def run(url: str, detection_size: int, fps: float, stop: threading.Event, sink: Sink, nvdec: bool = False) -> None:
    """Decode until `stop` is set (returns) or the source fails/ends (raises SourceError)."""
    import av

    try:
        container = _open(url, nvdec)
    except Exception as exc:  # noqa: BLE001
        if nvdec:
            log.info("NVDEC open failed (%s), retrying with CPU decode", exc)
            return run(url, detection_size, fps, stop, sink, nvdec=False)
        raise SourceError(mask_error(exc)) from exc
    try:
        if not container.streams.video:
            raise SourceError("В потоке нет видео")
        vs = container.streams.video[0]
        aus = container.streams.audio[0] if container.streams.audio else None
        resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_RATE) if aus is not None else None
        cutter = _AudioCutter()
        period = 1.0 / max(0.1, fps)
        next_t: Optional[float] = None
        t0: Optional[tuple[float, float]] = None  # (media seconds, server seconds) of the anchor packet
        last_media: Optional[float] = None
        opened = False
        streams = [vs] + ([aus] if aus is not None else [])

        def clock(media: Optional[float]) -> float:
            """Media time -> server time. A jump (camera restart, PTS wrap) re-anchors the clock so
            the sampler does not skip every frame until media time catches up."""
            nonlocal t0, last_media, next_t
            now = time.time()
            if media is None:
                return now
            if t0 is None or (last_media is not None and abs(media - last_media) > CLOCK_JUMP_S):
                if t0 is not None:
                    log.info("media clock jump %.1f s -> re-anchored", media - (last_media or 0.0))
                t0 = (media, now)
                next_t = None
            last_media = media
            return t0[1] + (media - t0[0])

        for packet in container.demux(*streams):
            if stop.is_set():
                return
            if packet.dts is None:  # flush packet at end of stream
                continue
            for frame in packet.decode():
                if packet.stream is vs:
                    if not opened:
                        decode = "nvdec" if getattr(vs.codec_context, "is_hwaccel", False) else "cpu"
                        sink.opened(OpenInfo((frame.width, frame.height), aus is not None, decode))
                        opened = True
                    ts = clock(frame.time)
                    if next_t is not None and ts < next_t:
                        continue  # sampled to the analysis rate
                    next_t = ts + period if next_t is None or ts - next_t > period else next_t + period
                    w, h = fit((frame.width, frame.height), detection_size)
                    image = frame.reformat(width=w, height=h, format="bgr24").to_ndarray()
                    tw, th = fit((w, h), TILE_MAX)
                    tile = image if (tw, th) == (w, h) else frame.reformat(width=tw, height=th, format="bgr24").to_ndarray()
                    sink.frame(DecodedFrame(image, tile, ts, time.time() * 1000.0))
                elif resampler is not None:
                    ts = clock(frame.time)
                    for out in resampler.resample(frame):
                        cutter.push(out.to_ndarray().astype("<i2").tobytes(), ts, sink)
                        ts += out.samples / AUDIO_RATE
        raise SourceError("Поток закончился")
    except SourceError:
        raise
    except Exception as exc:  # noqa: BLE001 - network drop, decode error
        if stop.is_set():
            return
        raise SourceError(mask_error(exc)) from exc
    finally:
        container.close()
