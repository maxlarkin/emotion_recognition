"""Inference worker: one spawn-context process that holds every model once and serves many sources.

Seam: source -> queue -> worker -> results.

* Parent side (`InferenceWorker`): register_source / submit_frame / submit_audio / results / stop.
  It never imports torch or touches CUDA; device selection happens inside the child.
  Code that starts a worker must run under `if __name__ == "__main__":` (spawn context).
* Child side: a receiver thread drains the inbound queue into per-source slots
  (video: size-1 slot, a newer frame overwrites the older one; audio: bounded FIFO), so a slow
  or silent source never blocks the others. The main thread takes everything pending, runs
  detector -> tracker -> FER (faces batched across sources) -> EMA -> fusion first (video
  latency has priority), then VAD -> SER (windows batched across sources), and publishes
  FrameResult / AudioResult objects. A failure is reported per source as an ErrorResult;
  the other sources of the same batch still get their results.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from app.config import MODEL_FILES, Settings, get_settings
from inference.detector import make_detector
from inference.shm_ring import FrameRef, FrameRing, RingReader
from inference.fer import CUDA_BATCH, FacialEmotion, crop_rgb
from inference.fusion import EMA_ALPHA, W_AUDIO, W_FACE, FusionWeights, TrackEMA, fuse
from inference.ser import MIN_SPEECH_RATIO, SpeechEmotion, SpeechWindow, Window
from inference.tracker import IoUTracker
from inference.types import (
    SAMPLE_RATE,
    AudioChunk,
    AudioResult,
    ErrorResult,
    FaceResult,
    FrameResult,
    VideoFrame,
    from_dict,
    to_dict,
    top_label,
)
from inference.vad import SileroVAD, VadStream

log = logging.getLogger(__name__)

Result = Union[FrameResult, AudioResult, ErrorResult]

SER_MAX_BATCH = 8  # speech windows per GigaAM forward pass
AUDIO_HISTORY = 16  # recent windows kept per source for media-time matching with frames
# A chunk starting this far (s) from where the previous one ended (a mic off->on gap, or ts going
# backwards) restarts the source's VAD stream and speech window: no stale audio in the next window.
AUDIO_GAP_S = 0.25


@dataclass
class WorkerConfig:
    device: str = "auto"  # auto | cpu | cuda
    ema_alpha: float = EMA_ALPHA
    w_face: float = W_FACE
    w_audio: float = W_AUDIO
    min_speech_ratio: float = MIN_SPEECH_RATIO
    # GigaAM encoder in fp16 + autocast, CUDA only. Measured on the RTX 3050: max abs deviation
    # from the CPU fp32 get_probs reference 2e-4 (fp32 on CUDA: 1e-5), within the 1e-2 gate.
    ser_fp16: bool = True
    audio_fifo_chunks: int = 32  # per-source audio FIFO (32 x 500 ms = 16 s)
    # Freshness is decided in media time (the caller's clock for both frame.ts and audio ts):
    # an audio window is used for a frame iff 0 <= frame.ts - window_end_ts <= audio_max_age_s.
    # The newest such window is attached to the FrameResult and, if it is speech, fused.
    # This stays correct for sources processed faster than real time (files) and for backlogs.
    audio_max_age_s: float = 2.0
    fer_cuda_batch: int = CUDA_BATCH  # fixed FER batch on CUDA (see inference.fer)
    inbound_maxsize: int = 256
    results_maxsize: int = 1024
    # Frames through per-source shared-memory rings (only a header crosses the queue). False sends
    # pickled frames, as before story 1.9.
    shm_frames: bool = field(default_factory=lambda: os.environ.get("SHM_FRAMES", "1") != "0")


# ------------------------------------------------------------------------------ device / files


def cuda_usable() -> bool:
    """Torch sees a GPU and ORT has the CUDA EP."""
    import onnxruntime
    import torch

    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    )


def select_device(requested: str, variant: str) -> str:
    """auto: CUDA only in the GPU image with CUDA usable, else CPU. cuda: fail if unusable."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unknown device {requested!r} (auto|cpu|cuda)")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_usable():
            raise RuntimeError("device 'cuda' requested, but CUDA is not usable (torch.cuda / CUDAExecutionProvider)")
        return "cuda"
    if variant != "gpu":
        return "cpu"  # CPU image: no probe
    try:
        return "cuda" if cuda_usable() else "cpu"
    except Exception as exc:  # noqa: BLE001 - a broken CUDA stack means "use the CPU"
        log.warning("CUDA probe failed, using CPU: %s", exc)
        return "cpu"


def check_model_files(settings: Settings) -> None:
    for key in MODEL_FILES:
        path = settings.model_path(key)
        if not path.is_file():
            raise FileNotFoundError(f"model file missing: {path} ({key}); MODELS_DIR={settings.models_dir}")


# ------------------------------------------------------------------------------ input validation


def validate_frame(image: Any) -> np.ndarray:
    """BGR uint8 [H, W, 3], non-empty; returns a private copy (mp.Queue pickles it later)."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"frame must be a numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"frame must be a non-empty uint8 HxWx3 BGR array, got {image.dtype} {image.shape}")
    return np.array(image, copy=True)


def validate_audio(samples: Any) -> np.ndarray:
    """PCM16 mono: little-endian bytes of even length, or a 1-D int16 array; returns a copy."""
    if isinstance(samples, (bytes, bytearray, memoryview)):
        raw = bytes(samples)
        if len(raw) % 2:
            raise ValueError(f"PCM16 audio bytes must have even length, got {len(raw)}")
        return np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if not isinstance(samples, np.ndarray):
        raise ValueError(f"audio must be bytes or a numpy array, got {type(samples).__name__}")
    if samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError(f"audio must be a 1-D int16 array (PCM16 mono 16 kHz), got {samples.dtype} {samples.shape}")
    return np.array(samples, copy=True)


# ------------------------------------------------------------------------------ per-source slots


class SourceSlots:
    """Thread-safe per-source inputs: video slot of size 1 (overwrite) and a bounded audio FIFO."""

    def __init__(self, audio_maxlen: int = 32):
        self.audio_maxlen = audio_maxlen
        self.lock = threading.Lock()
        self.event = threading.Event()  # set whenever something is pending
        self.frames: dict[str, Optional[Union[VideoFrame, FrameRef]]] = {}
        self.audio: dict[str, deque[AudioChunk]] = {}
        self.dropped_frames: dict[str, int] = {}
        self.dropped_audio: dict[str, int] = {}
        self.options: dict[str, dict] = {}  # per-source options given at registration (detect_every)
        # Ids unregistered since the last take(). A re-registered id stays here, so the engine
        # drops the old per-source state before it sees the new source's inputs.
        self.removed: set[str] = set()

    def register(self, source_id: str, options: Optional[dict] = None) -> None:
        with self.lock:
            self.options[source_id] = dict(options or {})
            if source_id not in self.frames:
                self.frames[source_id] = None
                self.audio[source_id] = deque(maxlen=self.audio_maxlen)
                self.dropped_frames[source_id] = 0
                self.dropped_audio[source_id] = 0

    def unregister(self, source_id: str) -> None:
        with self.lock:
            self.frames.pop(source_id, None)
            self.audio.pop(source_id, None)
            self.dropped_frames.pop(source_id, None)
            self.dropped_audio.pop(source_id, None)
            self.options.pop(source_id, None)
            self.removed.add(source_id)
        self.event.set()

    def registered(self, source_id: str) -> bool:
        with self.lock:
            return source_id in self.frames

    def put_frame(self, frame: Union[VideoFrame, FrameRef]) -> bool:
        with self.lock:
            if frame.source_id not in self.frames:
                return False
            if self.frames[frame.source_id] is not None:
                self.dropped_frames[frame.source_id] += 1  # stale frame dropped silently
            self.frames[frame.source_id] = frame
        self.event.set()
        return True

    def put_audio(self, chunk: AudioChunk) -> bool:
        with self.lock:
            fifo = self.audio.get(chunk.source_id)
            if fifo is None:
                return False
            if len(fifo) == fifo.maxlen:
                self.dropped_audio[chunk.source_id] += 1
            fifo.append(chunk)
        self.event.set()
        return True

    def take(self) -> tuple[list[Union[VideoFrame, FrameRef]], dict[str, list[AudioChunk]], set[str]]:
        with self.lock:
            frames = [f for f in self.frames.values() if f is not None]
            for sid in self.frames:
                self.frames[sid] = None
            audio = {sid: list(q) for sid, q in self.audio.items() if q}
            for q in self.audio.values():
                q.clear()
            removed, self.removed = self.removed, set()
            self.event.clear()
        return frames, audio, removed


# ------------------------------------------------------------------------------ engine (child)


class _SourceState:
    def __init__(self, ema_alpha: float):
        self.tracker = IoUTracker()
        self.ema = TrackEMA(ema_alpha)
        self.vad = VadStream()
        self.window = SpeechWindow()
        self.audio_history: deque[AudioResult] = deque(maxlen=AUDIO_HISTORY)  # by window_end_ts
        self.weights: Optional[FusionWeights] = None  # this source's own; None -> Engine.weights
        self.audio_next_ts: Optional[float] = None  # media time where the previous chunk ended
        # Detection cadence: detect on every Nth frame, carry the last boxes/ids in between.
        self.detect_every = 1
        self.frames_seen = 0
        self.last_boxes: Optional[list] = None
        self.last_ids: Optional[list] = None

    def reset_audio(self) -> None:
        self.vad = VadStream()
        self.window = SpeechWindow()


class Engine:
    """All models plus per-source state. Lives in the worker process (or a bench/test process)."""

    def __init__(self, settings: Settings, device: str, config: WorkerConfig):
        check_model_files(settings)
        self.device = device
        self.config = config
        self.weights = FusionWeights(config.w_face, config.w_audio)
        self.detector = make_detector(settings.model_path("yunet"), device)
        self.fer = FacialEmotion(settings.model_path("fer"), device=device, cuda_batch=config.fer_cuda_batch)
        self.vad = SileroVAD(settings.model_path("vad"))
        self.ser = SpeechEmotion(settings.model_path("ser"), device=device, fp16=config.ser_fp16)
        self.sources: dict[str, _SourceState] = {}

    def state(self, source_id: str) -> _SourceState:
        st = self.sources.get(source_id)
        if st is None:
            st = self.sources[source_id] = _SourceState(self.config.ema_alpha)
        return st

    def remove(self, source_id: str) -> None:
        self.sources.pop(source_id, None)

    def set_fusion(self, w_face: float, w_audio: float, source_id: Optional[str] = None) -> None:
        """Without source_id: the default weights (sources without their own). With it: that source only."""
        weights = FusionWeights(w_face, w_audio)  # validates
        if source_id is None:
            self.weights = weights
        else:
            self.state(source_id).weights = weights

    # ---- audio

    def _windows(self, sid: str, chunks: list[AudioChunk]) -> list[tuple[AudioResult, Window]]:
        st = self.state(sid)
        out = []
        for c in chunks:
            if st.audio_next_ts is not None and abs(c.ts - st.audio_next_ts) > AUDIO_GAP_S:
                st.reset_audio()  # discontinuity: drop pre-gap samples and the partial window
            st.audio_next_ts = c.ts + len(c.samples) / SAMPLE_RATE
            pend = st.vad.pending.shape[0]
            samples = np.asarray(c.samples, dtype=np.int16).astype(np.float32) / 32768.0
            for i, (chunk, prob) in enumerate(self.vad.feed(st.vad, samples)):
                end_ts = c.ts + ((i + 1) * chunk.shape[0] - pend) / SAMPLE_RATE
                w = st.window.push(chunk, prob, end_ts)
                if w is not None:
                    res = AudioResult(sid, w.speech_ratio >= self.config.min_speech_ratio, None, None, w.end_ts, w.speech_ratio)
                    out.append((res, w))
        return out

    def process_audio(self, audio: dict[str, list[AudioChunk]]) -> list[Union[AudioResult, ErrorResult]]:
        produced: list[Union[AudioResult, ErrorResult]] = []
        to_ser: list[tuple[AudioResult, Window]] = []
        for sid, chunks in audio.items():
            try:
                windows = self._windows(sid, chunks)
            except Exception as exc:  # noqa: BLE001 - isolate the failing source
                log.exception("audio failed for %s", sid)
                produced.append(ErrorResult(sid, "audio_failed", f"{type(exc).__name__}: {exc}"))
                continue
            speech = [(r, w) for r, w in windows if r.speech]
            for r, _ in speech[:-1]:  # backlog: SER only on the newest speech window
                r.skipped = True
            if speech:
                to_ser.append(speech[-1])
            produced.extend(r for r, _ in windows)

        failed: set[str] = set()
        for start in range(0, len(to_ser), SER_MAX_BATCH):
            part = to_ser[start : start + SER_MAX_BATCH]
            t0 = time.perf_counter()
            try:
                probs = self.ser.predict(np.stack([w.samples for _, w in part]))
            except Exception as exc:  # noqa: BLE001
                log.exception("SER batch failed")
                for r, _ in part:
                    failed.add(r.source_id)
                    produced.append(ErrorResult(r.source_id, "audio_failed", f"{type(exc).__name__}: {exc}"))
                continue
            ms = (time.perf_counter() - t0) * 1000.0
            for (r, _), p in zip(part, probs, strict=True):
                r.probs, r.label, r.ser_ms = to_dict(p), top_label(p), ms

        out: list[Union[AudioResult, ErrorResult]] = []
        for r in produced:
            if isinstance(r, AudioResult):
                if r.source_id in failed and r.speech and r.probs is None and not r.skipped:
                    continue  # its SER failed; reported as ErrorResult
                self.state(r.source_id).audio_history.append(r)
            out.append(r)
        return out

    def audio_for(self, st: _SourceState, frame_ts: float) -> Optional[AudioResult]:
        """Newest window with 0 <= frame_ts - window_end_ts <= audio_max_age_s (media time)."""
        best = None
        for r in st.audio_history:
            if r.speech and r.probs is None:
                continue  # skipped backlog window (SER ran only on a newer one): nothing to show or fuse
            age = frame_ts - r.window_end_ts
            if 0.0 <= age <= self.config.audio_max_age_s and (best is None or r.window_end_ts > best.window_end_ts):
                best = r
        return best

    # ---- video

    def process_frames(self, frames: list[VideoFrame]) -> list[Union[FrameResult, ErrorResult]]:
        if not frames:
            return []
        t_start = time.time()
        out: list[Union[FrameResult, ErrorResult]] = []
        per_frame = []
        crops: list[np.ndarray] = []
        # Detection runs serially: concurrent runs of one CUDA session failed in cuDNN (host allocation
        # errors) on the RTX 3050, and the worker died.
        for fr in frames:
            try:
                st = self.state(fr.source_id)
                t0 = time.perf_counter()
                detect = st.last_boxes is None or st.frames_seen % max(1, st.detect_every) == 0
                st.frames_seen += 1
                if detect:
                    boxes, det_own_ms = _timed_detect(self.detector, fr.image)
                    ids, expired = st.tracker.update(boxes, fr.ts)
                    st.last_boxes, st.last_ids = boxes, ids
                else:  # carried frame: last detection's boxes and ids, fresh crops for FER
                    boxes, ids, expired = st.last_boxes, st.last_ids, []
                    det_own_ms = 0.0
                det_ms = det_own_ms if detect else (time.perf_counter() - t0) * 1000.0
                frame_crops = [crop_rgb(fr.image, b) for b in boxes]
            except Exception as exc:  # noqa: BLE001 - isolate the failing source
                log.exception("frame failed for %s", fr.source_id)
                out.append(ErrorResult(fr.source_id, "frame_failed", f"{type(exc).__name__}: {exc}"))
                continue
            for tid in expired:
                st.ema.drop(tid)
            per_frame.append((fr, st, boxes, ids, len(crops), det_ms, detect))
            crops.extend(frame_crops)

        t0 = time.perf_counter()
        try:
            probs = self.fer.predict(crops)
        except Exception as exc:  # noqa: BLE001
            log.exception("FER batch failed")
            return out + [ErrorResult(fr.source_id, "frame_failed", f"{type(exc).__name__}: {exc}") for fr, *_ in per_frame]
        fer_ms = (time.perf_counter() - t0) * 1000.0

        for fr, st, boxes, ids, first, det_ms, detected in per_frame:
            audio = self.audio_for(st, fr.ts)
            audio_vec = from_dict(audio.probs) if audio is not None and audio.probs is not None else None
            faces = []
            for k, (box, tid) in enumerate(zip(boxes, ids, strict=True)):
                face = st.ema.update(tid, probs[first + k])
                fused = fuse(face, audio_vec, st.weights if st.weights is not None else self.weights)
                label = top_label(fused)
                faces.append(
                    FaceResult(
                        track_id=tid,
                        box=tuple(int(v) for v in box),  # type: ignore[arg-type]
                        face=to_dict(face),
                        face_label=top_label(face),
                        fused=to_dict(fused),
                        fused_label=label,
                        confidence=float(fused.max()),
                    )
                )
            timings = {
                "queue": max(0.0, (t_start - fr.submit_ts) * 1000.0) if fr.submit_ts else 0.0,
                "detect": det_ms,
                "detected": 1.0 if detected else 0.0,
                "fer": fer_ms,
                "batch_faces": float(len(crops)),
                "batch_frames": float(len(frames)),
                "total": (time.time() - t_start) * 1000.0,
            }
            out.append(FrameResult(fr.source_id, fr.frame_id, fr.ts, faces, audio, fr.submit_ts, timings))
        return out


# ------------------------------------------------------------------------------ child process


def _timed_detect(detector, image) -> tuple[list, float]:
    t0 = time.perf_counter()
    boxes = detector.detect(image)
    return boxes, (time.perf_counter() - t0) * 1000.0


def _put(outq, item) -> None:
    try:
        outq.put_nowait(item)
    except queue.Full:  # nobody is reading results; drop rather than stall inference
        pass


def _apply_controls(engine: Engine, slots: SourceSlots, removed: set[str], controls: deque, outq) -> None:
    """Drop removed sources first, then apply queued fusion configs (a config for a re-registered
    id belongs to the new source; one for a source that is gone is skipped, not resurrected)."""
    for sid in removed:
        engine.remove(sid)
    while controls:
        cfg = controls.popleft()
        sid = cfg.get("source_id")
        try:
            if sid is not None and not slots.registered(sid):
                continue
            engine.set_fusion(float(cfg["w_face"]), float(cfg["w_audio"]), sid)
        except Exception as exc:  # noqa: BLE001
            _put(outq, ErrorResult(None, "bad_config", f"{type(exc).__name__}: {exc}"))


class _LoopStats:
    """Logged every STATS_EVERY_S: where the worker's time goes (rounds, frames, detect, FER, audio)."""

    STATS_EVERY_S = 30.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.t0 = time.monotonic()
        self.rounds = self.frames = self.detected = 0
        self.read_ms = self.frames_ms = self.detect_ms = self.fer_ms = self.audio_ms = self.total_ms = 0.0

    def add(self, n_frames: int, results: list, read_ms: float, frames_ms: float, audio_ms: float, total_ms: float) -> None:
        if not n_frames and audio_ms < 1.0:
            return  # idle wake-up
        self.rounds += 1
        self.frames += n_frames
        self.read_ms += read_ms
        self.frames_ms += frames_ms
        self.audio_ms += audio_ms
        self.total_ms += total_ms
        fr = [r for r in results if isinstance(r, FrameResult)]
        self.detected += sum(int(r.timings.get("detected", 1)) for r in fr)
        self.detect_ms += sum(r.timings.get("detect", 0.0) for r in fr)
        if fr:
            self.fer_ms += fr[0].timings.get("fer", 0.0)
        elapsed = time.monotonic() - self.t0
        if elapsed >= self.STATS_EVERY_S:
            r = max(1, self.rounds)
            log.info(
                "loop: %.1f rounds/s, %.1f frames/round (%.0f%% detected); per round ms: total %.0f, shm read %.1f, "
                "frames %.0f (detect %.0f, fer %.0f), audio %.0f",
                self.rounds / elapsed, self.frames / r, 100.0 * self.detected / max(1, self.frames),
                self.total_ms / r, self.read_ms / r, self.frames_ms / r, self.detect_ms / r, self.fer_ms / r, self.audio_ms / r,
            )
            self.reset()


def _gpu_name() -> Optional[str]:
    """Shown in /api/health only: a failure here must not fail the worker start."""
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        log.warning("cannot read the GPU name", exc_info=True)
        return None


def _worker_main(inq, outq, config: WorkerConfig, settings: Settings) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s worker: %(message)s")
    try:
        device = select_device(config.device, settings.variant)
        engine = Engine(settings, device, config)
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which raises
        outq.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    gpu_name = _gpu_name() if device == "cuda" else None
    info = {"device": device, "gpu_name": gpu_name, "pid": os.getpid(), "ser_fp16": engine.ser.fp16}
    log.info("inference worker ready: %s", info)
    outq.put(("ready", info))

    slots = SourceSlots(config.audio_fifo_chunks)
    reader = RingReader()
    stop = threading.Event()
    controls: deque[dict] = deque()

    def handle(msg: Any) -> None:
        kind, payload = msg
        if kind in ("frame", "frame_shm"):
            if not slots.put_frame(payload):
                _put(outq, ErrorResult(payload.source_id, "unknown_source", f"unknown source {payload.source_id!r}"))
        elif kind == "audio":
            if not slots.put_audio(payload):
                _put(outq, ErrorResult(payload.source_id, "unknown_source", f"unknown source {payload.source_id!r}"))
        elif kind == "register":
            sid, options = payload if isinstance(payload, tuple) else (payload, {})
            slots.register(sid, options)
        elif kind == "unregister":
            slots.unregister(payload)
        elif kind == "config":
            controls.append(payload)
            slots.event.set()
        elif kind == "stop":
            stop.set()
        else:
            _put(outq, ErrorResult(None, "bad_message", f"unknown message kind {kind!r}"))

    def receiver() -> None:
        while not stop.is_set():
            try:
                msg = inq.get(timeout=0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError):
                stop.set()
                break
            except Exception as exc:  # noqa: BLE001 - e.g. an unpicklable payload
                log.exception("bad inbound message")
                _put(outq, ErrorResult(None, "bad_message", f"{type(exc).__name__}: {exc}"))
                continue
            try:
                handle(msg)
            except Exception as exc:  # noqa: BLE001 - malformed tuple / payload
                log.exception("bad inbound message")
                _put(outq, ErrorResult(None, "bad_message", f"{type(exc).__name__}: {exc}"))
        slots.event.set()

    rx = threading.Thread(target=receiver, name="receiver", daemon=True)
    rx.start()
    stats = _LoopStats()
    while not stop.is_set():
        slots.event.wait(0.1)
        t_round = time.perf_counter()
        taken, audio, removed = slots.take()
        _apply_controls(engine, slots, removed, controls, outq)
        for sid in removed:
            reader.forget(sid)
        t_read = time.perf_counter()
        frames = []
        for item in taken:  # resolve shared-memory refs: only the latest frame per source is copied
            if isinstance(item, FrameRef):
                image = reader.read(item)
                if image is None:
                    continue  # slot overwritten meanwhile: dropped like any stale frame
                item = VideoFrame(item.source_id, item.frame_id, item.ts, image, submit_ts=item.submit_ts)
            frames.append(item)
        for fr in frames:
            engine.state(fr.source_id).detect_every = int(slots.options.get(fr.source_id, {}).get("detect_every", 1))
        t_frames = time.perf_counter()
        results = engine.process_frames(frames)
        t_audio = time.perf_counter()
        for res in results:
            _put(outq, res)
        for res in engine.process_audio(audio):
            _put(outq, res)
        stats.add(len(frames), results, (t_frames - t_read) * 1000, (t_audio - t_frames) * 1000, (time.perf_counter() - t_audio) * 1000, (time.perf_counter() - t_round) * 1000)
    rx.join(timeout=1.0)
    reader.close()


# ------------------------------------------------------------------------------ parent handle


class WorkerStartError(RuntimeError):
    pass


class WorkerDied(RuntimeError):
    pass


class InferenceWorker:
    """Parent-side handle. Not thread-safe for concurrent start/stop; submit_* may be called from any thread."""

    def __init__(self, settings: Optional[Settings] = None, config: Optional[WorkerConfig] = None):
        self.settings = settings or get_settings()
        self.config = config or WorkerConfig()
        self._ctx = mp.get_context("spawn")
        self._inq: Any = None
        self._outq: Any = None
        self._proc: Optional[mp.process.BaseProcess] = None
        self._sources: set[str] = set()
        self._frame_ids: dict[str, itertools.count] = {}
        self._rings: dict[str, FrameRing] = {}  # parent-owned shared-memory rings, one per source
        self._lock = threading.Lock()
        self.info: dict[str, Any] = {}

    # ---- lifecycle

    def start(self, timeout: float = 600.0) -> "InferenceWorker":
        if self._proc is not None:
            raise RuntimeError("inference worker already started")
        self._inq = self._ctx.Queue(maxsize=self.config.inbound_maxsize)
        self._outq = self._ctx.Queue(maxsize=self.config.results_maxsize)
        self._sources.clear()
        self._frame_ids.clear()
        self.info = {}
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self._inq, self._outq, self.config, self.settings),
            name="inference-worker",
            daemon=True,
        )
        self._proc.start()
        deadline = time.monotonic() + timeout
        while True:
            try:
                kind, payload = self._outq.get(timeout=0.5)
                break
            except queue.Empty:
                if not self._proc.is_alive():
                    code = self._proc.exitcode
                    self._kill()
                    raise WorkerStartError(f"inference worker exited during start (code {code})") from None
                if time.monotonic() > deadline:
                    self._kill()
                    raise WorkerStartError(f"inference worker did not start within {timeout:.0f} s") from None
        if kind == "error":
            self._kill()
            raise WorkerStartError(f"inference worker failed to start: {payload}")
        self.info = payload
        return self

    @property
    def device(self) -> Optional[str]:
        return self.info.get("device")

    @property
    def alive(self) -> bool:
        proc = self._proc  # read once: a concurrent stop() sets it to None
        return proc is not None and proc.is_alive()

    def stop(self, timeout: float = 10.0) -> None:
        if self._proc is None:
            return
        if self._proc.is_alive():
            try:
                self._inq.put(("stop", None), timeout=1.0)
            except (queue.Full, ValueError, OSError):
                pass
            self._proc.join(timeout)
        self._kill()

    def _kill(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(5)
        for q in (self._inq, self._outq):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        with self._lock:
            rings, self._rings = list(self._rings.values()), {}
        for ring in rings:  # unlinked here: the worker never unlinks, even when it dies
            ring.close()
        self._proc = None
        self._inq = self._outq = None

    def __enter__(self) -> "InferenceWorker":
        return self.start() if self._proc is None else self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _require_running(self) -> None:
        if not self.alive:
            raise RuntimeError("inference worker is not running")

    def _send(self, msg: tuple, timeout: Optional[float] = None) -> None:
        """Put on the inbound queue: non-blocking without `timeout` (raises queue.Full).

        Raises RuntimeError if the worker is not running, also when a concurrent stop() closed the queue.
        """
        inq = self._inq  # read once: a concurrent stop() sets it to None
        if inq is None or not self.alive:
            raise RuntimeError("inference worker is not running")
        try:
            if timeout is None:
                inq.put_nowait(msg)
            else:
                inq.put(msg, timeout=timeout)
        except (ValueError, OSError) as exc:  # "Queue is closed"
            raise RuntimeError("inference worker is not running") from exc

    # ---- sources

    def register_source(self, source_id: str, detect_every: int = 1) -> None:
        """`detect_every`: run the face detector on every Nth frame of this source (carry boxes in between)."""
        self._require_running()
        with self._lock:
            self._sources.add(source_id)
            self._frame_ids[source_id] = itertools.count(1)
        self._send(("register", (source_id, {"detect_every": max(1, int(detect_every))})), timeout=5.0)

    def unregister_source(self, source_id: str) -> None:
        self._require_running()
        with self._lock:
            self._sources.discard(source_id)
            self._frame_ids.pop(source_id, None)
            ring = self._rings.pop(source_id, None)
        try:
            self._send(("unregister", source_id), timeout=5.0)
        finally:
            if ring is not None:
                ring.close()  # the worker keeps its mapping until it forgets the source

    def _check(self, source_id: str) -> None:
        if source_id not in self._sources:
            raise KeyError(f"unknown source {source_id!r}: call register_source() first")

    def submit_frame(
        self, source_id: str, image: np.ndarray, frame_id: Optional[int] = None, ts: Optional[float] = None
    ) -> bool:
        """Hand a BGR uint8 HxWx3 frame to the worker without blocking.

        Raises KeyError (unknown source), ValueError (bad frame), RuntimeError (worker not running).
        Returns False if the inbound queue is full (frame dropped).
        """
        self._check(source_id)
        image = validate_frame(image)
        self._require_running()
        if frame_id is None:
            frame_id = next(self._frame_ids[source_id])
        now = time.time()
        ts = now if ts is None else float(ts)
        if self.config.shm_frames:
            with self._lock:
                ring = self._rings.get(source_id)
                if ring is None:
                    ring = self._rings[source_id] = FrameRing()
            segment, slot, seq, shape = ring.write(image)
            msg: tuple = ("frame_shm", FrameRef(source_id, int(frame_id), ts, now, segment, slot, seq, shape))
        else:
            msg = ("frame", VideoFrame(source_id, int(frame_id), ts, image, submit_ts=now))
        try:
            self._send(msg)
            return True
        except queue.Full:
            return False

    def submit_audio(self, source_id: str, samples: Union[np.ndarray, bytes], ts: Optional[float] = None) -> bool:
        """PCM16 mono 16 kHz (1-D int16 array or little-endian bytes of even length).

        Raises KeyError / ValueError / RuntimeError like submit_frame; False if the queue is full.
        """
        self._check(source_id)
        pcm = validate_audio(samples)
        self._require_running()
        chunk = AudioChunk(source_id, time.time() if ts is None else float(ts), pcm)
        try:
            self._send(("audio", chunk))
            return True
        except queue.Full:
            return False

    def set_fusion(self, w_face: float, w_audio: float, source_id: Optional[str] = None) -> None:
        """Fusion weights: the default for every source, or only `source_id`'s when given.

        Invalid weights come back as ErrorResult(None, "bad_config"); the weights stay unchanged.
        """
        self._require_running()
        if source_id is not None:
            self._check(source_id)
        self._send(("config", {"w_face": w_face, "w_audio": w_audio, "source_id": source_id}), timeout=5.0)

    def results(self, timeout: float = 0.0, max_items: int = 10_000) -> list[Result]:
        """Drain available results; waits up to `timeout` seconds for the first one.

        Returns [] if the worker was never started or was stopped; raises WorkerDied if the
        process exited on its own.
        """
        if self._proc is None:
            return []
        if not self._proc.is_alive():
            code = self._proc.exitcode
            self._kill()
            raise WorkerDied(f"inference worker exited unexpectedly (code {code})")
        out: list[Result] = []
        try:
            out.append(self._outq.get(timeout=timeout) if timeout > 0 else self._outq.get_nowait())
        except queue.Empty:
            return out
        while len(out) < max_items:
            try:
                out.append(self._outq.get_nowait())
            except queue.Empty:
                break
        return out

