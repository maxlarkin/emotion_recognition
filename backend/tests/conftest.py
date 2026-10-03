"""Shared fixtures. Every test runs with a fake inference worker (no model processes are spawned),
except tests marked `models`, which use the real one."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import pytest

from inference.types import EMOTIONS, AudioResult, FaceResult, FrameResult


def one_face(image: np.ndarray) -> list[FaceResult]:
    h, w = image.shape[:2]
    face = {k: 0.0 for k in EMOTIONS}
    face["happiness"] = 0.81
    face["neutral"] = 0.19
    return [FaceResult(1, (w // 4, h // 4, w // 2, h // 2), face, "happiness", dict(face), "happiness", 0.81)]


class FakeWorker:
    """Parent-side InferenceWorker stand-in with the same surface.

    auto=True answers every frame at once; auto=False keeps a size-1 slot per source (a newer
    frame overwrites the older one, like the real worker) until `process()` is called.
    `audio[sid]` (an AudioResult) is attached to that source's frames under the real worker's
    media-time rule (0 <= frame ts - window_end_ts <= 2 s) and fused like the real engine, with
    the weights set per source by `set_fusion` (default 0.7 / 0.3). `fusion_queue_full` makes
    `set_fusion` raise queue.Full like a full inbound queue.
    """

    def __init__(self) -> None:
        self.auto = True
        self.faces_fn: Callable[[np.ndarray], list[FaceResult]] = one_face
        self.start_error: Optional[Exception] = None
        self.start_gate: Optional[threading.Event] = None
        self.sources: set[str] = set()
        self.registered: list[str] = []
        self.unregistered: list[str] = []
        self.submitted: list[tuple[str, int, float, tuple]] = []  # source, frame_id, ts, shape
        self.audio_submitted: list[tuple[str, int, float]] = []  # source, PCM byte count, ts (s)
        self.audio: dict[str, AudioResult] = {}
        self.fusion: dict[Optional[str], tuple[float, float]] = {}  # source_id (None = default) -> weights
        self.fusion_calls: list[tuple[float, float, Optional[str]]] = []
        self.fusion_queue_full = False
        self.slots: dict[str, tuple] = {}
        self.out: queue.Queue = queue.Queue()
        self.dead = False
        self.started = False
        self.stopped = False
        self.info = {"device": "cpu", "pid": 0}
        self.lock = threading.Lock()

    def start(self) -> "FakeWorker":
        if self.start_gate is not None:
            self.start_gate.wait()
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        return self

    def stop(self) -> None:
        self.stopped = True

    def register_source(self, sid: str) -> None:
        if self.dead:
            raise RuntimeError("inference worker is not running")
        with self.lock:
            self.sources.add(sid)
            self.registered.append(sid)

    def unregister_source(self, sid: str) -> None:
        if self.dead:
            raise RuntimeError("inference worker is not running")
        with self.lock:
            self.sources.discard(sid)
            self.slots.pop(sid, None)
            self.unregistered.append(sid)

    def submit_frame(self, sid: str, image: np.ndarray, frame_id=None, ts=None) -> bool:
        if sid not in self.sources:
            raise KeyError(sid)
        if self.dead:
            raise RuntimeError("inference worker is not running")
        with self.lock:
            self.submitted.append((sid, frame_id, ts, image.shape))
            item = (sid, frame_id, ts, image)
            if self.auto:
                self._emit(item)
            else:
                self.slots[sid] = item
        return True

    def submit_audio(self, sid: str, samples, ts=None) -> bool:
        if sid not in self.sources:
            raise KeyError(sid)
        if self.dead:
            raise RuntimeError("inference worker is not running")
        with self.lock:
            self.audio_submitted.append((sid, len(samples), ts))
        return True

    def set_fusion(self, w_face: float, w_audio: float, source_id: Optional[str] = None) -> None:
        if self.dead:
            raise RuntimeError("inference worker is not running")
        if source_id is not None and source_id not in self.sources:
            raise KeyError(source_id)
        if self.fusion_queue_full:
            raise queue.Full
        with self.lock:
            self.fusion_calls.append((w_face, w_audio, source_id))
            self.fusion[source_id] = (w_face, w_audio)

    def _emit(self, item: tuple) -> None:
        from inference.fusion import FusionWeights, fuse
        from inference.types import from_dict, to_dict, top_label

        sid, frame_id, ts, image = item
        audio = self.audio.get(sid)
        if audio is not None and not 0.0 <= ts - audio.window_end_ts <= 2.0:
            audio = None  # media-time rule of Engine.audio_for (audio_max_age_s = 2)
        weights = FusionWeights(*self.fusion.get(sid, self.fusion.get(None, (0.7, 0.3))))
        vec = from_dict(audio.probs) if audio is not None and audio.probs is not None else None
        faces = []
        for f in self.faces_fn(image):
            fused = fuse(from_dict(f.face), vec, weights)
            faces.append(FaceResult(f.track_id, f.box, f.face, f.face_label, to_dict(fused), top_label(fused), float(fused.max())))
        self.out.put(FrameResult(sid, frame_id, ts, faces, audio, time.time(), {"total": 1.0}))

    def process(self) -> None:
        with self.lock:
            items, self.slots = list(self.slots.values()), {}
            for item in items:
                self._emit(item)

    def push(self, result) -> None:
        """Emit an arbitrary worker result (e.g. an ErrorResult)."""
        self.out.put(result)

    def die(self) -> None:
        self.dead = True

    def results(self, timeout: float = 0.0, max_items: int = 10_000) -> list:
        from inference.worker import WorkerDied

        if self.dead:
            raise WorkerDied("inference worker exited unexpectedly (code -9)")
        out = []
        try:
            out.append(self.out.get(timeout=timeout) if timeout > 0 else self.out.get_nowait())
        except queue.Empty:
            return out
        while len(out) < max_items:
            try:
                out.append(self.out.get_nowait())
            except queue.Empty:
                break
        return out


@pytest.fixture(autouse=True)
def fake_worker(request, monkeypatch) -> Optional[FakeWorker]:
    if request.node.get_closest_marker("models"):
        yield None
        return
    import app.inference_hub as hub_mod

    fake = FakeWorker()
    monkeypatch.setattr(hub_mod, "WORKER_FACTORY", lambda: fake)
    yield fake
    if fake.start_gate is not None:
        fake.start_gate.set()  # release an abandoned start thread
