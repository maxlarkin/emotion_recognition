"""Shared fixtures. Every test runs with a fake inference worker (no model processes are spawned),
except tests marked `models`, which use the real one."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import pytest

from inference.types import EMOTIONS, FaceResult, FrameResult


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

    def _emit(self, item: tuple) -> None:
        sid, frame_id, ts, image = item
        self.out.put(FrameResult(sid, frame_id, ts, self.faces_fn(image), None, time.time(), {"total": 1.0}))

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
