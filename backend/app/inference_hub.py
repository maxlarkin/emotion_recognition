"""InferenceHub: owns the single InferenceWorker of this backend process.

* `start()` returns at once; a background thread spawns the worker and waits for the models,
  so /api/health and the UI stay reachable while models load. State: starting | ready | failed.
* One drain thread reads `worker.results()` and routes every result by `source_id` to the
  asyncio queue of the session that registered that source (`loop.call_soon_threadsafe`).
  If the worker dies, every session receives `WorkerGone` and the hub becomes `failed`.
* `WORKER_FACTORY` is the test hook: tests replace it with a fake worker.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

STARTING, READY, FAILED = "starting", "ready", "failed"


def _default_factory() -> Any:
    from inference.worker import InferenceWorker

    return InferenceWorker()


# Replaced by tests (a fake worker); called once per hub start, in the start thread.
WORKER_FACTORY: Callable[[], Any] = _default_factory

@dataclass(frozen=True)
class WorkerGone:
    """Put on a session queue when the worker process exits."""

    source_id: str


class HubUnavailable(RuntimeError):
    def __init__(self, state: str):
        super().__init__(state)
        self.state = state


@dataclass
class _Route:
    loop: asyncio.AbstractEventLoop
    queue: asyncio.Queue


class InferenceHub:
    def __init__(self, factory: Optional[Callable[[], Any]] = None):
        self._factory = factory
        self.state = STARTING
        self.error: Optional[str] = None
        self._worker: Any = None
        self._routes: dict[str, _Route] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._start_thread: Optional[threading.Thread] = None
        self._drain_thread: Optional[threading.Thread] = None

    # ---- lifecycle

    def start(self) -> None:
        self._start_thread = threading.Thread(target=self._start_worker, name="inference-start", daemon=True)
        self._start_thread.start()

    def _start_worker(self) -> None:
        try:
            worker = (self._factory or WORKER_FACTORY)()
            worker.start()
        except Exception as exc:  # noqa: BLE001 - reported through the state
            log.error("inference worker failed to start: %s", exc)
            self.error = f"{type(exc).__name__}: {exc}"
            self.state = FAILED
            return
        with self._lock:  # stop() reads _worker/_drain_thread under the same lock
            abandoned = self._stopping.is_set()  # shutdown arrived while the models were loading
            if not abandoned:
                self._worker = worker
                self._drain_thread = threading.Thread(target=self._drain, name="inference-drain", daemon=True)
                self._drain_thread.start()
                self.state = READY
        if abandoned:
            worker.stop()
            return
        log.info("inference worker ready: %s", getattr(worker, "info", {}))

    def stop(self) -> None:
        """Blocking; call from a thread. A start still loading models is abandoned (daemon process)."""
        with self._lock:
            self._stopping.set()
            drain = self._drain_thread
        if drain is not None:
            drain.join(timeout=5.0)
        with self._lock:
            worker, self._worker = self._worker, None
        if worker is not None:
            try:
                worker.stop()
            except Exception:  # noqa: BLE001
                log.exception("inference worker stop failed")
        with self._lock:
            self._routes.clear()

    def wait_ready(self, timeout: float) -> bool:
        """Test/bench helper: wait until the start thread finished (ready or failed)."""
        if self._start_thread is not None:
            self._start_thread.join(timeout)
        return self.state == READY

    # ---- results

    def _drain(self) -> None:
        from inference.worker import WorkerDied

        while not self._stopping.is_set():
            worker = self._worker
            if worker is None:
                return
            try:
                results = worker.results(timeout=0.2)
            except WorkerDied as exc:
                log.error("inference worker died: %s", exc)
                self.error = str(exc)
                self.state = FAILED
                self._worker = None
                with self._lock:
                    routes, self._routes = list(self._routes.items()), {}
                for sid, route in routes:
                    self._deliver(route, WorkerGone(sid))
                return
            except Exception:  # noqa: BLE001 - keep draining
                log.exception("reading inference results failed")
                self._stopping.wait(0.5)
                continue
            for res in results:
                sid = getattr(res, "source_id", None)
                with self._lock:
                    route = self._routes.get(sid) if sid is not None else None
                if route is None:
                    if sid is None:
                        log.warning("worker error: %s", res)
                    continue  # result of an unregistered source
                self._deliver(route, res)

    @staticmethod
    def _deliver(route: _Route, item: Any) -> None:
        try:
            route.loop.call_soon_threadsafe(route.queue.put_nowait, item)
        except RuntimeError:  # the session's loop is closed
            pass

    # ---- sources (thread-safe; may block briefly on the worker queue, so call them off the loop)

    def register(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> str:
        """Create a source whose results go to `queue` on `loop`; raises HubUnavailable unless ready."""
        worker = self._worker
        if self.state != READY or worker is None:
            raise HubUnavailable(self.state)
        source_id = uuid.uuid4().hex
        with self._lock:
            self._routes[source_id] = _Route(loop, queue)
        try:
            worker.register_source(source_id)
        except Exception as exc:  # noqa: BLE001 - worker gone
            log.warning("register_source failed: %s", exc)
            with self._lock:
                self._routes.pop(source_id, None)
            raise HubUnavailable(FAILED) from exc
        return source_id

    def unregister(self, source_id: str) -> None:
        with self._lock:
            known = self._routes.pop(source_id, None) is not None
        worker = self._worker
        if known and worker is not None:
            try:
                worker.unregister_source(source_id)
            except Exception as exc:  # noqa: BLE001 - worker gone: nothing to clean up
                log.info("unregister %s: %s", source_id, exc)

    def submit_frame(self, source_id: str, bgr: np.ndarray, frame_id: int, ts: float) -> bool:
        """Thread-safe. False if the frame was dropped (inbound queue full).

        Raises HubUnavailable if the worker is gone.
        """
        worker = self._worker
        if worker is None:
            raise HubUnavailable(self.state)
        try:
            return bool(worker.submit_frame(source_id, bgr, frame_id=frame_id, ts=ts))
        except (RuntimeError, ValueError, AttributeError, OSError) as exc:  # worker not running / queue closed
            raise HubUnavailable(FAILED) from exc

    def submit_audio(self, source_id: str, pcm: bytes, ts: float) -> bool:
        """Thread-safe. PCM16 LE mono 16 kHz, `ts` in seconds (first sample, same clock as frames).

        False if the chunk was dropped (inbound queue full). Raises HubUnavailable if the worker is gone.
        """
        worker = self._worker
        if worker is None:
            raise HubUnavailable(self.state)
        try:
            return bool(worker.submit_audio(source_id, pcm, ts=ts))
        except (RuntimeError, ValueError, AttributeError, OSError) as exc:  # worker not running / queue closed
            raise HubUnavailable(FAILED) from exc

    def set_fusion(self, source_id: str, w_face: float, w_audio: float) -> bool:
        """Fusion weights for this source only (validated by the caller). May block briefly.

        False if the worker's inbound queue stayed full. Raises HubUnavailable if the worker is gone.
        """
        worker = self._worker
        if worker is None:
            raise HubUnavailable(self.state)
        try:
            worker.set_fusion(w_face, w_audio, source_id=source_id)
            return True
        except queue.Full:
            return False
        except (RuntimeError, ValueError, AttributeError, OSError) as exc:  # worker not running / queue closed
            raise HubUnavailable(FAILED) from exc
