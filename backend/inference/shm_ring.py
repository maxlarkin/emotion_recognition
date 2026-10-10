"""Shared-memory frame rings (story 1.9): frames reach the worker without pickling the pixels.

The parent owns one `FrameRing` per source: a shared-memory segment with `SLOTS` slots and a small
header of per-slot sequence numbers (a seqlock). `write()` marks the slot odd, copies the frame,
marks it even again and returns a `FrameRef` (segment name, slot, sequence, shape) that crosses the
queue instead of the image. The worker's `RingReader.read()` copies the slot out and checks the
sequence before and after the copy; if the parent wrapped the ring meanwhile, the frame is dropped,
never returned torn. Only the parent unlinks segments (on close, unregister and worker exit).

The seqlock relies on numpy stores becoming visible in program order, which holds on x86-64 (the
target hardware). On weakly ordered CPUs (ARM) a torn read could pass the check.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SLOTS = 4
HEADER_BYTES = 64  # SLOTS x int64 sequence numbers, padded


@dataclass(frozen=True)
class FrameRef:
    """A frame waiting in a ring slot (sent instead of VideoFrame)."""

    source_id: str
    frame_id: int
    ts: float
    submit_ts: float
    segment: str
    slot: int
    seq: int
    shape: tuple[int, int, int]


class FrameRing:
    """Parent side. Writes and close are serialised by a lock: a restarted decode thread may overlap
    the old one for a moment, and close may come from another thread (unregister, stop)."""

    def __init__(self, slots: int = SLOTS):
        assert slots * 8 <= HEADER_BYTES, "header holds one int64 sequence per slot"
        self.slots = slots
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.capacity = 0
        self.next_slot = 0
        self.closed = False
        self._lock = threading.Lock()

    def _ensure(self, nbytes: int) -> None:
        if self.shm is not None and nbytes <= self.capacity:
            return
        self._release()
        self.capacity = nbytes
        self.shm = shared_memory.SharedMemory(create=True, size=HEADER_BYTES + self.slots * nbytes)
        self._seqs()[:] = 0

    def _seqs(self) -> np.ndarray:
        assert self.shm is not None
        return np.ndarray((self.slots,), dtype=np.int64, buffer=self.shm.buf[:HEADER_BYTES])

    def write(self, image: np.ndarray) -> Optional[tuple[str, int, int, tuple[int, int, int]]]:
        """Copy `image` (uint8 HxWx3) into the next slot; returns (segment, slot, seq, shape), or None
        once the ring is closed (the caller sends the frame another way or drops it)."""
        with self._lock:
            if self.closed:
                return None
            return self._write(image)

    def _write(self, image: np.ndarray) -> tuple[str, int, int, tuple[int, int, int]]:
        image = np.ascontiguousarray(image, dtype=np.uint8)
        self._ensure(image.nbytes)
        assert self.shm is not None
        slot = self.next_slot
        self.next_slot = (slot + 1) % self.slots
        seqs = self._seqs()
        seq = int(seqs[slot]) + 1
        seqs[slot] = seq  # odd: being written
        start = HEADER_BYTES + slot * self.capacity
        np.ndarray(image.shape, dtype=np.uint8, buffer=self.shm.buf[start : start + image.nbytes])[...] = image
        seqs[slot] = seq + 1  # even: stable
        return self.shm.name, slot, seq + 1, tuple(image.shape)  # type: ignore[return-value]

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self._release()

    def _release(self) -> None:
        shm, self.shm = self.shm, None
        if shm is not None:
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass


class RingReader:
    """Worker side: attaches to segments by name (never unlinks them) and copies frames out."""

    def __init__(self) -> None:
        self._segments: dict[str, shared_memory.SharedMemory] = {}
        self._by_source: dict[str, str] = {}
        self.dropped = 0

    def _attach(self, source_id: str, name: str) -> Optional[shared_memory.SharedMemory]:
        shm = self._segments.get(name)
        if shm is None:
            try:
                shm = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:  # already unlinked (source gone or ring resized)
                return None
            # Attaching registers the name with the resource tracker. A spawned worker shares the
            # parent's tracker, where the name is already registered (a set): nothing changes, and
            # the parent's unlink unregisters it. Unregistering here would drop the parent's entry.
            self._segments[name] = shm
        old = self._by_source.get(source_id)
        if old is not None and old != name:
            self._close(old)  # the ring was resized: the old segment is no longer written
        self._by_source[source_id] = name
        return shm

    def read(self, ref: FrameRef) -> Optional[np.ndarray]:
        """A private copy of the frame, or None if the slot was overwritten (counted in `dropped`)."""
        shm = self._attach(ref.source_id, ref.segment)
        if shm is None:
            self.dropped += 1
            return None
        h, w, c = ref.shape
        nbytes = h * w * c
        capacity = (shm.size - HEADER_BYTES) // SLOTS
        seqs = np.ndarray((SLOTS,), dtype=np.int64, buffer=shm.buf[:HEADER_BYTES])
        if int(seqs[ref.slot]) != ref.seq or nbytes > capacity:
            self.dropped += 1
            return None
        start = HEADER_BYTES + ref.slot * capacity
        image = np.ndarray(ref.shape, dtype=np.uint8, buffer=shm.buf[start : start + nbytes]).copy()
        if int(seqs[ref.slot]) != ref.seq:  # written while we copied
            self.dropped += 1
            return None
        return image

    def forget(self, source_id: str) -> None:
        name = self._by_source.pop(source_id, None)
        if name is not None:
            self._close(name)

    def _close(self, name: str) -> None:
        shm = self._segments.pop(name, None)
        if shm is not None:
            try:
                shm.close()
            except BufferError:  # a view is still alive; the mapping goes away with the process
                pass

    def close(self) -> None:
        for name in list(self._segments):
            self._close(name)
        self._by_source.clear()
