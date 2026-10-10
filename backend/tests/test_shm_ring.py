"""Shared-memory frame rings: write/read, resize, overwrite detection, cleanup."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inference.shm_ring import SLOTS, FrameRef, FrameRing, RingReader


def ref(sid, write_result, fid=1):
    seg, slot, seq, shape = write_result
    return FrameRef(sid, fid, 1.0, 1.0, seg, slot, seq, shape)


def frame(v, h=48, w=64):
    return np.full((h, w, 3), v, np.uint8)


def segment_exists(name: str) -> bool:
    return Path("/dev/shm", name.lstrip("/")).exists()


def test_write_then_read_returns_a_private_copy():
    ring, reader = FrameRing(), RingReader()
    try:
        r = ref("s", ring.write(frame(7)))
        img = reader.read(r)
        assert img is not None and img.shape == (48, 64, 3) and int(img[0, 0, 0]) == 7
        ring.write(frame(9))  # next slot; our copy is unaffected
        assert int(img[0, 0, 0]) == 7
    finally:
        reader.close()
        ring.close()


def test_overwritten_slot_is_dropped_not_torn():
    ring, reader = FrameRing(), RingReader()
    try:
        first = ref("s", ring.write(frame(1)))
        for v in range(2, SLOTS + 2):  # wrap the ring: slot of `first` rewritten
            ring.write(frame(v))
        assert reader.read(first) is None and reader.dropped == 1
    finally:
        reader.close()
        ring.close()


def test_resize_moves_to_a_new_segment_and_unlinks_the_old():
    ring, reader = FrameRing(), RingReader()
    try:
        small = ring.write(frame(1))
        old = small[0]
        assert reader.read(ref("s", small)) is not None
        big = ring.write(frame(2, 96, 128))
        assert big[0] != old and not segment_exists(old)
        img = reader.read(ref("s", big))
        assert img is not None and img.shape == (96, 128, 3)
    finally:
        reader.close()
        ring.close()


def test_close_unlinks_and_late_reads_drop():
    ring, reader = FrameRing(), RingReader()
    r = ref("s", ring.write(frame(3)))
    ring.close()
    assert not segment_exists(r.segment)
    assert reader.read(r) is None
    reader.close()


def test_reader_never_unlinks():
    ring, reader = FrameRing(), RingReader()
    try:
        r = ref("s", ring.write(frame(4)))
        assert reader.read(r) is not None
        reader.forget("s")
        reader.close()
        assert segment_exists(r.segment)  # still the parent's
    finally:
        ring.close()


def test_write_after_close_returns_none_and_creates_nothing():
    ring = FrameRing()
    ring.write(frame(1))
    name = ring.shm.name
    ring.close()
    assert ring.write(frame(2)) is None and ring.shm is None and not segment_exists(name)
