"""Per-source greedy IoU tracker: stable track ids, tracks expire after max_age seconds unseen."""

from __future__ import annotations

from dataclasses import dataclass

IOU_THRESHOLD = 0.3
MAX_AGE_S = 1.0

Box = tuple[int, int, int, int]  # x, y, w, h


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: Box
    last_seen: float


class IoUTracker:
    def __init__(self, iou_threshold: float = IOU_THRESHOLD, max_age_s: float = MAX_AGE_S):
        self.iou_threshold = iou_threshold
        self.max_age_s = max_age_s
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, boxes: list[Box], ts: float) -> tuple[list[int], list[int]]:
        """Assign a track id to every box (same order). Returns (ids, expired_track_ids)."""
        expired = [tid for tid, t in self.tracks.items() if ts - t.last_seen > self.max_age_s]
        for tid in expired:
            del self.tracks[tid]

        pairs = sorted(
            ((iou(b, t.box), i, tid) for i, b in enumerate(boxes) for tid, t in self.tracks.items()),
            reverse=True,
        )
        ids: list[int | None] = [None] * len(boxes)
        used: set[int] = set()
        for score, i, tid in pairs:
            if score < self.iou_threshold:
                break
            if ids[i] is not None or tid in used:
                continue
            ids[i] = tid
            used.add(tid)
        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self._next_id
                self._next_id += 1
            tid = ids[i]
            self.tracks[tid] = Track(tid, b, ts)
        return [int(t) for t in ids], expired  # type: ignore[arg-type]
