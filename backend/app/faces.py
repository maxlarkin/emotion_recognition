"""Face objects on the wire (interfaces.md): `decision` and `face_px` on top of the worker's FaceResult.

Story 1.7 decides only `abstain` (face narrower than the source's minimum, or a track younger than
1 s) and `none`; watch/alert/positive come with the event engine (story 1.10).
"""

from __future__ import annotations

from typing import Optional

from inference.types import FaceResult

MIN_TRACK_AGE_S = 1.0
WEBCAM_MIN_FACE_PX = 48


FORGET_AFTER_S = 1.0  # a track missing from results this long starts over (a single missed detection does not)


class TrackAges:
    """First-seen time per track of one source."""

    def __init__(self) -> None:
        self._first: dict[int, float] = {}
        self._last: dict[int, float] = {}

    def update(self, track_ids: list[int], ts: float) -> dict[int, float]:
        """Ages (s) of the given tracks at `ts` (seconds, the source's clock)."""
        for t in [t for t, seen in self._last.items() if ts - seen > FORGET_AFTER_S and t not in track_ids]:
            self._first.pop(t, None)
            self._last.pop(t, None)
        for t in track_ids:
            self._first.setdefault(t, ts)
            self._last[t] = ts
        return {t: max(0.0, ts - self._first[t]) for t in track_ids}

    def clear(self) -> None:
        self._first.clear()
        self._last.clear()


def decide(face_px: int, age_s: float, min_face_px: int) -> str:
    return "abstain" if face_px < min_face_px or age_s < MIN_TRACK_AGE_S else "none"


def face_to_wire(face: FaceResult, age_s: float = MIN_TRACK_AGE_S, min_face_px: int = WEBCAM_MIN_FACE_PX, scale: Optional[float] = None) -> dict:
    """`scale` rescales the box from detection pixels to the tile JPEG (streams); `face_px` stays in detection pixels."""
    face_px = int(face.box[2])
    box = [int(v) for v in face.box] if scale is None else [int(round(v * scale)) for v in face.box]
    return {
        "track_id": face.track_id,
        "box": box,
        "face": face.face,
        "face_label": face.face_label,
        "fused": face.fused,
        "fused_label": face.fused_label,
        "confidence": face.confidence,
        "decision": decide(face_px, age_s, min_face_px),
        "face_px": face_px,
    }
