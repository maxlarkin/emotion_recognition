"""Export rows -> CSV / JSON (interfaces.md, Export row)."""

from __future__ import annotations

import csv
import io
import json
from typing import Iterable

from inference.types import EMOTIONS

EXPORT_COLUMNS: tuple[str, ...] = (
    "session_id", "stream_id", "ts_ms", "source", "track_id", "box_x", "box_y", "box_w", "box_h",
    *(f"p_{k}" for k in EMOTIONS),
    "face_label", "audio_label", "fused_label", "speech",
)


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)  # floats: repr-precision, re-imported exactly


def to_csv(rows: Iterable[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(EXPORT_COLUMNS)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in EXPORT_COLUMNS])
    return buf.getvalue()


def to_json(rows: Iterable[dict]) -> str:
    return json.dumps([{c: r.get(c) for c in EXPORT_COLUMNS} for r in rows], ensure_ascii=False)
