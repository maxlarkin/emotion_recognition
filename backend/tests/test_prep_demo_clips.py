"""Pure parts of scripts/prep_demo_clips.py (no models, no network)."""

from __future__ import annotations

import numpy as np
import pytest

import hashlib
import json
from pathlib import Path

import scripts.prep_demo_clips as prep
from scripts.prep_demo_clips import (
    PROFILES,
    Candidate,
    Window,
    apply_geometry,
    candidates_of,
    check_stream_list,
    encode,
    ensure_source,
    find_windows,
    fit_geometry,
    in_band,
    merge_windows,
    plan_mix,
    windows_of,
    write_report,
)

REPO = Path(__file__).resolve().parents[2]


def series(values, fps=5):
    return [(i / fps, v) for i, v in enumerate(values)]


def test_find_windows_needs_two_held_seconds():
    assert find_windows(1, series([0.7] * 10), "negative") == []  # 1.8 s
    (w,) = find_windows(1, series([0.1] * 3 + [0.7] * 11 + [0.9] + [0.2] * 3), "negative")
    assert (w.start, w.end, w.peak, w.kind) == (0.6, 2.8, 0.9, "negative")
    assert w.duration == pytest.approx(2.2)


def test_find_windows_splits_on_a_dip_and_tolerates_short_gaps():
    ws = find_windows(2, series([0.8] * 12 + [0.1] + [0.8] * 12), "positive")
    assert len(ws) == 2
    gap = [(0.0, 0.8), (1.0, 0.8), (1.4, 0.8), (2.2, 0.8), (2.4, 0.8)]  # 0.8 s gap breaks the run
    assert find_windows(3, gap, "positive") == []


def face(track, w, neg=0.0, happy=0.0):
    fused = {k: 0.0 for k in ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")}
    fused["anger"], fused["happiness"] = neg, happy
    fused["neutral"] = max(0.0, 1 - neg - happy)
    return {"track": track, "box": [100, 50, w, w], "fused": fused}


def analysis(n=500, neg_from=50, neg_to=70):
    frames = []
    for i in range(n):
        t = i / 5
        neg = 0.8 if neg_from <= i < neg_to else 0.1
        frames.append({"t": t, "faces": [face(1, 120, neg=neg)] if i % 10 else [], "speech": False})
    return {"size": [854, 480], "has_audio": True, "frames": frames}


def test_windows_and_candidates_from_analysis():
    a = analysis()
    ws = windows_of(a)
    assert [w.kind for w in ws] == ["negative"] and ws[0].start == pytest.approx(10.2)  # t=10.0 has no face
    cands = candidates_of("src", a, ws, duration=100)
    assert [c.start for c in cands] == [0.0, 15.0]
    assert cands[0].windows == ws and cands[1].windows == []
    assert cands[0].face_ratio == pytest.approx(0.9, abs=0.01) and cands[0].median_face == 120


def cand(source, start, neg=0, pos=0, face=120, audio=True):
    wins = [Window("negative", 1, start + 1, start + 4, 0.8)] * neg + [Window("positive", 1, start + 5, start + 8, 0.8)] * pos
    return Candidate(source, start, start + 75, wins, 0.9, face, (400, 200), audio)


def test_plan_mix_counts_order_and_no_overlap():
    cands = [cand(f"s{i}", t, neg=(t == 0), pos=1, face=100 + 10 * i, audio=i < 7) for i in range(10) for t in range(0, 600, 75)]
    plan = plan_mix(cands)
    assert [p for p, _ in plan] == ["webcam"] * 6 + ["sector"] * 8 + ["security"] * 6  # clip order = camNN
    assert [p for p, _ in plan].count("webcam") == 6 and [p for p, _ in plan].count("sector") == 8 and [p for p, _ in plan].count("security") == 6
    chosen = [c for _, c in plan]
    assert not any(a.overlaps(b) for i, a in enumerate(chosen) for b in chosen[i + 1 :])
    assert all(c.has_audio for p, c in plan if p != "security")
    assert sum(1 for c in chosen if c.source == "s0") <= 3


def test_plan_mix_values_positive_like_negative_and_skips_wide_scenes_for_webcam():
    cands = [cand(f"n{i}", 0, neg=3) for i in range(10)] + [cand(f"p{i}", 0, neg=1, pos=2) for i in range(10)]
    cands += [cand(f"w{i}", 0, neg=2, pos=2, face=40) for i in range(10)]  # wide scenes: tiny faces
    cands += [cand(f"c{i}", 0, neg=2, pos=2, face=260) for i in range(10)]  # close-ups: too big for the webcam band
    plan = plan_mix(cands)
    webcam = [c for p, c in plan if p == "webcam"]
    assert all(60 <= c.median_face <= 200 for c in webcam)
    assert all(c.source.startswith("p") for c in webcam)  # neg+pos beats negative-only


def test_plan_mix_fails_loudly_when_short():
    with pytest.raises(RuntimeError, match="not enough segments"):
        plan_mix([cand("a", 0)])


@pytest.mark.parametrize("profile", ["webcam", "sector", "security"])
def test_geometry_lands_the_face_in_the_band(profile):
    g = fit_geometry((854, 480), face_px=120, face_center=(427, 200), profile=profile)
    out = apply_geometry(np.zeros((480, 854, 3), np.uint8), g)
    cw, ch = PROFILES[profile]["canvas"]
    assert out.shape == (ch, cw, 3)
    lo, hi = PROFILES[profile]["face"]
    assert lo <= 120 * g.scale <= hi


def test_webcam_crop_follows_the_face():
    g = fit_geometry((854, 480), face_px=130, face_center=(800, 100), profile="webcam")
    assert g.crop is not None and g.crop[0] == 854 - 640  # clamped to the right edge


def test_demo_stream_list_matches_the_clip_order():
    plan = [(p, None) for p in ["webcam"] * 6 + ["sector"] * 8 + ["security"] * 6]
    check_stream_list(plan, REPO / "demo-videos" / "demo-cams.json")  # raises on a mismatch
    with pytest.raises(SystemExit, match="do not match"):
        check_stream_list(list(reversed(plan)), REPO / "demo-videos" / "demo-cams.json")


def test_overlapping_windows_of_several_tracks_count_once():
    ws = [Window("negative", 1, 1.0, 4.0, 0.8), Window("negative", 2, 1.2, 3.8, 0.9), Window("positive", 3, 2.0, 5.0, 0.7), Window("negative", 4, 6.0, 9.0, 0.7)]
    out = merge_windows(ws)
    assert [(w.kind, w.start, w.end, w.peak) for w in out] == [("negative", 1.0, 4.0, 0.9), ("positive", 2.0, 5.0, 0.7), ("negative", 6.0, 9.0, 0.7)]


def test_band_check():
    assert in_band(130, "webcam") and in_band(84, "webcam") and not in_band(70, "webcam")
    assert not in_band(None, "security") and in_band(30, "security")


# ------------------------------------------------------------------------------ downloads


def src_entry(data480: bytes, original: bytes) -> dict:
    return {"id": "x", "url": "u480", "sha256": hashlib.sha256(data480).hexdigest(), "original_url": "uorig", "original_sha1": hashlib.sha1(original).hexdigest()}


def fake_fetch(files: dict, calls: list):
    def fetch(url, path):
        calls.append(url)
        path.write_bytes(files[url])
    return fetch


def test_ensure_source_verifies_falls_back_and_works_offline(tmp_path, monkeypatch):
    calls: list = []
    monkeypatch.setattr(prep, "fetch", fake_fetch({"u480": b"good480", "uorig": b"orig"}, calls))
    assert ensure_source(src_entry(b"good480", b"orig"), tmp_path).name == "x.480p.webm"
    assert ensure_source(src_entry(b"good480", b"orig"), tmp_path).name == "x.480p.webm"  # cached: no download
    assert calls == ["u480"]

    calls.clear()
    monkeypatch.setattr(prep, "fetch", fake_fetch({"u480": b"changed", "uorig": b"orig"}, calls))
    entry = src_entry(b"pinned480", b"orig")
    assert ensure_source(entry, tmp_path).name == "x.original.webm"  # transcode changed: original verified by SHA-1

    def offline(url, path):
        raise SystemExit("no network")
    monkeypatch.setattr(prep, "fetch", offline)
    assert ensure_source(entry, tmp_path).name == "x.original.webm"  # re-run without network


def test_ensure_source_refuses_a_bad_original(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "fetch", fake_fetch({"u480": b"changed", "uorig": b"tampered"}, []))
    with pytest.raises(SystemExit, match="x.original.webm"):
        ensure_source(src_entry(b"pinned", b"orig"), tmp_path)
    assert not (tmp_path / "x.original.webm").exists()  # not kept for the next run


# ------------------------------------------------------------------------------ report


def test_write_report_lists_relative_windows_and_gaps(tmp_path):
    c1 = Candidate("a", 60.0, 135.0, [Window("negative", 1, 70.0, 73.0, 0.9)], 0.9, 120, (100, 100), True)
    c2 = Candidate("b", 0.0, 75.0, [Window("positive", 1, 5.0, 8.0, 0.8)], 0.9, 120, (100, 100), True)
    write_report(tmp_path, {"a": {"title": "A"}, "b": {"title": "B"}}, [("webcam", c1), ("security", c2)], {"cam01": 140.0, "cam02": None})
    text = (tmp_path / "segments.md").read_text()
    assert "| cam01 | webcam | A | 01:00–02:15 | 140 px (90–200) | 00:10 (3.0 s, 0.90) | — |" in text
    assert "| cam02 | security | B | 00:00–01:15 | — (25–45) | — | 00:05 (3.0 s, 0.80) |" in text
    assert "Clips without a negative window: cam02." in text
    assert "Clips without a positive window: cam01." in text
    assert "without a detected face: cam02." in text


# ------------------------------------------------------------------------------ encoding


@pytest.mark.parametrize("profile", ["sector", "security"])
def test_encode_matches_the_profile(tmp_path, profile):
    import av

    from tests.test_rtsp_source import make_clip

    src = make_clip(tmp_path / "src.mkv", seconds=3.0, size=(854, 480), audio=True)
    cand = Candidate("s", 0.5, 2.5, [], 1.0, 120, (427, 200), True)
    out = tmp_path / "cam01.mp4"
    encode(src, cand, profile, (854, 480), out)
    assert out.exists() and not list(tmp_path.glob(".*part"))
    with av.open(str(out)) as c:
        v = c.streams.video[0]
        assert (v.codec_context.width, v.codec_context.height) == PROFILES[profile]["canvas"]
        assert float(v.average_rate) == 10.0
        frames = [f for f in c.decode(v)]
        assert len(frames) == 20 and [i for i, f in enumerate(frames) if f.key_frame] == [0]
    with av.open(str(out)) as c:
        if profile == "sector":
            a = c.streams.audio[0]
            assert a.rate == 16000 and a.channels == 1
            samples = sum(f.samples for f in c.decode(a))
            assert abs(samples - 2.0 * 16000) <= 1024  # audio covers the same span as video
        else:
            assert not c.streams.audio


def test_encode_rejects_a_source_that_ends_early(tmp_path):
    from tests.test_rtsp_source import make_clip

    src = make_clip(tmp_path / "src.mkv", seconds=1.0, size=(854, 480), audio=False)
    with pytest.raises(RuntimeError, match="only"):
        encode(src, Candidate("s", 0.0, 5.0, [], 1.0, 120, (427, 200), False), "security", (854, 480), tmp_path / "cam02.mp4")
    assert not (tmp_path / "cam02.mp4").exists() and not list(tmp_path.glob(".*part"))
