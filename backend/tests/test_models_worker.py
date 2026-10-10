"""Model tests (marker `models`): the I/O matrix through a real spawn-context worker serving 3 sources.

Run inside the backend image (weights in MODELS_DIR), see the story's Verification section.
Kept apart from test_models_engine.py so a worker process and an in-process Engine never coexist.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from app.config import MODEL_FILES, get_settings
from bench.stages import speech_audio, synthetic_frame
from inference.types import (
    EMOTIONS,
    AudioResult,
    ErrorResult,
    FrameResult,
    VideoFrame,
)

pytestmark = pytest.mark.models

SETTINGS = get_settings()
_missing = [k for k in MODEL_FILES if not SETTINGS.model_path(k).is_file()]
if _missing:
    pytest.skip(f"model files missing in {SETTINGS.models_dir}: {_missing}", allow_module_level=True)


@pytest.fixture(scope="module")
def device() -> str:
    from inference.worker import select_device

    return select_device("auto", SETTINGS.variant)


# ------------------------------------------------------------------------------ worker I/O matrix


class Collector:
    def __init__(self, worker):
        self.worker = worker
        self.items: list = []

    def wait(self, pred, timeout: float = 60.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for r in self.worker.results(timeout=0.2):
                self.items.append(r)
            hit = [r for r in self.items if pred(r)]
            if hit:
                return hit
        raise AssertionError(f"no matching result within {timeout} s; got {self.items[-5:]}")

    def frame(self, sid: str, fid: int, timeout: float = 60.0) -> FrameResult:
        return self.wait(lambda r: isinstance(r, FrameResult) and r.source_id == sid and r.frame_id == fid, timeout)[0]


@pytest.fixture(scope="module")
def worker():
    from inference.worker import InferenceWorker, WorkerConfig

    w = InferenceWorker(SETTINGS, WorkerConfig(device="auto")).start()
    for sid in ("s1", "s2", "s3"):
        w.register_source(sid)
    yield w
    w.stop()


@pytest.fixture(scope="module")
def col(worker):
    return Collector(worker)


FACE = synthetic_frame()
BLANK = np.full((360, 640, 3), 128, np.uint8)


def _check_probs(d: dict) -> None:
    assert tuple(d) == EMOTIONS
    assert sum(d.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in d.values())


def test_worker_reports_device(worker, device):
    assert worker.device == device
    assert worker.alive


def test_frame_with_faces(worker, col):
    worker.submit_frame("s1", FACE, frame_id=1, ts=100.0)
    r = col.frame("s1", 1)
    assert r.ts == 100.0 and len(r.faces) >= 1
    for f in r.faces:
        x, y, w, h = f.box
        assert 0 <= x and 0 <= y and x + w <= 640 and y + h <= 360 and w > 0 and h > 0
        _check_probs(f.face)
        _check_probs(f.fused)
        assert f.face_label == max(f.face, key=f.face.get)
        assert f.fused_label == max(f.fused, key=f.fused.get)
        assert f.confidence == pytest.approx(f.fused[f.fused_label])
    assert {"detect", "fer", "queue"} <= set(r.timings)


def test_frame_without_faces(worker, col):
    worker.submit_frame("s1", BLANK, frame_id=2, ts=100.2)
    assert col.frame("s1", 2).faces == []


def test_same_face_stable_track_then_expiry(worker, col):
    ids = []
    for i in range(5):
        worker.submit_frame("s2", FACE, frame_id=10 + i, ts=200.0 + i * 0.125)
        ids.append(col.frame("s2", 10 + i).faces[0].track_id)
    assert len(set(ids)) == 1
    worker.submit_frame("s2", FACE, frame_id=20, ts=200.5 + 1.5)  # unseen > 1 s
    assert col.frame("s2", 20).faces[0].track_id != ids[0]


def test_three_sources_latest_frame_wins(worker, col):
    n = 40
    for i in range(1, n + 1):
        for sid in ("s1", "s2", "s3"):
            worker.submit_frame(sid, FACE, frame_id=1000 + i, ts=300.0 + i / 8)
    for sid in ("s1", "s2", "s3"):
        col.frame(sid, 1000 + n)  # every source gets its latest frame
        got = [r for r in col.items if isinstance(r, FrameResult) and r.source_id == sid and r.frame_id > 1000]
        assert got and all(r.frame_id <= 1000 + n for r in got)
        ids = [r.frame_id for r in got]
        assert ids == sorted(ids)  # stale frames are dropped, never reordered


def test_unknown_source_keeps_worker_running(worker, col):
    with pytest.raises(KeyError):
        worker.submit_frame("nope", FACE)
    with pytest.raises(KeyError):
        worker.submit_audio("nope", np.zeros(8000, np.int16))
    worker.submit_frame("s3", BLANK, frame_id=5000, ts=400.0)
    assert col.frame("s3", 5000).faces == []
    assert worker.alive


def _feed_audio(worker, sid: str, pcm: np.ndarray, ts0: float) -> None:
    for k in range(0, pcm.shape[0], 8000):  # 500 ms chunks
        worker.submit_audio(sid, pcm[k : k + 8000], ts=ts0 + k / 16000)


def test_speech_window_and_fusion(worker, col):
    _feed_audio(worker, "s1", speech_audio(3.5), ts0=500.0)
    res = col.wait(lambda r: isinstance(r, AudioResult) and r.source_id == "s1")[0]
    assert res.speech is True and res.speech_ratio >= 0.5
    # first window completes with VAD chunk 94 (sample 48128), see SpeechWindow / Engine._windows
    assert res.window_end_ts == pytest.approx(500.0 + 48128 / 16000, abs=1e-6)
    _check_probs(res.probs)
    for k in ("disgust", "fear", "surprise"):
        assert res.probs[k] == 0.0
    assert res.label == max(res.probs, key=res.probs.get)

    worker.submit_frame("s1", FACE, frame_id=6000, ts=503.5)  # 0.49 s after the window (media time)
    fr = col.frame("s1", 6000)
    assert fr.audio is not None and fr.audio.speech
    face = np.array(list(fr.faces[0].face.values()))
    audio = np.array(list(fr.audio.probs.values()))
    np.testing.assert_allclose(list(fr.faces[0].fused.values()), 0.7 * face + 0.3 * audio, atol=1e-9)


def test_silence_fused_equals_face(worker, col):
    _feed_audio(worker, "s2", np.zeros(int(3.5 * 16000), np.int16), ts0=700.0)
    res = col.wait(lambda r: isinstance(r, AudioResult) and r.source_id == "s2")[0]
    assert res.speech is False and res.probs is None and res.label is None
    worker.submit_frame("s2", FACE, frame_id=7000, ts=703.5)
    fr = col.frame("s2", 7000)
    assert fr.audio is not None and fr.audio.speech is False
    assert fr.faces and fr.faces[0].fused == fr.faces[0].face


def test_no_error_results_so_far(col):
    assert not [r for r in col.items if isinstance(r, ErrorResult)]


def test_missing_model_file_fails_start(tmp_path):
    from inference.worker import InferenceWorker, WorkerStartError

    models = tmp_path / "models"
    for key, rel in MODEL_FILES.items():
        if key == "fer":
            continue
        (models / rel).parent.mkdir(parents=True, exist_ok=True)
        (models / rel).symlink_to(SETTINGS.model_path(key))
    settings = replace(SETTINGS, models_dir=models)
    with pytest.raises(WorkerStartError, match=r"enet_b0_8_va_mtl\.onnx"):
        InferenceWorker(settings).start(timeout=120)


# ------------------------------------------------------------------------------ worker: fusion config / isolation


def test_set_fusion_face_only_and_bad_config(worker, col):
    worker.set_fusion(1.0, 0.0)
    try:
        _feed_audio(worker, "s3", speech_audio(3.5), ts0=900.0)
        res = col.wait(lambda r: isinstance(r, AudioResult) and r.source_id == "s3" and r.window_end_ts > 900)[0]
        assert res.speech and res.probs is not None
        worker.submit_frame("s3", FACE, frame_id=9100, ts=903.5)
        fr = col.frame("s3", 9100)
        assert fr.audio is not None and fr.audio.speech
        assert fr.faces[0].fused == fr.faces[0].face
    finally:
        worker.set_fusion(0.7, 0.3)
    worker.set_fusion(-1.0, 0.0)
    col.wait(lambda r: isinstance(r, ErrorResult) and r.kind == "bad_config")
    worker.submit_frame("s3", BLANK, frame_id=9101, ts=910.0)
    col.frame("s3", 9101)
    assert worker.alive


def test_bad_frame_isolated_to_its_source(worker, col):
    # bypass submit_frame validation to inject a frame the detector cannot handle
    worker._inq.put(("frame", VideoFrame("s1", 9200, 1000.0, "broken")))  # type: ignore[arg-type]
    worker.submit_frame("s2", FACE, frame_id=9201, ts=1000.0)
    col.wait(lambda r: isinstance(r, ErrorResult) and r.source_id == "s1" and r.kind == "frame_failed")
    assert col.frame("s2", 9201).faces
    worker.submit_frame("s1", FACE, frame_id=9202, ts=1000.2)
    assert col.frame("s1", 9202).faces
    assert worker.alive




# ------------------------------------------------------------------------------ story 1.9: cadence and shared memory


def _segment_exists(name: str) -> bool:
    from pathlib import Path

    return Path("/dev/shm", name.lstrip("/")).exists()


def test_detect_every_reaches_the_worker(worker, col):
    worker.register_source("cad", detect_every=3)
    try:
        detected = []
        for i in range(4):  # 0.2 s apart: within the carry limit
            worker.submit_frame("cad", FACE, frame_id=i + 1, ts=300.0 + i * 0.2)
            detected.append(col.frame("cad", i + 1).timings["detected"])
        assert detected == [1.0, 0.0, 0.0, 1.0]
    finally:
        worker.unregister_source("cad")


def test_rings_are_unlinked_on_unregister_and_stop(worker, col):
    from inference.worker import InferenceWorker, WorkerConfig

    worker.register_source("ring")
    worker.submit_frame("ring", FACE, frame_id=1, ts=400.0)
    col.frame("ring", 1)
    name = worker._rings["ring"].shm.name
    assert _segment_exists(name)
    worker.unregister_source("ring")
    assert not _segment_exists(name)

    other = InferenceWorker(SETTINGS, WorkerConfig(device="cpu")).start()
    try:
        other.register_source("x")
        other.submit_frame("x", FACE, frame_id=1, ts=1.0)
        seg = other._rings["x"].shm.name
        assert _segment_exists(seg)
    finally:
        other.stop()
    assert not _segment_exists(seg)


def test_full_dev_shm_falls_back_to_pickled_frames(worker, col, monkeypatch):
    import inference.worker as worker_mod

    monkeypatch.setattr(worker_mod, "_shm_fits", lambda nbytes: False)
    worker.register_source("noshm")
    try:
        worker.submit_frame("noshm", FACE, frame_id=1, ts=500.0)
        assert col.frame("noshm", 1).faces  # still analysed
        assert "noshm" not in worker._rings
    finally:
        worker.unregister_source("noshm")
