"""Model tests (marker `models`): equivalence with the reference libraries and the I/O matrix
through a real spawn-context worker serving 3 sources.

Run inside the backend image (weights in MODELS_DIR), see the story's Verification section.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.config import MODEL_FILES, Settings, get_settings
from bench.stages import speech_audio, synthetic_frame
from inference.types import (
    EMOTIONS,
    AudioChunk,
    AudioResult,
    ErrorResult,
    FrameResult,
    VideoFrame,
    fer_probs_to_emotions,
    ser_probs_to_emotions,
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


# ------------------------------------------------------------------------------ equivalence


def _synthetic_rgb(h: int = 180, w: int = 150, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) * 127 / (w + h))], axis=-1)
    return np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)


def test_fer_matches_emotiefflib(device, monkeypatch):
    import emotiefflib.facial_analysis as fa

    from inference.fer import FacialEmotion

    monkeypatch.setattr(fa, "get_model_path_onnx", lambda name: str(SETTINGS.model_path("fer")))
    ref = fa.EmotiEffLibRecognizerOnnx("enet_b0_8_va_mtl")
    ours = FacialEmotion(SETTINGS.model_path("fer"), device=device)
    imgs = [_synthetic_rgb(seed=s) for s in range(3)]
    for img in imgs:
        _, scores = ref.predict_emotions(img, logits=False)
        expected = fer_probs_to_emotions(scores[:, :8])[0]
        got = ours.predict([img])[0]
        np.testing.assert_allclose(got, expected, atol=1e-3)
    # batching across crops does not change the result
    batch = ours.predict(imgs)
    np.testing.assert_allclose(batch, np.stack([ours.predict([i])[0] for i in imgs]), atol=1e-4)


def test_ser_matches_gigaam_get_probs(device, tmp_path, monkeypatch):
    import gigaam
    import gigaam.model
    import soundfile as sf
    import torch

    from inference.ser import SpeechEmotion
    from inference.worker import WorkerConfig

    t = np.arange(3 * 16000) / 16000.0
    rng = np.random.default_rng(0)
    sine = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 660 * t) + 0.01 * rng.standard_normal(t.size)
    clips = {"sine": (sine * 32767).astype(np.int16), "speech": speech_audio(3.0)}

    if shutil.which("ffmpeg") is None:
        # The image has no ffmpeg; decode the PCM16 wav exactly like gigaam.load_audio (s16le / 32768).
        def load_audio(path: str, sample_rate: int = 16000):
            data, sr = sf.read(path, dtype="int16")
            assert sr == sample_rate
            return torch.from_numpy(data.astype(np.float32) / 32768.0)

        monkeypatch.setattr(gigaam.model, "load_audio", load_audio)

    ckpt = SETTINGS.model_path("ser")
    ref = gigaam.load_model("emo", fp16_encoder=False, device="cpu", download_root=str(ckpt.parent))
    labels = [str(x) for x in ref.id2name]
    assert sorted(labels) == ["angry", "neutral", "positive", "sad"]

    fp16 = WorkerConfig().ser_fp16
    ours = SpeechEmotion(ckpt, device=device, fp16=fp16)
    tol = 1e-3 if not ours.fp16 else 1e-2  # fp16 on CUDA is accepted only within 1e-2
    for name, pcm in clips.items():
        path = tmp_path / f"{name}.wav"
        sf.write(str(path), pcm, 16000, subtype="PCM_16")
        with torch.inference_mode():
            ref_probs = ref.get_probs(str(path))
        expected = ser_probs_to_emotions(np.array([ref_probs[k] for k in labels]), labels)
        got = ours.predict(pcm.astype(np.float32)[np.newaxis, :] / 32768.0)[0]
        np.testing.assert_allclose(got, expected, atol=tol, err_msg=f"{name} (device={device}, fp16={ours.fp16})")


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


# ------------------------------------------------------------------------------ engine (no process)


@pytest.fixture(scope="module")
def engine(device):
    from inference.worker import Engine, WorkerConfig

    return Engine(SETTINGS, device, WorkerConfig())


def _chunks(sid: str, pcm: np.ndarray, ts0: float) -> list:
    return [AudioChunk(sid, ts0 + k / 16000, pcm[k : k + 8000]) for k in range(0, pcm.shape[0], 8000)]


def _own_pred(engine, frame, box) -> np.ndarray:
    from inference.fer import crop_rgb

    return engine.fer.predict([crop_rgb(frame, box)])[0]


def test_engine_ema_same_track(engine):
    bright = np.clip(FACE.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    r1 = engine.process_frames([VideoFrame("ema", 1, 0.0, FACE)])[0]
    r2 = engine.process_frames([VideoFrame("ema", 2, 0.125, bright)])[0]
    f1, f2 = r1.faces[0], r2.faces[0]
    assert f1.track_id == f2.track_id
    p1, p2 = _own_pred(engine, FACE, f1.box), _own_pred(engine, bright, f2.box)
    assert not np.allclose(p1, p2, atol=1e-4)  # the two frames really differ
    np.testing.assert_allclose(list(f1.face.values()), p1, atol=1e-6)
    np.testing.assert_allclose(list(f2.face.values()), 0.4 * p2 + 0.6 * p1, atol=1e-6)


def test_engine_cross_source_fer_offsets(engine):
    import cv2

    other = cv2.flip(FACE, 1)
    res = engine.process_frames(
        [VideoFrame("x1", 1, 0.0, FACE), VideoFrame("x2", 1, 0.0, BLANK), VideoFrame("x3", 1, 0.0, other)]
    )
    by = {r.source_id: r for r in res}
    assert by["x2"].faces == []
    for sid, frame in (("x1", FACE), ("x3", other)):
        f = by[sid].faces[0]
        np.testing.assert_allclose(list(f.face.values()), _own_pred(engine, frame, f.box), atol=1e-5)
    assert by["x1"].faces[0].face != by["x3"].faces[0].face


def test_engine_audio_across_sources(engine):
    res = engine.process_audio(
        {"c1": _chunks("c1", speech_audio(3.5), 0.0), "c2": _chunks("c2", np.zeros(56000, np.int16), 0.0)}
    )
    by = {r.source_id: r for r in res}
    assert by["c1"].speech and by["c1"].probs is not None and not by["c1"].skipped
    assert tuple(by["c1"].probs) == EMOTIONS
    assert by["c2"].speech is False and by["c2"].probs is None


def test_engine_audio_freshness_media_time(engine):
    win = engine.process_audio({"d1": _chunks("d1", speech_audio(3.5), 0.0)})[0]
    assert win.speech and win.probs is not None
    max_age = engine.config.audio_max_age_s
    fresh = engine.process_frames([VideoFrame("d1", 1, win.window_end_ts + 0.5, FACE)])[0]
    assert fresh.audio is win
    before = engine.process_frames([VideoFrame("d1", 2, win.window_end_ts - 0.1, FACE)])[0]
    assert before.audio is None and before.faces[0].fused == before.faces[0].face
    stale = engine.process_frames([VideoFrame("d1", 3, win.window_end_ts + max_age + 0.5, FACE)])[0]
    assert stale.audio is None
    assert stale.faces[0].fused == stale.faces[0].face


def test_engine_audio_backlog_ser_only_newest(engine):
    res = engine.process_audio({"e1": _chunks("e1", speech_audio(6.5), 0.0)})
    assert len(res) >= 3
    speech = [r for r in res if r.speech]
    assert len(speech) >= 2
    assert speech[-1].probs is not None and not speech[-1].skipped
    assert speech[-1].window_end_ts == max(r.window_end_ts for r in res)
    for r in speech[:-1]:
        assert r.skipped and r.probs is None and r.label is None


def test_engine_bad_frame_isolated(engine):
    res = engine.process_frames([VideoFrame("g1", 1, 0.0, "broken"), VideoFrame("g2", 1, 0.0, FACE)])  # type: ignore[arg-type]
    errs = [r for r in res if isinstance(r, ErrorResult)]
    frames = [r for r in res if isinstance(r, FrameResult)]
    assert [(e.source_id, e.kind) for e in errs] == [("g1", "frame_failed")]
    assert [f.source_id for f in frames] == ["g2"] and frames[0].faces
