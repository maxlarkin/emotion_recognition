"""Model tests (marker `models`), in-process: equivalence of FER / SER with the reference
libraries and the Engine pipeline. One Engine (one copy of every model) serves the whole module.

Run inside the backend image (weights in MODELS_DIR), see the story's Verification section.
Peak RAM about 3 GB; GigaAM weights are released explicitly (see release_gigaam).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from app.config import MODEL_FILES, get_settings
from bench.stages import speech_audio, synthetic_frame
from inference.types import (
    EMOTIONS,
    AudioChunk,
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



FACE = synthetic_frame()
BLANK = np.full((360, 640, 3), 128, np.uint8)


def _check_probs(d: dict) -> None:
    assert tuple(d) == EMOTIONS
    assert sum(d.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= 0 for v in d.values())


def release_gigaam(model) -> None:
    """Free a GigaAM model's weights. gigaam/hydra keep a traceback whose frames hold the model,
    so dropping the last reference frees nothing; moving the weights to the meta device does."""
    model.to_empty(device="meta")


@pytest.fixture(scope="module")
def engine(device):
    from inference.worker import Engine, WorkerConfig

    eng = Engine(SETTINGS, device, WorkerConfig())
    yield eng
    release_gigaam(eng.ser.model)  # ~1.4 GB; the worker tests that follow spawn a full model set


# ------------------------------------------------------------------------------ equivalence


def _synthetic_rgb(h: int = 180, w: int = 150, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) * 127 / (w + h))], axis=-1)
    return np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)


def test_fer_matches_emotiefflib(engine, monkeypatch):
    import emotiefflib.facial_analysis as fa

    monkeypatch.setattr(fa, "get_model_path_onnx", lambda name: str(SETTINGS.model_path("fer")))
    ref = fa.EmotiEffLibRecognizerOnnx("enet_b0_8_va_mtl")
    ours = engine.fer
    imgs = [_synthetic_rgb(seed=s) for s in range(3)]
    for img in imgs:
        _, scores = ref.predict_emotions(img, logits=False)
        expected = fer_probs_to_emotions(scores[:, :8])[0]
        got = ours.predict([img])[0]
        np.testing.assert_allclose(got, expected, atol=1e-3)
    # batching across crops does not change the result
    batch = ours.predict(imgs)
    np.testing.assert_allclose(batch, np.stack([ours.predict([i])[0] for i in imgs]), atol=1e-4)


def test_ser_matches_gigaam_get_probs(engine, device, tmp_path, monkeypatch):
    import gigaam
    import gigaam.model
    import soundfile as sf
    import torch

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

    ours = engine.ser  # built with WorkerConfig().ser_fp16, as in the worker
    tol = 1e-3 if not ours.fp16 else 1e-2  # fp16 on CUDA is accepted only within 1e-2
    try:
        for name, pcm in clips.items():
            path = tmp_path / f"{name}.wav"
            sf.write(str(path), pcm, 16000, subtype="PCM_16")
            with torch.inference_mode():
                ref_probs = ref.get_probs(str(path))
            expected = ser_probs_to_emotions(np.array([ref_probs[k] for k in labels]), labels)
            got = ours.predict(pcm.astype(np.float32)[np.newaxis, :] / 32768.0)[0]
            np.testing.assert_allclose(got, expected, atol=tol, err_msg=f"{name} (device={device}, fp16={ours.fp16})")
    finally:
        release_gigaam(ref)


# ------------------------------------------------------------------------------ engine (no process)




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
