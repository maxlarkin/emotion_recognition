"""Model-free guards for the inference core: label mapping, EMA, fusion, tracker, slots, windowing."""

from __future__ import annotations

import numpy as np
import pytest

from inference.fusion import FusionWeights, TrackEMA, fuse
from inference.ser import SpeechWindow
from inference.tracker import IoUTracker, iou
from inference.types import (
    EMOTIONS,
    FER_CLASSES,
    AudioChunk,
    VideoFrame,
    fer_probs_to_emotions,
    ser_probs_to_emotions,
    softmax,
)
import inference.worker as worker_mod
from bench.stages import violations
from inference.worker import InferenceWorker, SourceSlots, WorkerConfig, select_device, validate_audio, validate_frame
from inference.detector import clip_boxes


def test_emotion_keys_fixed():
    assert EMOTIONS == ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")


def test_fer_drops_contempt_and_renormalises():
    p8 = np.array([0.1, 0.5, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05])  # contempt = 0.5
    p7 = fer_probs_to_emotions(p8)
    assert p7.shape == (7,)
    assert p7.sum() == pytest.approx(1.0)
    # every remaining class is its 8-class probability / (1 - contempt)
    for i, key in enumerate(EMOTIONS):
        assert p7[i] == pytest.approx(p8[FER_CLASSES.index(key)] / 0.5)


def test_fer_batched_and_wrong_width():
    p = fer_probs_to_emotions(softmax(np.random.default_rng(0).normal(size=(3, 8))))
    assert p.shape == (3, 7)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        fer_probs_to_emotions(np.ones(7))


def test_ser_mapping():
    p = ser_probs_to_emotions(np.array([0.1, 0.2, 0.3, 0.4]), ["angry", "sad", "neutral", "positive"])
    d = dict(zip(EMOTIONS, p))
    assert d == pytest.approx(
        {"anger": 0.1, "sadness": 0.2, "neutral": 0.3, "happiness": 0.4, "disgust": 0, "fear": 0, "surprise": 0}
    )
    with pytest.raises(ValueError):
        ser_probs_to_emotions(np.array([1.0]), ["bored"])


def test_ema_alpha_and_first_observation():
    ema = TrackEMA(0.4)
    a, b = np.eye(7)[0], np.eye(7)[1]
    np.testing.assert_allclose(ema.update(1, a), a)
    np.testing.assert_allclose(ema.update(1, b), 0.4 * b + 0.6 * a)
    np.testing.assert_allclose(ema.update(2, b), b)  # tracks are independent
    ema.drop(1)
    assert 1 not in ema
    np.testing.assert_allclose(ema.update(1, b), b)
    with pytest.raises(ValueError):
        TrackEMA(0.0)


def test_fusion_with_speech():
    face = np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2])
    audio = np.array([0.5, 0.0, 0.0, 0.2, 0.2, 0.0, 0.1])  # mapped SER output
    fused = fuse(face, audio, FusionWeights(0.7, 0.3))
    np.testing.assert_allclose(fused, 0.7 * face + 0.3 * audio)
    assert fused.sum() == pytest.approx(1.0)
    # classes without an audio counterpart take the face term only, even if audio is non-zero there
    noisy = audio.copy()
    noisy[EMOTIONS.index("fear")] = 0.5
    np.testing.assert_allclose(fuse(face, noisy, FusionWeights(0.7, 0.3)), fused)


def test_fusion_silence_equals_face():
    face = np.array([0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.2])
    np.testing.assert_allclose(fuse(face, None, FusionWeights()), face)
    np.testing.assert_allclose(fuse(face, np.full(7, 1 / 7), FusionWeights(1.0, 0.0)), face)
    with pytest.raises(ValueError):
        FusionWeights(0.0, 0.0)


def test_fusion_weights_renormalised():
    face = np.full(7, 1 / 7)
    audio = np.array([1.0, 0, 0, 0, 0, 0, 0])
    fused = fuse(face, audio, FusionWeights(1.0, 1.0))
    assert fused.sum() == pytest.approx(1.0)
    assert fused[0] == pytest.approx((1 / 7 + 1.0) / 2)


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_tracker_stable_ids_and_expiry():
    tr = IoUTracker()
    ids, _ = tr.update([(10, 10, 50, 50), (200, 10, 50, 50)], ts=0.0)
    assert ids == [1, 2]
    ids, _ = tr.update([(205, 12, 50, 50), (12, 11, 50, 50)], ts=0.125)  # order swapped, moved a little
    assert ids == [2, 1]
    ids, expired = tr.update([(14, 12, 50, 50)], ts=0.9)  # track 2 unseen 0.775 s: kept
    assert ids == [1] and expired == []
    ids, expired = tr.update([(14, 12, 50, 50)], ts=1.2)  # track 2 unseen 1.075 s: expired
    assert ids == [1] and expired == [2]
    ids, _ = tr.update([(200, 10, 50, 50)], ts=1.3)
    assert ids == [3]  # a returning face gets a new id
    ids, expired = tr.update([(14, 12, 50, 50)], ts=2.5)  # 1 unseen 1.3 s -> expired, new id
    assert ids == [4] and set(expired) == {1, 3}


def test_tracker_low_iou_is_new_track():
    tr = IoUTracker()
    tr.update([(0, 0, 50, 50)], ts=0.0)
    ids, _ = tr.update([(40, 40, 50, 50)], ts=0.1)  # IoU ~ 0.01
    assert ids == [2]


def test_clip_boxes():
    boxes = clip_boxes(np.array([[-10, -5, 40, 30], [600, 300, 100, 100], [700, 10, 5, 5]]), 640, 360)
    assert boxes == [(0, 0, 30, 25), (600, 300, 40, 60)]


def _frame(sid, fid):
    return VideoFrame(sid, fid, float(fid), np.zeros((2, 2, 3), np.uint8))


def test_slot_overwrite_keeps_latest_per_source():
    slots = SourceSlots(audio_maxlen=2)
    for sid in ("s1", "s2", "s3"):
        slots.register(sid)
    for fid in range(1, 6):
        assert slots.put_frame(_frame("s1", fid))
    assert slots.put_frame(_frame("s2", 1))
    frames, audio, removed = slots.take()
    assert sorted((f.source_id, f.frame_id) for f in frames) == [("s1", 5), ("s2", 1)]
    assert slots.dropped_frames["s1"] == 4 and slots.dropped_frames["s3"] == 0
    assert audio == {} and removed == set()
    assert slots.take()[0] == []  # slots are emptied by take()


def test_audio_fifo_bounded_and_unknown_source():
    slots = SourceSlots(audio_maxlen=2)
    slots.register("s1")
    for i in range(3):
        assert slots.put_audio(AudioChunk("s1", float(i), np.zeros(8000, np.int16)))
    assert not slots.put_audio(AudioChunk("nope", 0.0, np.zeros(1, np.int16)))
    assert not slots.put_frame(_frame("nope", 1))
    _, audio, _ = slots.take()
    assert [c.ts for c in audio["s1"]] == [1.0, 2.0]  # oldest chunk dropped
    assert slots.dropped_audio["s1"] == 1
    slots.unregister("s1")
    assert slots.take()[2] == {"s1"}
    assert not slots.put_frame(_frame("s1", 1))


def test_speech_window_3s_hop_1s():
    win = SpeechWindow()
    chunk = np.zeros(512, np.float32)
    emitted = []
    for i in range(int(6.1 * 16000 / 512)):  # 6.1 s of chunks, first half speech
        w = win.push(chunk, 0.9 if i * 512 < 3 * 16000 else 0.1, end_ts=(i + 1) * 512 / 16000)
        if w is not None:
            emitted.append(w)
    assert len(emitted) == 4  # at 3 s, then every ~1 s
    assert all(w.samples.shape == (48000,) for w in emitted)
    assert emitted[0].end_ts == pytest.approx(3.0, abs=0.04)
    hops = np.diff([w.end_ts for w in emitted])
    np.testing.assert_allclose(hops, 1.0, atol=0.04)
    assert emitted[0].speech_ratio >= 0.95 and emitted[-1].speech_ratio < 0.5


def test_unknown_source_raises_without_starting_worker():
    w = InferenceWorker(config=WorkerConfig())
    with pytest.raises(KeyError):
        w.submit_frame("ghost", np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(KeyError):
        w.submit_audio("ghost", b"\x00\x00")
    w.stop()


def test_select_device_rejects_unknown():
    with pytest.raises(ValueError):
        select_device("tpu", "cpu")
    assert select_device("cpu", "gpu") == "cpu"


def test_reregister_resets_state():
    slots = SourceSlots()
    slots.register("s1")
    slots.take()
    slots.unregister("s1")
    slots.register("s1")  # before the next take(): the old engine state must still be dropped
    assert slots.put_frame(_frame("s1", 7))
    frames, _, removed = slots.take()
    assert removed == {"s1"} and [f.frame_id for f in frames] == [7]
    assert slots.dropped_frames["s1"] == 0


def test_unregister_drops_counters():
    slots = SourceSlots()
    slots.register("s1")
    slots.put_frame(_frame("s1", 1))
    slots.put_frame(_frame("s1", 2))
    slots.unregister("s1")
    assert "s1" not in slots.dropped_frames and "s1" not in slots.dropped_audio


def test_submit_validation():
    with pytest.raises(ValueError, match="even length"):
        validate_audio(b"\x00\x00\x01")
    with pytest.raises(ValueError, match="int16"):
        validate_audio(np.zeros(100, np.float32))
    with pytest.raises(ValueError, match="1-D"):
        validate_audio(np.zeros((2, 100), np.int16))
    with pytest.raises(ValueError, match="HxWx3"):
        validate_frame(np.zeros((10, 10), np.uint8))  # grayscale
    with pytest.raises(ValueError, match="HxWx3"):
        validate_frame(np.zeros((0, 10, 3), np.uint8))
    with pytest.raises(ValueError, match="HxWx3"):
        validate_frame(np.zeros((10, 10, 3), np.float32))
    img = np.zeros((4, 4, 3), np.uint8)
    out = validate_frame(img)
    img[0, 0, 0] = 9
    assert out[0, 0, 0] == 0  # copied
    np.testing.assert_array_equal(validate_audio(b"\x01\x00\xff\xff"), np.array([1, -1], np.int16))


def test_not_running_worker_rejects_calls():
    w = InferenceWorker(config=WorkerConfig())
    with pytest.raises(RuntimeError, match="not running"):
        w.register_source("s1")
    with pytest.raises(RuntimeError, match="not running"):
        w.set_fusion(0.7, 0.3)
    assert w.results() == []


def test_select_device_auto_cpu_image_does_not_probe(monkeypatch):
    def boom():
        raise RuntimeError("probe must not run")

    monkeypatch.setattr(worker_mod, "cuda_usable", boom)
    assert select_device("auto", "cpu") == "cpu"
    assert select_device("auto", "gpu") == "cpu"  # probe error -> CPU
    with pytest.raises(RuntimeError):
        select_device("cuda", "gpu")


def _report(**kw):
    rep = {
        "device": "cuda",
        "faces_in_frame": 1,
        "stages": {"detect_plus_fer": {"p95": 20.0}},
        "e2e": {"1_source": {"starved": False}, "3_sources": {"starved": False}},
        "peak_vram_mb": 1400.0,
        "vram_baseline_mb": 450.0,
        "vram_total_mb": 4096.0,
    }
    rep.update(kw)
    return rep


def test_violations_ok():
    assert violations(_report()) == []
    assert violations(_report(device="cpu", peak_vram_mb=None, vram_baseline_mb=None, vram_total_mb=None)) == []


def test_violations_each_condition():
    assert any("VRAM used" in v for v in violations(_report(peak_vram_mb=4700.0, vram_baseline_mb=500.0, vram_total_mb=8192.0)))
    assert any("95%" in v for v in violations(_report(peak_vram_mb=3950.0)))
    assert any("p95" in v for v in violations(_report(stages={"detect_plus_fer": {"p95": 120.0}})))
    starved = _report(e2e={"3_sources": {"starved": True}})
    assert any("starved" in v for v in violations(starved))
    assert any("starved" in v for v in violations(dict(starved, device="cpu")))
    assert any("no face" in v for v in violations(_report(faces_in_frame=0, device="cpu")))
    # GPU-only gates do not apply on the CPU
    assert violations(_report(device="cpu", stages={"detect_plus_fer": {"p95": 500.0}}, peak_vram_mb=4000.0)) == []


# ------------------------------------------------------------------------------ per-source fusion (stub models)


class _StubDetector:
    def detect(self, image):
        return [(10, 10, 50, 50)]


class _StubFER:
    FACE = np.array([0.05, 0.0, 0.05, 0.6, 0.1, 0.0, 0.2])

    def predict(self, crops):
        return np.stack([self.FACE for _ in crops]) if crops else np.zeros((0, 7))


def _stub_engine() -> "worker_mod.Engine":
    from inference.fusion import FusionWeights

    eng = worker_mod.Engine.__new__(worker_mod.Engine)  # no model files: detector / FER stubbed
    eng.device, eng.config = "cpu", WorkerConfig()
    eng.weights = FusionWeights(0.7, 0.3)
    eng.detector, eng.fer, eng.sources = _StubDetector(), _StubFER(), {}
    return eng


def _speech(sid: str, end_ts: float, probs=None, skipped: bool = False):
    from inference.types import AudioResult

    p = None if probs is None else dict(zip(EMOTIONS, probs))
    return AudioResult(sid, True, p, None if p is None else "anger", end_ts, 0.9, skipped=skipped)


def test_engine_fusion_weights_are_per_source():
    eng = _stub_engine()
    img = np.zeros((80, 80, 3), np.uint8)
    audio = [0.8, 0.0, 0.0, 0.1, 0.05, 0.0, 0.05]
    for sid in ("a", "b"):
        eng.state(sid).audio_history.append(_speech(sid, 10.0, audio))
    eng.set_fusion(1.0, 0.0, "a")
    out = {r.source_id: r for r in eng.process_frames([VideoFrame(s, 1, 10.5, img) for s in ("a", "b")])}
    fa, fb = out["a"].faces[0], out["b"].faces[0]
    assert fa.fused == fa.face  # face only for "a"
    expected = fuse(_StubFER.FACE, np.array(audio), FusionWeights(0.7, 0.3))
    np.testing.assert_allclose([fb.fused[k] for k in EMOTIONS], expected)  # "b" keeps the defaults
    # the global call still sets the default for sources without their own weights
    eng.set_fusion(0.5, 0.5)
    out = {r.source_id: r for r in eng.process_frames([VideoFrame(s, 2, 10.6, img) for s in ("a", "b")])}
    assert out["a"].faces[0].fused == out["a"].faces[0].face
    expected = fuse(_StubFER.FACE, np.array(audio), FusionWeights(0.5, 0.5))
    np.testing.assert_allclose([out["b"].faces[0].fused[k] for k in EMOTIONS], expected)
    with pytest.raises(ValueError):
        eng.set_fusion(0.0, 0.0, "a")
    assert eng.state("a").weights == FusionWeights(1.0, 0.0)  # unchanged


def test_engine_skips_backlog_window_without_probs():
    eng = _stub_engine()
    st = eng.state("s")
    st.audio_history.append(_speech("s", 10.0, skipped=True))  # SER ran only on a newer window
    assert eng.audio_for(st, 10.5) is None
    st.audio_history.append(_speech("s", 9.0, [1, 0, 0, 0, 0, 0, 0]))
    assert eng.audio_for(st, 10.5).window_end_ts == 9.0


class _StubVAD:
    """Silero VAD stand-in: same 512-sample chunking and leftover handling, every chunk is speech."""

    def feed(self, stream, samples):
        from inference.vad import CHUNK

        buf = np.concatenate([stream.pending, samples.astype(np.float32)])
        n = buf.shape[0] // CHUNK
        stream.pending = buf[n * CHUNK :].copy()
        return [(buf[i * CHUNK : (i + 1) * CHUNK], 1.0) for i in range(n)]


def _window_ends(eng, sid: str, starts: list[float], seconds: float = 0.5) -> list[float]:
    n = int(seconds * 16000)
    chunks = [AudioChunk(sid, ts, np.full(n, 1000, np.int16)) for ts in starts]
    return [w.end_ts for _, w in eng._windows(sid, chunks)]


def test_audio_gap_resets_vad_and_window():
    eng = _stub_engine()
    eng.vad = _StubVAD()
    # 2.5 s of continuous audio: no 3 s window yet
    assert _window_ends(eng, "s", [0.5 * k for k in range(5)]) == []
    # mic off -> on: the next chunk starts 7.5 s later. Without a reset the first window would end
    # at ~10.5 s, mixing 2.5 s of pre-gap audio; with it, the window needs 3 s of new audio.
    ends = _window_ends(eng, "s", [10.0 + 0.5 * k for k in range(7)])
    assert len(ends) == 1 and ends[0] == pytest.approx(10.0 + 48128 / 16000, abs=1e-6)


def test_audio_backwards_ts_resets_but_jitter_does_not():
    eng = _stub_engine()
    eng.vad = _StubVAD()
    # small arrival jitter (< 0.25 s) keeps the stream: 3.5 s of audio yields a window
    assert len(_window_ends(eng, "j", [0.0, 0.6, 0.9, 1.6, 2.0, 2.4, 3.1])) == 1
    eng2 = _stub_engine()
    eng2.vad = _StubVAD()
    assert _window_ends(eng2, "b", [100.0 + 0.5 * k for k in range(5)]) == []
    # ts jumps back by 50 s: restart, so 0.5 s more audio does not complete a window
    assert _window_ends(eng2, "b", [52.5]) == []


class _RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def remove(self, sid: str) -> None:
        self.calls.append(("remove", sid))

    def set_fusion(self, w_face: float, w_audio: float, sid=None) -> None:
        FusionWeights(w_face, w_audio)  # validates like the real engine
        self.calls.append(("fusion", sid, w_face, w_audio))


def test_controls_skip_config_for_unregistered_source():
    import queue
    from collections import deque

    slots = SourceSlots()
    slots.register("live")
    eng, outq = _RecordingEngine(), queue.Queue()
    controls = deque([
        {"w_face": 1.0, "w_audio": 0.0, "source_id": "gone"},
        {"w_face": 0.5, "w_audio": 0.5, "source_id": "live"},
        {"w_face": 0.6, "w_audio": 0.4, "source_id": None},
    ])
    worker_mod._apply_controls(eng, slots, set(), controls, outq)
    assert eng.calls == [("fusion", "live", 0.5, 0.5), ("fusion", None, 0.6, 0.4)]
    assert not controls and outq.empty()


def test_controls_remove_before_config_and_report_bad_weights():
    import queue
    from collections import deque

    slots = SourceSlots()
    slots.register("s")
    slots.unregister("s")
    slots.register("s")  # re-registered: the config belongs to the new source
    eng, outq = _RecordingEngine(), queue.Queue()
    controls = deque([{"w_face": 0.2, "w_audio": 0.8, "source_id": "s"}, {"w_face": -1.0, "w_audio": 0.0, "source_id": None}])
    worker_mod._apply_controls(eng, slots, {"s"}, controls, outq)
    assert eng.calls == [("remove", "s"), ("fusion", "s", 0.2, 0.8)]
    err = outq.get_nowait()
    assert (err.source_id, err.kind) == (None, "bad_config")
