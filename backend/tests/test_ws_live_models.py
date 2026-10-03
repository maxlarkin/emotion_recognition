"""/ws/live end-to-end with the real inference worker (marker `models`): frames of
bench/data/face.jpg must come back with at least one face, and bench/data/speech.wav chunks on
the same clock must give a vocal emotion and a fused one. Run inside the backend image."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import MODEL_FILES, get_settings
from sources.webcam import encode_audio, encode_frame

pytestmark = pytest.mark.models

SETTINGS = get_settings()
_missing = [k for k in MODEL_FILES if not SETTINGS.model_path(k).is_file()]
if _missing:
    pytest.skip(f"model files missing in {SETTINGS.models_dir}: {_missing}", allow_module_level=True)


def test_live_session_returns_faces(tmp_path, monkeypatch):
    import cv2

    from app.main import app
    from bench.stages import synthetic_frame

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    ok, buf = cv2.imencode(".jpg", synthetic_frame(), [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    jpeg = buf.tobytes()

    with TestClient(app) as client:
        hub = client.app.state.hub
        assert hub.wait_ready(600.0), hub.error
        with client.websocket_connect("/ws/live") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            assert ws.receive_json()["type"] == "session"
            result = None
            for fid in range(1, 11):
                capture_ts = time.time() * 1000.0
                ws.send_bytes(encode_frame(capture_ts, fid, jpeg))
                msg = ws.receive_json()  # one frame in flight: every frame is answered
                assert msg["type"] == "frame_result", msg
                assert msg["frame_id"] == fid and msg["capture_ts"] == capture_ts
                if msg["faces"]:
                    result = msg
                    break
            ws.send_text(json.dumps({"type": "stop"}))
    assert result is not None, "no face detected in 10 frames"
    face = result["faces"][0]
    x, y, w, h = face["box"]
    assert 0 <= x < 640 and 0 <= y < 360 and w > 0 and h > 0
    assert face["fused_label"] in face["fused"]
    assert 0.0 < face["confidence"] <= 1.0
    assert result["audio"] is None
    assert result["proc_ms"] > 0


EMOTION_KEYS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]


def _jpeg() -> bytes:
    import cv2

    from bench.stages import synthetic_frame

    ok, buf = cv2.imencode(".jpg", synthetic_frame(), [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buf.tobytes()


def _start(ws) -> None:
    ws.send_text(json.dumps({"type": "start"}))
    assert ws.receive_json()["type"] == "session"


def _send_speech(ws, base_ms: float, seconds: float) -> None:
    """speech.wav as 500 ms 0x02 chunks; capture_ts = first sample, on the frames' clock."""
    from bench.stages import speech_audio

    pcm = speech_audio(seconds)
    for k in range(0, pcm.shape[0], 8000):
        ws.send_bytes(encode_audio(base_ms + k / 16.0, pcm[k : k + 8000].astype("<i2").tobytes()))


def _frame_with(ws, jpeg: bytes, ts_ms: float, cond, first_id: int, timeout: float = 120.0) -> dict:
    """Send frames (one in flight) until a result with a face satisfies `cond`."""
    deadline = time.monotonic() + timeout
    fid = first_id
    while time.monotonic() < deadline:
        fid += 1
        ws.send_bytes(encode_frame(ts_ms + (fid - first_id), fid, jpeg))
        msg = ws.receive_json()
        assert msg["type"] == "frame_result", msg
        if msg["faces"] and cond(msg):
            return msg
        time.sleep(0.05)
    raise AssertionError("no matching frame_result in time")


def test_live_speech_gives_vocal_and_fused_emotion(tmp_path, monkeypatch):
    from app.main import app

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    jpeg = _jpeg()
    with TestClient(app) as client:
        hub = client.app.state.hub
        assert hub.wait_ready(600.0), hub.error
        with client.websocket_connect("/ws/live") as a, client.websocket_connect("/ws/live") as b, client.websocket_connect(
            "/ws/live"
        ) as c:
            for ws in (a, b, c):
                _start(ws)
            b.send_text(json.dumps({"type": "config", "fusion": {"w_face": 1, "w_audio": 0}}))
            base = time.time() * 1000.0
            # 4.5 s of speech: windows end at ~3.0 s and ~4.0 s; the frames come at 4.5 s (age <= 2 s).
            _send_speech(a, base, 4.5)
            _send_speech(b, base, 4.5)
            with_audio = lambda m: m["audio"] is not None  # noqa: E731
            ra = _frame_with(a, jpeg, base + 4500.0, with_audio, 0)
            rb = _frame_with(b, jpeg, base + 4500.0, with_audio, 0)
            # frames alone: no window is ever attached
            rc = _frame_with(c, jpeg, base + 4500.0, lambda m: True, 0)
            for ws in (a, b, c):
                ws.send_text(json.dumps({"type": "stop"}))

    audio = ra["audio"]
    assert audio["speech"] is True
    assert list(audio["probs"]) == EMOTION_KEYS
    assert sum(audio["probs"].values()) == pytest.approx(1.0, abs=1e-3)
    assert all(audio["probs"][k] == 0.0 for k in ("disgust", "fear", "surprise"))
    assert audio["label"] == max(audio["probs"], key=audio["probs"].get)
    assert 0 <= base + 4500.0 - audio["window_end_ts"] <= 2000.0
    face = ra["faces"][0]
    assert face["fused"] != face["face"]  # 0.7 * face + 0.3 * audio
    # the session with w_audio = 0 hears the same speech but fuses face only
    assert rb["audio"]["speech"] is True
    assert rb["faces"][0]["fused"] == rb["faces"][0]["face"]
    assert rc["audio"] is None
    assert rc["faces"][0]["fused"] == rc["faces"][0]["face"]
