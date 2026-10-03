"""/ws/live end-to-end with the real inference worker (marker `models`): frames of
bench/data/face.jpg must come back with at least one face. Run inside the backend image."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.config import MODEL_FILES, get_settings
from sources.webcam import encode_frame

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
