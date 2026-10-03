"""/ws/live I/O matrix with a fake inference worker (conftest.FakeWorker); no models are loaded."""

from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import health as health_mod
from app.config import MODEL_FILES
from app.ws import live as live_mod
from sources.webcam import AudioPacket, BadMessage, WebcamFrame, decode_jpeg, encode_frame, parse_message


def jpeg(width: int = 320, height: int = 240) -> bytes:
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 120, 200)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    return buf.tobytes()


JPEG = jpeg()


def wait_for(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.01)


@pytest.fixture()
def client(monkeypatch, tmp_path: Path):
    def _make(wait: bool = True) -> TestClient:
        root = tmp_path / "models"
        for rel in MODEL_FILES.values():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        monkeypatch.setenv("MODELS_DIR", str(root))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("APP_VARIANT", "cpu")
        health_mod._cached_device.cache_clear()
        from app.main import app

        return TestClient(app)

    yield _make
    health_mod._cached_device.cache_clear()


def ready(c: TestClient) -> TestClient:
    assert c.app.state.hub.wait_ready(5.0)
    return c


def start(ws) -> str:
    ws.send_text(json.dumps({"type": "start"}))
    msg = ws.receive_json()
    assert msg["type"] == "session", msg
    return msg["session_id"]


# ------------------------------------------------------------------------------ header


def test_header_round_trip():
    data = encode_frame(1727780000123.4, 4_000_000_000, JPEG)
    assert data[0] == 0x01 and len(data) == 13 + len(JPEG)
    assert struct.unpack("<d", data[1:9])[0] == 1727780000123.4  # little-endian f64
    assert struct.unpack("<I", data[9:13])[0] == 4_000_000_000  # little-endian u32
    frame = parse_message(data)
    assert frame == WebcamFrame(1727780000123.4, 4_000_000_000, JPEG)
    assert decode_jpeg(frame.jpeg).shape == (240, 320, 3)


def test_header_matches_client_layout():
    # Same bytes as frontend/src/live/protocol.ts encodeFrameHeader(1.5, 258).
    expected = bytes([0x01, 0, 0, 0, 0, 0, 0, 0xF8, 0x3F, 0x02, 0x01, 0, 0])
    assert encode_frame(1.5, 258, b"") == expected


def test_parse_audio_and_errors():
    assert isinstance(parse_message(b"\x02" + struct.pack("<d", 5.0) + b"\x00\x00"), AudioPacket)
    for bad in (b"", b"\x07" + bytes(20), b"\x01" + bytes(5), b"\x01" + struct.pack("<dI", float("nan"), 1)):
        with pytest.raises(BadMessage):
            parse_message(bad)
    for bad in (b"", b"not a jpeg"):
        with pytest.raises(BadMessage):
            decode_jpeg(bad)


# ------------------------------------------------------------------------------ matrix


def test_start_sends_session_and_registers(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        assert len(sid) == 32 and all(ch in "0123456789abcdef" for ch in sid)
        assert fake_worker.registered == [sid]


def test_frame_with_face(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        ws.send_bytes(encode_frame(1727780000123.4, 7, JPEG))
        msg = ws.receive_json()
    assert msg["type"] == "frame_result"
    assert msg["frame_id"] == 7
    assert msg["capture_ts"] == 1727780000123.4
    assert msg["proc_ms"] >= 0
    assert msg["audio"] is None
    (face,) = msg["faces"]
    assert face["box"] == [80, 60, 160, 120]  # pixels of the 320x240 JPEG
    assert face["face_label"] == face["fused_label"] == "happiness"
    assert face["confidence"] == pytest.approx(0.81)
    assert set(face["face"]) == set(face["fused"]) == {"anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"}
    assert face["track_id"] == 1
    # worker got BGR in JPEG size and ts in seconds
    (src, fid, ts, shape) = fake_worker.submitted[0]
    assert (src, fid, shape) == (sid, 7, (240, 320, 3))
    assert ts == pytest.approx(1727780000.1234)


def test_burst_answers_only_frames_the_worker_took(client, fake_worker):
    fake_worker.auto = False
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        start(ws)
        for fid in range(1, 6):
            ws.send_bytes(encode_frame(1000.0 + fid, fid, JPEG))
        wait_for(lambda: len(fake_worker.submitted) == 5)
        fake_worker.process()  # the worker takes only the newest frame
        first = ws.receive_json()
        ws.send_bytes(encode_frame(1006.0, 6, JPEG))
        wait_for(lambda: len(fake_worker.submitted) == 6)
        fake_worker.process()
        second = ws.receive_json()
    assert (first["frame_id"], first["capture_ts"]) == (5, 1005.0)
    assert (second["frame_id"], second["capture_ts"]) == (6, 1006.0)


def test_face_leaves(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        start(ws)
        ws.send_bytes(encode_frame(1.0, 1, JPEG))
        assert len(ws.receive_json()["faces"]) == 1
        fake_worker.faces_fn = lambda image: []
        ws.send_bytes(encode_frame(2.0, 2, JPEG))
        msg = ws.receive_json()
    assert msg["frame_id"] == 2 and msg["faces"] == []


def test_frame_before_start_reports_once(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        for fid in range(3):
            ws.send_bytes(encode_frame(1.0, fid, JPEG))
        assert ws.receive_json() == {"type": "error", "message": "Сессия не начата"}
        start(ws)  # the next message is the session: the error came only once
    assert fake_worker.submitted == []


@pytest.mark.parametrize(
    "data",
    [b"\x07" + bytes(20), b"\x01" + bytes(5), encode_frame(1.0, 1, b"definitely not a jpeg")],
    ids=["unknown_type", "short", "bad_jpeg"],
)
def test_bad_binary_keeps_socket(client, fake_worker, data):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        start(ws)
        ws.send_bytes(data)
        err = ws.receive_json()
        assert err["type"] == "error" and err["message"]
        ws.send_bytes(encode_frame(2.0, 2, JPEG))
        assert ws.receive_json()["frame_id"] == 2


def test_audio_and_config_are_ignored(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        start(ws)
        ws.send_bytes(b"\x02" + struct.pack("<d", 1.0) + bytes(16000))
        ws.send_text(json.dumps({"type": "config", "fusion": {"w_face": 0.5, "w_audio": 0.5}}))
        ws.send_bytes(encode_frame(3.0, 3, JPEG))
        assert ws.receive_json()["type"] == "frame_result"


def test_bad_control_message(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_text(json.dumps({"type": "dance"}))
        assert ws.receive_json()["type"] == "error"
        start(ws)


def test_start_while_loading(client, fake_worker):
    fake_worker.start_gate = threading.Event()
    with client() as c:
        assert c.get("/api/health").status_code == 200  # startup is not blocked by model loading
        with c.websocket_connect("/ws/live") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            msg = ws.receive_json()
            assert msg["type"] == "error" and msg["message"].startswith("Модели ещё загружаются")
            fake_worker.start_gate.set()
            assert c.app.state.hub.wait_ready(5.0)
            start(ws)


def test_start_after_worker_failed(client, fake_worker):
    from inference.worker import WorkerStartError

    fake_worker.start_error = WorkerStartError("model file missing")
    with client() as c:
        assert not c.app.state.hub.wait_ready(5.0)
        assert c.get("/api/health").status_code == 200  # the lifespan survived the failure
        with c.websocket_connect("/ws/live") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            msg = ws.receive_json()
    assert msg["type"] == "error" and msg["message"].startswith("Модуль распознавания недоступен")
    assert fake_worker.registered == []


def test_worker_dies_mid_session(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        start(ws)
        fake_worker.die()
        msg = ws.receive_json()
        assert msg == {"type": "error", "message": live_mod.MSG_WORKER_DIED}
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 1011


def test_disconnect_unregisters(client, fake_worker):
    with client() as c:
        ready(c)
        with c.websocket_connect("/ws/live") as ws:
            sid = start(ws)
        wait_for(lambda: fake_worker.unregistered == [sid])


def test_stop_unregisters_and_ends_session(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        ws.send_text(json.dumps({"type": "stop"}))
        wait_for(lambda: fake_worker.unregistered == [sid])
        ws.send_bytes(encode_frame(1.0, 1, JPEG))
        assert ws.receive_json() == {"type": "error", "message": "Сессия не начата"}
        sid2 = start(ws)  # a new session on the same socket
        assert sid2 != sid
        ws.send_bytes(encode_frame(2.0, 2, JPEG))
        assert ws.receive_json()["frame_id"] == 2


def test_two_sessions_are_isolated(client, fake_worker):
    with client() as c:
        ready(c)
        with c.websocket_connect("/ws/live") as a:
            sid_a = start(a)
            with c.websocket_connect("/ws/live") as b:
                sid_b = start(b)
                a.send_bytes(encode_frame(1.0, 101, JPEG))
                b.send_bytes(encode_frame(2.0, 201, JPEG))
                assert a.receive_json()["frame_id"] == 101
                assert b.receive_json()["frame_id"] == 201
            wait_for(lambda: fake_worker.unregistered == [sid_b])
            a.send_bytes(encode_frame(3.0, 102, JPEG))
            assert a.receive_json()["frame_id"] == 102
    assert sid_a != sid_b
    assert {s for s, *_ in fake_worker.submitted} == {sid_a, sid_b}


def test_worker_frame_failed_is_reported(client, fake_worker):
    from inference.types import ErrorResult

    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        fake_worker.push(ErrorResult(sid, "frame_failed", "boom"))
        assert ws.receive_json() == {"type": "error", "message": "Не удалось обработать кадр"}


def test_worker_unknown_source_is_not_sent(client, fake_worker):
    from inference.types import ErrorResult

    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        fake_worker.push(ErrorResult(sid, "unknown_source", "unknown source"))
        ws.send_bytes(encode_frame(1.0, 9, JPEG))
        assert ws.receive_json()["frame_id"] == 9


def test_repeated_start(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        ws.send_text(json.dumps({"type": "start"}))
        assert ws.receive_json() == {"type": "error", "message": "Сессия уже начата"}
    assert fake_worker.registered == [sid]


def test_deeply_nested_control_keeps_socket(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        ws.send_text("[" * 100_000 + "]" * 100_000)
        assert ws.receive_json()["type"] == "error"
        start(ws)


def test_shutdown_stops_worker(client, fake_worker):
    with client() as c:
        ready(c)
        assert not fake_worker.stopped
    assert fake_worker.stopped is True
