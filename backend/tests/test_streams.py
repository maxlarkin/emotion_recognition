"""Stream manager, /api/streams*, /api/system and /ws/streams with a fake RTSP source and the fake worker."""

from __future__ import annotations

import json
import struct
import threading
import time
from contextlib import contextmanager

import numpy as np
import pytest

from inference.types import EMOTIONS, FaceResult
from sources import rtsp
from tests.test_ws_live import client, ready, wait_for  # noqa: F401

URL = "rtsp://user:secret@192.0.2.10:554/main"


class FakeSource:
    """Replaces sources.rtsp.run: emits frames at `fps` until stopped; URLs with 'unreachable' fail."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float]] = []
        self.fail = threading.Event()  # set: the running source drops (raises) once

    def __call__(self, url, det, fps, stop, sink, nvdec=False):
        self.calls.append((url, det, fps))
        if "unreachable" in url:
            raise rtsp.SourceError("Нет ответа от камеры")
        sink.opened(rtsp.OpenInfo((1920, 1080), "noaudio" not in url, "cpu"))
        w, h = rtsp.fit((1920, 1080), det)
        tw, th = rtsp.fit((w, h), rtsp.TILE_MAX)
        while not stop.is_set():
            if self.fail.is_set():
                self.fail.clear()
                raise rtsp.SourceError("Поток закончился")
            now = time.time()
            sink.frame(rtsp.DecodedFrame(np.zeros((h, w, 3), np.uint8), np.zeros((th, tw, 3), np.uint8), now, now * 1000.0))
            stop.wait(1.0 / fps)


@pytest.fixture()
def source(monkeypatch):
    fake = FakeSource()
    monkeypatch.setattr("app.streams.rtsp.run", fake)
    monkeypatch.setattr("app.streams.BACKOFF_MAX_S", 0.2)
    return fake


@contextmanager
def running(make):
    """App with lifespan started and the (fake) worker ready."""
    with make() as c:
        ready(c)
        yield c


def add(c, **body) -> str:
    r = c.post("/api/streams", json={"name": "Вход", "url": URL, **body})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def row(c, sid: str) -> dict:
    return next(s for s in c.get("/api/streams").json() if s["id"] == sid)


def read_until(ws, cond, limit: int = 400):
    """Receive WS messages (text as dict, binary as bytes) until cond(msg) is true; returns the msg."""
    for _ in range(limit):
        msg = ws.receive()
        item = json.loads(msg["text"]) if msg.get("text") is not None else msg.get("bytes")
        if cond(item):
            return item
    raise AssertionError("message not received")


def is_result(sid=None):
    return lambda m: isinstance(m, dict) and m.get("type") == "stream_result" and (sid is None or m["stream_id"] == sid)


# ------------------------------------------------------------------------------ CRUD


def test_add_with_defaults_runs_and_masks(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        r = row(c, sid)
        assert r["url"] == "rtsp://user:***@192.0.2.10:554/main"
        assert r["camera_type"] == "sector" and r["site"] == ""
        assert r["settings"] == {"enabled": True, "analysis_fps": 5, "detection_size": "auto", "min_face_px": 48}
        assert r["source_resolution"] == [1920, 1080] and r["decode"] == "cpu" and r["has_audio"] is True
        assert source.calls[0] == (URL, 1920, 5.0)  # real url to the source, sector detection size
        sessions = c.get("/api/sessions").json()
        assert any(s["id"] == r["session_id"] and s["kind"] == "rtsp" and s["stream_id"] == sid for s in sessions)


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"name": "", "url": URL}, "Укажите название"),
        ({"name": "A", "url": "http://x"}, "rtsp://"),
        ({"name": "A", "url": URL, "camera_type": "ptz"}, "Тип камеры"),
        ({"name": "A", "url": URL, "settings": {"analysis_fps": 7}}, "analysis_fps"),
        ({"name": "A", "url": URL, "settings": {"enabled": 1}}, "enabled"),
        ({"url": URL}, "Укажите название"),
    ],
)
def test_bad_body_is_422_russian(client, source, body, detail):
    with running(client) as c:
        r = c.post("/api/streams", json=body)
        assert r.status_code == 422
        assert detail in r.json()["detail"]


def test_unreachable_reconnects_with_message(client, source):
    with running(client) as c:
        sid = add(c, url="rtsp://192.0.2.99/unreachable")
        wait_for(lambda: row(c, sid)["state"] == "reconnecting")
        wait_for(lambda: len(source.calls) >= 3)  # retried with (capped) backoff
        with c.websocket_connect("/ws/streams") as ws:
            st = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "stream_status" and m["stream_id"] == sid)
            assert st["state"] == "reconnecting"
            assert "Нет ответа от камеры" in st.get("message", "")


def test_drop_reconnects_same_session(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        session = row(c, sid)["session_id"]
        source.fail.set()
        wait_for(lambda: len(source.calls) >= 2)
        wait_for(lambda: row(c, sid)["state"] == "running")
        assert row(c, sid)["session_id"] == session


def test_disable_stops_and_finishes_session(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        session = row(c, sid)["session_id"]
        r = c.patch(f"/api/streams/{sid}", json={"settings": {"enabled": False}})
        assert r.status_code == 200 and r.json()["state"] == "stopped" and r.json()["session_id"] is None
        c.app.state.store.flush()
        assert next(s for s in c.get("/api/sessions").json() if s["id"] == session)["status"] == "finished"
        assert c.post(f"/api/streams/{sid}/reconnect").status_code == 409


def test_settings_change_restarts_only_that_stream(client, source):
    with running(client) as c:
        a, b = add(c), add(c, name="Касса", camera_type="webcam")
        wait_for(lambda: row(c, a)["state"] == "running" and row(c, b)["state"] == "running")
        session = row(c, a)["session_id"]
        n = len(source.calls)
        r = c.patch(f"/api/streams/{a}", json={"settings": {"analysis_fps": 3, "detection_size": 1280}})
        assert r.json()["settings"]["analysis_fps"] == 3
        wait_for(lambda: len(source.calls) == n + 1)
        assert source.calls[-1] == (URL, 1280, 3.0)
        assert row(c, a)["session_id"] == session
        assert row(c, b)["state"] == "running"
        # renaming does not restart the source
        c.patch(f"/api/streams/{a}", json={"name": "Новое имя"})
        time.sleep(0.2)
        assert len(source.calls) == n + 1 and row(c, a)["name"] == "Новое имя"


def test_masked_url_in_patch_keeps_the_stored_one(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        n = len(source.calls)
        c.patch(f"/api/streams/{sid}", json={"url": row(c, sid)["url"]})
        time.sleep(0.2)
        assert len(source.calls) == n
        assert c.app.state.streams.records[sid]["url"] == URL


def test_delete_finishes_session_and_404(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        session = row(c, sid)["session_id"]
        assert c.delete(f"/api/streams/{sid}").status_code == 204
        assert all(s["id"] != sid for s in c.get("/api/streams").json())
        c.app.state.store.flush()
        assert next(s for s in c.get("/api/sessions").json() if s["id"] == session)["status"] == "finished"
        assert c.get(f"/api/sessions/{session}/export?format=csv").status_code == 200
        assert c.delete(f"/api/streams/{sid}").status_code == 404
        assert c.patch(f"/api/streams/{sid}", json={"name": "x"}).json()["detail"] == "Поток не найден"


def test_reconnect_now_restarts(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        n = len(source.calls)
        assert c.post(f"/api/streams/{sid}/reconnect").status_code == 204
        wait_for(lambda: len(source.calls) == n + 1)


def test_connection_test(client, source, monkeypatch):
    monkeypatch.setattr("app.api.streams.rtsp.probe", lambda url: rtsp.OpenInfo((2560, 1440), False, "cpu"))
    with running(client) as c:
        assert c.post("/api/streams/test", json={"url": URL}).json() == {"ok": True, "resolution": [2560, 1440], "has_audio": False}
        assert c.post("/api/streams/test", json={"url": "http://x"}).json() == {"ok": False, "error": "Адрес должен начинаться с rtsp://"}

        def fail(url):
            raise rtsp.SourceError("Нет ответа от камеры")

        monkeypatch.setattr("app.api.streams.rtsp.probe", fail)
        assert c.post("/api/streams/test", json={"url": URL}).json() == {"ok": False, "error": "Нет ответа от камеры"}


# ------------------------------------------------------------------------------ persistence / seed


def test_streams_survive_restart(client, source):
    with running(client) as c:
        sid = add(c, site="Тверская", settings={"analysis_fps": 8})
    with running(client) as c:
        r = row(c, sid)
        assert r["site"] == "Тверская" and r["settings"]["analysis_fps"] == 8
        wait_for(lambda: row(c, sid)["state"] == "running")


def test_default_streams_seed_once(client, source, monkeypatch):
    monkeypatch.setenv("DEFAULT_STREAMS", json.dumps([{"name": "Демо", "url": "rtsp://mediamtx:8554/cam1", "camera_type": "webcam"}, {"name": ""}]))
    with running(client) as c:
        streams = c.get("/api/streams").json()
        assert [s["name"] for s in streams] == ["Демо"]
        c.delete(f"/api/streams/{streams[0]['id']}")
    with running(client) as c:
        assert c.get("/api/streams").json() == []


def test_invalid_default_streams_is_ignored(client, source, monkeypatch):
    monkeypatch.setenv("DEFAULT_STREAMS", "{not json")
    with running(client) as c:
        assert c.get("/api/streams").json() == []


# ------------------------------------------------------------------------------ results / WS


def test_ws_frame_then_result_with_tile_boxes(client, source):
    with running(client) as c, c.websocket_connect("/ws/streams") as ws:
        sid = add(c, camera_type="security", url="rtsp://192.0.2.5/noaudio")
        binary = read_until(ws, lambda m: isinstance(m, bytes))
        kind, n = struct.unpack_from("<BH", binary)
        assert kind == 0x11 and binary[3 : 3 + n].decode() == sid
        frame_id = struct.unpack_from("<I", binary, 3 + n)[0]
        assert binary[7 + n : 9 + n] == b"\xff\xd8"  # JPEG
        res = read_until(ws, is_result(sid))
        assert res["frame_id"] >= frame_id and res["audio"] is None
        assert res["decode_ts"] <= res["emit_ts"]
        face = res["faces"][0]
        # 1920-wide source is never upscaled to the 2560 security size; fake face = half the frame, tile 640 wide
        assert face["face_px"] == 960 and face["box"][2] == 320
        assert face["decision"] == "abstain"  # track younger than 1 s
        later = read_until(ws, lambda m: is_result(sid)(m) and m["faces"] and m["faces"][0]["decision"] == "none")
        assert later["fps"] > 0


def test_small_faces_abstain(client, source, fake_worker):
    def small(image):
        face = {k: 0.0 for k in EMOTIONS}
        face["neutral"] = 1.0
        return [FaceResult(1, (10, 10, 30, 30), face, "neutral", dict(face), "neutral", 1.0)]

    fake_worker.faces_fn = small
    with running(client) as c, c.websocket_connect("/ws/streams") as ws:
        add(c)
        t0 = time.time()
        while True:  # a result older than the 1 s track-age rule, so only the size can abstain
            res = read_until(ws, lambda m: is_result()(m) and m["faces"])
            if res["decode_ts"] / 1000.0 - t0 > 1.5:
                break
        assert res["faces"][0]["face_px"] == 30 and res["faces"][0]["decision"] == "abstain"


def test_subscribe_limits_frames(client, source):
    with running(client) as c, c.websocket_connect("/ws/streams") as ws:
        a = add(c)
        b = add(c, name="Касса")
        ws.send_text(json.dumps({"type": "subscribe", "stream_ids": [b]}))
        read_until(ws, is_result(b))
        for _ in range(30):
            m = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") in ("stream_result", "stream_status"))
            assert m.get("type") == "stream_status" or m["stream_id"] == b
        # statuses of unsubscribed streams still arrive
        c.patch(f"/api/streams/{a}", json={"settings": {"enabled": False}})
        st = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "stream_status" and m["stream_id"] == a and m["state"] == "stopped")
        assert st["state"] == "stopped"


def test_results_are_recorded_in_the_session(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        session = row(c, sid)["session_id"]
        time.sleep(0.6)
        c.app.state.store.flush()
        rows = c.get(f"/api/sessions/{session}/export?format=json").json()
        assert rows and all(r["stream_id"] == sid for r in rows)


def test_system_endpoint_and_ws_tick(client, source):
    with running(client) as c:
        s = c.get("/api/system").json()
        assert set(s) == {"server_ts", "cpu_pct", "ram_used_mb", "ram_total_mb", "gpu_util_pct", "vram_used_mb", "vram_total_mb"}
        assert s["ram_total_mb"] > 0
        with c.websocket_connect("/ws/streams") as ws:
            m = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "system")
            assert m["cpu_pct"] >= 0


def test_streams_wait_for_the_worker(client, source, fake_worker):
    fake_worker.start_gate = threading.Event()
    with client() as c:
        sid = add(c)
        wait_for(lambda: c.app.state.streams.runtimes[sid].message == "Модели ещё загружаются")
        assert row(c, sid)["state"] == "connecting"
        fake_worker.start_gate.set()
        wait_for(lambda: row(c, sid)["state"] == "running", timeout=8)


def test_connection_test_never_hangs(client, source, monkeypatch):
    monkeypatch.setattr("app.api.streams.TEST_TIMEOUT_S", 0.3)
    monkeypatch.setattr("app.api.streams.rtsp.probe", lambda url: time.sleep(2) or rtsp.OpenInfo((1, 1), False, "cpu"))
    with running(client) as c:
        t = time.monotonic()
        assert c.post("/api/streams/test", json={"url": URL}).json() == {"ok": False, "error": "Нет ответа от камеры"}
        assert time.monotonic() - t < 1.5


def test_slow_client_keeps_only_the_latest_frame_per_stream():
    from app.ws.streams import StreamsClient

    c = StreamsClient()
    c.push_frame("a", b"1", "r1")
    c.push_frame("a", b"2", "r2")
    c.push_frame("b", b"3", "r3")
    assert c.frames == {"a": (b"2", "r2"), "b": (b"3", "r3")}
    c.subscribe({"type": "subscribe", "stream_ids": ["b"]})
    assert list(c.frames) == ["b"]
    c.push_frame("a", b"4", "r4")
    assert list(c.frames) == ["b"]
    c.subscribe({"type": "subscribe", "stream_ids": "b"})  # malformed: ignored
    assert c.subscribed == {"b"}


def test_bad_ws_message_is_ignored(client, source):
    with running(client) as c, c.websocket_connect("/ws/streams") as ws:
        ws.send_text("{not json")
        ws.send_bytes(b"\x00")
        sid = add(c)
        read_until(ws, is_result(sid))


def test_worker_death_marks_streams_error(client, source, fake_worker):
    with running(client) as c, c.websocket_connect("/ws/streams") as ws:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        session = row(c, sid)["session_id"]
        n = len(source.calls)
        fake_worker.die()
        st = read_until(ws, lambda m: isinstance(m, dict) and m.get("type") == "stream_status" and m["state"] == "error")
        assert st["message"] == "Модуль распознавания недоступен"
        time.sleep(0.5)
        assert row(c, sid)["state"] == "error" and row(c, sid)["session_id"] is None
        assert len(source.calls) == n  # decoding stopped, not retried
        c.app.state.store.flush()
        assert next(s for s in c.get("/api/sessions").json() if s["id"] == session)["status"] == "finished"


def test_reenable_starts_a_new_session(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        first = row(c, sid)["session_id"]
        c.patch(f"/api/streams/{sid}", json={"settings": {"enabled": False}})
        n = len(source.calls)
        r = c.patch(f"/api/streams/{sid}", json={"settings": {"enabled": True}}).json()
        assert r["session_id"] not in (None, first)
        wait_for(lambda: len(source.calls) == n + 1 and row(c, sid)["state"] == "running")


def test_disabled_stream_is_never_started(client, source):
    with running(client) as c:
        sid = add(c, settings={"enabled": False})
        time.sleep(0.3)
        assert source.calls == [] and row(c, sid)["state"] == "stopped"
    with running(client) as c:
        time.sleep(0.3)
        assert source.calls == [] and row(c, sid)["state"] == "stopped"


def test_url_and_camera_type_change_restart(client, source):
    with running(client) as c:
        sid = add(c)
        wait_for(lambda: row(c, sid)["state"] == "running")
        c.patch(f"/api/streams/{sid}", json={"camera_type": "webcam"})
        wait_for(lambda: source.calls[-1] == (URL, 640, 5.0))
        c.patch(f"/api/streams/{sid}", json={"url": "rtsp://192.0.2.11/other"})
        wait_for(lambda: source.calls[-1] == ("rtsp://192.0.2.11/other", 640, 5.0))


def test_masked_password_edits_keep_the_secret(client, source):
    with running(client) as c:
        sid = add(c)
        r = c.patch(f"/api/streams/{sid}", json={"url": "rtsp://user:***@192.0.2.20:554/main"}).json()
        assert r["url"] == "rtsp://user:***@192.0.2.20:554/main"
        assert c.app.state.streams.records[sid]["url"] == "rtsp://user:secret@192.0.2.20:554/main"
        bad = c.post("/api/streams", json={"name": "A", "url": "rtsp://user:***@h/x"})
        assert bad.status_code == 422 and "пароль" in bad.json()["detail"]


def test_mask_url_edge_cases():
    from app.streams import mask_url

    assert mask_url("rtsp://u:p@[fe80::1]:554/x") == "rtsp://u:***@[fe80::1]:554/x"
    assert mask_url("rtsp://u:p@host:bad/x") == "rtsp://u:***@host:bad/x"  # no 500 on a broken port
    assert mask_url("rtsp://host/x") == "rtsp://host/x"
    assert mask_url("rtsp://u@host/x") == "rtsp://u@host/x"


def test_only_invalid_seed_entries_do_not_burn_the_seed(client, source, monkeypatch):
    monkeypatch.setenv("DEFAULT_STREAMS", json.dumps([{"name": "", "url": "x"}]))
    with running(client) as c:
        assert c.get("/api/streams").json() == []
    monkeypatch.setenv("DEFAULT_STREAMS", json.dumps([{"name": "Демо", "url": "rtsp://mediamtx:8554/cam1"}]))
    with running(client) as c:
        assert [s["name"] for s in c.get("/api/streams").json()] == ["Демо"]


def test_float_settings_are_rejected(client, source):
    with running(client) as c:
        r = c.post("/api/streams", json={"name": "A", "url": URL, "settings": {"analysis_fps": 5.0}})
        assert r.status_code == 422


def test_track_age_survives_one_missed_detection():
    from app.faces import TrackAges

    a = TrackAges()
    a.update([1], 0.0)
    a.update([], 0.4)  # detector missed the face once
    assert a.update([1], 0.8)[1] == pytest.approx(0.8)
    a.update([], 2.0)
    assert a.update([1], 2.2)[1] == 0.0  # gone for more than 1 s: a new track


def test_detect_every_follows_the_camera_type(client, source, fake_worker):
    with running(client) as c:
        web, sec = add(c, camera_type="webcam"), add(c, camera_type="security")
        wait_for(lambda: len(fake_worker.detect_every) == 2)
        by_stream = {sid: fake_worker.detect_every[c.app.state.streams.runtimes[sid].hub_sid] for sid in (web, sec)}
        assert by_stream == {web: 1, sec: 3}
