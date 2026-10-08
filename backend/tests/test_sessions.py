"""Session store + /api/sessions I/O matrix (story 5) through /ws/live with the fake worker."""

from __future__ import annotations

import csv
import io
import json
import time

import pytest

from app.export import EXPORT_COLUMNS
from app.store import SessionStore
from inference.types import EMOTIONS
from sources.webcam import encode_frame
from tests.test_ws_live import AUDIO_PROBS, JPEG, client, ready, silence, speech, start, wait_for  # noqa: F401

T0 = 1727780000000.0  # client capture_ts, ms


def frame(ws, ts_ms: float, fid: int) -> dict:
    ws.send_bytes(encode_frame(ts_ms, fid, JPEG))
    msg = ws.receive_json()
    assert msg["type"] == "frame_result" and msg["frame_id"] == fid
    return msg


def stop(ws) -> None:
    ws.send_text(json.dumps({"type": "stop"}))


def store(c):
    return c.app.state.store


def finished(c, sid: str) -> bool:
    store(c).flush()
    info = store(c).get(sid)
    return info is not None and info["status"] == "finished"


def record(c, fake_worker) -> str:
    """start, 2 frames with 1 face (speech window attached), 1 frame in silence, stop."""
    with c.websocket_connect("/ws/live") as ws:
        sid = start(ws)
        fake_worker.audio[sid] = speech(sid, end_ts=T0 / 1000 + 0.5)
        frame(ws, T0 + 600, 1)
        frame(ws, T0 + 700, 2)  # same window again
        fake_worker.audio[sid] = silence(sid, end_ts=T0 / 1000 + 1.0)
        frame(ws, T0 + 1100, 3)
        stop(ws)
        wait_for(lambda: finished(c, sid))
    return sid


def export(c, sid: str, fmt: str):
    r = c.get(f"/api/sessions/{sid}/export", params={"format": fmt})
    assert r.status_code == 200, r.text
    return r


# ------------------------------------------------------------------------------ matrix


def test_live_recorded_is_listed_finished(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
        r = c.get("/api/sessions")
    assert r.status_code == 200
    (item,) = r.json()
    assert item["id"] == sid and item["kind"] == "live" and item["status"] == "finished"
    assert item["stream_id"] is None and item["duration"] > 0 and item["created_at"] > 0


def test_export_csv(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
        r = export(c, sid, "csv")
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == f'attachment; filename="session-{sid}.csv"'
    reader = csv.DictReader(io.StringIO(r.text))
    assert tuple(reader.fieldnames) == EXPORT_COLUMNS and len(EXPORT_COLUMNS) == 20
    rows = list(reader)
    assert [(r["source"], float(r["ts_ms"])) for r in rows] == [
        ("audio", T0 + 500), ("face", T0 + 600), ("face", T0 + 700), ("audio", T0 + 1000), ("face", T0 + 1100),
    ]
    face = rows[1]
    assert face["session_id"] == sid and face["stream_id"] == ""
    assert face["track_id"] == "1" and [face[k] for k in ("box_x", "box_y", "box_w", "box_h")] == ["80", "60", "160", "120"]
    assert float(face["p_happiness"]) == pytest.approx(0.81) and float(face["p_anger"]) == 0.0
    assert face["face_label"] == "happiness" and face["fused_label"] == "happiness"
    assert face["audio_label"] == "neutral" and face["speech"] == "true"
    quiet_face = rows[4]
    assert quiet_face["audio_label"] == "" and quiet_face["speech"] == "false"
    audio = rows[0]
    assert {k: float(audio[f"p_{k}"]) for k in EMOTIONS} == AUDIO_PROBS
    assert audio["audio_label"] == "neutral" and audio["speech"] == "true"
    for k in ("track_id", "box_x", "box_y", "box_w", "box_h", "face_label", "fused_label", "stream_id"):
        assert audio[k] == ""


def test_export_json_same_objects_with_nulls(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
        r = export(c, sid, "json")
        csv_rows = list(csv.DictReader(io.StringIO(export(c, sid, "csv").text)))
    assert r.headers["content-type"].startswith("application/json")
    assert r.headers["content-disposition"] == f'attachment; filename="session-{sid}.json"'
    rows = json.loads(r.text)
    assert len(rows) == len(csv_rows) == 5
    assert all(list(row) == list(EXPORT_COLUMNS) for row in rows)
    audio, face = rows[0], rows[1]
    assert audio["track_id"] is None and audio["box_x"] is None and audio["face_label"] is None
    assert audio["stream_id"] is None and audio["speech"] is True
    assert face["track_id"] == 1 and face["speech"] is True and face["box_w"] == 160


def test_silence_window_row(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
        rows = json.loads(export(c, sid, "json").text)
    (quiet,) = [r for r in rows if r["source"] == "audio" and r["ts_ms"] == T0 + 1000]
    assert all(quiet[f"p_{k}"] == 0.0 for k in EMOTIONS)
    assert quiet["audio_label"] is None and quiet["speech"] is False


def test_repeated_window_gives_one_audio_row(client, fake_worker):
    with client() as c:
        ready(c)
        with c.websocket_connect("/ws/live") as ws:
            sid = start(ws)
            fake_worker.audio[sid] = speech(sid, end_ts=T0 / 1000)
            for i in range(10):
                frame(ws, T0 + 100 * (i + 1), i + 1)
            stop(ws)
            wait_for(lambda: finished(c, sid))
        rows = json.loads(export(c, sid, "json").text)
    assert [r["source"] for r in rows].count("audio") == 1
    assert [r["source"] for r in rows].count("face") == 10


def test_no_faces_no_face_rows(client, fake_worker):
    fake_worker.faces_fn = lambda image: []
    with client() as c:
        ready(c)
        with c.websocket_connect("/ws/live") as ws:
            sid = start(ws)
            frame(ws, T0, 1)
            frame(ws, T0 + 100, 2)
        wait_for(lambda: finished(c, sid))
        assert [s["id"] for s in c.get("/api/sessions").json()] == [sid]
        assert json.loads(export(c, sid, "json").text) == []
        assert export(c, sid, "csv").text.strip() == ",".join(EXPORT_COLUMNS)
        info = c.get(f"/api/sessions/{sid}").json()
    assert info["summary"] == {"dominant_emotion": None, "dominant_share": None, "faces_seen": 0, "speech_share": None}


def test_running_export_is_a_snapshot(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        frame(ws, T0, 1)
        frame(ws, T0 + 100, 2)
        assert c.get("/api/sessions").json()[0]["status"] == "running"
        rows = json.loads(export(c, sid, "json").text)
        assert [r["ts_ms"] for r in rows] == [T0, T0 + 100]
        info = c.get(f"/api/sessions/{sid}").json()
        assert info["status"] == "running" and info["progress"] is None
        frame(ws, T0 + 200, 3)
        assert len(json.loads(export(c, sid, "json").text)) == 3


def test_summary(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
        r = c.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    info = r.json()
    assert info["id"] == sid and info["status"] == "finished" and info["progress"] == 1
    assert info["summary"] == {"dominant_emotion": "happiness", "dominant_share": 1.0, "faces_seen": 1, "speech_share": 0.5}


def test_bad_format_422_russian(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
        r = c.get(f"/api/sessions/{sid}/export", params={"format": "xml"})
    assert r.status_code == 422
    assert "csv" in r.json()["detail"] and "формат" in r.json()["detail"]


@pytest.mark.parametrize(
    "method,path",
    [("get", "/api/sessions/nope"), ("get", "/api/sessions/nope/export?format=csv"),
     ("get", "/api/sessions/nope/export?format=xml"), ("delete", "/api/sessions/nope")],
)
def test_unknown_id_404(client, fake_worker, method, path):
    with client() as c:
        r = getattr(c, method)(path)
    assert r.status_code == 404 and r.json() == {"detail": "Сессия не найдена"}


def test_delete(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
        r = c.delete(f"/api/sessions/{sid}")
        assert r.status_code == 204
        assert c.get("/api/sessions").json() == []
        assert c.get(f"/api/sessions/{sid}/export?format=csv").status_code == 404
        conn = store(c)._rconn
        assert conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM audio_windows").fetchone()[0] == 0


def test_delete_running_409(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        r = c.delete(f"/api/sessions/{sid}")
        assert r.status_code == 409 and r.json() == {"detail": "Сессия ещё идёт"}
        assert [s["id"] for s in c.get("/api/sessions").json()] == [sid]


def test_disconnect_finishes(client, fake_worker):
    with client() as c:
        ready(c)
        with c.websocket_connect("/ws/live") as ws:
            sid = start(ws)
            frame(ws, T0, 1)
        wait_for(lambda: finished(c, sid))


def test_worker_death_finishes(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        fake_worker.die()
        assert ws.receive_json()["type"] == "error"
        wait_for(lambda: finished(c, sid))


def test_restart_finishes_stale_running(client, fake_worker, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    s = SessionStore(data / "app.db")
    s.create("stale", "live")
    s.record_frame("stale", 1, T0, [], None)
    s.flush()
    created_at, updated_at = s._read("SELECT created_at, updated_at FROM sessions WHERE id = 'stale'")[0]
    s.close()  # process "crashed": the row stays running
    time.sleep(0.3)
    with client() as c:
        (item,) = c.get("/api/sessions").json()
    assert item["id"] == "stale" and item["status"] == "finished"
    # Ends at the last write, not at the restart.
    assert item["duration"] == pytest.approx((updated_at - created_at) / 1000, abs=1e-3)
    assert item["duration"] < 0.25


def test_sessions_survive_restart(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
    with client() as c:
        assert [s["id"] for s in c.get("/api/sessions").json()] == [sid]
        assert len(json.loads(export(c, sid, "json").text)) == 5


def test_list_newest_first(client, fake_worker):
    with client() as c:
        ready(c)
        first = record(c, fake_worker)
        second = record(c, fake_worker)
        assert [s["id"] for s in c.get("/api/sessions").json()] == [second, first]


def test_store_failure_does_not_break_live(client, fake_worker, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk full")

    with client() as c:
        ready(c)
        for name in ("create", "record_frame", "finish"):
            monkeypatch.setattr(store(c), name, boom)
        with c.websocket_connect("/ws/live") as ws:
            start(ws)
            assert frame(ws, T0, 1)["faces"]
            stop(ws)
            start(ws)


def test_writer_survives_a_failed_write(tmp_path):
    s = SessionStore(tmp_path / "app.db")
    try:
        s.create("a", "live")
        bad_audio = {"speech": True, "probs": dict(AUDIO_PROBS), "label": "neutral", "window_end_ts": 1.0}
        s.record_frame("a", 1, 1.0, [{"box": [0, 0, 1, 1], "bad": object()}], bad_audio)  # not JSON-serialisable
        s.record_frame("a", 2, 2.0, [], {"speech": False, "probs": None, "label": None, "window_end_ts": 2.0})
        s.finish("a")
        assert s.flush()
        assert s.get("a")["status"] == "finished"
        rows = s.export_rows("a")
        assert [(r["source"], r["ts_ms"]) for r in rows] == [("audio", 2.0)]  # no window left by the bad frame
    finally:
        s.close()


def test_standalone_audio_window_is_stored(client, fake_worker):
    with client() as c, ready(c).websocket_connect("/ws/live") as ws:
        sid = start(ws)
        fake_worker.push(speech(sid, end_ts=T0 / 1000))
        skipped = speech(sid, end_ts=T0 / 1000 + 1)
        skipped.probs, skipped.label, skipped.skipped = None, None, True
        fake_worker.push(skipped)  # older window of a backlog: no SER probs, not stored
        wait_for(lambda: store(c).flush() and len(store(c).export_rows(sid)) >= 1)
        rows = json.loads(export(c, sid, "json").text)
    (row,) = rows
    assert row["source"] == "audio" and row["ts_ms"] == T0 and row["audio_label"] == "neutral" and row["speech"] is True
    assert {k: row[f"p_{k}"] for k in EMOTIONS} == AUDIO_PROBS


def test_store_open_failure_keeps_live(client, fake_worker, monkeypatch):
    import app.main as main_mod

    def broken(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(main_mod, "SessionStore", broken)
    with client() as c:
        assert c.get("/api/health").status_code == 200
        ready(c)
        with c.websocket_connect("/ws/live") as ws:
            start(ws)
            assert frame(ws, T0, 1)["faces"]
        r = c.get("/api/sessions")
    assert r.status_code == 503 and r.json() == {"detail": "Хранилище сессий недоступно, подробности в журнале сервера"}


def test_closed_store_gives_503(client, fake_worker):
    with client() as c:
        sid = record(ready(c), fake_worker)
        st = store(c)
        st.close()
        assert c.get("/api/sessions").status_code == 503
        assert c.get(f"/api/sessions/{sid}/export?format=csv").status_code == 503
        r = c.delete(f"/api/sessions/{sid}")
        assert r.status_code == 503 and r.json()["detail"].startswith("Хранилище сессий недоступно")


def test_live_without_store(client, fake_worker):
    with client() as c:
        ready(c)
        c.app.state.store = None
        with c.websocket_connect("/ws/live") as ws:
            start(ws)
            assert frame(ws, T0, 1)["faces"]
        assert c.get("/api/sessions").status_code == 503


# ------------------------------------------------------------------------------ pandas


def test_pandas_reads_exports(client, fake_worker, tmp_path):
    pd = pytest.importorskip("pandas")
    with client() as c:
        sid = record(ready(c), fake_worker)
        (tmp_path / "s.csv").write_bytes(export(c, sid, "csv").content)
        (tmp_path / "s.json").write_bytes(export(c, sid, "json").content)
    df_csv = pd.read_csv(tmp_path / "s.csv")
    df_json = pd.read_json(tmp_path / "s.json")
    assert list(df_csv.columns) == list(EXPORT_COLUMNS)
    assert list(df_json.columns) == list(EXPORT_COLUMNS)
    assert len(df_csv) == len(df_json) == 5
    assert df_csv["ts_ms"].dtype == float
    # read_json downcasts whole-number floats (these test timestamps) to int64; the file itself has floats.
    assert df_json["ts_ms"].dtype.kind in "fi"
    assert all(isinstance(r["ts_ms"], float) for r in json.loads((tmp_path / "s.json").read_text()))
    for df in (df_csv, df_json):
        audio = df[df["source"] == "audio"]
        assert audio[["track_id", "box_x", "box_y", "box_w", "box_h"]].isna().all().all()
    assert df_csv["speech"].dtype == bool
