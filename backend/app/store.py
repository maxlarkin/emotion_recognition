"""Session store: SQLite in DATA_DIR (CAP-4).

Writes never run on the event loop and never block it: callers enqueue, one writer thread drains
the queue and commits in batches. Reads use their own connection (WAL lets them run alongside the
writer) and are called from worker threads (`asyncio.to_thread`).

A frame keeps its WS face objects as JSON (the export derives rows; the timeline of a later story
returns the same objects). `frames.audio_ts` links a frame to its audio window; windows are
deduplicated by primary key.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from inference.types import EMOTIONS

log = logging.getLogger(__name__)

RUNNING = "running"
FINISHED = "finished"

BATCH_MAX = 500  # queued operations per commit
QUEUE_MAX_FRAMES = 20_000  # queued writes beyond which new frame records are dropped (writer far behind)
FLUSH_TIMEOUT_S = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    stream_id   TEXT,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,   -- server epoch ms
    updated_at  REAL NOT NULL,   -- server epoch ms of the last write for this session
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS frames (
    session_id TEXT NOT NULL,
    frame_id   INTEGER,
    ts_ms      REAL NOT NULL,
    faces      TEXT NOT NULL,    -- JSON list of WS face objects
    audio_ts   REAL              -- window_end_ts of the attached audio window
);
CREATE INDEX IF NOT EXISTS frames_session_ts ON frames (session_id, ts_ms);
CREATE TABLE IF NOT EXISTS audio_windows (
    session_id TEXT NOT NULL,
    ts_ms      REAL NOT NULL,    -- window_end_ts
    speech     INTEGER NOT NULL,
    probs      TEXT NOT NULL,    -- JSON, 7 keys (zeros in silence)
    label      TEXT,
    PRIMARY KEY (session_id, ts_ms)
);
CREATE TABLE IF NOT EXISTS streams (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    site        TEXT NOT NULL,
    url         TEXT NOT NULL,   -- with credentials; masked only in API responses
    camera_type TEXT NOT NULL,
    settings    TEXT NOT NULL,   -- JSON StreamSettings
    created_at  REAL NOT NULL    -- server epoch ms; list order
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_STOP = object()


class SessionRunning(Exception):
    """Delete refused: the session is still running."""


_INSERT_WINDOW = "INSERT OR IGNORE INTO audio_windows (session_id, ts_ms, speech, probs, label) VALUES (?, ?, ?, ?, ?)"


def _window_row(session_id: str, audio: dict) -> tuple:
    """WS audio block -> audio_windows row (7-key probs, zeros and no label in silence)."""
    speech = bool(audio.get("speech"))
    probs = audio.get("probs") or {}
    return (
        session_id,
        float(audio["window_end_ts"]),
        int(speech),
        json.dumps({k: float(probs.get(k, 0.0)) if speech else 0.0 for k in EMOTIONS}),
        audio.get("label") if speech else None,
    )


def _now_ms() -> float:
    return time.time() * 1000.0


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10.0, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


class SessionStore:
    _COLS = "id, kind, stream_id, status, created_at, updated_at, finished_at"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._wconn = _connect(self.path)
        self._wconn.execute("PRAGMA journal_mode = WAL")
        self._wconn.execute("PRAGMA synchronous = NORMAL")
        self._wconn.executescript(SCHEMA)
        n = self._finish_stale()
        if n:
            log.info("marked %d stale running session(s) as finished", n)
        self._rconn = _connect(self.path)
        self._rlock = threading.RLock()
        self._q: queue.Queue = queue.Queue()
        self._dropped = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="session-store", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ lifecycle

    def _finish_stale(self) -> int:
        """Sessions left `running` by a previous process become `finished` (called at open)."""
        cur = self._wconn.execute(
            "UPDATE sessions SET status = ?, finished_at = COALESCE(finished_at, updated_at) WHERE status = ?",
            (FINISHED, RUNNING),
        )
        return cur.rowcount

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(_STOP)
        self._thread.join(timeout=10.0)
        for conn in (self._wconn, self._rconn):
            try:
                conn.close()
            except sqlite3.Error:
                pass

    # ------------------------------------------------------------------ writes (non-blocking)

    def _put(self, op: Callable[[sqlite3.Connection], Any], fut: Optional[Future] = None) -> None:
        if self._closed:
            if fut is not None:
                fut.set_exception(RuntimeError("session store is closed"))
            return
        self._q.put((op, fut))

    def create(self, session_id: str, kind: str, stream_id: Optional[str] = None) -> None:
        now = _now_ms()

        def op(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT OR IGNORE INTO sessions (id, kind, stream_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, kind, stream_id, RUNNING, now, now),
            )

        self._put(op)

    def finish(self, session_id: str) -> None:
        now = _now_ms()

        def op(c: sqlite3.Connection) -> None:
            c.execute(
                "UPDATE sessions SET status = ?, finished_at = ?, updated_at = ? WHERE id = ? AND status = ?",
                (FINISHED, now, now, session_id, RUNNING),
            )

        self._put(op)

    def record_frame(self, session_id: str, frame_id: Optional[int], ts_ms: float, faces: list, audio: Optional[dict]) -> None:
        """One analysed frame: `faces` as sent on the WS, `audio` the WS audio block (or None)."""
        if self._q.qsize() >= QUEUE_MAX_FRAMES:
            self._dropped += 1
            if self._dropped % 1000 == 1:
                log.warning("session store is behind: %d frame record(s) dropped", self._dropped)
            return
        now = _now_ms()

        def op(c: sqlite3.Connection) -> None:
            # Serialised on the writer thread (the caller hands over `faces`/`audio` and does not mutate
            # them), all before the first execute so a failure leaves no half-written frame.
            faces_json = json.dumps(faces, ensure_ascii=False)
            win = _window_row(session_id, audio) if audio is not None else None
            if win is not None:
                c.execute(_INSERT_WINDOW, win)
            c.execute(
                "INSERT INTO frames (session_id, frame_id, ts_ms, faces, audio_ts) VALUES (?, ?, ?, ?, ?)",
                (session_id, frame_id, float(ts_ms), faces_json, win[1] if win else None),
            )
            c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

        self._put(op)

    def record_audio(self, session_id: str, audio: dict) -> None:
        """One audio window (the WS audio block) emitted on its own, not attached to a frame."""
        now = _now_ms()

        def op(c: sqlite3.Connection) -> None:
            c.execute(_INSERT_WINDOW, _window_row(session_id, audio))
            c.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

        self._put(op)

    def flush(self, timeout: float = FLUSH_TIMEOUT_S) -> bool:
        """Wait until everything queued so far is committed (read-your-writes for the API)."""
        fut: Future = Future()
        self._put(lambda c: None, fut)
        try:
            fut.result(timeout)
            return True
        except Exception:  # noqa: BLE001 - timeout or closed store: read what is there
            return False

    def delete(self, session_id: str) -> bool:
        """Blocking (call off the loop). False if unknown; raises SessionRunning if running."""
        fut: Future = Future()

        def op(c: sqlite3.Connection) -> bool:
            row = c.execute("SELECT status FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return False
            if row[0] == RUNNING:
                raise SessionRunning(session_id)
            c.execute("DELETE FROM frames WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM audio_windows WHERE session_id = ?", (session_id,))
            c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True

        self._put(op, fut)
        return fut.result(30.0)

    # ------------------------------------------------------------------ streams (CAP-10, blocking)

    def _write(self, op: Callable[[sqlite3.Connection], Any]) -> Any:
        """Run one write on the writer thread and wait for it (call off the loop)."""
        fut: Future = Future()
        self._put(op, fut)
        return fut.result(30.0)

    def load_streams(self) -> list[dict]:
        rows = self._read("SELECT id, name, site, url, camera_type, settings, created_at FROM streams ORDER BY created_at")
        return [
            {"id": r[0], "name": r[1], "site": r[2], "url": r[3], "camera_type": r[4], "settings": json.loads(r[5]), "created_at": r[6]}
            for r in rows
        ]

    def save_stream(self, rec: dict) -> None:
        """Insert or replace one stream record (keys as returned by load_streams)."""
        self._write(
            lambda c: c.execute(
                "INSERT OR REPLACE INTO streams (id, name, site, url, camera_type, settings, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (rec["id"], rec["name"], rec["site"], rec["url"], rec["camera_type"], json.dumps(rec["settings"]), rec["created_at"]),
            )
        )

    def delete_stream(self, stream_id: str) -> None:
        self._write(lambda c: c.execute("DELETE FROM streams WHERE id = ?", (stream_id,)))

    def get_meta(self, key: str) -> Optional[str]:
        rows = self._read("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0][0] if rows else None

    def set_meta(self, key: str, value: str) -> None:
        self._write(lambda c: c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)))

    def _run(self) -> None:
        c = self._wconn
        while True:
            item = self._q.get()
            batch = [item]
            while len(batch) < BATCH_MAX:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            stop = False
            done: list[tuple[Future, Any, Optional[BaseException]]] = []
            try:
                c.execute("BEGIN")
                for entry in batch:
                    if entry is _STOP:
                        stop = True
                        continue
                    op, fut = entry
                    try:
                        res = op(c)
                        if fut is not None:
                            done.append((fut, res, None))
                    except Exception as exc:  # noqa: BLE001 - one bad write must not stop the others
                        if fut is not None:
                            done.append((fut, None, exc))
                        if not isinstance(exc, SessionRunning):
                            log.warning("session store write failed: %s: %s", type(exc).__name__, exc)
                c.execute("COMMIT")
            except Exception as exc:  # noqa: BLE001 - e.g. disk full: log, keep the thread alive
                log.warning("session store commit failed: %s: %s", type(exc).__name__, exc)
                try:
                    c.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                done = [(f, None, exc) for f, _, _ in done]
            for fut, res, exc in done:
                if exc is None:
                    fut.set_result(res)
                else:
                    fut.set_exception(exc)
            if stop:
                return

    # ------------------------------------------------------------------ reads (blocking)

    def _read(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._rlock:
            return self._rconn.execute(sql, args).fetchall()

    @contextmanager
    def _snapshot(self):
        """Several reads that see one consistent state (the writer may commit in between otherwise)."""
        with self._rlock:
            self._rconn.execute("BEGIN")
            try:
                yield
            finally:
                self._rconn.execute("COMMIT")

    @staticmethod
    def _info(row: tuple) -> dict:
        sid, kind, stream_id, status, created_at, updated_at, finished_at = row
        end = finished_at if finished_at is not None else (_now_ms() if status == RUNNING else updated_at)
        return {
            "id": sid,
            "kind": kind,
            "stream_id": stream_id,
            "created_at": created_at,
            "duration": round(max(0.0, end - created_at) / 1000.0, 3),  # seconds
            "status": status,
        }

    def list(self) -> list[dict]:
        rows = self._read(f"SELECT {self._COLS} FROM sessions ORDER BY created_at DESC")
        return [self._info(r) for r in rows]

    def get(self, session_id: str) -> Optional[dict]:
        rows = self._read(f"SELECT {self._COLS} FROM sessions WHERE id = ?", (session_id,))
        return self._info(rows[0]) if rows else None

    def summary(self, session_id: str) -> Optional[dict]:
        """Session info + progress + summary, or None if unknown."""
        with self._snapshot():
            return self._summary(session_id)

    def _summary(self, session_id: str) -> Optional[dict]:
        info = self.get(session_id)
        if info is None:
            return None
        # Counted in SQLite (json_each over each frame's face list), not by parsing every frame here.
        fused = self._read(
            "SELECT json_extract(f.value, '$.fused_label') AS label, COUNT(*) AS n "
            "FROM frames, json_each(frames.faces) AS f WHERE frames.session_id = ? "
            "GROUP BY label ORDER BY n DESC, label LIMIT 1",
            (session_id,),
        )
        n_faces, n_tracks = self._read(
            "SELECT COUNT(*), COUNT(DISTINCT json_extract(f.value, '$.track_id')) "
            "FROM frames, json_each(frames.faces) AS f WHERE frames.session_id = ?",
            (session_id,),
        )[0]
        n_win, n_speech = self._read(
            "SELECT COUNT(*), COALESCE(SUM(speech), 0) FROM audio_windows WHERE session_id = ?", (session_id,)
        )[0]
        dominant, count = fused[0] if n_faces else (None, 0)
        return {
            **info,
            "progress": 1.0 if info["status"] == FINISHED else None,
            "summary": {
                "dominant_emotion": dominant,
                "dominant_share": count / n_faces if n_faces else None,
                "faces_seen": n_tracks,
                "speech_share": n_speech / n_win if n_win else None,
            },
        }

    def export_rows(self, session_id: str) -> Optional[list[dict]]:
        """Export rows (interfaces.md, Export row) ordered by ts_ms, or None if unknown."""
        with self._snapshot():
            return self._export_rows(session_id)

    def _export_rows(self, session_id: str) -> Optional[list[dict]]:
        info = self.get(session_id)
        if info is None:
            return None
        stream_id = info["stream_id"]
        windows = self._read(
            "SELECT ts_ms, speech, probs, label FROM audio_windows WHERE session_id = ? ORDER BY ts_ms", (session_id,)
        )
        by_ts = {ts: (bool(sp), label) for ts, sp, _, label in windows}
        empty = {"track_id": None, "box_x": None, "box_y": None, "box_w": None, "box_h": None}
        keyed: list[tuple[float, int, dict]] = []
        for ts, sp, probs_json, label in windows:
            probs = json.loads(probs_json)
            row = {"session_id": session_id, "stream_id": stream_id, "ts_ms": ts, "source": "audio", **empty}
            row.update({f"p_{k}": float(probs.get(k, 0.0)) for k in EMOTIONS})
            row.update({"face_label": None, "audio_label": label if sp else None, "fused_label": None, "speech": bool(sp)})
            keyed.append((ts, 0, row))
        frames = self._read(
            "SELECT ts_ms, faces, audio_ts FROM frames WHERE session_id = ? ORDER BY ts_ms, rowid", (session_id,)
        )
        for ts, faces_json, audio_ts in frames:
            speech, audio_label = by_ts.get(audio_ts, (None, None)) if audio_ts is not None else (None, None)
            for f in json.loads(faces_json):
                x, y, w, h = f["box"]
                row = {
                    "session_id": session_id, "stream_id": stream_id, "ts_ms": ts, "source": "face",
                    "track_id": f.get("track_id"), "box_x": x, "box_y": y, "box_w": w, "box_h": h,
                }
                probs = f.get("face") or {}
                row.update({f"p_{k}": float(probs.get(k, 0.0)) for k in EMOTIONS})
                row.update({
                    "face_label": f.get("face_label"), "audio_label": audio_label,
                    "fused_label": f.get("fused_label"), "speech": speech,
                })
                keyed.append((ts, 1, row))
        keyed.sort(key=lambda t: (t[0], t[1]))  # stable: frame order kept within a timestamp
        return [r for _, _, r in keyed]
