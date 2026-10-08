---
title: 'Session persistence and CSV/JSON export'
type: 'feature'
created: '2026-10-08'
status: 'done'
baseline_commit: '7ce875167fae2d1ac91524c8b2395be81f7051f4'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/interfaces.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Live results exist only on the socket. CAP-4 needs every finished session exportable as CSV and JSON that pandas re-imports. Stories 1.6 (rtsp), 1.9 (events) and 1.10 (file) need one session store.

**Approach:** SQLite store in `DATA_DIR`. Every live session, and every frame result and audio window it emits, is written off the event loop. REST `/api/sessions`, `/api/sessions/{id}`, `/api/sessions/{id}/export`, `DELETE /api/sessions/{id}` per `interfaces.md`. A «Сессии» card in the UI lists sessions with CSV/JSON download and delete.

## Boundaries & Constraints

**Always:**
- Export row exactly per `interfaces.md` (column order too). `stream_id` column present, empty for live. Face rows: box, the face probabilities, `face_label`, `fused_label`, plus `audio_label`/`speech` of the audio window attached to that frame (empty if none). Audio rows: one per distinct window, 7-key probs (zeros in silence), `audio_label` (empty in silence), `speech`, empty box/track/face/fused columns. Rows ordered by `ts_ms`.
- Live `ts_ms` = client `capture_ts` for face rows, `window_end_ts` for audio rows (same clock as the WS).
- Session id = the `session_id` sent on `/ws/live`. Kind `live`. Status `running` until `stop`, disconnect or worker death, then `finished`. On backend start, sessions left `running` become `finished`.
- Store writes never block or break live analysis: a write failure is logged, the session continues.
- Schema carries `kind` (`live|file|rtsp`) and `stream_id` so 1.6/1.10 add sources without migration.
- All error and UI text in Russian. 404 «Сессия не найдена».

**Never:**
- Changing `interfaces.md`, the WS payloads or `/api/health`.
- Timeline, media, files, streams, events endpoints (later stories).
- Persisting images or audio samples.
- A new runtime dependency (stdlib `sqlite3`, `csv`, `json`).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Live recorded | start, frames with 1 face, speech window, stop | `GET /api/sessions` lists it: kind `live`, `status: finished`, `duration` > 0 | N/A |
| Export CSV | finished session, `format=csv` | `text/csv` attachment; header = export columns; face + audio rows; `pandas.read_csv` gives 20 columns, no error | N/A |
| Export JSON | `format=json` | array of the same objects; nulls for empty cells; `pandas.read_json` works | N/A |
| Silence window | `audio.speech:false` | audio row: probs 0, `audio_label` empty, `speech` false | N/A |
| Repeated window | many frames carry the same `window_end_ts` | exactly one audio row | N/A |
| No faces | frames with 0 faces | no face rows; session still listed | N/A |
| Running export | session still running | export of rows so far (snapshot) | N/A |
| Summary | `GET /api/sessions/{id}` | `status`, `progress` (1 if finished, else null), summary: dominant fused emotion and its share of face rows, distinct `track_id` count, speech share of audio windows | empty session: nulls/0 |
| Bad format | `format=xml` | 422 | Russian `detail` |
| Unknown id | any `/api/sessions/{id}*` | 404 | «Сессия не найдена» |
| Delete | finished session | 204, gone from list and export | running: 409 «Сессия ещё идёт» |
| Disconnect | socket drops mid-session | status `finished` | N/A |
| Restart | backend restarts with a `running` row | row becomes `finished` | N/A |

</frozen-after-approval>

## Code Map

- `backend/app/config.py` -- `Settings.data_dir` (`/data`, `DATA_DIR`); DB at `data_dir / "app.db"`.
- `backend/app/main.py` -- `lifespan` creates `data_dir` and the hub; open the store here (`app.state.store`), close on shutdown; include the new router.
- `backend/app/ws/live.py` -- `_start` (register → `session` reply): create the session row. `stop` branch in `_on_text`, `run()` finally, `WorkerGone` in `_send_loop`: finish it. `_send_loop` → `_frame_result(item)` builds the wire dict with `capture_ts`; record that dict (faces as on the WS, `audio` block). Do not change wire output.
- `backend/inference/types.py` -- `EMOTIONS` order for the `p_*` columns.
- `backend/tests/test_ws_live.py` -- `client` fixture sets `DATA_DIR` to `tmp_path`; `start(ws)`, `ready(c)`, `JPEG` helpers to reuse. `FakeWorker.audio[sid]` attaches an `AudioResult`.
- `backend/tests/conftest.py` -- `FakeWorker`; no change expected.
- `frontend/src/App.tsx` -- layout: `LiveView` + health card. Add the sessions card below.
- `frontend/src/live/LiveView.tsx` -- `onSession` / `stop` callbacks; expose an optional `onSessionEnd` so the list refreshes.
- `frontend/src/api.ts` -- `fetchHealth` pattern (timeout, shape check) to copy for `fetchSessions`.
- `frontend/src/styles.css` -- `.card`, `button.secondary`, tokens; add a table style.

## Tasks & Acceptance

**Execution:**
- [x] `backend/app/store.py` -- `SessionStore`: sqlite3 (WAL), schema `sessions`, `frames` (session_id, frame_id, ts_ms, faces JSON, audio_ts), `audio_windows` (PK session_id+ts_ms, speech, probs JSON, label). One writer thread draining a queue, batched commits; reads on their own connection. `create/finish/record_frame/list/get/summary/export_rows/delete`, `finish_stale()` at open.
- [x] `backend/app/export.py` -- `EXPORT_COLUMNS`, rows → CSV (stdlib `csv`, `true`/`false` for `speech`, empty for null) and JSON.
- [x] `backend/app/api/sessions.py` -- the four endpoints; reads via `asyncio.to_thread`; `Content-Disposition: attachment; filename="session-<id>.<ext>"`.
- [x] `backend/app/ws/live.py`, `backend/app/main.py` -- wiring per Code Map; store absent or failing → live works unchanged.
- [x] `backend/tests/test_sessions.py` -- every matrix row through the WS + REST with the fake worker; pandas-free CSV/JSON round-trip check (`csv.DictReader`, `json.loads`) plus a pandas check skipped if pandas is absent.
- [x] `frontend/src/api.ts` -- `SessionInfo` type, `fetchSessions`, `deleteSession`, `exportUrl(id, format)`.
- [x] `frontend/src/sessions/SessionsView.tsx` -- «Сессии» card: table (Начало, Тип, Длительность, Статус), links «CSV», «JSON» (`download`), «Удалить» (disabled while running), «Обновить», empty state «Сессий пока нет». Refresh on mount and on live session start/end.
- [x] `frontend/src/sessions/format.ts` + `.test.ts` -- pure helpers: kind/status labels, duration text, date text.

**Acceptance Criteria:**
- Given a finished live session in the browser, when the user clicks «CSV» and «JSON», then both files download and `pandas.read_csv` / `pandas.read_json` load them with the 20 columns.
- Given the backend container is recreated, when the user opens the UI, then earlier sessions are still listed (volume `data`).
- Given a running session at ~8 results/s, when persisting, then UI p95 latency does not change noticeably versus story 1.4.

## Implementation Notes

- `created_at` is server epoch ms. `duration` is wall-clock seconds from `created_at` to `finished_at`, or to now while the session runs. A stale session's `finished_at` is its last write (`updated_at`).
- Summary is `{dominant_emotion, dominant_share, faces_seen, speech_share}`. For an empty session it is null, null, 0, null.
- On a bad format, an unknown id still returns 404 (the id check runs first).
- Reads `flush()` the writer queue first (up to 5 s), so they see what was written before them. Above 20 000 queued writes, new frame records are dropped and a warning is logged.
- `pandas==3.0.6` is in `requirements-dev.txt` (test only, not in the image).

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-08). Layers: blind-hunter (B), edge-case-hunter (E), verification-gap (V).

| # | Layer | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | B, E | An audio window is inserted before `json.dumps(faces)`, so a failed op leaves half a write committed | low | `store.py` `record_frame` op order. Fixed by serialising before any insert. | patch |
| 2 | B, E | A failed COMMIT drops up to 500 ops, including `create`/`finish`, so the session stays `running` | low | Needs a disk-full or I/O error. `finish_stale` repairs it at restart. A fix means retry logic. | reject |
| 3 | B, E | Export and summary reads are separate autocommit statements, so a running snapshot can be inconsistent | low | Windows are read before frames, so a frame committed in between gets `speech` null. Fixed by one read transaction. | patch |
| 4 | B, E | Export builds the whole file in memory | low | Live sessions are small. For a 1 h stream at about 17 MB of JSON the RAM spike is tolerable. Streaming adds complexity. Revisit with rtsp sessions (1.6). | reject |
| 5 | B | Dropped frame records leave no trace in the data | low | Only past 20 000 queued writes, which single-stream live never reaches. A per-session counter is new surface. | reject |
| 6 | B, E | A `flush()` timeout is silent; reads stall up to 5 s under backlog | low | Needs writer backlog, which a single live stream does not cause. | reject |
| 7 | B, E | Delete timeout, a closed store or a closed read connection gives an English 500 | low | `fut.result(30)`, `_put` RuntimeError and `sqlite3.ProgrammingError` are not caught. Direct mapping to 503. | patch |
| 8 | B, E | `close()` can close the connection under a busy writer; an op enqueued after `_STOP` hangs | low | Shutdown only, with a backlog longer than 10 s. | reject |
| 9 | B | Audio windows are stored only when attached to a frame | medium | The worker emits a standalone `AudioResult` (`worker.py:516`) that the hub routes to the session, and `_send_loop` ignores it. This deviates from the frozen "every … audio window it emits". | patch |
| 10 | B | A face without `box` makes the export return 500 | false | Faces come only from `face_to_wire`, which always sets `box`. | reject |
| 11 | B | «Удалить» deletes permanently with no confirmation | low | One click loses data. Fixed with `window.confirm`. | patch |
| 12 | B, E | A failed export is saved as a file holding the JSON error | low | Only for a session deleted meanwhile or a store that is down. A fix needs fetch plus blob logic. | reject |
| 13 | B, V | `fetchSessions`/`deleteSession` error text is untested | gap | Pre-verified: no test imports them. | patch |
| 14 | B, V | The list refresh on live start/end is untested | gap | No DOM harness in the frontend. The manual check covers it. | defer |
| 15 | B | The p95 acceptance criterion has no automated check | low | The added per-frame cost is one closure plus `queue.put`. The acceptance criterion is checked manually. | reject |
| 16 | B | The pandas test checks only columns and row count | low | Direct test assertions added. | patch |
| 17 | B | No foreign keys, no `created_at` index, summary parses JSON | low | No caller is harmed yet. Deletes clear the child tables explicitly. | reject |
| 18 | E | Frames are orphaned if `create` failed | low | Needs `create` to fail while later writes succeed. | reject |
| 19 | E | `None` `fused_label`/`track_id` skews the summary | false | Wire faces always carry both. | reject |
| 20 | E | `kind` is not validated against `KINDS` | low | Internal callers pass literals. | reject |
| 21 | E | One `busy` id lets a row be deleted twice while its delete is in flight | low | Needs two quick clicks on two rows. The 404 is shown in Russian. | reject |
| 22 | E | `onSessionEnd` fires when the server never confirmed a session | low | The only harm is one extra list fetch. | reject |
| 23 | E | The `fetchSessions` shape check skips `duration`/`status` | low | The server is the only producer. | reject |
| 24 | E | `progress` is null for a running session although the contract says 0–1 | false | The frozen matrix sets "1 if finished, else null". A change would edit the spec. | reject |
| 25 | V | The lifespan fallback when the store fails to open is untested | gap | Pre-verified. | patch |
| 26 | V | The duration of a recovered stale session is not pinned | gap | Pre-verified. | patch |
| 27 | V | The `QUEUE_MAX_FRAMES` cap is untested | gap | Pre-verified. Low risk at single-stream scope. | defer |

## Design Notes

Store the WS face objects as JSON per frame, not exploded rows: export derives rows; 1.10's timeline returns the same objects without a schema change. `frames.audio_ts` links a frame to its window, so face rows get `audio_label`/`speech` by lookup. Audio windows dedupe by primary key (`INSERT OR IGNORE`).

Size: ~20 streams × 8 fps × ~600 B ≈ 350 MB/h at full demo load; acceptable for the prototype, noted for 1.15.

## Verification

**Commands:**
- `cd backend && .venv/bin/pytest -q -m "not models"` -- expected: all pass.
- `cd frontend && npx vitest run && npm run build` -- expected: pass; URL guard empty.
- `docker build --network host` CPU and frontend images -- expected: success.

**Manual checks:**
- Browser via compose: run a live session, stop, download CSV/JSON, load both in pandas; `docker compose down && up` keeps the list.
