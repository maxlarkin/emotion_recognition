---
title: 'RTSP ingest with camera profiles and stream management'
type: 'feature'
created: '2026-10-10'
status: 'done'
baseline_commit: '5c8a473ae0fad7ee3190c4fc5255b94aaddf8d3a'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/interfaces.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/stack.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/camera-profiles.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The backend accepts only the browser webcam. CAP-10 needs RTSP cameras (emulated or real IP cameras) analysed in parallel through the same worker, managed at runtime, and shown on the grid and stream screens that story 1.6 built on mocks.

**Approach:** Server-side RTSP sources (PyAV, one decode thread per stream) feed the shared InferenceHub with per-stream camera profile and settings. A stream manager persists streams in SQLite, runs each as an `rtsp` session, reconnects with backoff and exposes `/api/streams*`, `/api/system` and `/ws/streams` exactly per `interfaces.md`. A `demo-cams` compose profile publishes three synthetic cameras. The frontend switches its `streams` and `system` groups from mocks to the real API.

## Boundaries & Constraints

**Always:**
- Contract exactly per `interfaces.md` (RTSP streams WS, REST streams/system rows, stream settings, face `decision`/`face_px`). `stream_result` boxes are in pixels of the JPEG sent just before; `face_px` in detection-size pixels.
- Ingest per `stack.md` RTSP ingest: TCP transport, 5 s open timeout; video sampled to `settings.analysis_fps` (default 5); frame scaled to the detection size (`auto` = camera type size from `camera-profiles.md`, longest side) before submit; tile JPEG longest side ≤ 640; audio track (if any) resampled to 16 kHz mono and sent in 500 ms chunks on the same media clock as frames; reconnect backoff 1, 2, 4 … 30 s.
- GPU image tries NVDEC (PyAV `hwaccel` cuda) and silently falls back to CPU decode per stream; the decode mode is logged and reported in the stream row as `decode: "nvdec"|"cpu"` (extra field, ignored by older clients).
- Decoding, scaling and JPEG encoding never run on the event loop. One stream's failure or slowness never stalls another stream or `/ws/live`.
- Decision in this story: `abstain` when `face_px < settings.min_face_px` or the track is younger than 1 s, else `none` (watch/alert/positive arrive with story 1.10). `/ws/live` faces get the same two fields (min face 48 px).
- Each stream run is a session (kind `rtsp`, `stream_id`) recording frames and audio windows like live sessions; `ts_ms` is server decode time (ms). Session spans start → stop/disable/delete; reconnects and settings changes keep it.
- Streams and their settings persist in SQLite; `DEFAULT_STREAMS` (JSON list of `{name, site, url, camera_type}`) seeds only an empty list. URLs keep credentials in the DB; API responses mask the password (`rtsp://user:***@host/...`); a PATCH whose `url` equals the masked form leaves the URL unchanged.
- `/api/system` and the `system` WS message: psutil CPU/RAM, NVML GPU util/VRAM, nulls without GPU.
- Russian `detail` texts for API errors; 404 «Поток не найден»; invalid body 422.
- New runtime dependency `av` (PyAV, pinned) installed in its own Dockerfile layer so the torch layer stays cached. Exactly one onnxruntime per image still holds.
- Frontend: `MOCKED.streams` and `MOCKED.system` become `false`; stubbed-fetch tests for every real streams/system adapter; the real `/ws/streams` client resyncs streams and incidents after reconnect and marks feeds stale while disconnected (deferred from 1.6).

**Never:**
- Event rule, incidents, `/api/settings`, stream interval log (story 1.10); 20-clip demo set and throughput tuning (story 1.8); file analysis (1.11).
- Recording RTSP video, storing frames or face images, ONVIF/PTZ, RTSP auth beyond URL credentials.
- One OS process per stream; a second inference worker; changing the `/ws/live` wire format beyond the two additive face fields.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Add stream | `POST /api/streams {name,url,camera_type}` reachable URL | `{"id"}`; `stream_status` connecting → running; `stream_result` + binary frame per analysed frame at ~analysis fps; session kind `rtsp` listed | N/A |
| Defaults | POST without `camera_type`/`site`/`settings` | `sector`, `""`, defaults `{enabled:true, analysis_fps:5, detection_size:"auto", min_face_px:48}` | N/A |
| Bad body | empty name, non-`rtsp(s)://` url, unknown camera_type, bad settings value | 422 | Russian `detail` |
| Unreachable | URL that never answers | state `connecting` then `reconnecting` with `message`; retries at 1, 2, 4 … 30 s | other streams unaffected |
| Stream drops | source stops mid-run | `reconnecting`; resumes `running` when the source returns, same session | N/A |
| Disable | `PATCH {settings:{enabled:false}}` | source stopped, state `stopped`, session finished | N/A |
| Settings change | `PATCH {settings:{analysis_fps:3}}` or `camera_type`/`url` | only that stream's source restarts; new fps/detection size apply; session kept | N/A |
| Delete | `DELETE /api/streams/{id}` | 204; source stopped; session finished and exportable; gone from list | unknown id 404 |
| Reconnect now | `POST /api/streams/{id}/reconnect` | backoff reset, immediate retry | disabled stream 409 «Анализ этой камеры выключен» |
| Test URL | `POST /api/streams/test {url}` | `{ok:true,resolution,has_audio}` within ~5 s, or `{ok:false,error}` Russian | never hangs > 6 s |
| Masked URL | GET list | password replaced by `***`; PATCH with the masked url keeps the stored one | N/A |
| Persistence | backend restart | streams back with settings, enabled ones start | N/A |
| Seed | empty table, `DEFAULT_STREAMS` set | streams created once; not re-added after delete + restart | invalid JSON logged, ignored |
| Small faces | face narrower than `min_face_px` | face `decision:"abstain"`; otherwise `none` after 1 s | N/A |
| No audio track | security camera | `audio: null` in results | N/A |
| WS subscribe | `{"type":"subscribe","stream_ids":[...]}` | only those streams' frames/results; status and system for all | bad message ignored |
| Slow client | client reads slower than streams produce | only the latest frame per stream is kept for that client | N/A |
| System | `GET /api/system`, WS each 1 s | psutil + NVML fields; GPU fields null in the CPU image | N/A |
| Worker not ready | hub loading or failed | streams stay `connecting`, retried | N/A |

</frozen-after-approval>

## Code Map

- `backend/app/inference_hub.py` -- `register(loop, queue)`, `submit_frame(sid, bgr, frame_id, ts)`, `submit_audio(sid, pcm, ts)`, `unregister`; results routed per source to an asyncio queue; `WorkerGone` on worker death; `WORKER_FACTORY` test hook. Reuse as is (thread-safe submits).
- `backend/inference/types.py` -- `FrameResult`, `FaceResult.box` in submitted-frame pixels, `AudioResult`; `ts` in seconds drives track expiry and audio matching (same clock for frames and audio).
- `backend/app/ws/live.py` -- `face_to_wire`, `audio_to_wire` (reuse for streams), `LiveConnection` pattern (send loop, store calls, WorkerGone). Add `decision`/`face_px` in `face_to_wire` via a small per-source track-age helper.
- `backend/app/store.py` -- `SessionStore.create(id, "rtsp", stream_id)`, `record_frame`, `record_audio`, `finish`; schema is `CREATE TABLE IF NOT EXISTS` (add a `streams` table the same way, no migration); writes go through the writer thread.
- `backend/app/main.py` -- lifespan creates store and hub; add the stream manager after them and stop it before the hub.
- `backend/app/api/sessions.py` -- router/`_store` pattern, Russian `detail`; follow it for streams/system routers.
- `backend/Dockerfile`, `requirements*.txt` -- add PyAV in a separate layer after the torch/onnxruntime layer.
- `docker-compose.yml`, `.env`, `.env.example`, `scripts/check-compose.sh` -- add `demo-cams` profile services and `DEFAULT_STREAMS` passthrough; keep "exactly one backend per profile" true.
- `backend/tests/` -- fake worker via `inference_hub.WORKER_FACTORY`; TestClient patterns in `test_ws_live.py`, `test_sessions.py`.
- `frontend/src/api/index.ts` (`MOCKED`), `real.ts`, `streamsSocket.ts`, `state/appData.ts` -- flip `streams`/`system`; add resync/stale handling; tests in `src/api/real.test.ts`, `src/state/appData.test.ts`.
- `bench/data/face.jpg`, `bench/data/speech.wav` (+ `ATTRIBUTION.txt`) -- inputs for the synthetic demo cameras.

## Tasks & Acceptance

**Execution:**
- [x] `backend/sources/rtsp.py` -- `RtspSource` thread: open (TCP, 5 s timeout, NVDEC try → CPU), decode, sample to fps, scale to detection size, tile (≤ 640) cache by frame_id, audio resample/chunk, media-clock timestamps, stop flag; `probe(url)` for the test endpoint -- ingest.
- [x] `backend/app/streams.py` -- `StreamManager`: CRUD + validation + masking, persistence via store, `DEFAULT_STREAMS` seed, per-stream run loop (hub register, backoff, status), result handling (face wire + decision/face_px with box rescale to tile, JPEG encode in a thread, store writes, fan-out to WS clients), session lifecycle -- core.
- [x] `backend/app/system_load.py` -- psutil/NVML snapshot shared by REST and WS -- system load.
- [x] `backend/app/api/streams.py`, `backend/app/api/system.py`, `backend/app/ws/streams.py` -- routes per `interfaces.md`; WS client with latest-per-stream buffer, subscribe filter, 1 s system ticks, initial statuses on connect -- API.
- [x] `backend/app/store.py`, `backend/app/main.py`, `backend/app/ws/live.py` -- `streams` table + accessors; wiring; `decision`/`face_px` on live faces -- integration.
- [x] `backend/requirements-stream.txt` (av pinned), `backend/Dockerfile` -- separate install layer -- dependency.
- [x] `docker-compose.yml`, `demo-videos/demo-cams.sh` (or emulator command), `.env.example`, `scripts/check-compose.sh` -- `demo-cams` profile: pinned `bluenviron/mediamtx:1.21.1-ffmpeg` for mediamtx and the emulator; three synthetic cameras (webcam 640×480 with audio, sector 1920×1080 with audio, security 2560×1440 small face, no audio) at 10 fps; example `DEFAULT_STREAMS` -- demo.
- [x] `backend/tests/test_streams.py` (+ `test_rtsp_source.py`) -- matrix rows with a fake worker and a fake/generated source (PyAV can decode a local file or `testsrc` through the same code path) -- tests.
- [x] `frontend/src/api/index.ts`, `real.ts`, `real.test.ts`, `streamsSocket.ts`, `state/appData.ts` (+ test) -- switch, adapters tested, resync/stale -- frontend.
- [x] `AGENTS.md` commands section -- how to run `demo-cams` (`--profile cpu --profile demo-cams`) -- docs.

**Acceptance Criteria:**
- Given `docker compose --profile cpu --profile demo-cams up` (and the GPU equivalent), when the UI opens, then the three demo cameras appear on `/streams` with analysed frames, boxes, fps and p50/p95, and their sessions are exportable after delete.
- Given the GPU profile on the RTX 3050, when a stream starts, then the log and stream row say whether NVDEC was used; with NVDEC unavailable the stream still runs on CPU decode. Record the outcome in Implementation Notes.
- Given three demo streams running, when one emulator stream is stopped, then the other two keep their fps and the stopped one shows «Подключение» and recovers when it returns.
- Given `pytest -q` and `npm test` / `npm run build`, then all pass and the `dist` URL grep is empty.

## Design Notes

Clocks: frames and audio of a stream use media time mapped once to server time (`t0_server + (pts − pts0)`), so the worker's media-time audio matching works; `decode_ts` (ms) is the server wall time when the frame was decoded, kept by `frame_id` for the result message and the session `ts_ms`.

Fan-out: the manager keeps, per WS client, `dict[stream_id → (jpeg, result)]` plus a wake event; the client task sends binary then JSON for each pending stream and drops anything overwritten meanwhile.

## Verification

**Commands:**
- `cd backend && .venv/bin/pytest -q` -- expected: all pass (new stream tests included).
- `cd frontend && npm test && npm run build && grep -rEo "https?://[^\"' )]*" dist | grep -vE "www\.w3\.org/"` -- expected: pass, grep empty.
- `docker build --network host --build-arg VARIANT=cpu -t emotion-recognition/backend:cpu backend` (and gpu) then `docker compose --profile cpu --profile demo-cams up -d` -- expected: three running streams via `curl -s localhost/api/streams`.

**Manual checks:**
- `/streams` grid and `/streams/:id` with demo cameras on CPU and GPU profiles; stop/start one emulator stream; NVDEC log line on GPU.

## Implementation Notes

- Implemented inline (no subagent). Backend: `sources/rtsp.py` (decode loop, `probe`), `app/streams.py` (manager + validation), `app/faces.py` (decision/face_px, shared with `/ws/live`), `app/system_load.py`, `app/api/streams.py`, `app/api/system.py`, `app/ws/streams.py`; `store.py` gains `streams` and `meta` tables. PyAV `av==18.1.0` in `requirements-stream.txt`, own Docker layer (placed before the weights layer, so a PyAV bump re-fetches weights but never torch).
- Demo: `demo-videos/demo-cams.sh` publishes cam1 webcam 640×480 + audio, cam2 sector 1920×1080 + audio, cam3 security 2560×1440 without audio, built live from `bench/data`; `demo-videos/demo-cams.json` is the `DEFAULT_STREAMS` list; AGENTS.md documents the commands. One pinned image `bluenviron/mediamtx:1.21.1-ffmpeg` serves MediaMTX and the emulator.
- NVDEC (AC2): on the RTX 3050 GPU profile all three demo streams opened with `decode=nvdec` (log `stream … opened: …, decode=nvdec`); the CPU image decodes on CPU. The PyAV 18.1.0 wheel ships FFmpeg with the cuda hwaccel.
- Isolation (AC3, GPU profile): killing the cam3 publisher → cam3 `reconnecting` («Поток закончился. Повтор через 1 с», then «Поток не найден по этому адресу») and back to `running` after ~6 s; cam1/cam2 kept producing (3.9 results/s each while cam3 was down).
- Throughput (recorded for story 1.8, not a 1.7 criterion): 3 demo streams at analysis_fps 5 give ~3.3–3.5 results/s each on GPU (p50 ~360 ms, p95 ~470–520 ms in the browser) and ~2.1/s on CPU (backend ~940 % CPU). All streams get the same rate: the worker is bound by YuNet on CPU at 1920/2560 detection sizes, which `stack.md` (Detection cost) assigns to story 1.8 (YuNet on ORT CUDA or detect every 2nd frame).
- Frontend: `MOCKED.streams`/`system` = false; adapters covered by `src/api/real.test.ts`; `/ws/streams` client reports open/closed, AppData clears feeds on close and reloads streams/incidents after a reconnect; a stream row is reread when it turns `running` (audio, resolution, decode are known only then). Mocked events still carry demo stream ids, so `knownIncidents` hides incidents of unknown streams until story 1.10; the mocked stream log answers for any id.
- Verified: `pytest -q` 153 passed (24 stream + 6 source tests new); `npm test` 124 passed; `npm run build` + URL grep empty; `scripts/check-compose.sh` four profiles ok; CPU and GPU profiles with demo-cams in Docker, grid and stream detail in Chrome with real frames, boxes, voice and latency.

## Spec Change Log

## Review Triage Log

Layers: blind-hunter (BH), edge-case-hunter (EC), verification-gap (VG). Route: P patched, D deferred, R rejected.

| # | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|
| 1 | `mask_url` 500 on a non-numeric port; IPv6 brackets lost (EC, BH) | high | `parts.port` raised in `row()`; now regex replaces only the password span, tested | P |
| 2 | Masked `***` saved as the password on edit, accepted on create (EC, BH) | medium | `unmask_url` restores the stored password; create rejects `:***@` (422), tested | P |
| 3 | Concurrent PATCH/DELETE race resurrects deleted streams / KeyError (BH, EC) | medium | Awaits between read and write; per-stream `asyncio.Lock` on update/delete/reconnect | P |
| 4 | Hub source leaked when halted during `register` (VG, BH, EC) | medium | Thread now unregisters and exits if stopped after registering | P |
| 5 | After WorkerGone the thread keeps decoding; status flips error→connecting (VG, BH, EC) | medium | WorkerGone now stops the thread, finishes the session, keeps `error`; tested | P |
| 6 | Results dropped from the session when the tile was evicted (BH) | medium | Decode time/scale kept apart from tiles (256 vs 16); recorded without a tile | P |
| 7 | Stale results of a halted session reach the next one (EC) | low | Queue drained in `_halt` | P |
| 8 | `_run` racing `_halt` (EC) | low | Mutations now serialised by the per-stream lock | P |
| 9 | Probe threads starve the default executor; «за 5 с» vs 6 s (BH, EC, VG) | medium | Own 2-thread pool, slot counter released when the probe really ends, text without a number | P |
| 10 | JPEG encoded for streams nobody watches; unbounded backlog (BH, EC) | low | Encode only for clients that want the stream; skip WS push above 4 queued results | P |
| 11 | PTS jump back/forward skips all frames (BH, EC) | medium | Clock re-anchors on jumps > 2 s and resets the sampler | P |
| 12 | Audio chunk timestamps drift after gaps (BH, EC) | low | Cutter restarts a chunk when packet time departs > 100 ms; tested | P |
| 13 | Probe returns ok with resolution 0×0 (EC) | low | Raises «Не удалось прочитать кадр» | P |
| 14 | Track age resets on one missed detection (EC) | low | Tracks forgotten only after 1 s unseen; tested | P |
| 15 | Seed flag burned by an all-invalid DEFAULT_STREAMS (BH) | low | Flag set only when a stream was saved; tested | P |
| 16 | `/ws/streams` deque drops statuses (BH) | medium | Statuses coalesced per stream, system latest-only, events queued | P |
| 17 | Other tabs not told about rename/delete; refresh storm on 20 cameras (BH) | low | Frontend rereads the list on any status change, coalesced (500 ms) | P |
| 18 | Floats accepted as settings; name length before strip; looser test URL check; unused `MSG_DISABLED` (BH) | low | Fixed; float test added | P |
| 19 | No test for /ws/live `decision`/`face_px` (VG, BH) | medium | `test_frame_decision_abstains_then_none` | P |
| 20 | No test for re-enable / disabled create+restart / url and camera-type restart (VG) | medium | Three tests added | P |
| 21 | Socket connection events untested (VG) | medium | `src/api/streamsSocket.test.ts` with a fake WebSocket | P |
| 22 | Subscribe test asserted nothing; small-face test could pass on track age (BH) | low | Asserts statuses of unsubscribed streams; reads a result older than 1.5 s | P |
| 23 | Test helpers imported from `test_ws_live` (BH) | low | Developer convenience, no named breakage; moving fixtures is more than a correction | R |
| 24 | Sprint status `in-progress` while spec `in-review`; `.env` lacks DEFAULT_STREAMS (BH) | false | Sprint moves to `review` at step 5 by design; DEFAULT_STREAMS stays out of `.env` on purpose (documented in `.env.example`, AGENTS.md) | R |
| 25 | No regression test for the media-clock jump (self) | low | Needs a clip with a PTS discontinuity; muxers reject one | D |
