---
title: 'Live webcam analysis with overlay and latency readout'
type: 'feature'
created: '2026-10-03'
status: 'done'
baseline_commit: 'f842166697d3634a065d0d36c4ec38180e9e8674'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/interfaces.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/stack.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The inference worker from 1.2 is not reachable from the app. `/ws/live` is a stub, and the UI shows only health. CAP-1 (webcam, boxes, emotion, p95 latency < 1 s shown in the UI) is the first demo milestone.

**Approach:** Start one `InferenceWorker` with the app. Implement the video and control parts of `/ws/live` per `interfaces.md`. Each live session is one worker source. Add a Russian live screen: the browser captures the webcam, sends JPEG frames, draws boxes, emotion and confidence on a canvas over the local `<video>`, and shows p50/p95 latency.

## Boundaries & Constraints

**Always:**
- Wire format exactly as `interfaces.md` Live session: `0x01` + f64 `capture_ts` (ms, client clock) + u32 `frame_id` + JPEG; `start`/`stop`; `session`, `frame_result`, `error`. Little-endian.
- `frame_result` echoes `frame_id` and `capture_ts`. `proc_ms` = server time from frame receipt to result send. Boxes are in pixels of the sent JPEG. `audio` is `null` while there is no audio block.
- Client: 8 fps by default, longest side ≤ 640 px, JPEG quality 0.8. Latency = render time of the overlay − `capture_ts`. p50/p95 over the last 60 s.
- One worker per backend process, started in the app lifespan without blocking startup. Models load once. JPEG decoding never runs on the event loop thread.
- Each session registers its own source and unregisters it on `stop`, on disconnect, or on error. Several sessions at once are isolated.
- All UI and `error.message` text in Russian. No new runtime network dependencies.

**Never:**
- Microphone capture, audio processing, the fusion `config` message, persistence, export. These belong to 1.4 and 1.5. In this story `0x02` chunks and `config` are accepted and ignored.
- Changing the `/api/health` payload or the `interfaces.md` contract.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Start | `{"type":"start"}`, worker ready | `{"type":"session","session_id":<32 hex>}` once; source registered | N/A |
| Frame with face | `0x01` frame after start | `frame_result` with that `frame_id`, faces with box, `face`, `face_label`, `fused`, `fused_label`, `confidence` | N/A |
| Frames faster than worker | burst of frames | results only for the frames the worker took; dropped ids get no reply | N/A |
| Face leaves | frames without faces | `faces: []`; the client clears boxes; stale overlay (no result > 1 s) is also cleared | N/A |
| Frame before start | `0x01` without start | ignored | `error` "Сессия не начата" once |
| Bad binary | unknown type, < 13 bytes, undecodable JPEG | frame skipped, socket stays open | `error` message |
| Worker not ready / failed | `start` while loading or after failure | no session | `error` "Модели ещё загружаются…" or "Модуль распознавания недоступен…" |
| Worker dies mid-session | worker exits | session ends | `error`, socket closed with code 1011 |
| Disconnect | client closes | source unregistered | N/A |
| Camera denied | `getUserMedia` rejects | UI message in Russian, no socket | N/A |

</frozen-after-approval>

## Code Map

- `backend/inference/worker.py` -- `InferenceWorker`: `start()` (blocking until models load), `register_source`, `unregister_source`, `submit_frame(source_id, bgr, frame_id, ts)`, `results()` (raises `WorkerDied`), `stop()`. `FrameResult`/`ErrorResult` in `inference/types.py`. `FaceResult` already has the wire face shape; `box` is `(x, y, w, h)` in frame pixels. `ts` is in seconds and drives track expiry, so pass `capture_ts / 1000`. Do not change the worker.
- `backend/inference/worker.py` docstring -- spawn context: never start the worker at import time. Uvicorn (`CMD` in `backend/Dockerfile`) is fine.
- `backend/app/main.py` -- `lifespan` already probes the device. Start the worker there in a background thread and stop it on shutdown.
- `backend/app/ws/live.py` -- stub to replace.
- `backend/tests/test_health.py` -- `make_client` builds `TestClient(app)` with fake model files. The lifespan must survive a failing worker start. Tests inject a fake worker and must not spawn models.
- `frontend/src/App.tsx`, `frontend/src/api.ts`, `frontend/src/styles.css` -- health card, design tokens, Russian text. Keep the health card and add the live screen beside it.
- `frontend/package.json` -- no test runner yet. Deferred-work asks for vitest with the first UI logic story.
- `backend/bench/data/face.jpg` -- a real face for model tests and the latency client.

## Tasks & Acceptance

**Execution:**
- [x] `backend/app/inference_hub.py` -- owns the worker. State is `starting|ready|failed`. A background thread starts the worker. One drain thread reads `results()` and routes `FrameResult`/`ErrorResult` by `source_id` to per-session asyncio queues via `loop.call_soon_threadsafe`. On `WorkerDied`, it notifies all sessions. A factory hook lets tests inject a fake worker.
- [x] `backend/sources/webcam.py` -- parse a binary message into a typed frame. Decode the JPEG to BGR with `cv2.imdecode`. Raise a typed error on bad input.
- [x] `backend/app/ws/live.py` -- protocol state machine per the matrix. Decode via `asyncio.to_thread`. Serialise `frame_result`. Clean up the source in `finally`.
- [x] `backend/app/main.py` -- start and stop the hub in the lifespan.
- [x] `backend/tests/test_ws_live.py` -- every matrix row except "Camera denied", using a fake worker and `TestClient.websocket_connect`; plus a header round-trip test.
- [x] `backend/tests/test_ws_live_models.py` (marker `models`) -- real worker, `face.jpg` frames, at least one face in the `frame_result`.
- [x] `backend/bench/live_ws.py` -- `python -m bench.live_ws --url ws://localhost/ws/live --seconds 60 [--image ...]`: a client that sends frames at 8 fps and prints p50/p95 of result − capture and the result rate. It gives a scripted latency number without a browser.
- [x] `frontend/src/live/protocol.ts` -- encode the frame header, parse and type `frame_result`.
- [x] `frontend/src/live/latency.ts` -- 60 s sliding window, p50/p95.
- [x] `frontend/src/live/overlay.ts` -- scale boxes from JPEG size to the displayed video. Draw the box and "эмоция NN%" with Russian emotion names.
- [x] `frontend/src/live/LiveView.tsx` -- Start/Stop buttons, `getUserMedia` video only, a capture loop that skips a frame while `ws.bufferedAmount` is high, the overlay canvas, the latency readout, the stale-overlay clear, and error states.
- [x] `frontend/src/App.tsx`, `frontend/src/styles.css` -- mount the live screen, with a layout for a 16:9 video.
- [x] `frontend/package.json` + `frontend/src/live/*.test.ts` -- add vitest. Test header encoding, the percentiles and window eviction, box scaling and labels.

**Acceptance Criteria:**
- Given `docker compose up` (CPU) and `--profile gpu up`, when the user opens http://localhost and presses Start, then their video shows boxes with a Russian emotion and confidence, and a p50/p95 readout updates.
- Given the GPU profile on the target laptop, when a live session runs for 60 s, then the UI p95 is < 1000 ms. Record the measured p50/p95 (UI and `bench.live_ws`) in Implementation Notes.
- Given two browser tabs in live sessions, when one tab closes, then the other keeps receiving results.
- Given `npm run build`, then the build passes and the `dist` URL grep from AGENTS.md is empty.

## Design Notes

The latency readout measures the full loop: capture → encode → WS → worker → WS → canvas paint. The client stamps the render time inside the `requestAnimationFrame` that draws the result. The overlay is drawn over the live video, not over the analysed frame, so a lagging box is visible as lag. That is the honest CAP-1 measure.

Background worker start keeps `/api/health` and the UI reachable while models load (tens of seconds on CPU). `start` gets a Russian "loading" error until the worker is ready.

## Verification

**Commands:**
- `cd backend && .venv/bin/pytest -q -m "not models"` -- expected: all pass.
- `docker run --rm --network host -v ./backend:/src:Z -w /src emotion-recognition/backend:cpu sh -c "pip install -q --target /tmp/pt pytest httpx && PYTHONPATH=/tmp/pt python -m pytest -q -p no:cacheprovider -m models"` -- expected: pass (repeat with `--gpus all` and the gpu image).
- `cd frontend && npx vitest run && npm run build` -- expected: pass; the URL grep is empty.
- With the GPU stack up: `python -m bench.live_ws --url ws://localhost/ws/live --seconds 60` -- expected: p95 < 1000 ms.

**Manual checks:**
- In the browser, the 60 s p95 shown in the UI is < 1000 ms. Boxes appear and disappear within 1 s when a face enters or leaves.

## Implementation Notes

- `sources/` is a new package; `backend/Dockerfile` now copies it. `requirements-dev.txt` gains `opencv-python-headless` (JPEG decode in the model-free `/ws/live` tests).
- `tests/conftest.py` installs a fake worker for every test not marked `models`, so `make_client` and the lifespan never spawn models. The old `/ws/live` stub test was removed from `test_health.py`.
- `/ws/live` runs a receive task and a send task in one anyio task group; the send task is the only socket writer. `proc_ms` is measured from message receipt to just before the `frame_result` is sent.
- Extra errors beyond the matrix (all Russian): repeated `start` ("Сессия уже начата"), malformed control JSON, unknown control type, worker `frame_failed` ("Не удалось обработать кадр"). `stop` keeps the socket open; a new `start` opens a new session.

**Measured latency (2026-10-03, local, no Docker: uvicorn + CPU device, i.e. not the target GPU profile):**
- `bench.live_ws --url ws://localhost:8000/ws/live --seconds 20`: result − capture p50 20.7 ms, p95 24.5 ms; `proc_ms` p50 18.5 / p95 21.8 ms; 8.0 results/s, face in 100 %. Two clients in parallel: 8.0/s each; the longer one kept receiving after the other closed.
- UI (headless Chrome, fake camera with `bench/data/face.jpg`, via `npm run dev`): p50 21 ms, p95 37 ms over ~15 s; box and "радость 100%" drawn on the face; overlay cleared on Stop.
- **Not yet measured:** GPU profile in Docker, UI p95 over a full 60 s through nginx (`http://localhost`), Docker-image `-m models` runs. The Docker daemon was not running during implementation (starting it needs sudo).

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-03): layers blind-hunter, edge-case-hunter, verification-gap.

| # | Layer | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | blind, edge | `_end_session_sync` calls `hub.unregister` on the event loop | low | `unregister_source` does `_inq.put(timeout=5.0)`; this is normally instant but can block the loop. Direct fix: `await asyncio.to_thread`. | patch |
| 2 | blind, edge | Hub start/stop race leaks the worker | low | `_start_worker` checks `_stopping` and only then assigns `_worker`; `stop()` in between sees no worker. A lock around check+assign fixes it. | patch |
| 3 | blind | A dead worker is never restarted; health does not show it | medium | Real, but restart and health signalling are not in this intent, and the health payload is frozen. Belongs to 1.9 hardening. | defer |
| 4 | blind | UI does not retry while models load | low | Matches the matrix row (Russian "loading" error, no session). Auto-retry is new behaviour. | reject |
| 5 | blind, edge | `_drain` spins on repeated non-WorkerDied errors | low | `continue` with no wait. Trivial `self._stopping.wait(0.5)`. | patch |
| 6 | blind, edge | Hidden tab: rAF paused, one huge sample skews p95 | medium | `render()` stamps the render time after the tab returns; the capture interval is throttled. Skip capture and samples while `document.hidden`. | patch |
| 7 | blind | Coalesced results bias p95 low; "Кадров за 60 с" counts samples | low | Coalescing needs two results inside one 16 ms frame at 8 fps, which is rare. The label is a direct rename. | patch (label only) |
| 8 | blind, edge | Client `sizes` map unbounded when no results come | low | Entries are deleted only on a result. Direct cap. | patch |
| 9 | blind | No session cap, unbounded `out` queue, no Origin check | low | `out` is fed at worker rate. A foreign page only gets results for frames it sends itself. Not in intent. | reject |
| 10 | blind, verif | No tests for ErrorResult → error / unknown_source silence, repeated start, shutdown stopping the worker | gap | Pre-verified by the verification layer: no test puts an ErrorResult on a session queue; `FakeWorker.stopped` is never asserted. | patch |
| 11 | blind, verif | Docker/GPU acceptance not run | gap | The Docker daemon was down during implementation. It is part of verification and runs before presentation. | patch (verification run) |
| 12 | blind | Story `in-review` vs sprint `in-progress` | false | The sprint sync to `review` happens at presentation (step 5), by design. | reject |
| 13 | blind | `epic-1-context.md` change unrelated to 1.3 | low | It is the fusion-wording follow-up of the spec fix. It will be committed separately as `docs:`. | patch (commit split) |
| 14 | blind | Bench deps and container command undocumented | low | `websockets` comes with `uvicorn[standard]` in the image. The fix is a spec/docs edit. | reject |
| 15 | blind, edge | No `onerror`; `noticeTimer` not cleared on Stop/unmount | low | The timer leak is real; clearing it is a direct fix. The 1006 wording is cosmetic. | patch (timer) |
| 16 | verif | `proc_ms` only asserted `>= 0` | gap | Diagnostic field; CAP-1 uses the client-side latency. | defer |
| 17 | verif | Overlay clear / stale rule has no automated test | gap | The repo covers it by the manual browser check listed in Verification. | defer |
| 18 | verif, edge | `ValueError`/`AttributeError` from `submit_frame` (closed queue after `_kill`) escapes `_on_binary` | low | `_require_running` and `put_nowait` can race with the drain thread's `_kill`. Widen the `except` in `hub.submit_frame`. | patch |
| 19 | verif | Pruned-frame fallback echoes `res.ts*1000`, not the exact `capture_ts` | low | Only after 512 pending frames without any result. Rare. | reject |
| 20 | edge | JPEG header with huge dimensions → OOM | low | Needs a crafted client on localhost. The fix is SOF parsing, which is new logic. | reject |
| 21 | edge | Non-JPEG images accepted | low | Harmless; the decoder is the same. | reject |
| 22 | edge | Deeply nested JSON → RecursionError ends the connection | low | `json.loads` raises RecursionError. Adding it to the `except` is direct. | patch |
| 23 | edge | Unmount during getUserMedia leaves the camera on | low | The cleanup does not bump `attemptRef`. One-line fix. | patch |
| 24 | edge | Camera track `ended` (unplugged) not handled | medium | The frozen last frame keeps being analysed with no UI message. Small `onended` → fatal Russian message. | patch |
| 25 | edge | Resolution change mid-session misplaces in-flight boxes | low | Rare; at most one result in flight. | reject |
| 26 | edge | bench: `--seconds <= --warmup`, `--fps <= 0` | low | Operator misuse of a dev tool; needs new guards. | reject |
| 27 | edge | bench: ConnectionClosed on the final stop discards measurements | low | Only when the worker died; the run is invalid then anyway. | reject |
| 28 | edge | bench receiver dies on a malformed message | low | The server is ours and the contract is fixed. | reject |

Patches applied by the implementer (12 items; the disconnect cleanup uses a shielded `run_in_executor`, because a cancelled `to_thread` dropped the job). Lead re-verified: units 64 passed ×3, models 21/21 (local CPU venv), vitest 15, build ok, URL grep empty. Docker/GPU verification still pending (daemon down).

Docker verification (lead, 2026-10-03, after patches):
- The frontend image build failed the Dockerfile URL guard: the UI text contained `http://localhost`, and the local AGENTS.md grep excludes localhost. Text changed to "по адресу localhost (без HTTPS камера доступна только там)". All three images now build with `--network host`.
- `-m models` inside the images: CPU 21/21, GPU (`--gpus all`) 21/21.
- GPU stack (`--profile gpu`, device cuda, RTX 3050 Laptop), `bench.live_ws` through nginx (`ws://frontend/ws/live`, run from the backend container) for 60 s: 480/480 results, 8.0/s, face in 100%, latency p50 22.8 ms / p95 23.9 ms, proc_ms p95 20.8 ms. PASS.
- Two concurrent clients on GPU (30 s and 10 s): both at 8.0/s, p95 about 24 ms; the longer one kept receiving after the shorter one closed.
- CPU stack: health ok (device cpu), 20 s bench p95 24.3 ms, 8.0/s.
- Pending for the human done-checkpoint: the 60 s p95 in the browser UI with a real webcam on the GPU profile.
