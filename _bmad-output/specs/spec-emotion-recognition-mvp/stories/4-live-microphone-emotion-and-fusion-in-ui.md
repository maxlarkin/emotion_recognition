---
title: 'Live microphone emotion and fusion in UI'
type: 'feature'
created: '2026-10-03'
status: 'done'
baseline_commit: 'd878e16cc631f24460e33de24ec5e878d5142919'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/interfaces.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/stack.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The worker already runs VAD → SER and late fusion, but `/ws/live` drops audio chunks (`0x02`) and the `config` message, and the browser never captures the microphone. CAP-2 asks for the vocal emotion and a fused emotion while the user speaks, "no speech" in silence (fused = facial), and a muted mic must not break video.

**Approach:** The live screen also captures the microphone, converts it to PCM16 mono 16 kHz, and sends 500 ms `0x02` chunks on the same socket. The server forwards them to the session's worker source and honours `config` per session. The UI shows the audio state, the facial emotion and the fused emotion, and has a mic on/off toggle.

## Boundaries & Constraints

**Always:**
- Wire format per `interfaces.md`: `0x02` + f64 `capture_ts` (ms, client clock, first sample of the chunk) + PCM16 LE. Use the same clock as video frames, because the worker matches audio to frames in media time.
- `frame_result.audio`: `null` when the worker attached no window. A speech window gives `speech: true`, `probs` with all 7 keys, and `label`. A silent window gives `speech: false`, `probs` with all 7 keys equal to 0, and `label: null`.
- `config` sets the fusion weights for this session only. Other sessions keep their own weights. Invalid weights (not numbers, negative, both 0) get a Russian `error`, and the current weights stay.
- Video works with no microphone, with the mic denied, or with the mic muted. In those cases the fused emotion equals the facial one.
- Capture at the device's native rate and resample to 16 kHz in an AudioWorklet (low-pass, then decimate). Do not force the `AudioContext` sample rate, because Firefox rejects a mismatched MediaStream source. The worklet ships in the bundle, with no CDN.
- All UI and error text is in Russian.

**Never:**
- Persistence, export, the RTSP or file paths.
- Changing `interfaces.md` or the `/api/health` payload.
- An audio-only session with no camera.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Speech | `0x02` chunks with speech, plus frames | after ≥ 3 s, `frame_result.audio` has `speech:true`, 7-key `probs` and `label`; `fused` = renormalise(0.7·face + 0.3·audio) | N/A |
| Silence | silent chunks plus frames | `audio.speech:false`, `probs` all 0, `label:null`; `fused == face` | N/A |
| No audio sent | frames only (mic off or muted) | `audio:null` (or `null` again 2 s after the last window); `fused == face` | N/A |
| Audio before start | `0x02` with no session | ignored | `error` "Сессия не начата" (once, shared with frames) |
| Bad audio | odd byte count or empty PCM | chunk skipped, socket open | `error` message |
| Config valid | `{"type":"config","fusion":{"w_face":1,"w_audio":0}}` | this session's fused == face; another session is unaffected | N/A |
| Config invalid | missing, negative or both 0 | weights unchanged | `error` "Некорректные веса объединения" |
| Mic denied or missing | `getUserMedia` audio rejects | the session runs video-only; the UI shows that the mic is unavailable | N/A |

</frozen-after-approval>

## Code Map

- `backend/app/ws/live.py:177` -- `AudioPacket` is dropped (`return`); `config` is a no-op at `:149`. `audio_to_wire` is at `:53`. `_on_binary` already handles "not started" and `HubUnavailable`.
- `backend/sources/webcam.py` -- `parse_message` already returns `AudioPacket(capture_ts ms, pcm bytes)`. The odd-length check belongs there.
- `backend/app/inference_hub.py` -- add `submit_audio` and `set_fusion(source_id, ...)`, mirroring `submit_frame` (same exception widening).
- `backend/inference/worker.py` -- `InferenceWorker.submit_audio(source_id, bytes, ts)` takes seconds. `set_fusion` is global today: `Engine.weights`, used in `process_frames` → `fuse(face, audio_vec, self.weights)`. Make the weights per source (`_SourceState`, defaulting to the config weights) by adding an optional `source_id` to the `("config", ...)` message. Keep the global call working: `test_inference_models.py:276` and `test_inference_units.py:243` use it. `AudioResult.probs` is `None` in silence. Worker-side `bad_config` errors carry `source_id=None`, so validate in `live.py`.
- `backend/tests/conftest.py` -- `FakeWorker` (with the `push` helper). Extend it with `submit_audio` and per-source fusion recording.
- `backend/bench/data/speech.wav` -- real speech (16 kHz) for the model test.
- `frontend/src/live/protocol.ts` -- `FrameResult.audio` is typed `unknown`. Add the audio type, its parsing and `encodeAudio`.
- `frontend/src/live/LiveView.tsx` -- `LiveRun` (socket, capture loop, render). The `getUserMedia` call at `:286` has `audio: false`. Keep the video request separate, so a denied mic never blocks video.
- `frontend/src/live/overlay.ts` -- `EMOTION_RU`, `faceLabel` (draws `fused_label`). Reuse them for the panel.

## Tasks & Acceptance

**Execution:**
- [x] `backend/inference/worker.py` -- per-source fusion weights via an optional `source_id` on `set_fusion`; the global behaviour is unchanged.
- [x] `backend/sources/webcam.py` -- reject empty or odd-length PCM with `BadMessage`.
- [x] `backend/app/inference_hub.py` -- `submit_audio`, `set_fusion(source_id, w_face, w_audio)`.
- [x] `backend/app/ws/live.py` -- forward audio (`ts = capture_ts / 1000`, off the loop), validate and apply `config`, serialise the audio block per Always.
- [x] `backend/tests/test_ws_live.py` + `conftest.py` -- every backend matrix row.
- [x] `backend/tests/test_ws_live_models.py` -- real worker: `speech.wav` chunks plus `face.jpg` frames on one clock give `speech:true` and 7-key probs, with `fused != face`. Frames alone give `audio:null` and `fused == face`.
- [x] `frontend/src/live/pcm-worklet.ts` -- AudioWorklet: mono mix, low-pass and decimate to 16 kHz, Int16, post 500 ms chunks.
- [x] `frontend/src/live/resample.ts` -- pure resampler used by the worklet, so it can be unit-tested.
- [x] `frontend/src/live/protocol.ts` -- `encodeAudio`, `AudioBlock` type and parsing.
- [x] `frontend/src/live/LiveView.tsx` -- optional mic capture, chunk timestamps (`now − chunk duration`), mic toggle «Микрофон вкл/выкл». Panel: «Голос»: «микрофон выключен», «нет доступа к микрофону», «ожидание…», «нет речи» or «<эмоция> NN%». «Лицо» and «Итог» show the largest face.
- [x] `frontend/src/live/*.test.ts` -- audio header bytes, resampler (rate, length, a 1 kHz tone keeps its amplitude, a 7 kHz tone is attenuated), audio block parsing, panel text selection.

**Acceptance Criteria:**
- Given the live screen with mic allowed (CPU and GPU profile), when the user speaks for 5 s, then «Голос» shows an emotion and «Итог» can differ from «Лицо». When silent, «Голос» shows «нет речи» and «Итог» equals «Лицо».
- Given a running session, when the mic is toggled off and on, then video results and latency continue without interruption, and «Голос» follows the toggle.
- Given the GPU profile, when speaking for 60 s, then the UI p95 stays < 1000 ms.

## Design Notes

Audio and frames share one clock (`performance.timeOrigin + performance.now()`). The worker attaches a window only if `0 ≤ frame.ts − window_end_ts ≤ 2 s`, so a wrong chunk timestamp shows up as `audio:null` while the user speaks. Stamp each chunk when the worklet message arrives, minus the chunk duration.

Silent windows get zero probs rather than `null`, so the "always 7 keys" contract line holds. `label: null` is what tells the UI "нет речи".

## Verification

**Commands:**
- `cd backend && .venv/bin/pytest -q -m "not models"` -- expected: all pass.
- `cd backend && MODELS_DIR=$(realpath ../.models) .venv/bin/pytest -q -m models` -- expected: pass. Repeat in both images per story 1.3.
- `cd frontend && npx vitest run && npm run build` -- expected: pass, and the Dockerfile URL guard passes (no `http://` in UI text).
- `docker build --network host` for the CPU, GPU and frontend images -- expected: success.

**Manual checks:**
- In the browser on the GPU profile: speech gives a vocal emotion, silence gives «нет речи», the mic toggle and a denied mic keep video running, and p95 is < 1000 ms.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-03): layers blind-hunter, edge-case-hunter, verification-gap.

| # | Layer | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | blind, edge | After mic off→on the worker's VAD `pending` and `SpeechWindow._chunks` keep pre-mute audio, so the first window mixes stale speech under a fresh `end_ts` | medium | `ser.py:37-51` keeps ≤ 3 s of chunks, and nothing resets them on a gap. Small fix in the worker: reset the source's VAD/window when a chunk's ts jumps past the expected end. | patch |
| 2 | blind, edge | «Микрофон выкл» keeps the track and AudioContext open (OS mic indicator stays on); chunks queued before "reset" can leak after unmute | medium | `MicCapture.setEnabled` only flips `track.enabled`. Close the capture on off and reopen it on on (the open path exists). | patch |
| 3 | blind | Default voice processing (EC/NS/AGC) may change prosody | maybe-false | No evidence that it degrades GigaAM on this setup; would need an A/B on recorded speech. Would be low at most. | reject |
| 4 | blind, edge | `self.fusion` survives `stop` and is re-applied to the next session | medium | Frozen intent: "`config` sets the fusion weights for this session only". Clear it on `stop`. | patch |
| 5 | blind | Stored weights are applied after the `session` reply, so the first frames use the defaults | low | `_start` sends `session` before `_apply_fusion`. A direct reorder. | patch |
| 6 | blind | `config` can stall the receive task for up to 5 s on a full queue | low | Needs 256 queued inbound messages; rare. The fix changes the worker API timeout. | reject |
| 7 | blind | Worklet allocates per render quantum | low | No demonstrated dropout; a refactor of the resampler buffers. | reject |
| 8 | blind, edge | Main-thread chunk timestamps jitter under jank | low | Jitter is tens of ms against a 0–2 s match window. A sample-counter clock is new logic. | reject |
| 9 | blind | Hint hardcodes 70%/30% | false | The UI never sends `config`, so the defaults are always in effect for the UI session. | reject |
| 10 | blind, verif | Missing tests: the matrix row "`null` again 2 s after the last window"; `MSG_FUSION_BUSY` (queue.Full) | gap | Pre-verified: no test covers either. The matrix audit requires the row. | patch |
| 11 | blind | Story/sprint status vs checked tasks | false | Status moves at presentation by design. | reject |
| 12 | blind | `parseAudio` defaults a missing `window_end_ts` to 0 | low | `protocol.ts:106`. Reject the block instead: direct. | patch |
| 13 | edge | Huge JSON integer weight → `math.isfinite` raises OverflowError and closes the connection | low | Confirmed: `math.isfinite(10**400)` raises. Direct fix in `parse_fusion`. | patch |
| 14 | edge | Weights like 1e308 overflow `fuse` → NaN | low | Finite huge weights pass. Direct fix: normalise by the sum in `parse_fusion` (fusion renormalises anyway, so semantics are unchanged). | patch |
| 15 | edge | `self.fusion` overwritten even when applying it was busy | low | Same handler as #4/#5; store the weights only after success. | patch (with #4) |
| 16 | edge | `ctx.resume()` may never settle | maybe-false | Would need a browser run without user activation; the context is created after a granted `getUserMedia`, which browsers treat as allowed. Low at most. | reject |
| 17 | edge | No upper bound on the PCM chunk size | low | Needs a crafted local client; adds a guard. | reject |
| 18 | verif | Worker control loop (config for an unregistered source, removals before configs) untested | gap | Filed as defer: UUID ids, at most a small per-stop leak; needs a loop-body test harness. | defer |
| 19 | verif | Browser audio path untested: the shared clock, chunk start = arrival − duration, worklet 8000-sample chunking/reset | gap | Pre-verified: no vitest imports `mic.ts` or `pcm-worklet.ts`. Extract pure helpers and test them. | patch |

Patches applied by the implementer (all 10 patch rows). Lead re-verified (2026-10-03):
- Units 93 passed ×2. `-m models` 22/22 in the local venv, the CPU image and the GPU image (`--gpus all`). vitest 43 passed. Build ok; the Dockerfile URL guard passes. All three images build with `--network host`.
- Browser smoke test on the GPU stack through nginx: headless Chrome with a fake camera (`face.jpg`) and a fake mic (looping `bench/data/speech.wav`).
  - After 4 s «Голос» showed «ожидание…». From 8 s on it showed «нейтрально 52–82%», and «Итог» showed «радость 75–84%» while «Лицо» showed «радость 100%». So fusion is visible.
  - Mic off: «Голос» showed «микрофон выключен» and «Итог» equalled «Лицо». Video kept running at 7.8 results/s.
  - Mic on again: «Голос» was back within 4 s.
  - Over the whole run: p50 30 ms, p95 45 ms, no UI errors.
- Not covered by the smoke test: silence in the browser (the fake mic loops speech; the backend tests cover the silence row), a denied mic (covered by the panel logic and tests), and a real webcam and mic, which is left to the human.
