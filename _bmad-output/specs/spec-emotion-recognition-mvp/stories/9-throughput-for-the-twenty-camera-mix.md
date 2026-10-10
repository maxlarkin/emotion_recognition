---
title: 'Throughput for the twenty-camera mix'
type: 'feature'
created: '2026-10-11'
status: 'in-review'
baseline_commit: 'a25dd69af089a780aa02377513b03f8cc87fda73'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/stack.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/camera-profiles.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** With the story-1.8 mix (6 webcam, 8 sector, 6 security) the GPU profile collapses. Within 90 s only 2/20 streams produce results, at ~0.02 results/s, with 11–14 s latency, 6 % GPU and host RAM at 13.6/15 GB. CAP-10 needs every stream at ≥ 5 analysed fps, `proc_ms` p95 ≤ 100 ms and decode-to-result p95 < 1000 ms for 10 min. There are two causes. YuNet runs on the CPU (69 ms at 1920, 164 ms at 2560). Full-size frames are pickled into a 256-deep queue, which puts gigabytes in flight.

**Approach:** Three steps, measured after each:
1. The GPU image detects with the YuNet ONNX on the ONNX Runtime CUDA EP. Measured 2.7 / 12 / 20.5 ms at 640 / 1920 / 2560.
2. Detect every Nth frame per source: N = 1 for webcam, 3 for sector/security. On the other frames the last boxes are carried and FER and fusion still run.
3. Frames reach the worker through per-source shared-memory rings, and only a small header crosses the queue.

## Boundaries & Constraints

**Always:**
- Same seams and contracts: one worker process; the `FrameResult`/`frame_result`/`stream_result` payloads; the latest-frame-slot semantics; `face_px`/`decision` behaviour; `/ws/live` unchanged for clients.
- GPU detection is equivalent to `cv2.FaceDetectorYN` (same score/NMS thresholds; boxes within 2 px of OpenCV on the bench face and on a few frames of the demo clips). If the CUDA EP is unavailable or the session fails, it falls back to OpenCV with a log line. The CPU image keeps OpenCV.
- `detect_every` is a per-source option given at registration (default 1). The stream manager derives it from `camera_type` (webcam 1, sector 3, security 3); `/ws/live` and file sessions use 1. A carried frame reuses the last detection's boxes and track ids, and fresh crops of the current frame go to FER. A track missed by the next detection pass disappears as before. `timings` report whether the frame was detected.
- Shared-memory transport: the parent owns one ring per source (4 slots, resized when the frame size changes). A seqlock (sequence written before and after each slot write) lets the child copy a slot and detect an overwrite; an overwritten read drops that frame and never yields a wrong result. Segments are unlinked on unregister, worker stop and worker death; the child never unlinks. Audio stays on the queue.
- Measurements on the RTX 3050, GPU profile, demo-cams mix with all 20 streams enabled. Step 1 and step 2 runs last 3 min each; the final run lasts 10 min. Record per-stream analysed fps (min/median), `proc_ms` p95 from worker timings, decode-to-result p95, GPU util, VRAM, backend CPU and RSS, host RAM. A CPU-image run of 3 min is recorded for reference. The table goes into Implementation Notes and `docs/benchmark-report.md` (a new "20-camera mix" section).
- A script `backend/bench/streams_ws.py` records `/ws/streams` for N seconds and prints these numbers, so the runs are reproducible. `proc_ms` per stream comes from a new additive stream-row field `proc_p95_ms`: the p95 of the worker's `timings.total` over the last 30 s, documented in `interfaces.md`.

**Never:**
- Lowering the profile detection sizes or the analysis fps of the demo mix to pass.
- A second worker process, model duplication, or changing the CAP-11 rule.
- Leaking shared-memory segments in `/dev/shm` after stop, stream delete or worker death.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| GPU detect | GPU image, bench face | same boxes as OpenCV (± 2 px), same count | CUDA EP missing → OpenCV + log |
| Odd size | frame 854×480 or 2560×1440 | padded to a multiple of 32 internally, boxes in frame pixels, clipped | N/A |
| Carried frame | sector source, frame 2 of 3 | result with the frame-1 boxes/ids, fresh FER, `timings.detected = 0` | N/A |
| New face between detections | face appears on a carried frame | shows up at the next detection (≤ N−1 frames later) | N/A |
| Live webcam | `/ws/live` session | detects every frame (N = 1) | N/A |
| Ring write | frame sizes change mid-stream | ring reallocated; old segment unlinked | N/A |
| Overwrite during read | parent wraps the ring while the child copies | frame dropped (counted), no result for it | N/A |
| Unregister / stop / death | stream deleted, backend stopped, worker killed | no `psm_*` segments of the backend left in `/dev/shm` | N/A |
| 20-stream run | GPU profile, 20 streams, 10 min | min fps ≥ 5, `proc_ms` p95 ≤ 100 ms, decode-to-result p95 < 1000 ms | otherwise numbers + named bottleneck in the report |

</frozen-after-approval>

## Code Map

- `backend/inference/detector.py` -- `FaceDetector` (OpenCV). Add `CudaYuNet` with the same `detect(image) -> boxes` surface and a `make_detector(model_path, device)` factory. YuNet 2026may ONNX: input `input [1,3,H,W]` (dynamic H/W, batch 1); outputs `cls_{8,16,32} [1,A,1]`, `obj_*`, `bbox_* [1,A,4]`, `kps_* [1,A,10]`.
- `backend/inference/worker.py` -- `Engine.__init__` (detector creation), `Engine.process_frames` (detect → crop → tracker → FER → EMA → fusion), `_SourceState` (add frame counter, last boxes/ids, detect_every), `SourceSlots.put_frame`, `_worker_main.handle` (`register` payload, new `frame_shm`), `InferenceWorker.register_source/submit_frame/_kill`.
- `backend/inference/tracker.py` -- unchanged; carried frames skip `update`.
- `backend/app/inference_hub.py` -- `register(loop, queue)` → `register(loop, queue, detect_every=1)`.
- `backend/app/streams.py` -- passes `detect_every` from `camera_type` in `_thread` when registering.
- `backend/bench/stages.py` -- existing per-stage bench; add CUDA YuNet stage numbers.
- `backend/tests/conftest.py` -- `FakeWorker.register_source(sid)`: accept the new keyword.
- `docs/benchmark-report.md` -- does not exist yet (story 1.14 writes the full report); create it with the 20-camera section only.

## Tasks & Acceptance

**Execution:**
- [x] `backend/inference/detector.py` (+ `tests/test_detector_cuda.py`, models marker) -- `CudaYuNet` (pad to 32, NCHW float32, priors, `sqrt(cls·obj)`, decode, NMS 0.3, top-k, clip) + parity test + factory with fallback -- GPU detection.
- [x] `backend/inference/worker.py`, `app/inference_hub.py`, `app/streams.py`, `tests/conftest.py` (+ unit tests) -- `detect_every` per source with carried boxes -- detection cadence.
- [x] `backend/inference/shm_ring.py` (+ tests: write/read, resize, overwrite detection, cleanup), `worker.py` -- shared-memory frames -- transport.
- [x] `backend/app/streams.py`, `interfaces.md` -- `proc_p95_ms` in the stream row -- inference-time evidence.
- [x] `backend/bench/streams_ws.py` -- `/ws/streams` recorder (fps, decode-to-result p50/p95, `proc_p95_ms` from `/api/streams`, system load) -- measurement.
- [x] `docs/benchmark-report.md`, Implementation Notes -- step table (baseline from story 1.8, step 1, step 2, step 3 / final 10 min, CPU reference) -- evidence.

**Acceptance Criteria:**
- Given the GPU profile with all 20 demo streams, when it runs for 10 min, then min per-stream fps ≥ 5, `proc_ms` p95 ≤ 100 ms and decode-to-result p95 < 1000 ms. If not met, the report gives the numbers and the bottleneck.
- Given `pytest -q` and capped `pytest -m models`, then all pass, including the CUDA parity test on the GPU image.
- Given the backend stopped after a run, then `/dev/shm` holds none of its segments.

## Design Notes

Worker-side handling of a shared-memory frame:

```python
seq0 = ring.seq(slot)          # even = stable
img = ring.view(slot, shape).copy()
if seq0 != header.seq or ring.seq(slot) != seq0: drop()   # overwritten meanwhile
```

The parent writes `seq+1` (odd) before the copy into the slot and `seq+2` after it, then sends the header with `seq+2`.

## Verification

**Commands:**
- `cd backend && .venv/bin/pytest -q` and capped `-m models` -- expected: pass.
- `docker run --rm --gpus all emotion-recognition/backend:gpu python -m pytest ...` for the CUDA parity test -- expected: pass.
- `bench/streams_ws.py --seconds 600` against the GPU profile with 20 streams -- expected: CAP-10 numbers.

## Implementation Notes

- **CAP-10 is not met on the RTX 3050 laptop.** The final 10-min run with 20 streams gave 1.08 fps per stream (decode→result p95 1346 ms, round time p95 1328 ms). 8 mixed streams reach 4.8–4.9 fps. The full table, the per-part breakdown and the options are in `docs/benchmark-report.md`.
- **Step 1 (CUDA YuNet).** `inference/detector.py` has `CudaYuNet` with OpenCV-equivalent post-processing; the parity test (`tests/test_detector_cuda.py`, GPU image) passes 4/4, boxes within 2 px. A single ORT session fed alternating sizes gave no results at all: re-planning costs ~1.4 s per round. One session per padded size (≤ 6) fixed it (46 ms per round of three sizes). Input preparation uses `cv2.copyMakeBorder` + `cv2.dnn.blobFromImage`.
- **Step 2 (detect every N).** `detect_every` reaches the worker with `register` (webcam 1, sector/security 3 from `streams.DETECT_EVERY`; `/ws/live` 1). Carried frames reuse the boxes and ids and run fresh FER; `timings.detected` marks them. Measured: 53–54 % of frames detected, as planned.
- **Step 3 (shared memory).** `inference/shm_ring.py` (4-slot ring per source, seqlock, parent-owned segments). It is on by default (`SHM_FRAMES=1`; `0` restores pickled frames for comparison). The compose backends get `shm_size: 1gb`. No `psm_*` segments remain after stop (checked after every run and after the model tests). The reader does not unregister from the resource tracker: the spawned worker shares the parent's tracker, and unregistering there dropped the parent's entry.
- **Tried and reverted:** parallel detection in 4 threads on one CUDA session. cuDNN failed with `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED` and the worker was killed. Detection stays serial (comment in `Engine.process_frames`).
- **Evidence tooling:** `bench/streams_ws.py` (recorder, exit 0 only when CAP-10 holds); `proc_p95_ms` in the stream row (`interfaces.md` updated); worker `loop:` log line every 30 s with the per-part breakdown.
- **Observed but unexplained:** one backend container restart during the capacity check (not flagged as an OOM kill; logs lost on `down`); one NVDEC open "generic error" with CPU-decode fallback.
- **Verified:** `pytest -q` 195 passed; capped `pytest -m models` 22 passed (real worker over shared memory); CUDA parity 4/4 in the GPU image; GPU runs as in the report; CPU reference run 0.21 fps (audio dominates there).

## Spec Change Log

## Review Triage Log
