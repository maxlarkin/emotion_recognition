---
title: 'Inference core with per-stage benchmark'
type: 'feature'
created: '2026-10-02'
status: 'done'
baseline_commit: '02dca2e6bed1b675e02ab109a4ea705628cf3854'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/stack.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** No model runs yet. Stories 1.3 (webcam), 1.4 (mic), 1.6 (RTSP, 3+ parallel streams) and 1.8 (files) all need one inference engine. That engine has to hold every model once within 4 GB VRAM and serve several sources at the same time. Nobody has yet measured whether it is fast enough.

**Approach:** Build the `backend/inference/` package and a worker process. The pipeline is: detector → tracker → FER → EMA. Audio goes through VAD → SER, then late fusion. Each source gets its own slots, and faces are batched across sources. Add a bench command that measures every stage and memory use on CPU and GPU. Nothing is wired into the API yet.

## Boundaries & Constraints

**Always:**
- Use the models, parameters and fusion defaults in `stack.md`. Output emotion keys are always `anger, disgust, fear, happiness, sadness, surprise, neutral`.
  - FER: drop contempt, then renormalise.
  - SER: map `angry→anger, sad→sadness, neutral→neutral, positive→happiness`. The other classes get 0.
- The worker is a separate process started with the `spawn` context. It loads every model exactly once. CUDA is used only inside the worker, never in the API process.
- Each source has a video slot of size 1, where a new frame overwrites the old one, and a bounded audio FIFO. One slow or silent source never blocks the others.
- Device selection: GPU image with CUDA available → CUDA execution provider and torch `cuda`. Otherwise CPU. The worker reports which device it chose.
- Models are loaded only from `MODELS_DIR`, with no network access.

**Never:**
- FastAPI routes, WebSocket wiring, RTSP/PyAV decoding, persistence, or UI. Those belong to later stories.
- Using the EmotiEffLib or GigaAM high-level APIs (`predict_emotions`, `get_probs(path)`) in the hot path. Use them only in tests, as an equivalence reference.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Frame with faces | BGR frame from source `s1` | `FrameResult(source_id, frame_id, ts, faces=[{track_id, box, face probs (7), face_label, fused probs, fused_label, confidence}])`, plus the latest audio block for `s1` | N/A |
| Frame without faces | frame with no detections | `faces=[]` | N/A |
| Same face over frames | consecutive frames | stable `track_id`; EMA α=0.4 applied | a track disappears after 1 s unseen |
| Three sources at once | `s1..s3` submit faster than the worker | each source gets results for its latest frame; stale frames are dropped silently | N/A |
| Speech window | 3 s of PCM16 16 kHz, VAD speech ≥ 50% | `AudioResult(speech=true, probs (7 keys), label)`; fused = 0.7·face + 0.3·audio | N/A |
| Silence | VAD speech < 50% | `speech=false`, fused == face | N/A |
| Unknown source | submit before `register_source` | `KeyError`-style error returned to the caller | the worker keeps running |
| Model file missing | path absent at worker start | the worker fails to start with a clear message naming the file | N/A |

</frozen-after-approval>

## Code Map

- `backend/app/config.py:11` -- `MODEL_FILES` / `Settings.model_path` give the weight paths. Reuse them; do not duplicate paths.
- `backend/app/api/health.py:40` -- the `probe_cuda` subprocess pattern. Device logic in the worker must mirror its rule: torch CUDA available and `CUDAExecutionProvider` present.
- Verified in the CPU image (2026-10-02):
  - `gigaam.load_model("emo", download_root=...)` returns `GigaAMEmo` with `id2name = ['angry','sad','neutral','positive']`. `get_probs` does `prepare_wav(path) → forward(wav[1,T], length[1]) → avg_pool1d over time → head → softmax`. Reproduce that from a float32 16 kHz numpy window (no file).
  - `enet_b0_8_va_mtl.onnx`: input `input [N,3,224,224]` float, output `output [N,10]`. Indices 0–7 are emotion logits in the order `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise`; 8–9 are valence/arousal (ignored). Preprocess an RGB crop: resize to 224, divide by 255, normalise with mean `[0.485,0.456,0.406]` and std `[0.229,0.224,0.225]`, CHW.
  - Silero `silero_vad.onnx`: inputs `input [B,T]`, `state [2,B,128]`, `sr` (int64); outputs `output [B,1]`, `stateN`. Keep state per source; use 512-sample chunks at 16 kHz.
  - OpenCV 5.0.0 provides `cv2.FaceDetectorYN`.
- No inference code exists yet. `backend/inference/__init__.py`, `backend/bench/__init__.py` and `backend/eval/__init__.py` are placeholders.

## Tasks & Acceptance

**Execution:**
- [x] `backend/inference/types.py` -- dataclasses `VideoFrame`, `AudioChunk`, `FaceResult`, `AudioResult`, `FrameResult`, `EMOTIONS` tuple, label-mapping helpers -- the shared contract for later stories.
- [x] `backend/inference/detector.py` -- YuNet wrapper: set the input size per frame, score threshold 0.7, return boxes clipped to the frame -- face boxes.
- [x] `backend/inference/tracker.py` -- per-source IoU tracker (IoU ≥ 0.3, drop a track after 1 s unseen) -- stable track ids.
- [x] `backend/inference/fer.py` -- ORT session, batched crops → 7-key probabilities -- facial emotion.
- [x] `backend/inference/vad.py`, `backend/inference/ser.py` -- Silero VAD with per-source state; GigaAM-Emo on a 3 s window with 1 s hop, mapped to 7 keys -- vocal emotion.
- [x] `backend/inference/fusion.py` -- per-track EMA and weighted late fusion, configurable weights -- `stack.md` Fusion.
- [x] `backend/inference/worker.py` -- spawn-context process: `register_source`, `submit_frame`, `submit_audio`, `results()`, `stop()`, `device`; per-source slots; FER batched across sources -- the seam for every source.
- [x] `backend/bench/stages.py` -- `python -m bench.stages [--device auto|cpu|cuda] [--seconds N] [--json out]`. Report p50/p95 ms for detection on a 640×360 frame, FER at batch 1/4/8, VAD per 512-sample chunk and SER per 3 s window; worker end-to-end frame latency for 1 and 3 synthetic sources; peak VRAM (NVML, whole device) and peak RSS. Exit code 2 with a message if VRAM > 4096 MB or detection+FER p95 > 100 ms on CUDA -- the story's stop condition.
- [x] `backend/tests/test_inference_units.py` -- model-free tests: label mapping and renormalisation, EMA, fusion and silence rule, tracker id stability and expiry, slot overwrite -- fast guards.
- [x] `backend/tests/test_inference_models.py` (marker `models`) -- equivalence tests on a synthetic image and sine wav: FER probabilities vs EmotiEffLib `predict_emotions`, and SER vs `get_probs(file)`, within 1e-3. Also every matrix row through a real worker with 3 sources -- correctness against the reference libraries.
- [x] `backend/requirements.txt` -- add `psutil`, `nvidia-ml-py` (pinned) -- bench memory metrics.

**Acceptance Criteria:**
- Given the CPU image, when model tests and bench run, then all tests pass and the bench prints every stage with exit code 0.
- Given the GPU image on the RTX 3050, when the bench runs, then it reports device `cuda`, all models together use ≤ 4096 MB VRAM, and detection+FER p95 ≤ 100 ms. Otherwise the run stops and reports the numbers.
- Given 3 sources in one worker, when frames arrive at 8 fps each, then every source keeps receiving results and none is starved.

## Implementation Notes

- Docker: run as `sg docker -c '...'` if the shell lacks the docker group. Build images with `docker build --network host` (host VPN MTU); see AGENTS.md. If the daemon is down, ask the user to run `sudo systemctl enable --now docker`. Never use sudo yourself.

- Decisions (implementer, verified by lead 2026-10-02):
  - GigaAM on CUDA runs fp16 (autocast). The largest deviation from CPU fp32 `get_probs` is 2e-4, which is within the 1e-2 gate.
  - FER on CUDA always runs a fixed batch of 4 with zero padding. With the ORT CUDA EP, every change of batch shape costs about 65 ms. Before this fix, 3 sources ran at about 4 fps with p95 about 410 ms. CPU batches are not padded.
  - YuNet and Silero VAD run on the CPU in both images. The pip build of OpenCV has no CUDA DNN, and VAD is faster on the CPU per 32 ms chunk.
  - A frame carries the latest audio window only if that window is ≤ 2 s old. `AudioResult` is also emitted on its own.
  - `bench/data/` holds `face.jpg`, a public-domain NASA photo, and `speech.wav`, MIT-licensed from silero-vad. Attribution is in `ATTRIBUTION.txt`. Synthetic audio never reached 50% VAD speech.
  - The image has no ffmpeg. The SER equivalence test therefore reads the wav the way `gigaam.load_audio` does (s16le / 32768).
  - Review round: audio freshness is now decided in media time (`0 <= frame.ts - window_end_ts <= audio_max_age_s`, last 16 windows kept per source). Under backlog only the newest speech window runs SER; older windows get `skipped=True`. `results()` raises `WorkerDied` on an unexpected exit.
  - Code that starts `InferenceWorker` must sit under `if __name__ == "__main__":` (spawn context).
- Verification (lead re-run):
  - Unit tests: 34 passed.
  - CPU image: 12/12 model tests pass. Detection+FER p95 is 15.8 ms. With 3 sources, the minimum is 7.1 fps and p95 is 156 ms. Peak RSS is 2.1 GB.
  - GPU image: 12/12 model tests pass. Device is cuda. Detection+FER p95 is 14.7 ms. With 3 sources, the minimum is 7.9 fps and p95 is 43 ms. Peak VRAM is 1392 MB for the whole device, of which 446 MB was the baseline. Peak RSS is 2.9 GB. No STOP condition was hit.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-02): layers blind-hunter, edge-case-hunter, verification-gap.

| # | Layer | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | blind, edge | unregister→register before take() keeps old engine state | medium | `SourceSlots.register` discards the id from `removed`, so `Engine.remove` never runs. Applies to RTSP reconnects with a reused id. | patch |
| 2 | blind | Per-source bookkeeping (`dropped_*`, `_frame_ids`) never freed | low | Same root as #1: grows with every new id. A direct deletion fixes it. | patch (with #1) |
| 3 | blind, edge | One bad frame/chunk fails the whole cross-source batch; ErrorResult has `source_id=None` | medium | There is a single try/except around `process_frames`/`process_audio`. CAP-10 multi-source isolation is at risk. | patch |
| 4 | blind, edge | No input validation in `submit_*` (odd bytes, float/2-D audio, non-BGR frame) | low | `np.frombuffer` raises on odd bytes; float audio becomes silence. Direct guards fix it. | patch (with #3) |
| 5 | edge | Caller mutates the image buffer after `submit_frame` → torn frame | low | `mp.Queue` pickles in the feeder thread later. One copy fixes it. | patch |
| 6 | edge | Receiver thread dies on an unexpected exception and the worker looks alive | medium | Only Empty/EOF/OSError are caught. | patch |
| 7 | blind, edge | Lifecycle: double `start`, use after stop raises ValueError, a dead worker is invisible, start-error path leaks queues | medium | Read in `InferenceWorker`; `alive` is never consulted. | patch |
| 8 | edge | `auto` device fails start if `cuda_usable` raises | low | Unlike `probe_cuda`, it does not swallow errors. A trivial try fixes it. | patch |
| 9 | blind | Audio backlog (16 s FIFO) can block video for seconds on CPU | medium | SER takes about 250 ms per window on CPU (bench), and up to ~14 windows per source are processed in one pass. | patch |
| 10 | blind | Audio freshness uses worker monotonic time, not media time | medium | `_fresh_audio` uses `time.monotonic()`. File sources (1.8) process faster than real time. | patch |
| 11 | blind | Tuning constants duplicated in `WorkerConfig`; `MIN_SPEECH_RATIO` unused; worker rebuilds `Settings` and loses `data_dir` | low | Drift risk for the next story that edits the defaults. Direct correction. | patch |
| 12 | blind, edge, verif | Bench VRAM gate unreachable on a 4 GB card; `starved` and zero faces don't affect the exit code | medium | NVML `used` ≤ total < 4096. `starved` is printed only. | patch |
| 13 | edge | Stage subprocess crash hangs the bench for 30 min | low | `q.get(timeout=1800)`. A direct poll loop fixes it. | patch |
| 14 | edge | Schedule catch-up bursts distort fps; errors from the previous run leak into the next | low | `next_frame += 1/FPS` without skipping; errors are not filtered by sid. | patch |
| 15 | verif | EMA through the engine is unobserved (identical frames) | gap | Pre-verified by the layer. | patch |
| 16 | verif | Cross-source FER offset is unobserved | gap | Pre-verified by the layer. | patch |
| 17 | verif | SER cross-source mapping is unobserved | gap | Pre-verified by the layer. | patch |
| 18 | verif | Freshness rule has no test | gap | Pre-verified by the layer. | patch |
| 19 | verif, blind | `set_fusion` path untested | gap | Pre-verified by the layer. | patch |
| 20 | verif | `window_end_ts` tolerance hides the `- pend` term | gap | Pre-verified by the layer: ±0.05 covers the 32 ms error. | patch |
| 21 | edge | SER fp16 tolerance 1e-2 vs the spec's 1e-3 | false | Spec Design Notes allow fp16 "within 1e-2". | reject |
| 22 | blind, edge | Tracker never expires if `ts` goes backwards | low | Caller-controlled clock. Re-registration now resets the source (#1). A guard adds a branch. | reject |
| 23 | edge | FIFO overflow splices discontiguous audio | low | Only after a 16 s backlog, which #9 now bounds. Gap detection is new logic. | reject |
| 24 | edge | Register/unregister `queue.Full` after 5 s diverges state | low | 256-deep inbound queue; unlikely. | reject |
| 25 | blind | Window `speech_ratio` covers 128 extra samples | low | 0.27% of the window. Negligible. | reject |
| 26 | blind | `_put` silently drops results when the queue is full | low | 1024-deep queue, drained by the caller. Counters would be new surface. | reject |
| 27 | blind | Model-test docstring says "inside the image" | false | Verification mounts the source into the image, which is accurate. | reject |
| 28 | blind | numpy not pinned in dev; nvidia-ml-py in the CPU image | low | The constraint pins numpy under uv. NVML is a few hundred KB and returns null on CPU. | reject |
| 29 | verif | `interfaces.md` audio probs show 4 keys; the worker emits 7 (story matrix) | medium | The contract companion disagrees with the story matrix. The fix is a spec update, not code. | defer |
| 30 | verif | `stack.md` fusion wording ("renormalise after fusion") vs the implemented `0.7·face + 0.3·audio` | low | The story matrix pins the implemented formula. A doc clarification is needed in the spec. | defer |

Patches applied by the implementer; lead re-verified: units 41 passed; models 20/20 (CPU and GPU); bench exit 0 on both (GPU: det+FER p95 14.6 ms, 3 sources min 7.9 fps, p95 43 ms, VRAM 1396/4096 MB, baseline 450).

## Design Notes

Tests that need models are not in the image (`.dockerignore` excludes `tests`). Run them by mounting the backend source into the built image. pytest is installed per run with `pip install --target /tmp/pt` over the host network (the image venv ignores user site-packages); this is test-time only, and the image stays offline at runtime.

GigaAM on CUDA: use fp16 only if the SER equivalence test still passes within 1e-2. Otherwise use fp32. Record the choice.

## Verification

**Commands:**
- `cd backend && .venv/bin/pytest -q -m "not models"` -- expected: all pass.
- `docker run --rm --network host -v ./backend:/src:Z -w /src emotion-recognition/backend:cpu sh -c "pip install -q --target /tmp/pt pytest httpx && PYTHONPATH=/tmp/pt python -m pytest -q -p no:cacheprovider -m models && python -m bench.stages --seconds 20"` -- expected: pass; bench exit 0.
- Same with `--gpus all emotion-recognition/backend:gpu` and `python -m bench.stages --device cuda` -- expected: device cuda, VRAM ≤ 4096 MB, detection+FER p95 ≤ 100 ms.
