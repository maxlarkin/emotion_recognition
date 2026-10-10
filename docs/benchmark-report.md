# Benchmark report

Story 1.14 writes the full capacity report. This file starts with the 20-camera mix measurements of
story 1.9.

## 20-camera mix on the target laptop (story 1.9)

**Setup:**
- Laptop: RTX 3050 Laptop GPU with 4 GB, 16 GB RAM. GPU profile, `demo-cams` with the story-1.8 clips.
- Mix: 6 webcam (640×480), 8 sector (1920×1080) and 6 security (2560×1440) streams, all at 10 fps. 14 of them carry audio.
- Analysis at 5 fps per stream with the camera-profile detection sizes. NVDEC decode.
- Backend container memory capped at 8 GB.
- Recorder: `python -m bench.streams_ws --seconds N` (results/s per stream, decode-to-result latency, `proc_p95_ms` of `/api/streams`, system load).
- `proc_ms` is the worker's time for the whole round, from taking the frames to the results. It covers all streams' frames together, so it is not a per-frame inference time.

| Step | Run | Streams | fps per stream (min / median) | decode→result p50 / p95 | proc p95 (max) | GPU util | VRAM | Notes |
|---|---|---|---|---|---|---|---|---|
| Baseline (story 1.8): YuNet on CPU, pickled frames | 90 s | 20 | 0.01 / 0.02 (2 of 20 streams answered) | 11.2 s / 14.4 s | — | 6 % | — | host RAM 13.6 / 15 GB, gigabytes of frames queued |
| 1: YuNet on the ORT CUDA EP, one session | 180 s | 20 | 0 | — | — | — | 4.0 GB | no result at all |
| 1b: one CUDA session per input size | 180 s | 20 | 1.74 / 1.74 | 468 ms / 1043 ms | 976 ms | 37 % | 4.05 GB | with detect-every-3, pickled frames (`SHM_FRAMES=0`) |
| 3: + shared-memory frames | 180 s | 20 | 1.54 / 1.54 | 661 ms / 864 ms | 703 ms | 33 % | 4.07 GB | |
| Final | **600 s** | 20 | **1.08 / 1.08** | 818 ms / 1346 ms | 1328 ms | 29 % | 4.09 GB | slower than the 3-min run |
| Capacity check | 120 s | 8 (2 webcam, 3 sector, 3 security) | 4.78 / 4.88 | — / 356 ms | 296 ms | 40 % | — | close to 5 fps |
| CPU image (reference) | 180 s | 20 | 0.21 / 0.21 | — / 2817 ms | 2768 ms | — | — | per round: detect 959 ms, FER 572 ms, **audio 3271 ms** (GigaAM SER on CPU) |

CAP-10 (20 streams, ≥ 5 fps each, inference p95 ≤ 100 ms, decode-to-result p95 < 1000 ms) is **not met**
on this laptop. About 8 mixed streams reach ~5 fps.

### Where the time goes

The worker logs one line per 30 s (`loop:`). Final run, one round processes all 20 streams' frames:

| Part | ms per round |
|---|---|
| detection (≈ 53 % of frames detected: webcam every frame, sector/security every 3rd) | 505–537 |
| FER (all faces of the round, CUDA, fixed batch 8) | 113–122 |
| audio (VAD + GigaAM SER for 14 audio streams) | 146–163 |
| shared-memory reads | 28–31 |
| **total** | **813–846**, i.e. 1.2 rounds/s |

**Findings:**
- **Detection dominates.** Measured alone, YuNet takes 3.7 / 20 / 35 ms per frame at 640 / 1920 / 2560 on CUDA, including input preparation. In the running system each call costs 1.5–2.5 times more: 20 NVDEC decodes, FER and SER share the same small GPU, and the per-round cost grew over the 10-minute run.
- **One CUDA session must not be fed alternating input sizes.** ORT re-plans the graph on every shape change: one round of the three sizes took 1407 ms with one shared session and 46 ms with one session per size. Story 1.9 keeps one session per padded size.
- **Running one session from several threads breaks cuDNN** on this GPU (`CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED`, and the worker died). Detection stays serial.
- **Shared-memory frames** remove gigabytes of pickled frames from the queue and lower latency. They do not raise fps, because the worker is not waiting for input.
- **VRAM is the hard limit.** The models, three YuNet sessions, NVDEC and the desktop together use 4.05–4.09 of 4.0 GB.
- During the session the backend container restarted once, while streams were being re-enabled for the capacity check. Docker did not flag it as an OOM kill, and the logs were lost when the stack was removed. One NVDEC open failed with a "generic error" and fell back to CPU decode.

### What would reach the target

Changes outside story 1.9's constraints, for a decision:
1. **Smaller detection sizes for the demo.** For example 1280 for sector and 1920 for security. Faces then shrink to ~33–66 px and ~19–34 px, which conflicts with the camera-profile face-size goals.
2. **Detect less often on large profiles.** Every 5th or 6th frame instead of every 3rd; faces carried up to ~1 s.
3. **Fewer streams per laptop.** About 8 mixed streams at 5 fps, or 20 streams at ~1–1.7 fps.
4. **A bigger GPU for the 20-stream target.** For example 8 GB or more, with more SMs. This feeds the CAP-9 hardware plan.
5. **Pipeline work.** SER in its own thread or at a slower cadence, and overlapping CPU preparation with GPU runs through IO binding. Expected gain is tens of percent, not 4×.
