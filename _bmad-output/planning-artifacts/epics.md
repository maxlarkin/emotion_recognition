# Epics: emotion_recognition MVP

Derived from `_bmad-output/specs/spec-emotion-recognition-mvp/stories.yaml` (source of truth). Contract: that folder's `SPEC.md` and its companions. Change stories via bmad-spec, then regenerate this file.

## Epic 1: Real-time emotion recognition MVP

Working offline Docker prototype: live webcam + mic emotion recognition, parallel analysis of 3+ RTSP streams, file analysis, export, plus regulation and scaling documents (SPEC CAP-1..CAP-10).

### Story 1.1: Repository skeleton and Docker compose with CPU and GPU profiles

Create backend/frontend/docs layout, FastAPI app with /api/health, React+Vite shell served by nginx, and docker compose with default CPU profile and gpu profile, model weights baked at build (CAP-5 base, stack.md Runtime).

**Dev notes:** Verify `docker compose up` and `docker compose --profile gpu up` both start and /api/health reports device. Download all model weights (YuNet, EmotiEffLib enet_b0_8_va_mtl, Silero VAD, GigaAM emo) in the backend Dockerfile so runtime needs no network.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "1"). Checkpoints: spec=true, done=true.

### Story 1.2: Inference core with per-stage benchmark

Implement detector, IoU tracker, FER, VAD, SER, smoothing and late fusion as one worker process that serves several sources (per-source slots, faces batched across sources), with unit tests and a bench command reporting per-stage ms and peak VRAM/RAM (stack.md Models, Processing, Runtime; CAP-6 first half).

**Dev notes:** First confirm GigaAM-Emo label list via get_probs and record it; map labels per SPEC Assumptions. Stop and report if all models do not fit 4 GB VRAM together or if FER+detector exceed 100 ms per frame on GPU profile.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "2"). Checkpoints: spec=true, done=true.

### Story 1.3: Live webcam analysis with overlay and latency readout

Browser capture, WebSocket /ws/live protocol, latest-frame slot to worker, canvas overlay of boxes and emotions, and p50/p95 latency display (CAP-1, interfaces.md Live session).

**Dev notes:** This is the demo milestone: measure p95 latency over 60 s on the target laptop and include the number in the completion report.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "3"). Checkpoints: spec=false, done=true.

### Story 1.4: Live microphone emotion and fusion in UI

Stream PCM16 audio chunks, run VAD-gated SER windows, fuse with face probabilities, show audio state and fused emotion (CAP-2).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "4"). Checkpoints: spec=false, done=false.

### Story 1.5: Session persistence and CSV/JSON export

Persist live session results to SQLite in the data volume, add sessions list and export endpoints (including the stream_id column) and UI download (CAP-4, interfaces.md REST and Export row).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "5"). Checkpoints: spec=false, done=false.

### Story 1.6: RTSP multi-stream ingest with demo cameras

Add RTSP source processes feeding the shared worker, /api/streams, /ws/streams and rtsp sessions, plus the demo-cams compose profile (MediaMTX + ffmpeg emulator) with three openly licensed clips and ATTRIBUTION.md (CAP-10 backend, interfaces.md RTSP streams, stack.md RTSP ingest and Demo cameras).

**Dev notes:** Find openly licensed (Creative Commons or public domain) clips with close-up faces and Russian speech; record source URL, author and licence per clip in demo-videos/ATTRIBUTION.md; stop and report if no suitable clips are found. Clips must be usable offline after build.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "6"). Checkpoints: spec=true, done=false.

### Story 1.7: Multi-camera grid screen with system load

Russian multi-camera page: tile per stream with analysed frames, boxes, facial/vocal/fused emotions, per-stream FPS and p50/p95 latency, stream add/remove by RTSP URL, status and overall CPU/RAM/GPU/VRAM load via /api/system (CAP-10 UI).

**Dev notes:** Demo milestone: run 3 emulated cameras for 10 min on the GPU profile and report per-stream fps and p95 latency in the completion report.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "7"). Checkpoints: spec=false, done=true.

### Story 1.8: Recorded video upload, analysis and synced playback

Upload endpoint, ffmpeg-based file job with progress, synced playback overlay and emotion timeline chart, export reuse (CAP-3).

**Dev notes:** Cover edge cases: no audio track, no faces, unsupported codec returns failed status with a message.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "8"). Checkpoints: spec=false, done=true.

### Story 1.9: Offline and cross-platform hardening

GPU-to-CPU fallback with UI warning, offline verification with network disabled (including the demo-cams profile), Windows Docker Desktop (WSL2) run check, README run guide (CAP-5 full).

**Dev notes:** Grep built frontend for external URLs; any CDN or remote font reference fails the story.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "9"). Checkpoints: spec=false, done=true.

### Story 1.10: Multi-stream capacity benchmark and report

Extend bench to run N parallel 720p@5fps RTSP streams from the demo-cams emulator with and without audio until real time breaks, on CPU and GPU, and write docs/benchmark-report.md (CAP-6 full).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "10"). Checkpoints: spec=false, done=false.

### Story 1.11: Offline accuracy evaluation script and report

Script computing UAR, macro-F1 and confusion matrices for FER and SER on public test subsets, writing docs/evaluation-report.md (CAP-7).

**Dev notes:** Datasets are not bundled in images; script takes local dataset paths. Document how to obtain each dataset.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "11"). Checkpoints: spec=true, done=false.

### Story 1.12: Media quality regulation document

Write docs/media-requirements.md from research.md section 4.2 plus CAP-7 results, with evidence per parameter and measurement conditions for accuracy targets (CAP-8).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "12"). Checkpoints: spec=false, done=true.

### Story 1.13: Scaling plan for 20+ cameras

Write docs/scaling-plan.md comparing single-server and multi-node options with hardware and cost, grounded in benchmark-report.md and research.md section 3 (CAP-9).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "13"). Checkpoints: spec=false, done=true.
