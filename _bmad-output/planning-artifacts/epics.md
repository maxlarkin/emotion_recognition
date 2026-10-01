# Epics: emotion_recognition MVP

Derived from `_bmad-output/specs/spec-emotion-recognition-mvp/stories.yaml` (source of truth). Contract: that folder's `SPEC.md` and its companions. Change stories via bmad-spec, then regenerate this file.

## Epic 1: Real-time emotion recognition MVP

Working offline Docker prototype: live webcam + mic emotion recognition, file analysis, export, plus regulation and scaling documents (SPEC CAP-1..CAP-9).

### Story 1.1: Repository skeleton and Docker compose with CPU and GPU profiles

Create backend/frontend/docs layout, FastAPI app with /api/health, React+Vite shell served by nginx, and docker compose with default CPU profile and gpu profile, model weights baked at build (CAP-5 base, stack.md Runtime).

**Dev notes:** Verify `docker compose up` and `docker compose --profile gpu up` both start and /api/health reports device. Download all model weights (YuNet, EmotiEffLib enet_b0_8_va_mtl, Silero VAD, GigaAM emo) in the backend Dockerfile so runtime needs no network.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "1"). Checkpoints: spec=true, done=true.

### Story 1.2: Inference core with per-stage benchmark

Implement detector, IoU tracker, FER, VAD, SER, smoothing and late fusion as a worker-process module with unit tests, plus a bench command reporting per-stage ms and peak VRAM/RAM (stack.md Models and Processing; CAP-6 first half).

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

Persist live session results to SQLite in the data volume, add sessions list and export endpoints and UI download (CAP-4, interfaces.md REST and Export row).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "5"). Checkpoints: spec=false, done=false.

### Story 1.6: Recorded video upload, analysis and synced playback

Upload endpoint, ffmpeg-based file job with progress, synced playback overlay and emotion timeline chart, export reuse (CAP-3).

**Dev notes:** Cover edge cases: no audio track, no faces, unsupported codec returns failed status with a message.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "6"). Checkpoints: spec=false, done=true.

### Story 1.7: Offline and cross-platform hardening

GPU-to-CPU fallback with UI warning, offline verification with network disabled, Windows Docker Desktop (WSL2) run check, README run guide (CAP-5 full).

**Dev notes:** Grep built frontend for external URLs; any CDN or remote font reference fails the story.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "7"). Checkpoints: spec=false, done=true.

### Story 1.8: Multi-stream capacity benchmark and report

Extend bench to run N parallel 720p@5fps file-fed streams with and without audio until real time breaks, on CPU and GPU, and write docs/benchmark-report.md (CAP-6 full).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "8"). Checkpoints: spec=false, done=false.

### Story 1.9: Offline accuracy evaluation script and report

Script computing UAR, macro-F1 and confusion matrices for FER and SER on public test subsets, writing docs/evaluation-report.md (CAP-7).

**Dev notes:** Datasets are not bundled in images; script takes local dataset paths. Document how to obtain each dataset.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "9"). Checkpoints: spec=true, done=false.

### Story 1.10: Media quality regulation document

Write docs/media-requirements.md from research.md section 4.2 plus CAP-7 results, with evidence per parameter and measurement conditions for accuracy targets (CAP-8).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "10"). Checkpoints: spec=false, done=true.

### Story 1.11: Scaling plan for 20+ cameras

Write docs/scaling-plan.md comparing single-server and multi-node options with hardware and cost, grounded in benchmark-report.md and research.md section 3 (CAP-9).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "11"). Checkpoints: spec=false, done=true.
