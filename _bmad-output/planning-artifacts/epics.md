# Epics: emotion_recognition MVP

Derived from `_bmad-output/specs/spec-emotion-recognition-mvp/stories.yaml` (source of truth). Contract: that folder's `SPEC.md` and its companions. Change stories via bmad-spec, then regenerate this file.

## Epic 1: Real-time emotion recognition MVP

Working offline Docker prototype for a quality manager: live webcam + mic emotion recognition, parallel analysis of 20 RTSP streams of three camera types, an emotion event journal, file analysis, export, plus benchmark, accuracy (face-size sweep), per-camera-type media regulation and a 30–50 camera hardware plan (SPEC CAP-1..CAP-11).

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

### Story 1.6: UI shell, design system and screens on mock data

Rebuild the frontend per ui.md and the adopted EmoSense mockups: router, tokens and bundled fonts, toolbar menus, all screens (grid, stream detail, live, journal, sessions, settings, analyze) on a typed API module whose unbuilt endpoints are served by mocks shaped like interfaces.md. Existing live webcam and sessions features keep working against the real backend.

**Dev notes:** Use mocks only for endpoints the backend does not serve yet; live (/ws/live), health and sessions stay real. Grep dist for external URLs. Report which endpoints are mocked.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "6"). Checkpoints: spec=false, done=true.

### Story 1.7: RTSP ingest with camera profiles and stream management

Add RTSP sources feeding the shared worker with per-stream camera_type, profile detection size, 5 fps sampling and NVDEC decode with CPU fallback, plus /api/streams, /ws/streams, rtsp sessions and the demo-cams profile running a few clips (CAP-10 backend, interfaces.md RTSP streams, stack.md RTSP ingest, camera-profiles.md), plus stream management (site and settings, PATCH, reconnect, connection test, SQLite stream list, per-face face_px).

**Dev notes:** camera_type defaults to sector. Verify NVDEC decode via PyAV hwaccel on the RTX 3050 and record whether it works; CPU decode must remain the fallback. Switch the frontend stream endpoints from mocks to the real API.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "7"). Checkpoints: spec=true, done=false.

### Story 1.8: Twenty-camera demo set and throughput

Build the 20-stream demo mix (clip prep script encoding per camera profile, emulator looping with -c copy, DEFAULT_STREAMS) and tune the pipeline until the mix meets CAP-10 numbers on the GPU profile (camera-profiles.md 20-stream demo mix, stack.md Demo cameras and Detection cost).

**Dev notes:** Find openly licensed (Creative Commons or public domain) clips with faces, Russian speech and clear negative and positive moments; record source URL, author and licence per clip in demo-videos/ATTRIBUTION.md; stop and report if no suitable clips are found. Run the mix for 10 min; if any stream falls below 5 fps or proc_ms p95 exceeds 100 ms, apply the detection fallback from stack.md and report the numbers before and after.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "8"). Checkpoints: spec=true, done=true.

### Story 1.9: Multi-camera grid and stream detail on live data with system load

Wire the grid and stream detail screens to /ws/streams and /api/system: 20 tiles with analysed frames, boxes, facial/vocal/fused emotions, per-stream FPS and p50/p95, visitor cards, camera/processing block, add/remove and settings from the UI (CAP-10 UI, ui.md).

**Dev notes:** Demo milestone: run the 20-stream mix for 10 min on the GPU profile and report per-stream fps and p95 latency in the completion report.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "9"). Checkpoints: spec=false, done=true.

### Story 1.10: Emotion event engine, incidents and journal

Implement the event rule and per-face decision states, settings API, event persistence with incident state and ack, the event WS message, /api/events with filters and export, the stream interval log, and wire the journal, bell, incident banner and settings screens (CAP-11, stack.md Emotion events, interfaces.md).

**Dev notes:** Events come from every source, including webcam and file sessions. Never store face images. Show in the completion report that the demo clips' negative and positive moments each produce an event.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "10"). Checkpoints: spec=true, done=true.

### Story 1.11: Recorded video upload queue, analysis and synced playback

Upload endpoint with queued file jobs and progress, synced playback overlay, emotion chart, episode timeline and table from session events, export reuse; wire the analyze screen (CAP-3, ui.md File analysis).

**Dev notes:** Cover edge cases: no audio track, no faces, unsupported codec returns failed status with a message.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "11"). Checkpoints: spec=false, done=true.

### Story 1.12: Offline and cross-platform hardening

GPU-to-CPU fallback with UI warning, offline verification with network disabled (including the 20-camera demo-cams profile and the event journal), Windows Docker Desktop (WSL2) run check, README run guide (CAP-5 full).

**Dev notes:** Grep built frontend for external URLs; any CDN or remote font reference fails the story.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "12"). Checkpoints: spec=false, done=true.

### Story 1.13: Per-profile capacity benchmark and report

Extend bench to measure detection per profile size, decode cost CPU vs NVDEC, and the maximum parallel streams per camera profile at 5 fps with and without audio on CPU and GPU, writing docs/benchmark-report.md (CAP-6 full).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "13"). Checkpoints: spec=false, done=false.

### Story 1.14: Accuracy evaluation with face-size sweep

Script computing UAR, macro-F1 and confusion matrices for FER and SER on public test subsets plus the FER face-size sweep (16-100 px) for 7 classes and 3 groups, writing docs/evaluation-report.md (CAP-7).

**Dev notes:** Datasets are not bundled in images; script takes local dataset paths. Document how to obtain each dataset.

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "14"). Checkpoints: spec=true, done=false.

### Story 1.15: Media quality regulation per camera type

Write docs/media-requirements.md with minimum and recommended parameters per camera type, derived from the CAP-7 sweep through pixels-per-metre and research.md section 4.2 (CAP-8, camera-profiles.md).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "15"). Checkpoints: spec=false, done=true.

### Story 1.16: Hardware plan for 30-50 cameras and outlook to 3000

Write docs/scaling-plan.md with per-profile node capacity from benchmark-report.md, single-server vs multi-node hardware and cost for 30-50 cameras, edge-plus-centre architecture for 3000, the alert and ticket adapter seam, and one recommended stack (CAP-9).

**Spec:** `_bmad-output/specs/spec-emotion-recognition-mvp/` (stories.yaml id "16"). Checkpoints: spec=false, done=true.
