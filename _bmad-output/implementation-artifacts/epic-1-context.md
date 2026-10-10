# Epic 1 Context: Real-time emotion recognition MVP

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Deliver an offline, Docker-only prototype for a quality manager who oversees client service points. It recognises the 7 basic emotions in real time from faces (primary) and Russian speech (secondary). Sources are the browser webcam and microphone, uploaded video files, and 20 parallel RTSP streams of three camera types (webcam, sector, security). The system writes a journal of negative and positive emotion events and exports results. It also produces evidence-backed documents: a benchmark, an accuracy evaluation with a face-size sweep, a media-quality regulation per camera type, and a hardware plan for 30–50 cameras with an outlook to 3000. It is judged as a working demo on one laptop (4 GB VRAM, 16 GB RAM) with the network unplugged. Accuracy is measured and reported, not required to meet a target.

## Stories

- Story 1.1: Repository skeleton and Docker compose with CPU and GPU profiles
- Story 1.2: Inference core with per-stage benchmark
- Story 1.3: Live webcam analysis with overlay and latency readout
- Story 1.4: Live microphone emotion and fusion in UI
- Story 1.5: Session persistence and CSV/JSON export
- Story 1.6: UI shell, design system and screens on mock data
- Story 1.7: RTSP ingest with camera profiles and stream management
- Story 1.8: Twenty-camera demo clip set
- Story 1.9: Throughput for the twenty-camera mix
- Story 1.10: Multi-camera grid and stream detail on live data with system load
- Story 1.11: Emotion event engine, incidents and journal
- Story 1.12: Recorded video upload queue, analysis and synced playback
- Story 1.13: Offline and cross-platform hardening
- Story 1.14: Per-profile capacity benchmark and report
- Story 1.15: Accuracy evaluation with face-size sweep
- Story 1.16: Media quality regulation per camera type
- Story 1.17: Hardware plan for 30-50 cameras and outlook to 3000

## Requirements & Constraints

- **Live webcam:** p95 latency from capture to overlay render is under 1 s over 60 s on the GPU profile, and the UI shows it. Face boxes appear or disappear within 1 s.
- **Microphone:** shows the vocal emotion and a fused emotion while the user speaks, and "no speech" otherwise; in silence the fused emotion equals the facial one. Muting the mic must not break video.
- **Files:** mp4/webm/mkv/avi up to 10 min, with visible progress, synced playback overlay and an emotion-over-time chart. A file with no audio or no faces finishes cleanly with a note; an unsupported codec gives `failed` with a message.
- **Export:** every finished session (live, file, rtsp) exports CSV and JSON, and pandas must re-import both.
- **20 streams (GPU profile):** the demo mix runs for 10 min with each stream at ≥ 5 analysed fps, `proc_ms` p95 ≤ 100 ms, and decode-to-result p95 < 1 s. Losing or stopping one stream does not affect the others; a lost stream reconnects. Streams can be added by URL and camera type at runtime without a restart.
- **Events:** a sustained emotion on a face track produces one event within 1 s of the rule holding. Events come from every source. Negative events are incidents with state new/confirmed/false_alarm that the manager acknowledges. The journal filters by kind, state, camera and time and exports. The demo clips' scripted negative and positive moments must each produce an event.
- **Offline:** no CDN, remote fonts or model-hub calls at runtime. Weights are baked in at build. The built frontend must contain no external URLs.
- **CPU-only mode must work**, and GPU falls back to CPU with a visible UI warning. The 20-stream targets apply to GPU only.
- **Budget:** all models together fit in 4 GB VRAM and 16 GB RAM.
- **Privacy:** no face recognition, embeddings, identities or stored face images.
- UI text is in Russian. Runs on Linux and Windows (Docker Desktop, WSL2).
- **Non-goals:** fine-tuning, alert or ticket delivery (only document the seam), more than 20 streams, auth, ONVIF/PTZ/recording, annotated MP4 export, emotions beyond the 7.

## Technical Decisions

- **Models (pretrained, no fine-tuning):**
  - Face detection: YuNet via OpenCV `FaceDetectorYN`.
  - FER: EmotiEffLib `enet_b0_8_va_mtl` on ONNX; drop contempt and renormalise to 7 classes.
  - VAD: Silero.
  - SER: GigaAM-Emo, with labels mapped angry→anger, sad→sadness, neutral→neutral, positive→happiness; disgust, fear and surprise are always 0 from audio.
- **Emotion keys** are always `anger, disgust, fear, happiness, sadness, surprise, neutral`.
- **Pipeline:** every source feeds one inference worker process through "source → queue → worker → results", outside the FastAPI event loop.
  - Each source has a latest-frame slot of size 1, which drops stale frames. Audio and file jobs use bounded FIFOs.
  - The worker batches faces across sources.
  - RTSP decoding runs outside the worker. With 20 streams, use threads or a small process pool, not one process per stream.
- **Processing:**
  - Live JPEG at 5–10 fps, longest side ≤ 640 px.
  - IoU tracker with EMA smoothing, α = 0.4.
  - Audio is PCM16 mono 16 kHz in 500 ms chunks. SER runs on a 3 s window with a 1 s hop, only when ≥ 50% of the window is speech.
  - Fusion: `renormalise(0.7·face + 0.3·audio)`. With no speech, w_audio = 0.
- **RTSP:**
  - PyAV over TCP with a 5 s open timeout. The GPU profile uses NVDEC (`hwaccel` cuda) with CPU decode as the fallback.
  - Video is sampled at 5 fps and scaled to the camera profile's detection size (webcam 640, sector 1920, security 2560). FER crops come from that frame. Tiles go to the browser at ≤ 640 px.
  - Reconnect backoff is 1, 2, 4… s, capped at 30 s. `camera_type` defaults to `sector`.
  - If CPU YuNet cannot hold the mix, run YuNet on the ORT CUDA EP or detect every 2nd frame and let the tracker carry boxes.
- **Per-face decision** (`alert, watch, abstain, positive, none`) and `face_px` travel on every face object; small faces (< stream `min_face_px`) and young tracks are `abstain`.
- **Streams** carry `site` and per-stream `settings` (enabled, analysis_fps, detection_size, min_face_px); global event/journal settings live behind `/api/settings`. Both persist in SQLite.
- **Event rule:** applied per track on the smoothed fused probabilities.
  - Negative when the sum of anger, disgust, fear and sadness is ≥ 0.6 for ≥ 2 s. Positive when happiness is ≥ 0.6 for ≥ 2 s.
  - The same kind is not repeated on the same track within 30 s. Thresholds are configurable.
  - An event stores the peak emotion, the mean group probability and the duration, in SQLite.
- **Contracts are fixed:** the `/ws/live`, `/ws/streams` and REST payloads, the export row and the event object are defined in `interfaces.md`.
  - The export row includes `stream_id`, which is empty for webcam and file sessions.
  - Session kinds are `live`, `file` and `rtsp`.
- **Runtime:**
  - Python 3.11 base image `python:3.11-slim` for both images. The GPU image takes CUDA 12.6 from the torch cu126 wheels.
  - Exactly one onnxruntime package per image.
  - SQLite and uploads live in the named volume `data`.
  - nginx serves the SPA and proxies `/api` and `/ws`.
  - The `gpu` compose profile uses `gpus: all`.
  - The `demo-cams` profile runs mediamtx and an ffmpeg emulator that loops pre-encoded clips with `-c copy`. `DEFAULT_STREAMS` lists the 20 URLs.
  - System load comes from psutil plus NVML; the GPU fields are null without a GPU.
- **Demo mix:**
  - 6 webcam streams: 640×480, audio, faces ≥ 90 px.
  - 8 sector streams: 1080p, audio, faces 50–100 px.
  - 6 security streams: 1440p, no audio, faces 25–45 px.
  - All at 10 fps. Clips are openly licensed and recorded in `demo-videos/ATTRIBUTION.md`.
- **Docs:**
  - CAP-8 converts the CAP-7 face-size sweep (16–100 px) through pixels-per-metre into per-camera-type minimum resolutions. Face px = 0.16 m · ppm.
  - CAP-9 sizes hardware from the measured per-profile stream capacity.

## UX & Interaction Patterns

- **Design system:** adopted EmoSense package — dark flat theme from `tokens.css`, radius 0, shadows only on dropdowns/active items, IBM Plex fonts bundled via `@fontsource`, no UI kit, no native selects in settings. Mockups in `planning-artifacts/ux-emosense/mockups` are the visual reference; the spec's `ui.md` wins on conflict.
- **Shell:** sticky 44 px toolbar with click-open menus (Потоки, История, Настройки, Анализ видео), bell with new-incident count, health warning banner. React Router routes: `/streams`, `/streams/:id`, `/live`, `/history/events`, `/history/sessions`, `/settings/:tab`, `/analyze/:sessionId?`.
- **Streams:** sidebar grouped by site with search and a «Веб-камера» entry; grid of 20 tiles with system load; stream detail with incident banner, tiles, video + visitor cards, interval journal and collapsible camera/processing block.
- **Settings:** tabs События, Камеры (bulk edit), Журнал, Система; one row per setting; Save/Cancel with unsaved-changes guard.
- **File analysis:** upload queue with progress; result = player + overlay, emotion chart, tiles, episode timeline and table synced with each other.
- **Data access:** one typed API module; endpoints not built yet served by mocks shaped like `interfaces.md`.

## Cross-Story Dependencies

- **1.1–1.5 are done.** They provide the compose skeleton, the shared worker, live video and live audio. Later stories reuse the worker seam; they do not fork it.
- **1.5 (persistence and export)** underpins file sessions (1.12), rtsp sessions (1.7) and the event store (1.11). Add the `stream_id` column now.
- **UI:** 1.6 builds every screen on mocks; 1.7, 1.10, 1.11 and 1.12 replace mocks with real endpoints and must keep the 1.6 API module shapes.
- **Streams:** 1.7 adds RTSP backend ingest and stream management. 1.8 builds the 20-clip mix, 1.9 tunes throughput (CUDA YuNet, detect-every-N, shared-memory frames). 1.10 wires the grid and detail screens to live data.
- **Events:** 1.11 needs fused smoothed results from every source. The event WS message travels on `/ws/streams`.
- **Hardening:** 1.13 verifies offline operation for everything, including the demo-cams profile and the journal.
- **Docs chain:** 1.14 (benchmark) feeds 1.17 (hardware plan), and 1.15 (face-size sweep) feeds 1.16 (media regulation).
