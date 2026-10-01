# Epic 1 Context: Real-time emotion recognition MVP

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Deliver, within 1.5 months, a working offline Docker prototype that recognizes human emotions in real time from a webcam and microphone, can analyse uploaded video files, and exports results, plus the documents an educational mandate requires: a media quality regulation and a justified plan for 20+ parallel cameras. The team has only a laptop (NVIDIA GPU with 4 GB VRAM, 16 GB RAM), so everything runs locally on open pretrained models without fine-tuning. The MVP is judged as a working demo, not on accuracy (95–98% is out of reach on in-the-wild data).

## Stories

- Story 1.1: Repository skeleton and Docker compose with CPU and GPU profiles
- Story 1.2: Inference core with per-stage benchmark
- Story 1.3: Live webcam analysis with overlay and latency readout
- Story 1.4: Live microphone emotion and fusion in UI
- Story 1.5: Session persistence and CSV/JSON export
- Story 1.6: Recorded video upload, analysis and synced playback
- Story 1.7: Offline and cross-platform hardening
- Story 1.8: Multi-stream capacity benchmark and report
- Story 1.9: Offline accuracy evaluation script and report
- Story 1.10: Media quality regulation document
- Story 1.11: Scaling plan for 20+ cameras

## Requirements & Constraints

- **Live video:** face boxes plus one of 7 basic emotions with confidence. On the GPU profile, p95 latency from capture to overlay render must be under 1000 ms over a 60 s session, and it is shown in the UI. Faces gain or lose boxes within 1 s of entering or leaving the frame.
- **Live audio:** while the user speaks, show the vocal emotion and a fused emotion. In silence, show "no speech" and the fused emotion equals the facial one. Muting the mic must not break video.
- **File analysis:** mp4/webm/mkv/avi up to 10 min, with visible progress. Playback shows synced boxes and emotions plus an emotion-over-time chart. A file with no audio or no faces finishes without error and reports that. An unsupported codec yields `failed` with a message.
- **Export:** CSV and JSON for any finished session, re-importable by pandas.
- **Ops:** `docker compose up` (CPU) and `docker compose --profile gpu up` serve the app at `http://localhost` on Linux and on Windows (Docker Desktop, WSL2). The GPU profile falls back to CPU with a visible warning. The whole system works with the network disconnected: no runtime calls to CDNs, model hubs or remote fonts. Model weights are baked into images at build time.
- **Hardware budget:** all models must load together within 4 GB VRAM / 16 GB RAM. CPU-only must work.
- **UI text is in Russian.** Camera and mic are captured in the browser, never passed into the container.
- **Measurement:** one benchmark command reports ms per frame (detection, FER), ms per SER window, peak VRAM/RAM, and the maximum number of parallel 720p@5fps streams with and without audio, on both CPU and GPU. One evaluation command reports UAR, macro-F1 and confusion matrices for FER (RAF-DB or AffectNet subset) and SER (RESD, Dusha subsets) as markdown. Datasets are not bundled.
- **Docs:** `docs/media-requirements.md` (video and audio parameters, each with an evidence source, plus the measurement conditions for accuracy targets) and `docs/scaling-plan.md` (capacity per node from the benchmark, single-server vs multi-node with hardware and cost, one recommendation).
- **Non-goals:** fine-tuning, auth/multi-user, multi-stream demo, RTSP ingest, annotated MP4 export, cloud/K8s, emotions beyond the 7 basic ones, valence/arousal. Assume one user and one live session at a time.

## Technical Decisions

- **Models (pretrained, no fine-tuning):**
  - Faces: YuNet via OpenCV `FaceDetectorYN` (2026may onnx with OpenCV 5.x, or 2023mar with 4.x).
  - Facial emotion: EmotiEffLib `enet_b0_8_va_mtl` on the ONNX backend. Drop the contempt class and renormalise to 7.
  - Voice activity: Silero VAD at 16 kHz.
  - Speech emotion: GigaAM-Emo via `gigaam.load_model("emo")`. Fallback is emotion2vec+ base, only if GigaAM fails on CPU or the memory budget.
- **Emotion keys** are always `anger, disgust, fear, happiness, sadness, surprise, neutral`. Audio labels map angry→anger, sad→sadness, neutral→neutral, positive→happiness. Fear, disgust and surprise come from the face only. Confirm the GigaAM label list via `get_probs` first.
- **Processing defaults:**
  - Client sends JPEG at 8 fps (5–10), longest side ≤ 640 px, quality ~0.8.
  - An IoU tracker assigns stable track ids. Facial probabilities get a per-track EMA (α = 0.4).
  - Audio is PCM16 mono 16 kHz in 500 ms chunks. SER runs on a 3 s window with a 1 s hop, only when VAD marks speech in at least 50% of the window.
  - Fusion: `p = 0.7·p_face + 0.3·p_audio` per shared class, with w_audio = 0 when there is no speech, then renormalise.
  - Files are decoded with PyAV/ffmpeg, video sampled at 5 fps, audio resampled to 16 kHz mono and analysed in the live windows.
- **Process model:**
  - Inference runs in one worker process holding all models, outside the FastAPI event loop.
  - It talks to the backend through `multiprocessing` queues: a latest-frame slot (size 1) for live video, and bounded FIFOs for audio and file jobs.
  - The `source → queue → worker → results` seam is mandatory so RTSP or multi-camera adapters can be added later without a rewrite.
- **Runtime stack:**
  - Backend: Python 3.11+, FastAPI, Uvicorn, ONNX Runtime (CPU EP, or CUDA EP for gpu), PyTorch only where GigaAM needs it, OpenCV, PyAV, SQLite.
  - Frontend: React + TypeScript + Vite, with a canvas overlay over the local `<video>` and a locally bundled chart library.
- **Docker:**
  - The `frontend` service is nginx serving the SPA and proxying `/api` and `/ws` to `backend`.
  - Profile `gpu` uses a CUDA base image and `gpus: all`. A named volume `data` holds SQLite and uploads. Ports are configurable via `.env` (default 80).
- **Layout:**
  - `backend/app/` (api, ws, sessions, export)
  - `backend/inference/` (detector, fer, vad, ser, fusion, tracker, worker)
  - `backend/bench/`, `backend/eval/`
  - `frontend/src/` (live, upload, session view, export)
  - `docs/`
- **Interfaces (fixed contract):**
  - `/ws/live` binary frames: type `0x01` + float64 `capture_ts` + uint32 `frame_id` + JPEG. Audio: `0x02` + float64 `capture_ts` + PCM16 LE.
  - `/ws/live` control JSON: `start`, `stop`, and `config` with fusion weights.
  - Server sends `session`, `frame_result` (faces with track_id, box relative to the sent JPEG, face/fused probs and labels, confidence; audio block with speech/probs/label) and `error`. Only the latest frame is answered.
  - REST: `/api/health` (status, device, models), `/api/files`, `/api/sessions[/{id}[/timeline|/media|/export?format=csv|json]]`, `DELETE /api/sessions/{id}`.
  - File status values are `queued|processing|done|failed`. Possible notes are `no_audio_track` and `no_faces_detected`.
  - Export row columns: `session_id, ts_ms, source(face|audio), track_id, box_x..box_h, p_<7 emotions>, face_label, audio_label, fused_label, speech`. Audio rows have empty box columns and zeros for unmapped classes.
- **Versions:** re-check library versions at install time.

## UX & Interaction Patterns

- Live view: the user enables the camera and sees their own video with a canvas overlay. Boxes are scaled from JPEG coordinates to the displayed size. Each face is labelled with its emotion and confidence. The UI shows a p50/p95 latency readout computed as `render_time − capture_ts`.
- Audio indicator: "no speech" vs vocal emotion, alongside the fused emotion.
- Upload flow: progress while processing, then synced playback with the overlay, an emotion timeline chart, and CSV/JSON download.
- GPU-unavailable warning is visible in the UI. All UI text is in Russian.

## Cross-Story Dependencies

- 1.1 (skeleton, compose, baked weights, `/api/health`) underpins everything.
- 1.2 (inference core + bench) is required by 1.3, 1.4, 1.6, 1.8 and 1.9. It must stop and report if the models exceed 4 GB VRAM together or if detector+FER exceed 100 ms per frame on GPU.
- 1.3 sets up `/ws/live` and the latest-frame slot. 1.4 extends it with audio and fusion.
- 1.5 adds persistence and export. 1.6 reuses that export and the timeline for file sessions.
- 1.7 hardens all features already built (offline, fallback, Windows).
- 1.8 extends the 1.2 bench and produces `docs/benchmark-report.md`, which 1.11 depends on.
- 1.9 produces `docs/evaluation-report.md`, which 1.10 depends on.
- 1.10 and 1.11 also draw on the project's research document (media table and scaling sections).
