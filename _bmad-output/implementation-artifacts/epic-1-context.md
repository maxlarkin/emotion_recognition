# Epic 1 Context: Real-time emotion recognition MVP

<!-- Compiled from planning artifacts. Edit freely. Regenerate with compile-epic-context if planning docs change. -->

## Goal

Deliver within 1.5 months a working, fully offline Docker prototype that recognises the 7 basic emotions in real time from face video and Russian speech on a single laptop (NVIDIA GPU with 4 GB VRAM, 16 GB RAM, CPU-only must also work). The demo covers: a live webcam and microphone session with an overlay; parallel analysis of at least 3 RTSP camera streams on one multi-camera screen; analysis of uploaded video files; and CSV/JSON export. It also delivers the documents that justify the system: a benchmark report, an accuracy evaluation report, a media quality regulation, and a scaling plan for 20+ cameras. The project is judged as a working demo, not on accuracy. Open pretrained models are used with no fine-tuning, and the 95–98% accuracy targets are explicitly out of reach.

## Stories

- Story 1.1: Repository skeleton and Docker compose with CPU and GPU profiles
- Story 1.2: Inference core with per-stage benchmark
- Story 1.3: Live webcam analysis with overlay and latency readout
- Story 1.4: Live microphone emotion and fusion in UI
- Story 1.5: Session persistence and CSV/JSON export
- Story 1.6: RTSP multi-stream ingest with demo cameras
- Story 1.7: Multi-camera grid screen with system load
- Story 1.8: Recorded video upload, analysis and synced playback
- Story 1.9: Offline and cross-platform hardening
- Story 1.10: Multi-stream capacity benchmark and report
- Story 1.11: Offline accuracy evaluation script and report
- Story 1.12: Media quality regulation document
- Story 1.13: Scaling plan for 20+ cameras

## Requirements & Constraints

- **Live webcam:** box plus emotion and confidence for each face. On the GPU profile, p95 latency from capture to overlay render must stay < 1000 ms over 60 s, and the UI shows it. Faces gain or lose boxes within 1 s of entering or leaving the frame.
- **Live audio:** while the user speaks, the UI shows the vocal and fused emotions. In silence it shows "no speech" and the fused emotion equals the facial one. Muting the mic must not break video.
- **Multi-stream RTSP:**
  - At least 3 emulated 720p cameras with audio are analysed in parallel for 10 min on the GPU profile. Each stream keeps p95 < 1000 ms and ≥ 5 analysed fps.
  - Each tile shows frames, boxes, facial/vocal/fused emotions, fps and p50/p95. The screen shows overall CPU, RAM, GPU and VRAM load.
  - Streams are isolated: stopping or losing one does not affect the others. A lost stream shows its status and reconnects automatically. Streams can be added by URL at runtime without a restart.
- **Files:** mp4/webm/mkv/avi up to 10 min, with visible progress, synced playback overlay and an emotion-over-time chart. Files with no audio or no faces finish successfully and report it. An unsupported codec gives a failed status with a message.
- **Export:** CSV and JSON for any finished webcam, file or RTSP session. Both must be re-importable by pandas.
- **Ops:**
  - The CPU profile runs with `docker compose up`, the GPU profile with `docker compose --profile gpu up`, at `http://localhost`, on Linux and on Windows (Docker Desktop, WSL2).
  - Everything works with the network unplugged, including the emulated cameras. The GPU profile falls back to CPU with a visible warning.
- **Benchmark and evaluation:**
  - One benchmark command reports per-stage ms (detection, FER, SER per window), peak VRAM/RAM, and the maximum number of parallel 720p@5fps streams with and without audio, on both CPU and GPU.
  - One offline evaluation command reports UAR, macro-F1 and confusion matrices: FER on a RAF-DB/AffectNet subset, SER on RESD and Dusha subsets. Datasets are not bundled; the script takes local paths.
- **Documents:**
  - The media regulation covers video parameters (resolution, face px, interocular distance, fps, compression, pose, occlusion, light) and audio parameters (sample rate, bit depth, codec/bitrate, SNR, segment length), with evidence for each and the measurement conditions.
  - The scaling plan uses the measured per-node capacity, compares single-server and multi-node options with hardware and cost, and recommends one.
- **Hard constraints:**
  - All models must fit together in 4 GB VRAM / 16 GB RAM.
  - No CDN, remote fonts, model hub or package installs at runtime. Weights are baked in at build and verified by checksum.
  - All UI text is in Russian.
  - The local camera and mic are captured in the browser (`getUserMedia`), with no device passthrough. RTSP is ingested on the server.
- **Non-goals:** auth or multi-user access; running ≥ 20 cameras; ONVIF, PTZ or RTSP recording; RTSP auth beyond credentials in the URL; annotated MP4 export; valence/arousal; cloud or Kubernetes.

## Technical Decisions

- **Models:**
  - Face detection: YuNet via OpenCV `FaceDetectorYN`, with the model version matched to the OpenCV major version.
  - FER: EmotiEffLib `enet_b0_8_va_mtl` on ONNX. It has 8 classes; drop contempt and renormalise to 7.
  - VAD: Silero VAD at 16 kHz.
  - SER: GigaAM-Emo, fp16 on GPU. Confirm its labels via `get_probs`. Fallback only: emotion2vec+ base.
- **Emotion keys** are always `anger, disgust, fear, happiness, sadness, surprise, neutral`. Audio labels map angry→anger, sad→sadness, neutral→neutral, positive→happiness. Fear, disgust and surprise come from the face only.
- **Processing:**
  - IoU tracker for stable track ids.
  - Per-track EMA on facial probabilities (α 0.4, configurable).
  - Audio is PCM16 mono 16 kHz in 500 ms chunks. SER runs on a 3 s window with a 1 s hop, only if VAD marks speech in ≥ 50% of the window.
  - Fusion is `w_face·p_face + w_audio·p_audio` (defaults 0.7 / 0.3). `w_audio` is 0 without speech. Face-only classes use the face probability. Renormalise after fusion.
- **Single inference worker process** holds all models and runs outside the FastAPI event loop. Every source (webcam WS, RTSP, file) feeds it through the "source → queue → worker → results" seam. It keeps a per-source latest-frame slot of size 1 that drops stale frames, plus bounded FIFOs for audio and file jobs, and batches faces across sources. Models load once.
- **Live input:** the client sends JPEG at 5–10 fps (default 8), longest side ≤ 640 px, quality ~0.8. The server answers only the latest frame; dropped frames get no reply.
- **RTSP ingest:**
  - One source process per stream decodes with PyAV (TCP, 5 s open timeout), samples to 8 fps (minimum 5), and resamples audio to 16 kHz mono windows. Decoding never runs inside the worker.
  - Reconnect backoff is 1, 2, 4… s, capped at 30 s. `DEFAULT_STREAMS` env lists the startup streams.
- **Demo cameras:** compose profile `demo-cams` runs a pinned `bluenviron/mediamtx` plus an ffmpeg emulator. The emulator loops `demo-videos/cam{1,2,3}.mp4` (H.264 + AAC, `-re -stream_loop -1`) to `rtsp://mediamtx:8554/cam{1,2,3}`. Clips must be openly licensed, with close-up faces and Russian speech, and attributed in `demo-videos/ATTRIBUTION.md`.
- **System load:** `psutil` for CPU/RAM. NVML (`nvidia-ml-py`) for GPU/VRAM, which returns null on CPU or with no GPU.
- **Files:** decoded with PyAV/ffmpeg, video sampled at 5 fps, audio windowed like live audio.
- **Fixed protocol contract** (change only via a spec update):
  - `/ws/live`:
    - Binary video frames: `0x01`, f64 `capture_ts`, u32 `frame_id`, then JPEG.
    - Binary audio chunks: `0x02`, f64 `capture_ts`, then PCM16.
    - JSON control messages: start, stop, config.
    - The server sends `frame_result` (faces, audio), `session` and `error`.
  - `/ws/streams` (view-only, one socket for the whole grid):
    - Binary `0x11` frames: u16 id length, `stream_id`, u32 `frame_id`, then JPEG ≤ 640 px. Each is sent just before its `stream_result`.
    - JSON messages: `stream_status` (connecting/running/reconnecting/stopped/error) and `system` once per second with `server_ts`.
    - Optional `subscribe`.
    - Latency is `render − decode_ts`, corrected by the clock offset taken from `server_ts`.
  - REST:
    - `/api/health` reports the device and loaded models. `/api/system` is also exposed.
    - Endpoints: `/api/files` (POST), `/api/streams` (GET, POST, DELETE), and `/api/sessions` with list, detail, timeline, media, export and delete.
    - Session kinds are `live|file|rtsp`. File statuses are `queued|processing|done|failed`, with notes `no_audio_track` and `no_faces_detected`.
    - Stream URLs are returned with credentials masked.
  - The export row has the fixed column order `session_id, stream_id, ts_ms, source(face|audio), track_id, box_*, p_<7 keys>, face_label, audio_label, fused_label, speech`. `stream_id` is empty for webcam and file sessions. Audio rows have empty box columns and zeros for classes with no audio counterpart.
- **Runtime:** Python 3.11, FastAPI/Uvicorn, ONNX Runtime with exactly one package per image (`onnxruntime` for CPU, `onnxruntime-gpu` for GPU), PyTorch only where GigaAM needs it, OpenCV, PyAV, SQLite in the named volume `data`, which also holds uploads.
- **Frontend:** React + TS + Vite. Canvas overlay on the local `<video>`. Chart library bundled locally. nginx serves the SPA and proxies `/api` and `/ws`.
- **Docker:** one backend Dockerfile with `VARIANT=cpu|gpu`. The GPU profile uses a CUDA base image and `gpus: all`. Port 80 is configurable via `.env`.

## UX & Interaction Patterns

- Russian-only UI.
- **Live screen:** local video with a canvas overlay of boxes, emotion and confidence; a p50/p95 latency readout; an audio state indicator ("no speech" or the vocal emotion); the fused emotion.
- **Multi-camera screen:** one tile per stream, add/remove by RTSP URL, per-tile status, and an overall system load panel.
- **Upload flow:** progress, then playback with a synced overlay and an emotion timeline chart.
- **Sessions:** list with CSV/JSON download.
- **GPU fallback:** a visible warning.

## Cross-Story Dependencies

- 1.1 (done) provides the compose, the images and the baked weights that everything else builds on.
- 1.2's multi-source worker underpins 1.3, 1.4, 1.6 and 1.8.
- 1.3 must come before 1.4, which reuses `/ws/live`.
- 1.5's persistence and export, including `stream_id`, are reused by 1.6 (rtsp sessions) and 1.8.
- 1.6 must come before 1.7 (grid UI) and 1.10, which drives N streams from the demo-cams emulator.
- 1.9 verifies offline operation across all of it, including demo-cams.
- 1.10 → 1.13: the measured capacity feeds the scaling plan.
- 1.11 → 1.12: accuracy results feed the media regulation.
- Done-checkpoint stories, where the agent stops for human review: 1.1, 1.2, 1.3, 1.7, 1.8, 1.9, 1.12, 1.13.
- Stop-and-report triggers:
  - 1.2: the models don't fit in 4 GB VRAM, or FER plus detector take more than 100 ms per frame on GPU.
  - 1.6: no suitable licensed clips can be found.
