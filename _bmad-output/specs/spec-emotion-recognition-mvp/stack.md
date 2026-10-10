# Stack and model choices

Decisions of record for implementers. Evidence and alternatives live in `research.md` (section numbers given). Re-check versions at install time: the research flagged version claims as stale after 2026-11-01.

## Models (no fine-tuning)

| Role | Choice | Notes | Research |
|---|---|---|---|
| Face detection | YuNet via OpenCV `FaceDetectorYN` | Use `face_detection_yunet_2026may.onnx` with OpenCV 5.x, or the 2023mar model with OpenCV 4.x. Detects faces from ~10 px. | §1 |
| Facial emotion (FER) | EmotiEffLib `enet_b0_8_va_mtl`, ONNX backend (`pip install emotiefflib`) | 8 classes; drop contempt and renormalise to 7. Upgrade option: `enet_b2_8_best` (+1.5 pt AffectNet, ~3× slower). | §1 |
| Voice activity | Silero VAD | 16 kHz; <1 ms per 30 ms chunk on CPU. | §2 |
| Speech emotion (SER) | GigaAM-Emo (`gigaam.load_model("emo")`) | Russian; Dusha classes angry/sad/neutral/positive. Confirm the label list via `get_probs` in the first story. ONNX export available; fp16 on GPU. | §2 |
| Fallback SER | emotion2vec+ base | Only if GigaAM-Emo fails on CPU or memory budget. 9 labels; Russian quality unverified. | §2 |

## Processing parameters

- **Live video:** client sends JPEG at 5–10 fps (default 8), longest side ≤ 640 px, quality ~0.8.
- **Face tracking:** IoU tracker assigns stable track ids. Detection runs every frame for one stream; detect-every-N is optional for files.
- **Temporal smoothing:** EMA per track over facial probabilities (default α = 0.4, configurable).
- **Live audio:** PCM16 mono 16 kHz, 500 ms chunks. SER runs on a sliding 3 s window (hop 1 s), only when VAD marks speech in ≥ 50% of the window.
- **Fusion:** over all 7 keys, `p = renormalise(w_face·p_face + w_audio·p_audio)`. Defaults: w_face = 0.7, w_audio = 0.3. `p_audio` is 0 for disgust, fear and surprise (no SER class), so those keys get only the weighted face term. No speech: w_audio = 0, and the fused distribution equals the face distribution.
- **RTSP ingest:**
  - One source per stream decodes with PyAV (TCP transport, 5 s open timeout). The GPU profile decodes with NVDEC (PyAV `hwaccel` cuda) and falls back to CPU decode. With 20 streams, sources may be threads or a small process pool instead of one process each, to stay within 16 GB RAM.
  - Video is sampled to the analysis rate (default 5 fps). The audio track, if present, is resampled to 16 kHz mono and windowed like live audio.
  - Each stream has a `camera_type` (`camera-profiles.md`). Frames are scaled to that profile's detection size, and FER crops come from that frame.
  - Frames and audio go to the shared inference worker. A per-stream latest-frame slot drops stale frames, and the worker batches faces across streams.
  - On stream loss: reconnect with backoff of 1, 2, 4… seconds, capped at 30 s.
- **Detection cost at large profiles:** YuNet cost grows with pixel count, so `sector` (1920) and `security` (2560) detection dominate CPU time. If CPU YuNet cannot hold the 20-stream mix at 5 fps, run the YuNet ONNX on ONNX Runtime CUDA EP, or detect every 2nd frame and let the tracker carry boxes. CAP-6 decides.
- **Demo cameras:**
  - Compose profile `demo-cams` adds two services. `mediamtx` runs `bluenviron/mediamtx`, pinned. `camera-emulator` runs ffmpeg, which loops 20 clips to `rtsp://mediamtx:8554/cam1..cam20` with `-re -stream_loop -1 -c copy`.
  - Clips are pre-encoded once per camera profile (resolution, 10 fps, H.264, AAC where the profile has audio) by a prep script, so the laptop does no live re-encoding. The mix and per-type geometry are in `camera-profiles.md`.
  - The default stream list (env `DEFAULT_STREAMS`) points at those 20 URLs with their `camera_type`.
  - Clips are openly licensed, with faces, Russian speech, and moments of clear negative and positive emotion. `demo-videos/ATTRIBUTION.md` records the source, author and licence of each.
- **Emotion events (CAP-11):**
  - Groups: negative = anger, disgust, fear, sadness; positive = happiness; neutral and surprise raise no event.
  - Rule per face track on the smoothed fused probabilities: a negative event when the negative-group sum is ≥ 0.6 for ≥ 2 s; a positive event when happiness is ≥ 0.6 for ≥ 2 s. The same kind is not repeated on the same track within 30 s. The negative emotion subset, both thresholds, the duration and the cooldown come from `/api/settings` (defaults as here) and apply without restart.
  - Faces below the stream's `min_face_px` (`face_px` at detection size) and tracks younger than 1 s are `abstain`: they feed neither the rule nor an open event.
  - Per-face `decision`: `alert` while the track's negative event is open (from emit until the negative-group sum drops below the threshold), `watch` when the negative-group sum is ≥ threshold but not yet held for the duration, `abstain` as above, `positive` when happiness is ≥ threshold, else `none`.
  - Negative events start in state `new`; `POST /api/events/{id}/ack` sets `confirmed` or `false_alarm`. The state is stored only; MVP thresholds do not learn from it.
  - The event is emitted within 1 s of the rule holding and stored in SQLite. An event carries the peak emotion, the mean group probability over the held interval, and the duration. No face image is stored.
- **System load:** `psutil` for CPU and RAM. NVML (`nvidia-ml-py`) for GPU utilisation and VRAM; it reports null in the CPU image or when no GPU is present.
- **Latency budget:** per-frame inference (`proc_ms`) p95 ≤ 100 ms with 20 streams; RTSP decode-to-result p95 < 1000 ms.
- **File analysis:** decode with ffmpeg (PyAV). Video sampled at 5 fps. Audio resampled to 16 kHz mono and analysed in the same windows as live audio.

## Runtime

- **Backend:** Python 3.11+, FastAPI, Uvicorn, ONNX Runtime (CPU EP, or CUDA EP in the GPU profile), PyTorch only where GigaAM requires it, OpenCV, PyAV, SQLite (SQLAlchemy or plain `sqlite3`).
- **Inference process:** one worker process holds all models. FastAPI talks to it through `multiprocessing` queues: latest-frame slot of size 1 for live video, bounded FIFO for audio and file jobs.
- **Frontend:** React + TypeScript + Vite, React Router (browser history; nginx falls back to `index.html`). Plain CSS over the `tokens.css` custom properties, no UI kit. IBM Plex fonts from `@fontsource` packages. Canvas overlay on top of the local `<video>` element; charts drawn in-house (SVG/canvas), no CDN. Screens and states per `ui.md`.
- **Settings and streams persistence:** global settings and the stream list (name, site, url, camera_type, settings) are stored in SQLite and survive restarts; `DEFAULT_STREAMS` seeds the list only when it is empty.
- **Docker:** `docker compose` with services `frontend` (nginx serving the built SPA and proxying `/api` and `/ws` to the backend) and `backend`.
  - Profile `gpu` uses `gpus: all`; the default profile is CPU-only. Both images share the `python:3.11-slim` base: the GPU image gets CUDA 12.6 / cuDNN 9 libraries from the torch cu126 `nvidia-*` wheels, not from a CUDA base image.
  - Weights are downloaded during `docker build` and baked into the image.
  - A named volume `data` holds SQLite and uploads.
- **Port:** app served at `http://localhost:80`. Ports are configurable via `.env`.

## Repository layout (target)

```
backend/   app/ (api, ws, sessions, export, streams), sources/ (webcam, rtsp, file), inference/ (detector, fer, vad, ser, fusion, tracker, worker), bench/, eval/
frontend/  src/ (app shell + router, api/ with mocks, ui/ components, streams grid + detail, live, history, settings, analyze)
docs/      media-requirements.md (CAP-8), scaling-plan.md (CAP-9), benchmark-report.md, evaluation-report.md
backend/   app/events (CAP-11 rule engine + journal API)
demo-videos/ (clips + ATTRIBUTION.md, prep script for per-profile encodes), docker-compose.yml, backend/Dockerfile (+ gpu variant/arg), frontend/Dockerfile
```
