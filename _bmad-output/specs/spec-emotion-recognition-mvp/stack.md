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
  - One source process per stream decodes with PyAV (TCP transport, 5 s open timeout).
  - Video is sampled to the analysis rate (default 8 fps, minimum 5). The audio track is resampled to 16 kHz mono and windowed like live audio.
  - Frames and audio go to the shared inference worker. A per-stream latest-frame slot drops stale frames, and the worker batches faces across streams.
  - On stream loss: reconnect with backoff of 1, 2, 4… seconds, capped at 30 s.
- **Demo cameras:**
  - Compose profile `demo-cams` adds two services. `mediamtx` runs `bluenviron/mediamtx`, pinned. `camera-emulator` runs ffmpeg, which loops `demo-videos/cam{1,2,3}.mp4` as H.264 + AAC to `rtsp://mediamtx:8554/cam{1,2,3}` with `-re -stream_loop -1`.
  - The default stream list (env `DEFAULT_STREAMS`) points at those three URLs.
  - Clips are openly licensed, with close-up faces and Russian speech. `demo-videos/ATTRIBUTION.md` records the source, author and licence of each.
- **System load:** `psutil` for CPU and RAM. NVML (`nvidia-ml-py`) for GPU utilisation and VRAM; it reports null in the CPU image or when no GPU is present.
- **File analysis:** decode with ffmpeg (PyAV). Video sampled at 5 fps. Audio resampled to 16 kHz mono and analysed in the same windows as live audio.

## Runtime

- **Backend:** Python 3.11+, FastAPI, Uvicorn, ONNX Runtime (CPU EP, or CUDA EP in the GPU profile), PyTorch only where GigaAM requires it, OpenCV, PyAV, SQLite (SQLAlchemy or plain `sqlite3`).
- **Inference process:** one worker process holds all models. FastAPI talks to it through `multiprocessing` queues: latest-frame slot of size 1 for live video, bounded FIFO for audio and file jobs.
- **Frontend:** React + TypeScript + Vite. Canvas overlay on top of the local `<video>` element. Chart library bundled locally, no CDN.
- **Docker:** `docker compose` with services `frontend` (nginx serving the built SPA and proxying `/api` and `/ws` to the backend) and `backend`.
  - Profile `gpu` uses the CUDA base image and `gpus: all`; the default profile is CPU-only.
  - Weights are downloaded during `docker build` and baked into the image.
  - A named volume `data` holds SQLite and uploads.
- **Port:** app served at `http://localhost:80`. Ports are configurable via `.env`.

## Repository layout (target)

```
backend/   app/ (api, ws, sessions, export, streams), sources/ (webcam, rtsp, file), inference/ (detector, fer, vad, ser, fusion, tracker, worker), bench/, eval/
frontend/  src/ (live, multicam grid, upload, session view, export)
docs/      media-requirements.md (CAP-8), scaling-plan.md (CAP-9), benchmark-report.md, evaluation-report.md
demo-videos/ (clips + ATTRIBUTION.md), docker-compose.yml, backend/Dockerfile (+ gpu variant/arg), frontend/Dockerfile
```
