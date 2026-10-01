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
- **Fusion:** per shared class, `p = w_face·p_face + w_audio·p_audio`. Defaults: w_face = 0.7, w_audio = 0.3; w_audio = 0 when there is no speech. Classes without an audio counterpart use the face probability only. Renormalise after fusion.
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
backend/   app/ (api, ws, sessions, export), inference/ (detector, fer, vad, ser, fusion, tracker, worker), bench/, eval/
frontend/  src/ (live, upload, session view, export)
docs/      media-requirements.md (CAP-8), scaling-plan.md (CAP-9), benchmark-report.md, evaluation-report.md
docker-compose.yml, backend/Dockerfile (+ gpu variant/arg), frontend/Dockerfile
```
