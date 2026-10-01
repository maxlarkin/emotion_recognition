---
id: SPEC-emotion-recognition-mvp
companions:
  - stack.md
  - interfaces.md
  - architecture-diagrams.md
  - ../../planning-artifacts/research/technical-realtime-emotion-recognition-2026-10-01/research.md
sources: []
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Real-time emotion recognition MVP (video + audio)

## Why

Mandate plus vision. An educational project must deliver, within 1.5 months, a working prototype that recognizes human emotions in real time from video and audio. It must come with a media quality regulation and a justified path to 20+ parallel cameras. The team has only a laptop (4 GB VRAM NVIDIA GPU, 16 GB RAM) and no GPU server. The MVP must therefore run locally, offline, on open pretrained models with no fine-tuning. Research (`research.md`) selected those models and showed that 95–98% accuracy is unattainable on in-the-wild data. The MVP is judged as a working demo, not on accuracy.

## Capabilities

- **CAP-1**
  - **intent:** User opens the app, enables the webcam, and sees their own live video with a box around each face and that face's current emotion (one of 7 basic) with confidence.
  - **success:** On the target laptop (GPU profile), p95 latency from frame capture to rendering of that frame's overlay is < 1000 ms over a 60 s session. The latency is shown in the UI. Faces entering or leaving the frame gain or lose boxes within 1 s.
- **CAP-2**
  - **intent:** System listens to the microphone during a live session, detects speech, recognizes vocal emotion, and fuses it with the facial emotion.
  - **success:** While the user speaks, the UI shows the vocal emotion and a fused emotion. In silence, the audio indicator shows "no speech" and the fused emotion equals the facial one. Muting the mic does not break the video pipeline.
- **CAP-3**
  - **intent:** User uploads a recorded video file and gets the same face and voice emotion analysis over its whole duration.
  - **success:**
    - An mp4 / webm / mkv / avi file of up to 10 min is analysed with visible progress.
    - The result plays back with face boxes and emotions synced to the video, plus an emotion-over-time chart.
    - A file with no audio track or no faces finishes without error and reports that fact.
- **CAP-4**
  - **intent:** User exports the results of any live or file session.
  - **success:** For any finished session, CSV and JSON download. Each row holds timestamp, face track id, box, per-emotion probabilities, facial label, audio label, and fused label. Files are re-importable by pandas without errors.
- **CAP-5**
  - **intent:** Operator starts the whole system with one Docker command on Linux or Windows and uses it without internet.
  - **success:**
    - `docker compose up` (CPU profile) and `docker compose --profile gpu up` (GPU profile) bring the app up at `http://localhost` on Linux and on Windows (Docker Desktop, WSL2).
    - With the network disconnected, CAP-1–CAP-4 still work.
    - The GPU profile falls back to CPU with a visible warning if no GPU is available.
- **CAP-6**
  - **intent:** Team measures runtime cost on the target machine to ground scaling claims.
  - **success:** One command runs a benchmark and writes a report covering:
    - ms per frame for detection and FER;
    - ms per audio window for SER;
    - peak VRAM and RAM;
    - maximum sustainable number of parallel 720p@5 fps streams, with and without audio.
    
    The report is produced for both the CPU and GPU profiles.
- **CAP-7**
  - **intent:** Team measures the recognition quality of the chosen models on public test data.
  - **success:** One command runs an offline evaluation script. It reports UAR, macro-F1 and a confusion matrix for FER on a RAF-DB or AffectNet test subset, and for SER on RESD and Dusha test subsets. Results are saved as a markdown report.
- **CAP-8**
  - **intent:** Team delivers a media quality regulation: the minimum video and audio parameters for reliable recognition.
  - **success:** A document in the repo specifies the following, each with its evidence source:
    - resolution, face size in px, interocular distance, frame rate, compression, head pose, occlusion, lighting;
    - audio sample rate, bit depth, codec/bitrate, SNR, segment length;
    - the measurement conditions under which accuracy targets are stated.
    
    The draft table in `research.md` §4.2 is the seed, and CAP-7 results are added where available.
- **CAP-9**
  - **intent:** Team delivers a justified deployment plan for ≥20 parallel camera streams.
  - **success:** A document in the repo gives the stream-per-node capacity measured by CAP-6. It compares single-server and multi-node options with hardware and cost, and recommends one, citing `research.md` §3.

## Constraints

- **Target hardware:** laptop with an NVIDIA GPU of 4 GB VRAM and 16 GB RAM. Every model must load together within these limits. No GPU server or cloud at runtime.
- **CPU-only must work.** Docker GPU passthrough is not guaranteed on every Linux or Windows machine.
- **Fully offline at runtime.** No calls to external services, CDNs or model hubs once the images are built. Model weights ship inside the image.
- **Linux and Windows via Docker only.** No native installs required from the user.
- **Fixed stack:** FastAPI backend and React frontend.
- **UI text is in Russian.**
- **Camera and microphone are captured in the browser,** not in the container. Device passthrough into Docker on Windows is not viable.
- **Pretrained open models only, no fine-tuning in MVP.** Choices and alternatives are listed in `stack.md`.
- **Inference runs outside the FastAPI event loop.** The "source → queue → worker → results" seam is mandatory so RTSP or multi-camera sources can be added without a rewrite.
- **Delivery: 1.5 months.**

## Non-goals

- Fine-tuning on Russian media content (planned after the MVP; datasets are listed in `research.md`).
- Authentication, user accounts, multi-user access.
- Running ≥20 cameras, or any multi-stream demo, in the MVP. The CAP-9 document is the accepted evidence for the multi-camera requirement.
- RTSP/IP-camera ingest in the MVP (the seam exists, the adapter does not).
- Meeting the 95% / 98% accuracy targets.
- Exporting annotated video (burned-in overlay MP4).
- Cloud deployment, Kubernetes, horizontal scaling infrastructure.
- Emotions beyond the 7 basic ones; valence/arousal; micro-expressions.

## Success signal

On the target laptop, with the network unplugged, a fresh `docker compose --profile gpu up` is followed by this demo:
1. The user opens `http://localhost`, enables the camera and sees their video.
2. Face boxes and emotions update with p95 latency under 1 s.
3. When the user speaks, a vocal emotion appears.
4. The user uploads a recorded clip, gets its analysis, and downloads CSV/JSON.

The regulation (CAP-8) and scaling plan (CAP-9) documents are in the repo, backed by the benchmark (CAP-6) and evaluation (CAP-7) reports.

## Assumptions

- "Offline" means runtime only. Images are built once with internet access, or transferred with `docker save` / `docker load`.
- One user and one live session at a time.
- CSV and JSON export satisfies "export". Annotated MP4 is not needed.
- The contempt class of the 8-class FER model is dropped and the remaining probabilities are renormalised to the 7 basic emotions.
- Audio labels map as follows: angry → anger, sad → sadness, neutral → neutral, positive → happiness. Fear, disgust and surprise come from the face only.
