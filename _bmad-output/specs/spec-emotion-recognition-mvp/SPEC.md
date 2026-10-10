---
id: SPEC-emotion-recognition-mvp
companions:
  - stack.md
  - interfaces.md
  - camera-profiles.md
  - ui.md
  - architecture-diagrams.md
  - ../../planning-artifacts/research/technical-realtime-emotion-recognition-2026-10-01/research.md
sources: []
---

> **Canonical contract.** This SPEC and the files in `companions:` are the complete, preservation-validated contract for what to build, test, and validate. Source documents listed in frontmatter are for traceability — consult them only if you need narrative rationale or prose color this contract intentionally omits.

# Real-time emotion recognition MVP (video + audio)

## Why

A quality manager oversees client service points spread across cities and countries. Each point has cameras with audio of three types: a webcam on the manager's PC facing the client, high-resolution sector cameras, and security dome cameras (`camera-profiles.md`). Video is primary and audio is secondary. Negative client emotions must reach the manager quickly; positive ones are only recorded. The near-term scale is 30–50 cameras, later up to 3000.

This educational project must deliver, within 1.5 months, a working prototype that recognises emotions in real time and processes 20 parallel camera streams on one laptop (4 GB VRAM NVIDIA GPU, 16 GB RAM). It must also answer two questions with evidence: which camera resolution is the lowest that still gives reliable recognition for each camera type (target 80%), and what local hardware handles 30–50 cameras. The system runs offline on open pretrained models with no fine-tuning. Research (`research.md`) shows 95–98% is unattainable on in-the-wild data. The MVP is judged as a working demo; accuracy numbers are measured, not required.

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
    - Several files can be queued at once; each shows its own status and progress.
    - The result plays back with face boxes and emotions synced to the video, plus an emotion-over-time chart, an episode timeline and an episode table, with selection synced between timeline, table and playback.
    - A file with no audio track or no faces finishes without error and reports that fact.
- **CAP-4**
  - **intent:** User exports the results of any webcam, file or RTSP-stream session.
  - **success:** For any finished session, CSV and JSON download. Each row holds timestamp, face track id, box, per-emotion probabilities, facial label, audio label, and fused label. Files are re-importable by pandas without errors.
- **CAP-5**
  - **intent:** Operator starts the whole system with one Docker command on Linux or Windows and uses it without internet.
  - **success:**
    - `docker compose up` (CPU profile) and `docker compose --profile gpu up` (GPU profile) bring the app up at `http://localhost` on Linux and on Windows (Docker Desktop, WSL2).
    - With the network disconnected, CAP-1–CAP-4, CAP-10 (with emulated cameras) and CAP-11 still work.
    - The GPU profile falls back to CPU with a visible warning if no GPU is available.
- **CAP-6**
  - **intent:** Team measures runtime cost on the target machine to ground the 20-stream demo and the 30–50 camera hardware sizing.
  - **success:** One command runs a benchmark and writes a report covering:
    - ms per frame for detection (at each camera profile's detection size) and FER;
    - ms per audio window for SER;
    - decode cost per stream for each camera profile, CPU decode vs NVDEC;
    - peak VRAM and RAM;
    - maximum sustainable number of parallel streams per camera profile at 5 fps, with and without audio.

    The report is produced for both the CPU and GPU profiles.
- **CAP-7**
  - **intent:** Team measures the recognition quality of the chosen models on public test data, including how quality falls as faces get smaller.
  - **success:** One command runs an offline evaluation script and saves a markdown report with:
    - UAR, macro-F1 and a confusion matrix for FER on a RAF-DB or AffectNet test subset, and for SER on RESD and Dusha test subsets;
    - a face-size sweep: the FER test faces downscaled to 16, 24, 32, 48, 64 and 100 px, with UAR per size for 7 classes and for 3 groups (negative / neutral / positive);
    - the smallest face size that reaches the 80% target, or a statement that it is not reached.
- **CAP-8**
  - **intent:** Team delivers a media quality regulation: the minimum video and audio parameters for reliable recognition, per camera type.
  - **success:** A document in the repo specifies the following, each with its evidence source:
    - for each camera type in `camera-profiles.md`: minimum and recommended resolution, lens / horizontal field of view, distance to the client, mounting height and angle, frame rate, bitrate;
    - face size in px, interocular distance, head pose, occlusion, lighting, compression;
    - audio sample rate, bit depth, codec/bitrate, SNR, segment length;
    - the measurement conditions under which accuracy targets are stated.

    Minimum resolutions follow from the CAP-7 face-size sweep converted through pixels-per-metre. The `research.md` §4.2 table is the seed.
- **CAP-9**
  - **intent:** Team delivers a justified local (on-prem) deployment plan for 30–50 cameras, with an outlook to 3000.
  - **success:** A document in the repo gives:
    - stream-per-node capacity per camera profile, measured by CAP-6;
    - hardware requirements (CPU, GPU, RAM, disk, network) for 30–50 cameras, comparing at least one single-server and one multi-node option with cost;
    - an architecture outlook for 3000 cameras (edge processing at sites, events to a centre);
    - the seam where alert delivery (SMS, messenger, email) and ticket creation plug into CAP-11 events;
    - one recommended technology stack, citing `research.md` §3.
- **CAP-10**
  - **intent:** Operator connects 20 RTSP video streams of mixed camera types, either emulated cameras or real IP cameras given by URL. The system analyses all of them in parallel (face, audio where present, fusion). The operator watches them on one multi-camera screen.
  - **success:**
    - On the target laptop (GPU profile), the 20-stream demo mix (`camera-profiles.md`) is analysed simultaneously for 10 min.
    - Each stream sustains ≥ 5 analysed fps. Per-frame inference time p95 ≤ 100 ms. Per-stream decode-to-result p95 < 1000 ms.
    - Each tile shows the analysed frames with face boxes, facial/vocal/fused emotions, the stream's analysis FPS and p50/p95 latency.
    - The screen shows overall CPU, RAM, GPU utilisation and VRAM.
    - Stopping one stream, or losing its connection, does not affect the others. A lost stream shows its status and reconnects when the source returns.
    - Adding a stream by RTSP URL and camera type in the UI starts its analysis without restarting the system. Changing a stream's settings (analysis on/off, fps, detection size, minimum face size) applies without restarting the system.
    - Opening a stream shows a detail screen with a card per visitor (decision state, negative probability, facial and fused emotion), the vocal emotion, an interval journal, and a small-face warning when faces are below the stream's minimum size (`ui.md`).
- **CAP-11**
  - **intent:** The manager sees a journal of client emotion events: negative events that need attention and positive events recorded as facts.
  - **success:**
    - A sustained negative or positive emotion on a face track creates one event by the event rule (`stack.md`), within 1 s of the rule holding. A different client produces a new track and its own events.
    - Events from every source are stored and appear live in a Russian journal screen with camera, time, emotion, confidence and duration. Negative events are visually highlighted.
    - A negative event is an incident with state new, confirmed or false alarm. The UI shows the count of new incidents and lets the manager confirm or reject an incident from the stream screen and the journal; the state persists.
    - The journal filters by kind, state, camera and time range and exports CSV and JSON.
    - In the 20-stream demo, the scripted negative and positive moments in the demo clips each produce an event.

## Constraints

- **Target hardware:** laptop with an NVIDIA GPU of 4 GB VRAM and 16 GB RAM. Every model must load together within these limits, and the 20-stream demo runs on it. No GPU server or cloud at runtime.
- **CPU-only must work.** Docker GPU passthrough is not guaranteed on every Linux or Windows machine. The 20-stream targets apply to the GPU profile only.
- **Fully offline at runtime.** No calls to external services, CDNs or model hubs once the images are built. Model weights ship inside the image.
- **Emotions only.** No face recognition, no face embeddings, no stored face images or identities.
- **Linux and Windows via Docker only.** No native installs required from the user.
- **Fixed stack:** FastAPI backend and React frontend.
- **UI text is in Russian.**
- **UI follows the adopted EmoSense design system** (`ui.md`, `tokens.css`): dark flat theme, no rounded corners, shadows only on dropdowns and active elements, fonts and all assets bundled locally.
- **The local webcam and microphone are captured in the browser,** not in the container. Device passthrough into Docker on Windows is not viable. Network RTSP sources are ingested server-side.
- **Pretrained open models only, no fine-tuning in MVP.** Choices and alternatives are listed in `stack.md`.
- **Inference runs outside the FastAPI event loop.** All sources (webcam, RTSP, files) feed one inference worker through the "source → queue → worker → results" seam. Models load once, and RTSP decoding runs outside the worker.
- **Delivery: 1.5 months.**

## Non-goals

- Fine-tuning on Russian media content (planned after the MVP; datasets are listed in `research.md`).
- Person identification, re-identification across cameras, or tracking repeat visits.
- Delivering alerts by SMS, messenger or email, creating tickets in a control system, and configuring alert recipients or channels. The MVP stops at the event journal and incident states; CAP-9 documents the seam.
- Self-calibrating thresholds from incident feedback, data retention policies, and storing incident video clips.
- Running more than 20 streams in the MVP. CAP-9 covers 30–50 and 3000.
- Authentication, user accounts, multi-user access.
- ONVIF discovery, PTZ control, recording RTSP video, and RTSP authentication beyond credentials in the URL.
- Reaching 80% (or 95% / 98%) accuracy. Targets are measured and reported, lower results are acceptable.
- Exporting annotated video (burned-in overlay MP4) and storing face snapshots with events.
- Cloud deployment, Kubernetes, horizontal scaling infrastructure.
- Emotions beyond the 7 basic ones (the design package's contempt is dropped); valence/arousal; micro-expressions.

## Success signal

On the target laptop, with the network unplugged, a fresh `docker compose --profile gpu up` is followed by this demo:
1. The user opens `http://localhost`, enables the camera and sees their video.
2. Face boxes and emotions update with p95 latency under 1 s.
3. When the user speaks, a vocal emotion appears.
4. The user uploads a recorded clip, gets its analysis, and downloads CSV/JSON.
5. The operator starts the 20 emulated cameras of three types. The multi-camera screen shows all 20 analysed in parallel at ≥ 5 fps with per-stream latency under 1 s and the CPU/GPU load.
6. The journal screen shows negative and positive events from those cameras as they happen; the manager confirms one incident and marks another false, and exports the journal.

The regulation (CAP-8) and hardware plan (CAP-9) documents are in the repo, backed by the benchmark (CAP-6) and evaluation (CAP-7) reports.

## Assumptions

- **CAP-10 measured result, accepted on 2026-10-11.** Story 1.9 measured the target laptop (RTX 3050, 4 GB) with CUDA YuNet, detection on every 3rd frame for large profiles and shared-memory frames. The 20-stream mix runs at ~1–1.7 analysed fps per stream (1.08 in the 10-min run), and about 8 mixed streams reach ~5 fps. The bottleneck is detection under GPU contention, with VRAM at its limit. The 20-stream targets stay as goals; these numbers feed CAP-9. Details: `docs/benchmark-report.md`.

- "Offline" means runtime only. Images are built once with internet access, or transferred with `docker save` / `docker load`.
- One user and one webcam session at a time. RTSP streams run in parallel with it, but the 20-stream demo does not need the webcam.
- CSV and JSON export satisfies "export". Annotated MP4 is not needed.
- The contempt class of the 8-class FER model is dropped and the remaining probabilities are renormalised to the 7 basic emotions.
- Audio labels map as follows: angry → anger, sad → sadness, neutral → neutral, positive → happiness. Fear, disgust and surprise come from the face only.
- Emotion groups: negative = anger, disgust, fear, sadness; positive = happiness; neutral = neutral, surprise.
- "Tens of milliseconds" means per-frame inference time. Network buffering and the event rule's 2 s debounce add to the time until an event, on purpose.
- A client change needs no special handling beyond face tracks: a new person gets a new track and new events.
- Demo clips are openly licensed videos (e.g. Creative Commons) with faces and Russian speech. They live in `demo-videos/` with an `ATTRIBUTION.md` and are available offline.
- Camera geometry used for face-size estimates is listed in `camera-profiles.md` and is replaced by measured values when real cameras are known.

## Open questions

- Security cameras at 6–8 m give ~20–30 px faces at 2K with a 2.8 mm lens and look down steeply. Accept low quality there, or require a 6–8 mm lens or ROI streams?
- Production webcams at remote points: `getUserMedia` needs HTTPS outside localhost. Publish webcams through an RTSP agent, or serve the UI over HTTPS? (Post-MVP.)
- 152-FZ: does processing client video for emotions, without identification, need client notice or consent at service points? Legal check before production.
- Staff in frame: everyone in view, including the manager facing the webcam, is analysed as a client. Workplace emotion recognition of employees is prohibited by EU AI Act art. 5(1)(f) at EU points. Add per-camera staff exclusion zones before production?
