# Architecture diagrams

## MVP (one laptop, Docker)

```mermaid
flowchart LR
  subgraph Browser
    CAM[getUserMedia camera+mic] --> VID[local video element]
    CAM --> ENC[JPEG 8 fps + PCM16 16 kHz]
    OVL[canvas overlay + charts]
  end
  subgraph frontend [frontend container: nginx]
    SPA[React SPA]
  end
  subgraph backend [backend container]
    WS[FastAPI /ws/live] --> SLOT[latest-frame slot / audio FIFO]
    API[FastAPI /api] --> JOBS[file job queue]
    SLOT --> W[inference worker process]
    JOBS --> W
    W --> DET[YuNet] --> TRK[IoU tracker] --> FER[EmotiEffLib]
    W --> VAD[Silero VAD] --> SER[GigaAM-Emo]
    FER --> FUS[smoothing + late fusion]
    SER --> FUS
    FUS --> RES[results]
    RES --> WS
    RES --> DB[(SQLite + uploads volume)]
  end
  ENC -- WebSocket --> WS
  WS -- frame_result JSON --> OVL
  SPA -.served to.-> Browser
```

## Multi-camera RTSP and events (CAP-10, CAP-11, built in MVP)

```mermaid
flowchart LR
  subgraph demo [profile demo-cams]
    EMU[camera-emulator: ffmpeg -c copy loops 20 pre-encoded clips] --> MTX[MediaMTX rtsp://mediamtx:8554/cam1..cam20]
  end
  IPC[real IP cameras] -.RTSP.-> SRC
  MTX --> SRC[RTSP source per stream: PyAV/NVDEC decode, 5 fps sample, scale to camera_type detection size, audio 16 kHz]
  SRC --> SLOTS[per-stream latest-frame slots + audio FIFOs]
  SLOTS --> W[shared inference worker: batched faces, VAD+SER per stream, fusion]
  W --> RES[results bus]
  RES --> WSS[FastAPI /ws/streams] --> GRID[React multi-camera grid]
  RES --> DB[(SQLite sessions kind rtsp)]
  RES --> EVT[event rule per track] --> DB
  EVT --> WSS
  EVT --> JRN[React event journal via /api/events]
  MON[psutil + NVML] --> WSS
```

## Target for 30–50 cameras and outlook to 3000 (CAP-9, not built in MVP)

```mermaid
flowchart LR
  subgraph site [service point, edge node for large rollouts]
    CAMS[webcam / sector / security cameras] --> EDGE[decode + detect + FER + event rule]
  end
  subgraph centre [on-prem centre]
    MTX[MediaMTX] --> D1[decode nodes: NVDEC, 5 fps sample]
    D1 --> Q[(NATS / Redis Streams)]
    Q --> I1[inference workers: batched faces, TensorRT/ORT]
    I1 --> EV[event rule]
    EV --> PG[(PostgreSQL events)]
    EV --> NOTIF[alert adapters: SMTP, messenger, SMS gateway]
    EV --> TICK[ticket system webhook]
    PG --> API2[FastAPI WS/SSE] --> UI[React journal + grid]
  end
  CAMS -.30–50 cameras: RTSP to centre.-> MTX
  EDGE -.3000 cameras: events only.-> Q
```

The MVP's `source → queue → worker → results` interfaces map one-to-one onto this picture. At 30–50 cameras video goes to the centre. At 3000, bandwidth rules that out, so sites run edge nodes and send only events.
