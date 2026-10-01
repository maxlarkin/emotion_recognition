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

## Seam for scaling (CAP-9 target, not built in MVP)

```mermaid
flowchart LR
  C1[IP cameras RTSP] --> MTX[MediaMTX]
  MTX --> D1[decode node 1: HW decode, 5 fps sample]
  MTX --> D2[decode node N]
  D1 --> Q[(Redis Streams / NATS)]
  D2 --> Q
  Q --> I1[inference worker: batched faces]
  Q --> I2[inference worker N]
  I1 --> R[(results bus)]
  I2 --> R
  R --> API2[FastAPI WS/SSE] --> UI[React dashboard]
  R --> PG[(PostgreSQL events)]
```

The MVP's `source → queue → worker → results` interfaces map one-to-one onto this picture. In the MVP the browser WebSocket is the only source adapter.
