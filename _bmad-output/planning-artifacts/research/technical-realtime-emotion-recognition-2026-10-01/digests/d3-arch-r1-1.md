# D3 — Real-time video+audio inference pipeline architecture (round 1, digest 1)

Accessed: 2026-10-01 for every source. Budget used: 20 tool calls, about 12 sources.

## (a) Findings

| # | Claim | Source URL | Publisher | Pub date | Confidence | Class |
|---|---|---|---|---|---|---|
| F1 | aiortc's latest version is 1.15.0, released 2026-07-13. It requires Python >=3.10, supports Opus/PCMU/PCMA audio and VP8/H.264 video, and is maintained by the pyrtc org. The most recent commit to the repo is dated 2026-07-17. The GitHub Releases page is empty, so PyPI is the release channel. | https://pypi.org/project/aiortc/ ; GitHub API commits | PyPI / aiortc | 2026-07-13 | high | version |
| F2 | aiortc has a known pitfall: blocking inside `VideoStreamTrack.recv()` with OpenCV or detection work makes latency build up from 4-5 s to around 30 s. The `RemoteStreamTrack` queue grows when the consumer is slower than the incoming fps, and delay accumulates over time (issues #1192 and #642, discussion #1238). The usual fix is to run inference in a separate thread or process, consume only the latest frame, and drop the rest. | https://github.com/aiortc/aiortc/issues/1192 ; https://github.com/aiortc/aiortc/issues/642 ; https://github.com/aiortc/aiortc/discussions/1238 | GitHub issues (community) | various, 2019-2025 | medium (from search snippets; I did not read each thread) | pattern |
| F3 | fastrtc's latest PyPI release is 0.0.34 (2025-11-24). Its last commit on main is dated 2026-01-12. The repo has 4.6k stars, 66 open issues and 16 open PRs. It supports WebRTC and WebSocket, mounts on FastAPI, and has a YOLOv10 video demo. It is pre-1.0 and has had no release for about 10 months or commit for about 9 months, which looks stalled. | https://pypi.org/project/fastrtc/ ; https://github.com/gradio-app/fastrtc | PyPI / Gradio-HF | 2025-11-24 | high | version / ecosystem |
| F4 | onnxruntime-gpu 1.30.0 was released 2026-09-10. It requires Python >=3.11 and ships wheels for Linux x86-64 and Windows, including free-threaded builds for 3.13 and 3.14. | https://pypi.org/project/onnxruntime-gpu/ | PyPI / Microsoft | 2026-09-10 | high | version |
| F5 | MediaMTX v1.21.1 was published 2026-09-20. It ingests and republishes RTSP and serves WebRTC, HLS and RTMP, with on-demand sources. | https://github.com/bluenviron/mediamtx/releases (date confirmed via the GitHub API) | bluenviron | 2026-09-20 | high | version |
| F6 | Frigate's documented detector inference times: RTX 3050 runs YOLOv9-tiny-320 in 8 ms, small-320 in 10 ms and small-640 in 28 ms. GTX 1070 runs YOLOv9-small-320 in 16 ms. Intel Iris Xe (OpenVINO) runs MobileNetV2 in about 10 ms and YOLOv9-tiny-320 in 6 ms. Intel N100 runs MobileNetV2 in about 15 ms. Frigate calls video decoding "CPU-intensive" and says hardware acceleration is critical when running multiple cameras. | https://docs.frigate.video/frigate/hardware/ | Frigate (official docs) | living doc | high (community-measured numbers curated in the docs) | performance |
| F7 | Frigate recommends 5 fps for detection, saying "the default of 5 is correct for the vast majority of cameras". It recommends using a low-resolution substream (for example 1280x720 at 5 fps) for detection and the main stream only for recording. | https://docs.frigate.video/frigate/camera_setup/ | Frigate | living doc | high | pattern |
| F8 | OpenCV Zoo benchmarks on an Intel i7-12700K CPU: YuNet face detection runs at about 0.69 ms (160x120 input, FP32) and MobileFaceNet facial expression recognition at about 1.79 ms (112x112). On Jetson Orin Nano with CUDA FP16, YuNet runs at about 5.0 ms and the FER model at about 6.8 ms. The CPU path is already fast enough for many streams of face crops. | https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md | OpenCV | living doc | high (input sizes are small, so real 720p detection costs more) | performance |
| F9 | DeepStream 9.1's performance page reports only end-to-end model FPS on Jetson Thor and similar devices (for example RT-DETR at 208 FPS and PeopleNet at 349 FPS with rendering off). It gives no per-stream count for a 720p reference pipeline that I could extract. | https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Performance.html | NVIDIA | DS 9.1 | medium | performance |
| F10 | immers.cloud's cheapest T4 instance (teslat4-1.4.8.60: 4 vCPU, 8 GB RAM, 1 x T4 16 GB) costs $0.23/h, about $164 per month. 4 vCPU/16 GB costs $0.24/h. Prepay discounts of 15% for 30 days and 25% for 60 days apply to GPU instances. | https://en.immers.cloud/gpu/t4/ | immers.cloud | current page | high | pricing |
| F11 | gpuindex.ru (an aggregator) lists T4 hourly prices: Selectel 4 vCPU/32 GB at 19.18 RUB/h, immers.cloud 4/8 at 20.34 RUB/h, Beget 6/32 at 34.93 RUB/h, Yandex Cloud 4/16 at 78.70 RUB/h, Yandex DataSphere at 168.48 RUB/h. It also lists an RTX 4090 from 17.99 RUB/h. | https://gpuindex.ru/gpus/nvidia-t4 | gpuindex.ru (aggregator) | undated | medium (aggregator; the official Selectel and Yandex pages did not render prices) | pricing |
| F12 | immers.cloud's price list includes RTX 3080 10 GB, A2 16 GB, T4 16 GB, RTX 2080 Ti and others, and marks the T4 "EOL". Prices load through JavaScript, so I could not attribute them per GPU from the HTML. | https://immers.cloud/prices/ | immers.cloud | current | medium | pricing |

## (b) Recommended MVP pipeline and scaling options

**Labelling:** the architecture below is my synthesis from F1-F12. The stream-count arithmetic is an estimate and has not been measured.

### MVP (laptop with 4 GB VRAM, 16 GB RAM)

**Transport.** Recommended: WebSocket from React, carrying JPEG frames at 5-10 fps plus 16 kHz PCM16 audio chunks of about 0.5-1 s.

- Why: the evidence on the alternatives points this way. fastrtc is pre-1.0 and has had no release since Nov 2025 (F3). aiortc is maintained (F1), but it brings real latency-buildup pitfalls that must be engineered around (F2).
- Analysis does not need 30 fps (F7), so the main thing WebRTC buys, smooth 30 fps media, matters little. The browser shows its own local `<video>` and only overlays results.
- Backend OpenCV capture of the laptop webcam is fine for a demo, but it does not generalise to the browser/remote case.

**Backend.**

- FastAPI WebSocket handler. Keep a latest-frame-only slot (drop stale frames) and hand frames to one inference worker process through a `multiprocessing` queue of size 1. This is the "don't block recv" lesson from F2.
- Models: face detector (YuNet or SCRFD-class) and FER CNN, both on ONNX Runtime (F4) with the CUDA EP. These models fit easily in 4 GB, and CPU alone is viable (F8).
- Speech emotion recognition runs in a second worker on audio windows.
- Results go back as JSON on the same socket.

**Capacity estimate (unverified).**

- Per-frame cost on CPU is about 3-10 ms (F8). An RTX 3050-class GPU detector at 320 takes about 8-10 ms (F6).
- At 5 fps per stream, one 4 GB GPU should handle 10-20 streams on inference alone.
- The real bottleneck at 720p RTSP scale is likely H.264 decode on the CPU (F6). Use camera substreams (F7) and NVDEC/VAAPI decode.

### Scaling to 20+ IP cameras

| Option | Fit | Evidence |
|---|---|---|
| A. One box: MediaMTX (F5) in front of the cameras, 1-2 decode processes using PyAV/FFmpeg with hardware decode on 720p substreams, frames sampled to 5 fps (F7), and face crops from all streams batched into one ONNX/TensorRT worker. | Best budget fit for about 20 cameras | F5, F6, F7, F8 |
| B. Horizontal: N decode/infer workers behind Redis Streams or NATS, results to UI via WebSocket/SSE, events in Postgres. | Use only when one box saturates (likely 40-50+ cameras) | No evidence gathered this run (pattern only) |
| C. DeepStream/Triton | Highest throughput, but a heavy NVIDIA-only stack | No stream counts found (F9) |
| D. Rented GPU: T4 at about 19-20 RUB/h, roughly 14-15k RUB per month at 24/7 | Fallback if no local server | F10, F11 |

## (c) Leads worth chasing

- Frigate GitHub discussions with measured "N cameras on GTX 1650 / 3050" reports, covering decode plus detect CPU and GPU load.
- NVDEC concurrent-session limits on consumer GeForce cards. The driver caps NVENC, but NVDEC caps need checking.
- Throughput from ONNX Runtime TensorRT EP vs CUDA EP on a GTX 1650 for SCRFD/YuNet.
- Whether fastrtc has a successor or fork, given the activity gap.
- Official Selectel and Cloud.ru GPU price pages. These need a JS-rendering fetch.

## (d) Looked for and could not find

- A per-stream DeepStream count for 720p on a small GPU (F9 lacks it).
- Measured FPS on a GTX 1650 / 4 GB laptop GPU specifically.
- Official hourly GPU prices from Yandex Cloud, Selectel and Cloud.ru. The pages are JS-rendered or returned 404, so I fell back to the gpuindex aggregator.
- Current Triton, OpenVINO and TensorRT versions (not checked, out of budget).
- Practitioner write-ups comparing Redis Streams, NATS and Kafka at 20-50 cameras (not searched, out of budget).
- A head-to-head latency benchmark of WebSocket vs WebRTC for inference uplink.
