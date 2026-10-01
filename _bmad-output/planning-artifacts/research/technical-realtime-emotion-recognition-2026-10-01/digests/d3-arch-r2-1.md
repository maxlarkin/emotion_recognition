# Digest d3-arch-r2-1: Small-hardware capacity for multi-camera face analytics (round 2)

Accessed: 2026-10-01 (all sources). Budget: ~19 tool calls, 9 sources read in full or as excerpts.
Hardware target: laptop with a 4 GB NVIDIA GPU (GTX 1650 or RTX 3050 laptop), 16 GB RAM; 20 or more camera streams.

## (a) Findings

| # | Claim | Source URL | Publisher | Pub date | Confidence | Class |
|---|---|---|---|---|---|---|
| F1 | NVDEC H.264 1080p (4:2:0) decode throughput per architecture: Pascal 694 fps @1544 MHz, **Turing 771 fps @1860 MHz**, **Ampere 748 fps @1665 MHz**, Ada 903, Blackwell 2172. | https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html | NVIDIA (Video Codec SDK 13.0 docs) | SDK 13.0, undated page | High (official), but measured at high clocks, not on laptop parts | performance |
| F2 | "All GeForce products consist of a single NVDEC." There is no fixed decode-session limit: an application "can decode multiple videos simultaneously", "subject to the hardware performance limit and available memory". | same as F1 | NVIDIA | same | High | performance / pattern |
| F3 | The NVENC (encode) session limit on GeForce is separate. It was raised from 2 to 3 (2020), then 5 (2023), then 8 per system on non-qualified GPUs. One secondary source says 12 as of 2025 (not verified against the primary). NVDEC is not limited this way (F2). | https://videocardz.com/newz/nvdia-geforce-gpus-now-support-up-to-8-concurrent-nvenc-encoding-sessions ; https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html (search snippet only) | VideoCardz / NVIDIA | 2023 / undated | Medium (8 limit), Low (12) | version |
| F4 | NVIDIA moderator on a GTX 1650 4 GB running DeepStream 6.2: "The GTX 1650 is consumer card. It is less than half of T4's decoding capability. For 3840x2160@30fps h264 videos, the limitation may be less than 4.2 streams." The user reached at most 8 sources of 4K with decode only (18 fps), and adding more caused GPU memory problems because decoder buffers sit in VRAM. | https://forums.developer.nvidia.com/t/limited-number-of-sources-due-to-decoding/270897 | NVIDIA Developer Forums | 2023-10-27 to 2023-11-06 | Medium (vendor staff, informal) | performance |
| F5 | DeepStream reference pipeline: 1080p30 input, ResNet18 TrafficCamNet detector at 960x544 INT8, IOU tracker, 2 ResNet18 secondary classifiers, batch 32, no rendering. H.264 1080p30 streams per platform: **T4 31** (GPU 61%), **A2 31**, A10 43, L4 68, A4000 24, **Jetson Orin Nano 8**, Orin NX 13, AGX Orin 15. H.265: T4 45, Orin Nano 13. | https://archive.docs.nvidia.com/metropolis/deepstream/7.0/dev-guide/text/DS_Performance.html | NVIDIA (archived DeepStream docs; page is titled 6.4 but sits at the 7.0 URL) | archive page; NVIDIA's own numbers | High for those platforms. No GeForce or laptop entry. | performance |
| F6 | The search snippet says DeepStream 6.3 gave 43 H.264 and 75 H.265 streams on T4. The figures differ between versions, so treat stream counts as about ±30% depending on the release. | https://archive.docs.nvidia.com/metropolis/deepstream/6.3/dev-guide/text/DS_Performance.html (search snippet, not fetched) | NVIDIA | archive | Low to Medium | performance |
| F7 | SCRFD on RTX 2080 SUPER, TensorRT fp16, batch 1, 640x640. Inference / total detection time: scrfd_500m 0.8 / 1.9 ms; **scrfd_2.5g 1.1 / 2.2 ms**; scrfd_10g 2.0 / 3.3 ms; retinaface_r50 8.4 / 12.3 ms. Face-embedding CNN w600k_mbf (MobileFaceNet, 112x112): 0.7 ms at batch 1, **9.9 ms at batch 64** (about 0.15 ms per face). ResNet50 recognizer: 1.9 ms at batch 1, 33.2 ms at batch 64. | https://github.com/SthPhoenix/InsightFace-REST | SthPhoenix (GitHub README) | README, undated; author-stated hardware | Medium (one repo, hardware stated, not a 4 GB GPU) | performance |
| F8 | SCRFD 640: 11.0 ms per image with ONNX Runtime GPU and 9.1 ms with TensorRT 8 (1,000 runs). The GPU is **not stated**. | https://github.com/yangjian1218/scrfd_onnx_tensorrt/blob/master/readme.md | yangjian1218 (GitHub) | undated | Low (no hardware stated) | performance |
| F9 | Frigate ONNX detector on **RTX 3050**: YOLOv9-tiny 320 **8 ms**, YOLOv9-small 320 10 ms, YOLOv9-small 640 28 ms, RF-DETR 320 10 ms, YOLO-NAS 320 about 12 ms. RTX 3070: 6, 8, 25 ms. Frigate also says video decode is one of the most CPU-heavy tasks, and that hardware decode lets the same hardware handle 2-3x more cameras (that second figure is from a search snippet). | https://docs.frigate.video/frigate/hardware/ ; https://docs.frigate.video/configuration/hardware_acceleration_video/ | Frigate NVR docs | current docs | Medium-High (community-measured; RTX 3050 variant (desktop or laptop) not stated) | performance |
| F10 | wav2vec2-base-960h (95M parameters), PyTorch on CPU only, 8-core/12-thread Intel CPU, 16 GB RAM, Windows 11. Mean latency **0.376 s per utterance** (mean utterance 4.58 s), RTF **0.083**. | https://github.com/Siddhessh511/asr-model-benchmark | Siddhessh511 (GitHub) | undated (recent; the PyTorch version string "2.14.0+cpu" was reported but not verified) | Medium-Low (one hobby repo, exact CPU not named) | performance |
| F11 | wav2vec2 on edge devices: RTF about 1.4-1.5 on Raspberry Pi (about 1.0 when quantized) and about 0.22-0.23 on the Jetson Nano GPU, i.e. 3-5x better than the Pi. | https://arxiv.org/pdf/2202.05993 | Gondi (Meta), arXiv | 2022-02 | Medium (search-result abstract only, not fully read) | performance |
| F12 | Unattributed claim: "Wav2Vec2 on GPU ~3-5 ms per 1 s of audio" (batched throughput, not streaming latency). | search-result synthesis, probably https://arxiv.org/pdf/2503.23439 (not opened) | n/a | 2025 | **Low**, needs a primary source | performance |
| F13 | FP16 rule of thumb: about 2 bytes per parameter for weights, so about 2 GB of VRAM per 1B parameters, plus activations and runtime overhead. | https://www.runpod.io/articles/guides/gpu-memory-sizing-guide-for-llm-inference | RunPod (vendor guide) | undated | Medium (general rule, not speech-specific) | pattern |
| F14 | CPU software decode: transcoding 1080p30 H.264 on an i5-5575R used about 30% CPU. Forum reports put one stream at 2-5% CPU, jumping to 25-30% at three streams. These are anecdotal and under-specified. | Intel Community / NVIDIA forum (search snippets: https://community.intel.com/t5/Media-Intel-Video-Processing/QSV-CPU-usage/m-p/1079246 , https://forums.developer.nvidia.com/t/opencv-h264-decoder-high-cpu-usage/154216/2) | forums | various, old | **Low** | performance |

## (b) Capacity estimate (arithmetic shown; assumptions marked)

Assumptions, not evidenced this run:
- A1: TU117 (GTX 1650) and GA107 (RTX 3050 laptop) carry Turing-class and Ampere-class NVDEC respectively, each with one engine (F2). NVDEC throughput on a laptop is assumed to be about 0.7-1.0x the F1 table because clocks are lower.
- A2: A GTX 1650 laptop has roughly 1/3 to 1/4 of the inference throughput of an RTX 2080 SUPER, and an RTX 3050 laptop roughly 1/2 to 1/3. These are training-data beliefs, used only as scale factors.
- A3: With H.264, every frame must be decoded at the camera's native fps, even if analytics run at a lower fps (unless decoding keyframes only).

### B1. GPU decode (NVDEC), decode only

| Case | Arithmetic | Result |
|---|---|---|
| Turing NVDEC ceiling, 1080p30 | 771 fps ÷ 30 = 25.7; × 0.7-1.0 (A1) | **about 18-25 streams** |
| Cross-check (F4): GTX 1650 below 4.2 × 4K30 | 4K = 4 × 1080p pixels, so 4.2 × 4 = 16.8 → **below about 17 × 1080p30** | consistent, but takes the lower end |
| Cross-check (F4 + F5): below half of T4 | T4 pipeline gives 31 H.264 1080p30 streams (inference-bound, so decode-only is higher); half = 15.5 | **about 15** |
| **Working figure for a GTX 1650, 1080p30** | min(17, 18-25) | **about 15 streams (conservative); 20 is unlikely** |
| 720p30 (0.44 × the pixels of 1080p) | 15 ÷ 0.444 ≈ 34 | **about 30 or more 720p30 streams** (decode only; VRAM is the next limit, F4) |
| 720p at 15 fps camera substream | 34 × 2 = 68 | well above 20; decode stops being the bottleneck |
| Ampere (RTX 3050 laptop) | 748 ÷ 30 = 24.9; × 0.7-1.0 | **about 17-25 × 1080p30** (no forum cross-check found) |

**Two-source status:** the GTX 1650 1080p30 figure rests on F1 and F4 (two independent NVIDIA sources that agree on about 15-25). The RTX 3050 figure rests on F1 alone.

### B2. GPU inference budget per second (20 streams, analytics at 5 fps)

| Stage | Arithmetic | GPU time per wall-clock second |
|---|---|---|
| Face detection SCRFD-2.5g, 640, batch 1 | 2080S total 2.2 ms (F7) × 3-4 (A2) ≈ 7-9 ms; 20 × 5 = 100 calls → 0.7-0.9 s | **0.7-0.9 s (GTX 1650)**; about 0.45-0.65 s (3050) |
| Same detector, analytics at 10 fps | 200 calls × 7-9 ms | 1.4-1.8 s, **does not fit** on a GTX 1650 at batch 1 |
| Emotion CNN (MobileFaceNet-class proxy, 112x112), 3 faces per frame | 2080S batch 64: 0.155 ms per face (F7) × 4 ≈ 0.6 ms; 100 frames × 3 faces = 300 faces → 0.19 s | about 0.2 s |
| Detector proxy on RTX 3050 (F9, a heavier YOLOv9-tiny-320 at 8 ms) | 100 × 8 ms = 0.8 s | an upper bound if the detector is YOLO-class |

Total at 20 streams and 5 fps on a GTX 1650 comes to about 0.9-1.1 s of GPU time per second, which is **at or over 100% before audio**. On an RTX 3050 laptop it is about 0.65-0.85 s, which fits only marginally.

### B3. Speech emotion model (if one audio channel per stream, 3 s windows, no overlap)

| Case | Arithmetic | Result |
|---|---|---|
| wav2vec2-base on CPU (F10, RTF 0.083 on an 8-core CPU) | 20 streams × 3 s audio = 60 s of audio per 3 s; × 0.083 = 5.0 s of compute per 3 s wall-clock → load 1.66 | **does not fit** on the CPU for 20 streams; about 12 streams max (1 ÷ 0.083) using all cores |
| Large model (about 300M parameters) on CPU | about 3x base (training-data belief) → about 4 streams | not viable for 20 |
| fp16 VRAM, weights only (F13) | base 95M × 2 B = 0.19 GB; large 317M × 2 B = 0.63 GB | fits in 4 GB, but competes with decode buffers (F4) |
| GPU SER (F11: Jetson Nano RTF about 0.22, the weakest GPU) | a GTX 1650 is far stronger than a Nano; for real-time, 20 × RTF must stay below the GPU time left after B2, which is about 0 on a GTX 1650 | SER on GPU at 20 streams is **not supportable** on a 4 GB laptop alongside video |

### B4. Sizing conclusion (estimate)

| Configuration | Supported streams (estimate) | Binding constraint |
|---|---|---|
| GTX 1650 4 GB laptop, video only, 1080p30 in, 5 fps analytics | **about 10-15** | NVDEC (B1) and GPU inference (B2) |
| Same, with 720p/15 fps substreams | **about 12-16** | GPU inference (B2) |
| RTX 3050 4 GB laptop, video only, 720p substreams | **about 16-20** (marginal) | GPU inference and VRAM |
| Either laptop plus SER for all streams | **about 8-12 video + SER** | CPU or GPU SER load (B3) |
| **20 or more streams with SER** | **2 laptop-class nodes (about 10-12 each)**, or 1 server-class GPU (T4/A2/L4 tier runs 31-68 1080p30 streams in a heavier DeepStream pipeline, F5) | — |

Every figure must be validated on the target machine with `nvidia-smi dmon` (dec / sm / mem columns) before committing.

## (c) Looked for but not found

- Direct GTX 1650 / RTX 3050 laptop benchmarks for SCRFD, RetinaFace, YuNet, ResNet-18, MobileNetV2 or EfficientNet-B0 with ONNX Runtime CUDA or TensorRT. None had both the hardware and the numbers stated, so B2 relies on scaling from a 2080S (assumption A2).
- An NVDEC table from NVIDIA for laptop SKUs, or a measured count of 1080p30 sessions on a GTX 1650 or RTX 3050 laptop. Only the 4K moderator statement (F4) exists.
- Confirmation that the GTX 1650's NVDEC generation matches the Turing table (A1).
- A DeepStream performance entry for any GeForce or laptop GPU.
- Credible measured CPU cores per 1080p or 720p H.264 software decode with FFmpeg. Only anecdotes (F14); the Medium decode benchmark returned 403.
- Measured latency and VRAM for wav2vec2, WavLM or HuBERT (base or large) on a small GPU for 2-3 s windows. F12 is unattributed. No fp16 VRAM measurement was found; the F13 rule of thumb is used instead.
- WavLM / HuBERT CPU latency numbers. Only wav2vec2-base is covered (F10).
