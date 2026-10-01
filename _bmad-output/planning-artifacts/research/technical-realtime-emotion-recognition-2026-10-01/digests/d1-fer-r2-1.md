# Digest d1-fer-r2-1: FER models, round 2 (targeted gaps)

Accessed for all sources: 2026-10-01. Budget used: 14 tool calls, 9 sources read.
Classes: P = primary (official repo or original paper), S = secondary. PR = peer-reviewed, PP = preprint.

## (a) Findings

| # | Claim | Source URL | Publisher | Pub date | Conf. | Class |
|---|---|---|---|---|---|---|
| 1 | EmotiEffLib README gives AffectNet accuracy only (no RAF-DB column): enet_b0_8_best_afew 60.95 (8-cls) / 64.63 (7-cls); enet_b0_8_best_vgaf 61.32 / 64.57; enet_b0_8_va_mtl 61.93 / 64.94; enet_b2_7 – / 66.34; enet_b2_8 63.03 / 66.29; enet_b2_8_best 63.125 / 66.51; mobilenet_7.h5 – / 64.71 | https://github.com/sb-ai-lab/EmotiEffLib | sb-ai-lab (GitHub) | README, last edited 2026-06-23 | High | P |
| 2 | EmotiEffLib license is Apache-2.0. The README shows PyPI version v1.1.1. The README mentions wins at ABAW-8 and ABAW-10 | same | sb-ai-lab | same | High (license); Medium (version string came from a badge) | P |
| 3 | EmotiEffLib commits since 2025-09: 2025-09-08 "Fixed issue with mobilefacenet"; 2025-09-10 "Bump version and move setup.py to pyproject.toml"; 2025-09-26 ONNX/Torch header fixes and xtensor bump (C++ side); 2026-03-16 "Uploaded script for EXPR, VA and AU challenges at ABAW-10"; 2026-05-17 test update; 2026-05-27 and 2026-06-23 README updates. None of the commit messages announce a new FER model | https://github.com/sb-ai-lab/EmotiEffLib/commits/main | sb-ai-lab | 2025-09 to 2026-06 | High | P |
| 4 | arXiv 2607.12774 (Bakin & Savchenko, submitted 2026-07-14, PP) is "HSEmotion Team at the 11th ABAW Challenge". For the multi-task track it uses frozen lightweight extractors MT-EmotiDDAMFN and MT-EmotiEffNet-B0 with separate heads, plus post-processing: temporal Gaussian smoothing, per-class expression bias, AffectNet blending, per-AU threshold tuning and weighted backbone fusion. The abstract announces no new model release and gives no RAF-DB or AffectNet numbers | https://arxiv.org/abs/2607.12774 | arXiv | 2026-07-14 | Medium (abstract only) | P, PP |
| 5 | Correction: arXiv 2603.08034 is NOT an HSEmotion paper. It is Jun Yu et al., "Solution to the 10th ABAW Expression Recognition Challenge" (2026-03-09, PP): a multimodal audio-visual Transformer with 60.79% accuracy and F1 0.5029 on the Aff-Wild2 validation set. It has no lightweight image model and is not relevant to the MVP | https://arxiv.org/abs/2603.08034 | arXiv | 2026-03-09 | High | P, PP |
| 6 | DDAMFN (Electronics 12(17):3595, MDPI, 2023, PR) reports 67.03% on AffectNet-7, 64.25% on AffectNet-8 and 91.35% on RAF-DB. The MDPI page returned 403. These numbers come from a search-result summary of the MDPI/ResearchGate listing, not from the paper text | https://www.mdpi.com/2079-9292/12/17/3595 | MDPI Electronics | 2023 | Medium | S (paper snippet) |
| 7 | LibreFace license is the "USC research license" (non-permissive research license, not OSI). Install: `pip install --upgrade libreface`. Runs on CPU or CUDA (`--device cuda:0`/`cpu`) and uses MediaPipe for face and landmark detection. The README shows FER results on FACES and CAFE, not on RAF-DB or AffectNet | https://github.com/ihp-lab/LibreFace | ihp-lab (USC) | not dated | High | P |
| 8 | LibreFace paper (WACV 2024, PR; arXiv 2308.10713): AU intensity PCC 0.63 on DISFA, "7% higher than OpenFace 2.0" and "runs two times faster". The abstract claims competitive FER on AffectNet, FFHQ and RAF-DB but gives no numbers | https://arxiv.org/abs/2308.10713 | WACV / arXiv | 2023-08-18 | Medium | P, PR |
| 9 | YuNet on the opencv_zoo benchmark: Intel i7-12700K CPU, input 160x120, FP32 mean 0.69 ms, INT8 mean 0.79 ms. Jetson Orin Nano CUDA FP32 mean 5.23 ms, slower than its own CPU at 2.59 ms. The 160x120 input is small, so latency at 640x480 will be higher | https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md | OpenCV | not dated | High | P |
| 10 | SCRFD README (VGA resolution): SCRFD_500M WIDER easy/med/hard 90.57/88.12/68.51, 0.57M params, 3.6 ms; SCRFD_2.5G 93.78/92.16/77.87, 0.67M params, 4.2 ms; SCRFD_10G 95.16/93.87/83.05, 3.86M params, 4.9 ms. A separate CPU figure exists for 500M: about 28.3 ms single-thread PyTorch on a Ryzen 9 3950X at 640x480. The extract did not confirm the hardware behind the 3.6/4.2/4.9 ms column. I believe it is a GPU, but that is unverified | https://github.com/deepinsight/insightface/tree/master/detection/scrfd | InsightFace | not dated | Medium | P |
| 11 | Savchenko SISY 2021 (arXiv 2103.17107, PR): MobileNet, EfficientNet and RexNet multi-task models trained on cropped faces without margins. Near-SOTA on AffectNet, and 4.5% over prior single models on AFEW/VGAF. The abstract gives no RAF-DB number | https://arxiv.org/abs/2103.17107 | IEEE SISY / arXiv | 2021-10-04 (v final) | Medium | P, PR |
| 12 | Frame-skip practice: run detection every Nth frame and let a tracker fill the gaps. One project reports that detecting every 3rd frame is 2.25x faster for a 4.7-point coverage loss (anonymization context). Risks: missing faces that newly enter the frame, and degradation at low FPS or with fast motion. The SORT repo has an issue on tracking with frame skips | https://github.com/ArgusObscura/Lethe/pull/11 ; https://github.com/abewley/sort/issues/22 ; https://arxiv.org/pdf/2309.02666 | GitHub PR / issue; arXiv | various | Low-Medium (search summary; pages not read) | S |

## (b) Updated finalist table

| Model | AffectNet-7 | AffectNet-8 | RAF-DB | License | Install | Notes |
|---|---|---|---|---|---|---|
| EmotiEffLib enet_b0_8_va_mtl | 64.94 | 61.93 | not reported in README | Apache-2.0 | pip (v1.1.1) | Lightest. Repo maintained (last commit 2026-06); no new FER weights since 2025-09 |
| EmotiEffLib enet_b2_8(_best) | 66.29 (66.51) | 63.03 (63.125) | not reported in README | Apache-2.0 | pip | About 1.5 to 2 points better than B0 on AffectNet |
| DDAMFN | 67.03 | 64.25 | 91.35 | not checked this run | GitHub code (not checked) | Highest accuracy in this table; the paper is peer-reviewed (MDPI 2023). HSEmotion's ABAW-11 paper uses an "MT-EmotiDDAMFN" extractor, release status unverified |
| LibreFace | no numbers found | no numbers found | no numbers found | USC research license (restrictive) | `pip install libreface` | Built-in MediaPipe detection. The restrictive license is a demerit |
| Detector: YuNet | — | — | — | MIT (unverified this run) | OpenCV built-in | 0.69 ms on i7-12700K CPU at 160x120 |
| Detector: SCRFD_500M / 2.5G | — | — | — | InsightFace (not checked) | ONNX | 3.6 / 4.2 ms at VGA (hardware unconfirmed); WIDER-hard 68.5 / 77.9 |

Lead unchanged: EmotiEffLib enet_b0_8_va_mtl or enet_b2_8 with YuNet. DDAMFN is the main accuracy alternative if its license and weights are acceptable.

## (c) Looked for, not found

- An EmotiEffLib/HSEmotion RAF-DB accuracy figure from the README, the SISY 2021 abstract or search results. The IEEE TAC 2022 paper was not fetched (budget).
- New lightweight pretrained FER weights from ABAW-10 or ABAW-11. None announced. The MT-EmotiDDAMFN weight release is unverified.
- DDAMFN++. No source found. The DDAMFN license and repo were not checked.
- LibreFace RAF-DB/AffectNet numbers and FPS (the abstract has none; the full paper was not read).
- Measured ONNX Runtime latency for EfficientNet-B0 or B2 at 224x224 on a laptop CPU or GPU. Not found; benchmark locally.
- The hardware behind the SCRFD Infer(ms) column, and YuNet latency at 640x480 or on an x86 CUDA GPU.
- A primary source on ByteTrack/SORT applied specifically to faces. Only secondary or anecdotal sources found.
