# D1 Digest: Real-time FER models and face detectors for a student MVP (round 1)

Researcher: subagent d1-fer-r1-1 · Accessed: 2026-10-01 · Budget used: ~12 sources, 20 tool calls

## (a) Findings

| # | Claim | Source URL | Publisher | Pub date | Accessed | Confidence | Class |
|---|---|---|---|---|---|---|---|
| 1 | EmotiEffLib (formerly HSEmotion; Savchenko, HSE / sb-ai-lab) gives these AffectNet accuracies: mobilenet_7 64.71% (AffectNet-7); enet_b0_8_best_afew 64.63% / 60.95% (AffectNet-7 / AffectNet-8); enet_b0_8_best_vgaf 64.57% / 61.32%; enet_b2_7 66.34% (AffectNet-7); enet_b2_8_best 66.51% / 63.125% | https://github.com/sb-ai-lab/EmotiEffLib | GitHub (official repo) | living README | 2026-10-01 | high (repo's own numbers; the papers behind them are peer-reviewed, IEEE TAC 2022 and ICML 2023) | accuracy |
| 2 | EmotiEffLib model sizes: mobilenet_7 14 MB, enet_b0 16 MB, enet_b2 30 MB. Latency: 16±5 / 59±26 / 191±18 ms, measured on a **Samsung Fold 3 (Snapdragon 888 CPU, Android)**. A laptop GPU should be much faster, but that is not yet measured | https://github.com/av-savchenko/face-emotion-recognition | GitHub (official) | living README | 2026-10-01 | high (numbers) / low (laptop extrapolation) | speed |
| 3 | EmotiEffLib supports PyTorch and ONNX backends and has Python and C++ APIs. Install with `pip install emotiefflib`. Models expect timm 0.9.* (old v0.1 needed timm==0.4.5) | https://pypi.org/project/emotiefflib/ | PyPI | 2025-09-10 | 2026-10-01 | high | version |
| 4 | Latest EmotiEffLib release is 1.1.1 (2025-09-10); the previous one was 1.0 (2025-02-24). There has been **no PyPI release for about 12 months**, so check repo commits | https://pypi.org/project/emotiefflib/ | PyPI | 2025-09-10 | 2026-10-01 | high | version |
| 5 | EmotiEffLib is Apache-2.0, "no limitation for both academic and commercial usage" | https://github.com/sb-ai-lab/EmotiEffLib | GitHub | living | 2026-10-01 | high | license |
| 6 | The HSEmotion team reports 1st place in Expression Recognition at ABAW-8 (CVPR 2025 workshop). Their ABAW-8 pipeline puts EmotiEffLib frame features into an MLP and uses the pretrained FER model to pick high-score frames (preprint, arXiv 2503.10399, 2025-03-13) | https://arxiv.org/abs/2503.10399 | arXiv (preprint, workshop paper) | 2025-03-13 | 2026-10-01 | medium | landscape |
| 7 | Savchenko's earlier ABAW work smooths frame-level EmotiNet predictions over time with box and Gaussian filters (seen in a search snippet; I did not read the full text) | https://arxiv.org/pdf/2507.02205 | arXiv (preprint) | 2025-07 | 2026-10-01 | low | landscape |
| 8 | OpenFace 3.0 (CMU/MIT) does landmarks, AUs, gaze and emotion in one multitask model. It was published at IEEE FG 2025 (peer-reviewed), arXiv 2506.02891 (2025-06-03) | https://arxiv.org/abs/2506.02891 | arXiv / IEEE FG 2025 | 2025-06-03 | 2026-10-01 | high | landscape |
| 9 | OpenFace 3.0 AffectNet-8 accuracy: 0.59 single-task, 0.60 MTL with uncertainty. In the same table DDAMFN scores 0.65 and EfficientFace 0.60 | https://arxiv.org/html/2506.02891v1 | arXiv | 2025-06 | 2026-10-01 | high | accuracy |
| 10 | Compute table in the OpenFace 3.0 paper (Threadripper 1920X CPU), in params / GFLOPs / ms per frame on CPU: OpenFace 3.0 29.4M / 4.2 / 38 ms; LibreFace 22.5M / 3.7 / 33 ms; Py-Feat 49.3M / 8.2 / 82 ms; OpenFace 2.0 44.8M / 7.4 / 75 ms | https://arxiv.org/html/2506.02891v1 | arXiv | 2025-06 | 2026-10-01 | medium (authors benchmarking competitors) | speed |
| 11 | OpenFace 3.0 uses RetinaFace MobileNet-0.25 for detection and an EfficientNet backbone pretrained on VGGFace2. Code is "freely available for research purposes" | https://arxiv.org/html/2506.02891v1 | arXiv | 2025-06 | 2026-10-01 | medium | license |
| 12 | DDAMFN (MobileFaceNet-based, MDPI *Electronics* 2023, peer-reviewed). The repo provides RAF-DB and AffectNet checkpoints in DDAMFN++/checkpoints_ver2.0, which the authors themselves call "suboptimal". The README shows no license | https://github.com/SainingZhang/DDAMFN | GitHub | 2023+ | 2026-10-01 | medium | license |
| 13 | trpakov/vit-face-expression: ViT-base (85.8M params) fine-tuned on FER2013, 71.16% test accuracy, Apache-2.0, about 16.7k downloads/month. It is a single-dataset model card, not a paper | https://huggingface.co/trpakov/vit-face-expression | Hugging Face | undated | 2026-10-01 | medium | accuracy |
| 14 | EMO-AffectNetModel (Ryumina et al., SPIIRAS, *Neurocomputing* 2022, peer-reviewed): ResNet50 (VGGFace2) static model scores 66.4% on AffectNet val (66.5% with label smoothing). The static+LSTM dynamic model reports UAR of RAVDESS 65.8, CREMA-D 60.6, SAVEE 76.1, **RAMAS (Russian) 44.3**, IEMOCAP 25.1, Aff-Wild2 51.6. Repo is MIT, but weights are "scientific use only" | https://github.com/ElenaRyumina/EMO-AffectNetModel | GitHub | 2022 | 2026-10-01 | high | accuracy |
| 15 | RAMAS (Neurodata Lab) is a Russian acted multimodal corpus: about 7 h of close-up face video, 10 native-Russian actors aged 18-28, 6 basic emotions plus domination/submission, at least 5 annotators per scenario | https://peerj.com/preprints/26688/ | PeerJ Preprints | 2018 | 2026-10-01 | medium (old, preprint) | landscape |
| 16 | YuNet (OpenCV Zoo) scores AP 0.8844 / 0.8656 / 0.7503 (easy/medium/hard) on WIDER Face val. It detects faces of about 10x10 to 300x300 px and is MIT-licensed. There are int8 and block-quantized variants. A **new `face_detection_yunet_2026may.onnx`** (dynamic input) is required for OpenCV 5.x with the ONNX Runtime engine; the 2023mar version is for OpenCV 4.x. The paper is in *Machine Intelligence Research* 20(5), 2023 | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | GitHub (OpenCV) | 2026-05 | 2026-10-01 | high | version |
| 17 | MediaPipe Face Detector offers BlazeFace short-range (for selfie/webcam), full-range, and Sparse full-range (about 60% smaller). Input is 128x128, fp16. Latency on Pixel 6 is 2.94 ms on CPU and 7.41 ms on GPU (short-range). Python is supported, code is Apache-2.0. Page updated 2026-08-17 | https://developers.google.com/edge/mediapipe/solutions/vision/face_detector | Google | 2026-08-17 | 2026-10-01 | high | speed |
| 18 | Recent lightweight FER papers keep posting about 92% on RAF-DB (e.g. FERMam 92.13%, 26.99M params, 5.97 GFLOPs), but these are journal papers with no confirmed pip-ready weights | https://pmc.ncbi.nlm.nih.gov/articles/PMC13129035/ | PMC (journal) | 2025-26 | 2026-10-01 | low (snippet only) | landscape |

## (b) Finalist comparison

| Finalist | AffectNet-7 | AffectNet-8 | RAF-DB | FER2013 | Params / size | Speed (source hardware) | ONNX | Install | License | Activity |
|---|---|---|---|---|---|---|---|---|---|---|
| **EmotiEffLib enet_b0_8_best_vgaf / _afew** | 64.6% | 61.0–61.3% | not in README | – | 16 MB (~4M params, unverified) | 59 ms (phone CPU) | yes | `pip install emotiefflib` | Apache-2.0 | v1.1.1 2025-09; ABAW-8 winner team |
| **EmotiEffLib enet_b2_8_best** | 66.5% | 63.1% | not in README | – | 30 MB | 191 ms (phone CPU) | yes | same | Apache-2.0 | same |
| **EmotiEffLib mobilenet_7** (CPU fallback) | 64.7% | – | – | – | 14 MB | 16 ms (phone CPU) | yes | same (TF .h5 origin) | Apache-2.0 | same |
| **OpenFace 3.0** (all-in-one) | – | 59–60% | not retrieved | – | 29.4M | 38 ms/frame (desktop CPU, whole pipeline) | not verified | "single line" pip (claim) | research-only | FG 2025 |
| **EMO-AffectNetModel** (ResNet50 + LSTM) | 66.4% (val; class count not stated) | – | – | – | ResNet50 (~25M) | not reported | TorchScript | manual, Google Drive weights | MIT code, weights scientific-use | 2022, low activity |
| DDAMFN++ (backup, best accuracy) | not retrieved (MDPI returned 403) | 0.65 (per OpenFace 3.0 table, 8-class) | not retrieved | – | MobileFaceNet-based (small) | not reported | no | manual | none stated | checkpoints called "suboptimal" by authors |
| trpakov ViT (not recommended) | – | – | – | 71.16% | 85.8M | heavy for 4 GB | via HF | `transformers` | Apache-2.0 | HF card |

Detector shortlist: **YuNet** (OpenCV built-in `FaceDetectorYN`, MIT, ~ms-level) or **MediaPipe BlazeFace short-range** (webcam-tuned, 2.94 ms on Pixel 6 CPU). Use RetinaFace-mobilenet0.25 if you need alignment landmarks robust to pose (OpenFace 3.0 uses it).

**Preliminary recommendation (evidence-limited):** EmotiEffLib `enet_b0_8_best_vgaf` (ONNX) + YuNet or MediaPipe, with EMA/Gaussian temporal smoothing of the probabilities. Upgrade to `enet_b2_8` if the GPU budget allows. Its VGGFace2-pretrained timm EfficientNet is easy to fine-tune later on Russian media.

## (c) Leads worth chasing

1. Get DDAMFN / DDAMFN++ RAF-DB and AffectNet numbers from the paper (MDPI blocked; try arXiv or Semantic Scholar).
2. Find EmotiEffLib RAF-DB numbers: check Savchenko's IEEE TAC 2022 / ICML 2023 papers. Also check GitHub commit dates since 2025-09.
3. HSEmotion at ABAW-11 (arXiv 2607.12774, July 2026) and the ABAW-10 expression solution (arXiv 2603.08034): what is new in 2026, and is there a newer EmotiEffLib model?
4. AffectFuse (arXiv 2607.16546) temporal modelling: is it light enough for an MVP?
5. YuNet and SCRFD-500M CPU ms on x86: check the opencv_zoo benchmark table and the insightface SCRFD README.
6. LibreFace repo: check license, pip, and FER accuracy (22.5M params, 33 ms CPU per the OpenFace 3.0 table).
7. POSTER++ weights and speed. It is likely too heavy, but verify.
8. Russian-specific: Ryumina/SPIIRAS later works (e.g. OCEAN-AI, MASAI), and whether any Russian TV/film FER corpus exists.

## (d) Looked for but not found

- **RAF-DB and FER2013 accuracy for EmotiEffLib**: not in the README.
- **DDAMFN exact numbers**: MDPI returned 403.
- **Laptop CPU/GPU FPS for any finalist**: only phone CPU (EmotiEffLib), a desktop Threadripper (OpenFace 3.0) and Pixel 6 (MediaPipe) figures exist. No 4 GB laptop GPU benchmark was found.
- **Py-Feat, py-fer, DeepFace emotion, POSTER++, LibreFace accuracy**: not retrieved this round, apart from the OpenFace 3.0 compute table.
- **Face tracking to skip detection on some frames**: no source retrieved. My belief, unverified: an IOU/centroid tracker or OpenCV KCF/CSRT between detector calls every N frames is common practice.
- **FER models trained on Russian/Slavic faces**: none found. The only Russian evidence is the RAMAS corpus (acted, 2018) and EMO-AffectNet's cross-corpus RAMAS UAR of 44.3%, which suggests a domain gap.
