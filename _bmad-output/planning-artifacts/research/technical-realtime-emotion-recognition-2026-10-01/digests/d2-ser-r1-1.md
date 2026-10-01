# D2 — Russian SER + audio-visual fusion (round 1, digest 1)

Accessed: 2026-10-01 for all sources. Tool calls: ~19. Sources read: 13.
Scope: open, locally-run SER for Russian, no fine-tuning, 4 GB VRAM shared with face model; simple A/V fusion.

## (a) Findings

| # | Claim | Source | Publisher | Pub date | Conf. | Class |
|---|---|---|---|---|---|---|
| 1 | Aniemore/wavlm-emotion-russian-resd: 7 labels (anger, disgust, enthusiasm, fear, happiness, neutral, sadness); RESD test WA 0.8071, macro-F1 0.8026; 317M params, ~1207 MiB fp32; MIT; ~48.7k downloads/last month; quantized INT8/FP8/INT4 variants listed | https://huggingface.co/Aniemore/wavlm-emotion-russian-resd | Aniemore (HF model card) | card undated in fetch (current as of access) | high | accuracy/license/version |
| 2 | Same model on Dusha podcast test: WA 0.104, macro-F1 0.112; on CAMEO test macro-F1 0.210. Card itself states training data is acted, class-balanced and model over-predicts emotional classes on spontaneous Russian audio (neutral ~93% in real-world material) | same as #1 | Aniemore | — | high | accuracy (acted vs in-the-wild) |
| 3 | Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition: same 7 labels; RESD UA 0.718 / WA 0.721 / macro-F1 0.721; 316M params; MIT; ~5.7k downloads/month; Dusha podcast macro-F1 0.095, CAMEO 0.225 | https://huggingface.co/Aniemore/wav2vec2-xlsr-53-russian-emotion-recognition | Aniemore | — | high | accuracy/license |
| 4 | Aniemore README lists voice models wav2vec2-xlsr-53 (73%), wav2vec2-resd (75%), wavlm-resd (82%), hubert-resd (75%), unispeech-sat-resd (72%); audio+text variants 81-83%; README label list says "interest" where HF cards say "enthusiasm"; claims "<5 s per 3-s clip" processing (hardware unspecified); MIT | https://github.com/aniemore/Aniemore | Aniemore GitHub | README, undated | medium (speed claim unsourced/hardware unknown) | accuracy/speed/license |
| 5 | RESD: Russian, studio-quality acted dialogues, 7 emotions, ~4 h (Aniemore); emotion2vec paper cites 1,396 samples, 200 speakers | #4 and https://arxiv.org/html/2312.15185v1 | Aniemore; arXiv | 2023-12 | medium | dataset |
| 6 | emotion2vec+ (seed/base ~90M/large ~300M): 9 labels angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown; FunASR/ModelScope inference, 16 kHz, utterance or frame-level; repo MIT badge; last news Oct 2024; language mix of pseudo-labeled training data not stated | https://github.com/ddlBoJack/emotion2vec ; https://huggingface.co/emotion2vec/emotion2vec_plus_large | Alibaba/SJTU authors (GitHub, HF) | repo news Oct 2024 (flag: ~2 yr old) | high (labels/sizes); low (license of weights — HF card shows generic "model-license") | landscape/license/version |
| 7 | emotion2vec (base, 93.8M, frozen + linear probe) on RESD: WA 64.75 / UA 65.04 / WF1 64.53 vs WavLM-base 56.17 WA; IEMOCAP 71.79 WA (5-fold session) — note this is with a RESD-trained linear head, NOT zero-shot emotion2vec+ | https://arxiv.org/html/2312.15185v1 | arXiv preprint (later ACL Findings 2024 — unverified this run) | 2023-12-23 | high (numbers); preprint | accuracy |
| 8 | No published zero-shot emotion2vec+ result on Russian (RESD/Dusha) found | searches this run | — | — | medium (absence) | accuracy |
| 9 | Dusha: ~350 h, ~300k clips; Crowd 255.7 h / 201,850 files (acted, crowd-sourced), Podcast 90.9 h / 102,113 files (spontaneous); 4 emotions positive, sad, angry, neutral (+ "other"); 8,308 speakers; raw podcast audio withheld for licensing, features (30 GB) provided; license documents in repo (PDF, terms not read) | https://github.com/salute-developers/golos/tree/master/dusha | SaluteDevices GitHub | 2022-2023 | high (size); low (license terms) | dataset |
| 10 | Dusha paper (Interspeech 2023, peer-reviewed): podcast is heavily neutral (81,104 of 90,462 aggregated podcast files neutral ≈ 90%); MobileNetV2+self-attention baseline: Crowd UA 0.83 / WA 0.76 / F1 0.77; Podcast UA 0.89 / WA 0.53 / F1 0.54; same arch on IEMOCAP4 UA/WA/F1 0.59 | https://www.isca-archive.org/interspeech_2023/kondratenko23_interspeech.pdf | ISCA | 2023-08 | high | dataset/accuracy |
| 11 | waveletdeboshir/whisper-base-ser-dusha: Whisper-base encoder + head, 20.7M params; labels angry, sad, neutral, positive, other; Dusha crowd WA 0.836 / macro-F1 0.843; no podcast eval; license not stated in fetch; ~4.8k downloads/month | https://huggingface.co/waveletdeboshir/whisper-base-ser-dusha | community HF card | undated in fetch | medium (self-reported, crowd only) | accuracy/speed (size) |
| 12 | xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned: base hubert-large-ls960-ft (~316M — size inferred, unverified); labels neutral, angry, positive, sad, other; acc 0.86, balanced acc 0.76, macro-F1 0.81 (test split unspecified); Apache-2.0; last update 2023-06-10; ~224k downloads/month (unusually high, possibly automated) | https://huggingface.co/xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned | community HF card | 2023-06-10 (flag age) | medium | accuracy/license/version |
| 13 | Other Dusha fine-tunes exist: xbgoose hubert-base and wavlm-large, nixiieee/whisper-small-emotion-classifier-dusha (not read) | WebSearch results | HF | — | low | landscape |
| 14 | RAMAS: first Russian multimodal corpus, ~7 h, 10 semi-professional actors, dyadic; video of faces, speech, mocap, EDA/PPG; anger, disgust, happiness, sadness, fear ("scare"), surprise (+ neutral per other refs, unverified); SPECOM 2018 | https://peerj.com/preprints/26688/ (via search snippet), Dusha Interspeech paper §2.2 | PeerJ preprint / SPECOM 2018 | 2018 | medium (license/access not verified) | dataset |
| 15 | Silero VAD: <1 ms per 30+ ms chunk on one CPU thread; ~2 MB JIT; 8/16 kHz; MIT | https://github.com/snakers4/silero-vad | snakers4 GitHub | version/date not shown | high | speed/license |
| 16 | EmotiEffLib (face): Apache-2.0, v1.1.1, PyTorch + ONNX, AffectNet 7/8-class models, mobile CPU 16-191 ms/inference; repo shows no audio or A/V fusion code | https://github.com/sb-ai-lab/EmotiEffLib | sb-ai-lab GitHub | undated | high | landscape/license |
| 17 | HSEmotion (EmotiEffLib authors) ABAW-8 2025: fuse frame-level facial descriptors + acoustic features + ASR text embeddings, aggregate, feed simple MLP (one hidden layer); audio contribution not isolated in abstract | https://arxiv.org/abs/2503.10399 | arXiv / CVPR 2025 ABAW workshop | 2025-03-13 | medium | landscape (fusion) |

## (b) Finalist comparison (Russian SER, no fine-tuning)

| Finalist | Labels (map to 7 basic) | Russian accuracy (source) | In-the-wild evidence | Size / VRAM est. | License | Maintenance |
|---|---|---|---|---|---|---|
| Aniemore wavlm-emotion-russian-resd | anger, disgust, enthusiasm→happiness/surprise?, fear, happiness, neutral, sadness — no surprise | RESD WA 0.807 / F1 0.803 | Dusha podcast F1 0.11 (very poor; over-predicts emotion) | 317M; ~1.2 GB fp32, ~0.6 GB fp16 (est.); INT8/INT4 variants exist | MIT | HF card recently updated with cross-eval + quantized variants |
| Aniemore wav2vec2-xlsr-53-russian | same 7 | RESD F1 0.72 | Dusha podcast F1 0.095 | 316M | MIT | older, fewer downloads |
| emotion2vec+ base / large | angry, disgusted, fearful, happy, neutral, sad, surprised (+other/unknown) — covers all 7 basic | no zero-shot Russian number found; emotion2vec base + RESD linear head 64.8 WA | none found | ~90M / ~300M | MIT (repo); HF weight license ambiguous | last news Oct 2024 |
| whisper-base-ser-dusha | angry, sad, neutral, positive, other — no fear/disgust/surprise | Dusha crowd F1 0.843 (self-reported) | trained on full Dusha incl. podcast, but podcast metric not reported | 20.7M (lightest; CPU-friendly) | not stated | unknown |
| xbgoose hubert-large Dusha | neutral, angry, positive, sad, other | F1 0.81, acc 0.86 (split unspecified) | unknown | ~316M (est.) | Apache-2.0 | last update 2023-06 |

Tentative read (from evidence above): no single open model covers all 7 classes AND has evidence of working on spontaneous Russian. Aniemore = full-ish label set but acted-only; Dusha models = 4 classes but trained on real Russian speech incl. podcasts; emotion2vec+ = full label set, multilingual, but Russian performance unverified. A pragmatic MVP option is a Dusha-trained model (4 classes) for audio, with fear/disgust/surprise left to the face channel — this is an inference, not a sourced recommendation.

Latency: no measured GPU/CPU latency for 2-3 s windows found for any finalist. Unverified belief: ~300M-param SSL models run a 3 s window in tens of ms on a modest GPU fp16 and a few hundred ms on CPU; 20M Whisper-base encoder is CPU-viable. Must be benchmarked locally.

## Fusion notes (Q4)
- Evidence only for: Silero VAD is cheap enough (<1 ms/chunk CPU) to gate SER per window (#15); HSEmotion uses MLP-on-aggregated-features fusion (#17), which needs training. For no-training MVP, weighted late fusion of probabilities over a shared label space (with VAD gating audio weight to 0 when no speech) is the default — this design is NOT sourced this run.
- Dusha podcast clips were VAD-sliced into ~5 s segments (#10) — suggests 2-5 s windows are within training distribution for Dusha models.
- EmotiEffLib has no audio/multimodal model (#16). Emotion-LLaMA / MER baselines not checked (budget).

## Accuracy gap (Q5)
- Acted→spontaneous: Aniemore models fall from macro-F1 ~0.72-0.80 on RESD to ~0.10 on Dusha podcast (#2, #3) — dramatic. Dusha baseline WA drops 0.76 (crowd) → 0.53 (podcast) within its own corpus (#10).
- Cross-lingual: emotion2vec frozen features + linear head reach only ~65% on RESD vs 72% WA IEMOCAP (#7) — not a clean cross-lingual measure.

## (c) Leads worth chasing
1. Run emotion2vec+ base/large zero-shot on RESD test + a Dusha podcast sample locally — the key unknown.
2. Read nixiieee/whisper-small-emotion-classifier-dusha and xbgoose wavlm-large Dusha cards for podcast-split metrics.
3. Dusha license PDF terms (crowd audio redistribution, commercial use).
4. EmoBox (emotion2vec authors' multilingual benchmark) — may contain RESD/Dusha numbers for many SSL models.
5. Emotion-LLaMA / MER2024-25 baselines, Qwen2-Audio / Qwen2.5-Omni SER on Russian, and 4-bit VRAM — not checked.
6. Savchenko CVPRW 2025 ABAW paper (403 on CVF; try arXiv) for quantified audio gain over face-only.
7. RAMAS access/license (application-based?).

## (d) Looked for, not found
- Measured latency / VRAM for any Russian SER model on 2-3 s windows.
- Zero-shot emotion2vec+ accuracy on Russian.
- Dusha podcast-test metrics for community Dusha models.
- License for whisper-base-ser-dusha; Dusha license terms (not read).
- Silero VAD current version/date.
- Any 2025-2026 Russian-specific SER release newer than these; audio-LLM SER results for Russian.
- Quantified audio+video late-fusion gains in a real-time setting.
