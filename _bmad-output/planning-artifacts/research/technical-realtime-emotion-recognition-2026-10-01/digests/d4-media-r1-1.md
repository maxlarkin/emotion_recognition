# D4: Media quality requirements and how realistic the accuracy targets are (round 1, digest 1)

Accessed: 2026-10-01 (all sources). Budget used: about 20 tool calls, 9 sources read in full or in part, the rest from search snippets only (marked "snippet").
Firewall note: project context shaped the questions only. Every claim below traces to a retrieved source. Where no source was found, the item is marked **UNVERIFIED** and is a hypothesis to check later.

## (a) Findings

| # | Claim | Source URL | Publisher / type | Pub date | Confidence | Class |
|---|---|---|---|---|---|---|
| F1 | FER accuracy falls steadily as face resolution drops. Faces were downsampled from 100x100 px (bicubic). **RAF-DB, 7 classes, DRGFER:** 89.24% at 100 px, 88.23% at 50 px, 85.30% at 25 px, 81.91% at about 17 px, 77.35% at 12.5 px. A naive multi-scale-trained baseline drops to 66.85% at 12.5 px. FERPlus shows the same pattern (84.12% to 75.01%). | https://arxiv.org/html/2404.06365v1 | Ou et al., arXiv **preprint** | 2024-04-09 | high (numbers read from tables) | media-param / accuracy |
| F2 | So accuracy holds roughly flat (about 1 point lost) down to about 50 px face width and loses about 4 points at 25 px. Below about 17 px it collapses unless the model is trained for low resolution. | derived from F1 | (inference from F1) | n/a | medium | media-param |
| F3 | Face-recognition literature: faces with a tight box under 32x32 px "begin to present significant accuracy challenges". Low-resolution faces are typically under 32x24 px with an eye-to-eye distance of about 10 px. | https://arxiv.org/pdf/1805.11519 | "Face Recognition in Low Quality Images: A Survey", arXiv **preprint** (snippet only) | 2018 | medium | media-param |
| F4 | IEC/EN 62676-4 DORI pixel densities: Detect 25 px/m, Observe 62.5 px/m, Recognize 125 px/m, Identify 250 px/m. The **IEC 62676-4:2025** revision reportedly adds more levels: Overview 20, Outline 40, Discern 80, Perceive 125, Characterise 250, Validate 500, Scrutinise 1500 px/m. | https://www.backstreet-surveillance.com/cctv-industry-experts/dori-cctv.html ; https://www.jvsg.com/pixel-density/ | Vendor / industry blogs (**secondary**, not the standard text) | n.d. / 2025 | medium (DORI values); low (2025 level names: one secondary source; the 2025 edition may supersede the DORI naming) | standard |
| F5 | Back-of-envelope (my arithmetic, not a source): an adult face is about 0.15-0.16 m wide (**UNVERIFIED** anthropometric figure). At 250 px/m ("Identify") that gives about 40 px of face width. At 500 px/m it gives about 80 px. Combined with F1, "Identify" density sits at the low edge for FER (around the 25-50 px band). About 500 px/m is needed to reach the plateau of 50 px or more. | derived F1 + F4 | inference | n/a | low-medium | media-param |
| F6 | Audio codec study, 3 codecs (MP3, AAC, Opus forced to SILK) at many bitrates, data resampled to 16 kHz, metric UAR on EMO-DB, eNTERFACE, PolishEMO and others. With **matched** training, UAR stays at the uncompressed level. Above 64 kbit/s there is little or no UAR loss for any codec. With **mismatched** training (clean train, compressed test) UAR drops sharply. Opus showed the largest drop, about **-20% on average**, which the authors attribute to a DC-offset/high-pass interaction. Augmenting training with encoded audio recovers the loss (for example AAC +35% with ComParE features). | https://www.isca-archive.org/interspeech_2019/oates19_interspeech.pdf | Oates et al., Interspeech 2019, **peer-reviewed** | 2019 | high | media-param |
| F7 | Earlier work cited in F6: MP3 at 32 kbit/s or more keeps SER accuracy (SVM, Siegert/Requardt). For AMR, AMR-WB and AMR-WB+, accuracy does not always fall as bitrate falls. | same as F6 (related-work section) | peer-reviewed (citing) | 2019 | medium | media-param |
| F8 | Neural speech codecs at low bitrates cut SER macro-F1 on IEMOCAP from about 64% to about 53%. | https://arxiv.org/html/2605.11098 (AffectCodec); also ACL Findings 2026 https://aclanthology.org/2026.findings-acl.442.pdf | arXiv preprint, then ACL Findings 2026 (**peer-reviewed**), snippet only | 2026-05 | medium | media-param |
| F9 | **Dusha** (Russian SER): 346.6 h, 303,963 files, 8,308 speakers. Crowd part (acted by crowd workers): 255.7 h. Podcast part (natural speech): 90.9 h. 4 classes: positive, sad, angry, neutral. Audio normalised to **16-bit, 16 kHz**. Open licence (EN and RU licence files are in the repo). Each sample has 3 or more raw annotator labels. | https://github.com/salute-developers/golos/tree/master/dusha ; https://www.isca-archive.org/interspeech_2023/kondratenko23_interspeech.pdf | SberDevices, Interspeech 2023, **peer-reviewed** | 2023 | high | dataset |
| F10 | Dusha baseline (shallow model): **Crowd test** UA 0.83, WA 0.76, F1 0.77. **Podcast test** UA 0.89, WA 0.53, F1 0.54. The same architecture on IEMOCAP gives UA 0.59, WA 0.59, F1 0.59. Natural (podcast) speech is far harder than crowd-acted speech even with only 4 classes. | kondratenko23 PDF, Table 3 | peer-reviewed | 2023 | high | accuracy |
| F11 | **RAMAS**: Russian, acted, dyadic. 10 semi-professional actors (5 male, 5 female), 6 basic emotions plus neutral (per snippet), about 7 h. Includes close-up face video, speech, motion capture, EDA and PPG. Described as an "open database". | https://peerj.com/preprints/26688/ ; SPECOM 2018 | PeerJ **preprint** plus SPECOM 2018 (peer-reviewed), snippet only | 2018 | medium | dataset |
| F12 | **AffectNet**: over 1M web images, about half manually annotated, 7 or 8 categories plus valence/arousal. **Two human annotators agreed on only 60.7%** of 36k images used for the agreement study. | https://arxiv.org/abs/1708.03985 (PDF text checked) | Mollahosseini et al., IEEE TAFFC, **peer-reviewed** | 2017 | high | accuracy |
| F13 | Static in-the-wild FER SOTA reported in recent papers: POSTER++ RAF-DB 92.21%, AffectNet-7 67.49%. Another method reports RAF-DB 92.44%. Recent methods cluster around RAF-DB 89-92% and AffectNet-7 64-68%. | search snippets (ResearchGate table; arXiv 2410.17622) | mixed, aggregator snippet | 2023-2026 | medium (not checked against the original tables) | accuracy |
| F14 | Dynamic (video) in-the-wild FER: a 2025 method reports DFEW WAR 76.62% / UAR 65.25% and MAFW WAR 58.70% (11 classes). | search snippet (arXiv 2507.15765 or 2412.13168) | arXiv **preprints**, snippet | 2025 | medium | accuracy |
| F15 | Occlusion by face masks: the Residual Masking Network drops from 74.14% to 35.65% and the Amend-Representation Module from 90.42% to 45.45% on masked test data. Eye occlusion (glasses) hurts less. Training on masked data limits the loss to an average drop of about 10%. | https://link.springer.com/article/10.1007/s11042-022-13559-8 | Multimedia Tools & Applications, **peer-reviewed**, snippet | 2022 | medium | media-param |
| F16 | Occlusion and pose test subsets exist: Occlusion-RAF-DB, -FERPlus, -AffectNet (735, 605 and 682 images), and Pose subsets at 30° or more and 45° or more of yaw/pitch (Wang et al., RAN, IEEE TIP 2020). I did not retrieve the per-subset accuracy numbers. | https://arxiv.org/abs/1905.04075 ; https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2023.1250706/full | peer-reviewed | 2019/2023 | high (subsets exist); numbers missing | media-param |

## (b) Draft requirements table (round 1, to validate)

| Parameter | Minimum | Recommended | Evidence ref / status |
|---|---|---|---|
| Face width in frame (px) | 48 px (about 25-50 px band costs about 1-4 accuracy points) | 100 px or more (128-224 is the typical model input) | F1, F2, F3. The upper value is a convention, **UNVERIFIED** |
| Inter-ocular distance (px) | about 20 px (hypothesis: about 0.4 x face width) | 40-60 px | Ratio **UNVERIFIED**. ICAO/ISO 19794-5 values not retrieved |
| Scene pixel density | 250 px/m (DORI Identify / 2025 "Characterise") | 500 px/m (2025 "Validate") | F4, F5 |
| Head pose (yaw/pitch) | 30° or less (standard benchmarks split at 30° and 45°) | 15° or less | F16. Accuracy-drop numbers **not retrieved** |
| Occlusion | No medical/face mask on the lower face | No occlusion. Glasses acceptable | F15 |
| Frame rate | **UNVERIFIED**: 10-15 fps hypothesised for macro-expressions | 25-30 fps. Micro-expression datasets use 100-200 fps (**UNVERIFIED**) | No source retrieved (gap) |
| Video compression | **UNVERIFIED** | **UNVERIFIED** | Gap. Treat by analogy with audio: augment training with the deployment codec |
| Lighting | **UNVERIFIED** | Frontal, even lighting | Gap |
| Audio sample rate | 16 kHz | 16 kHz, 16-bit (Dusha's format) | F6, F9. Effect of 8 kHz telephony not quantified here |
| Audio codec/bitrate | MP3 32 kbit/s or more (older SVM result); 64 kbit/s or more for any of MP3/AAC/Opus | Lossless PCM/FLAC, or 64 kbit/s or more **and** codec-matched training augmentation | F6, F7, F8 |
| Audio SNR / reverberation | **UNVERIFIED** | **UNVERIFIED** | Gap |
| Segment length | **UNVERIFIED** (Dusha/IEMOCAP utterances are typically a few seconds) | **UNVERIFIED** | Gap |

## Assessment: are ≥95% (streaming) and ≥98% (recorded) realistic?

- Evidence points to **no, for in-the-wild data with 7 or more classes**:
  - The best static FER models reach about 92% on RAF-DB and about 67% on AffectNet-7 (F13).
  - Video FER reaches about 77% WAR on DFEW and about 59% on MAFW (F14).
  - Natural Russian speech (Dusha podcast) gives F1 0.54 even with 4 classes (F10).
- The labels themselves cap accuracy. Two AffectNet annotators agree on only 60.7% of images (F12), so "accuracy" above human agreement on such data is not meaningful.
- Values of 95-100% appear only on acted/lab sets (CK+, JAFFE, RAVDESS, KDEF; the snippet in search 1 cites 97.34% on KDEF). Those numbers are not retrieved from original papers here, so they are **UNVERIFIED in this run**.
- **Conditions under which ≥95% is plausible (hypothesis):**
  - posed or acted expressions
  - frontal faces of 100 px or more
  - few classes (for example 3-4 classes, or a binary negative vs non-negative)
  - per-clip or per-session aggregation rather than per-frame
  - the metric is plain accuracy on a held-out set **from the same dataset**
- Requirements should name the metric (UAR/macro-F1, not WA), the number of classes, the dataset type, and speaker/subject-independent splits.

## (c) Leads worth chasing

1. ISO/IEC 19794-5 / ICAO 9303 / ISO/IEC 29794-5 face-quality: minimum inter-eye distance (commonly quoted as 90 or 120 px, **UNVERIFIED**).
2. GOST R 51558-2014 (Russian CCTV) or GOST R/IEC 62676-4: pixel density for identification.
3. Get the official IEC 62676-4:2025 summary from webstore.iec.ch to confirm the new levels.
4. RAN paper (arXiv 1905.04075) Tables: accuracy on Pose-30/45 and Occlusion subsets.
5. Studies of FER under H.264/JPEG compression and frame-rate subsampling. Also check DFER models' 16-frame sampling, which suggests a low effective fps is tolerated.
6. SER under noise/reverb: CHiME-style SNR sweeps (0/5/10/20 dB), 8 vs 16 kHz telephony (García et al., call-centre SER, cited in F6).
7. ABAW 2025/2026 challenge expression F1 (Aff-Wild2) and the newest DFEW/MAFW SOTA (Sensors 26(17):5384, 2026, which returned 403 on fetch).
8. RESD (Russian Emotional Speech Dialogues, Aniemore) size, licence and SOTA. RAMAS access terms (licence request via Neurodata Lab, which may be defunct).
9. Human agreement on RAF-DB (40 annotators), DFEW and Dusha (raw labels are available, so agreement can be computed).

## (d) Looked for but not found this run

- Quantified effect of fps and of H.264 bitrate/JPEG quality on FER: no source retrieved.
- Quantified accuracy drop for illumination: not retrieved. For head pose: only the subset definitions, not the numbers.
- SNR, reverberation and 8 kHz effects on SER; minimum segment duration; mic distance: not retrieved.
- ICAO/ISO 19794-5 and GOST texts: not retrieved (no primary standard text read; DORI comes from secondary industry pages only).
- RESD details and any Russian-face FER video dataset other than RAMAS: not retrieved.
- Primary-paper checks for CK+/JAFFE/RAVDESS around 98-100% and for AffectNet-8 / Aff-Wild2 SOTA: not done.
