# D2 SER Round 2 (targeted gaps): Russian SER

Accessed date for all sources: 2026-10-01. Budget used: 15 tool calls, 9 sources read.
Classes: P = primary (official repo, paper, or model card by the author); S = secondary (mirror or aggregator); U = unverified.

## (a) Findings

| # | Claim | Source URL | Publisher | Pub date | Confidence | Class |
|---|---|---|---|---|---|---|
| 1 | EmoBox covers 32 emotion datasets in 14 languages and evaluates 10 pre-trained speech models. It was accepted at INTERSPEECH 2024, so it is peer-reviewed. | https://arxiv.org/abs/2406.07162 | Ma et al. (emotion2vec authors), arXiv/INTERSPEECH | 2024-06-11 | High | P |
| 2 | EmoBox includes RESD (Russian, acted, 7 emotions, 200 speakers, 1,396 utterances, 2.3 h). It does **not** include Dusha. | https://github.com/emo-box/EmoBox | EmoBox repo | 2024 (repo) | High | P |
| 3 | The EmoBox README has no per-dataset (RESD) numbers for emotion2vec, WavLM, HuBERT or Whisper. The results are only in images on the README and on the project leaderboard site, which I could not retrieve. | https://github.com/emo-box/EmoBox | EmoBox repo | 2024 | High (absence) | P |
| 4 | GigaAM-Emo exists. It is the GigaAM encoder fine-tuned on Dusha and is loaded with `gigaam.load_model("emo")` plus `get_probs(path)`, which returns a per-emotion probability dict. The repo is MIT-licensed. The emo model is in the v1 line (`v1_ssl`, `emo`, `v1_ctc`, `v1_rnnt`) and was released in April 2024. The README says it "outperforms existing models by 15% Macro F1". | https://github.com/salute-developers/GigaAM | SaluteDevices (Sber) | 2024-04 (emo release); repo current | High | P |
| 5 | Reported Dusha results for GigaAM-Emo: Crowd UA 0.90 / WA 0.87 / MF1 0.84; **Podcast UA 0.90 / WA 0.76 / MF1 0.67**. Dusha baseline: Crowd MF1 0.77, Podcast MF1 0.54. TIM-Net: Crowd 0.78, Podcast 0.55. Note: on the podcast split UA is higher than WA, so the column labels may be swapped in the original. Use MF1. | https://huggingface.co/niobures/GigaAM/blob/main/README.md (mirror of the official GigaAM README) | Mirror of SaluteDevices README | 2024 | Medium-High (mirror, matches the official 15% claim) | S |
| 6 | GigaAM-Emo inference: encoder, then avg-pool over time, then a linear head, then softmax. The code sets no length limit for emo; ASR switches to longform at 25 s. The labels come from the config and are not visible in the code. | https://raw.githubusercontent.com/salute-developers/GigaAM/main/gigaam/model.py | SaluteDevices | current main | High | P |
| 7 | GigaAM-Emo labels are angry / sad / neutral / positive (4 classes). | Search-engine summary only (no primary page read) | aggregator | n/a | Low | U |
| 8 | waveletdeboshir/whisper-base-ser-dusha: **no license field** in the YAML front matter, so treat it as all-rights-reserved and unusable without permission. It has 20.7M params and 5 labels (angry, sad, neutral, positive, other), and is evaluated **only on Dusha crowd test**: WA 0.836, F1-macro 0.843. No podcast metrics. About 4.8k downloads last month. | https://huggingface.co/waveletdeboshir/whisper-base-ser-dusha/raw/main/README.md | waveletdeboshir (HF) | not shown | High | P |
| 9 | nixiieee/whisper-small-emotion-classifier-dusha: Apache-2.0 license, 88.5M params, 5 labels (neutral, angry, positive, sad, other). It was trained on `nixiieee/dusha_balanced` (updated 2025-06-18) and scores eval acc 0.772, balanced acc 0.806, F1 0.804. The eval set is its own balanced split, with **no crowd/podcast breakdown**. 17 downloads last month. | https://huggingface.co/nixiieee/whisper-small-emotion-classifier-dusha | nixiieee (HF) | 2025 | High (card); Low comparability | P |
| 10 | xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned: base model hubert-large-ls960-ft, 5 labels incl. "other". Test results: acc 0.86, balanced acc 0.76, macro F1 0.81. The **split is not stated** (no podcast number). Trained on half the train set for 2 epochs. More than 224k downloads. No license stated in the fetched summary. | https://huggingface.co/xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned | xbgoose (HF) | not shown | Medium | P |
| 11 | oruk-bench PR #1 (2026-09-16) adds xbgoose HuBERT-Dusha (316M), Aniemore WavLM-RESD (317M) and wav2vec2-large-superb-er (316M) for evaluation on Dusha Crowd and Podcast, RESD and SUPERB-ER. **No results were published yet**: the maintainers run the official numbers. Watch this for a head-to-head podcast comparison. | https://github.com/Oruk-AI/oruk-bench/pull/1 | Oruk-AI (GitHub PR) | 2026-09-16 | High (existence) | P |
| 12 | I found no published zero-shot evaluation of emotion2vec+ on Russian or any other non-zh/en language. Papers only use emotion2vec+ as a zero-shot ECA metric in TTS (e.g. MPE-TTS). | https://arxiv.org/html/2505.18453v1 | arXiv preprint | 2025-05 | Medium (absence) | P |

## (b) Updated finalist table

| Model | Labels | Params | Dusha podcast (spontaneous) | Other metric | License | Notes for MVP |
|---|---|---|---|---|---|---|
| **GigaAM-Emo** (SaluteDevices) | 4 (angry/sad/neutral/positive, unverified) | not stated; GigaAM encoder (my belief is about 240M, unverified) | **MF1 0.67** (only podcast number found) | Crowd MF1 0.84 | MIT (repo) | Best documented spontaneous-speech result. Pip package, simple API. **New front-runner.** |
| xbgoose hubert-large dusha | 5 (+other) | 316M | not reported | macro F1 0.81 (split unstated) | not confirmed | Widely used. Pending oruk-bench results. |
| waveletdeboshir whisper-base-ser-dusha | 5 (+other) | 20.7M | not reported | crowd F1 0.843 | **none stated**, legal risk | Smallest and fastest, but the license blocks it. |
| nixiieee whisper-small dusha | 5 (+other) | 88.5M | not reported | F1 0.804 (balanced own split) | Apache-2.0 | Low adoption. Metrics not comparable. |
| Aniemore wavlm-emotion-russian-resd | 7 | 317M | poor (round 1) | RESD (acted) | (round 1) | Acted-domain only. |
| emotion2vec+ | 9 | (round 1) | unknown | none on Russian found | (round 1) | Russian performance unverified. Would need an in-house check on Dusha podcast. |

## (c) Looked for, not found

- EmoBox per-dataset RESD numbers for emotion2vec, WavLM, HuBERT and Whisper. They are only in leaderboard images and on the site, which I did not retrieve. Dusha is not in EmoBox.
- Any zero-shot or cross-lingual evaluation of emotion2vec+ on Russian.
- Podcast-split metrics for the xbgoose, nixiieee and waveletdeboshir models. None of the cards report them. Results from oruk-bench are pending (PR 2026-09-16).
- GigaAM-Emo parameter count and labels from a primary page, and any v2/v3 emo update. The README lists emo only under v1.
- Any 2025–2026 Russian SER release from Yandex or MTS AI. I did not search for these because the budget ran out.
- **Q5 latency:** I found no measured CPU or small-GPU latency for wav2vec2, WavLM, HuBERT or GigaAM on 2–5 s clips. This is still open and should be measured locally.
