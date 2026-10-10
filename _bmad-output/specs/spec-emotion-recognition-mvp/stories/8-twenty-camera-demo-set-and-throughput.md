---
title: 'Twenty-camera demo clip set'
type: 'feature'
created: '2026-10-10'
status: 'in-review'
baseline_commit: 'fa5a23aca532177fd32395a4cacecffd727400f9'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/camera-profiles.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/stack.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** CAP-10 and CAP-11 need a 20-stream demo mix (6 webcam, 8 sector, 6 security) with faces, Russian speech and clear negative and positive moments. Today `demo-cams` publishes 3 synthetic streams of one still photo, re-encoded live.

**Approach:** Build the mix from openly licensed Russian videos on Wikimedia Commons. Acted short films supply negative and positive moments; studio interviews supply natural speech and laughter. A prep script runs once with network inside the backend image. It downloads pinned sources and verifies them, mines negative/positive windows with the project's own models, cuts segments and encodes each for its camera profile. The emulator loops the 20 files with `-c copy`. Throughput tuning is story 1.9.

## Boundaries & Constraints

**Always:**
- Mix and geometry per `camera-profiles.md`: webcam 640×480 with audio (faces ≥ 90 px), sector 1920×1080 with audio (faces 50–100 px), security 2560×1440 without audio (faces 25–45 px). All clips are 10 fps H.264 with closed GOPs (AAC where the profile has audio), 60–120 s long and loopable. The emulator runs `-re -stream_loop -1 -c copy` per stream, with no live re-encode.
- Sources are openly licensed (CC0, CC BY, CC BY-SA, public domain) and pinned in `demo-videos/sources.json`: Commons file page, download URL, SHA-1 from the Commons API, licence, author. Political and war content is excluded. Candidates: short films «Допрос», «Автобус» (TembAz), «Самоволка» (ВГИК), «Время жить, время умирать», «С чистого листа»; «Страна FM» studio interviews; «Интервью по средам» for 640×480 material.
- Window mining uses the project models (detector + FER, SER where there is speech) on the downloaded sources. A window is negative when the fused negative group is ≥ 0.6 for ≥ 2 s on one face, positive when happiness is ≥ 0.6 for ≥ 2 s (the CAP-11 rule). Each of the 20 clips contains at least one window of either kind where the sources allow it. `demo-videos/segments.md` lists, per clip: source, time range, windows found (kind, start, duration, peak), and the clips with no negative window.
- Profile geometry is derived from measured face sizes: the prep script scales/crops/pads each segment so the median detected face lands in the profile's band, and the report states the achieved median.
- Encoded clips and downloads are not committed (`.gitignore`). `demo-videos/ATTRIBUTION.md` gets one row per source (page URL, author, licence, segments used). After prep, the demo runs fully offline.
- `demo-videos/demo-cams.json` lists the 20 streams (`rtsp://mediamtx:8554/cam1..cam20`, name, site, camera_type) for `DEFAULT_STREAMS`. The synthetic story-1.7 cameras remain as a no-download fallback (`demo-cams.sh` falls back to them when `clips/` is empty).
- AGENTS.md documents the prep command and the demo command.

**Never:**
- Committing video or audio files; downloading anything at demo runtime.
- Changing the pipeline, contracts or event rule (stories 1.9 and 1.11).
- Footage of private individuals in private settings; political or war content.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Prep run | network, Docker, models in the image | `demo-videos/clips/cam01..cam20.mp4`, `segments.md`, downloads cached in `demo-videos/cache/` | SHA-1 mismatch or a failed download stops with the file name; nothing half-written stays in `clips/` |
| Prep again | cache present, network off | re-encodes from the cache | N/A |
| Geometry | each encoded clip | resolution/fps/audio per profile; median face within the band (± 10 %) | out-of-band clip reported in `segments.md` |
| Loop | emulator plays a clip end to end | no decode errors at the loop seam; timestamps continue | N/A |
| Demo up | `--profile gpu|cpu --profile demo-cams` with clips | 20 streams published; with `DEFAULT_STREAMS` seeded, 20 streams `running` | a missing clip: that publisher logs and retries, the others run |
| No clips | `clips/` empty | the 3 synthetic cameras as in story 1.7 | log line explaining how to run prep |

</frozen-after-approval>

## Code Map

- `demo-videos/demo-cams.sh`, `demo-cams.json`, `ATTRIBUTION.md` -- story 1.7 synthetic set (3 streams, live x264); replace with the clip loop and keep the synthetic set as a fallback.
- `docker-compose.yml` -- `mediamtx` and `camera-emulator` (image `bluenviron/mediamtx:1.21.1-ffmpeg`, has ffmpeg/libx264/aac); the emulator mounts `./demo-videos:/demo:ro,z`.
- `backend/inference/worker.py` -- `Engine` (detector, tracker, FER, VAD/SER, fusion) can run in-process for mining. `InferenceWorker` or `Engine` with `Settings(models_dir=/models)` inside the backend image; `bench/stages.py` shows how to load models.
- `backend/sources/rtsp.py` -- PyAV decode helpers (`fit`); the backend image has PyAV 18.1 with FFmpeg and libx264 encoders for prep encoding.
- `backend/app/streams.py` -- `DEFAULT_STREAMS` seeding (only into an empty list, once).
- Measured: the YuNet CPU detector takes 6 / 69 / 164 ms at 640 / 1920 / 2560, so mining runs on the GPU image when available (FER on CUDA) and on downscaled analysis frames.

## Tasks & Acceptance

**Execution:**
- [x] `demo-videos/sources.json` -- pinned sources with SHA-1, licence, author, page URL -- reproducible inputs.
- [x] `backend/scripts/prep_demo_clips.py` (+ unit tests for the pure parts: window finder, geometry fit, mix plan) -- download/verify, analyse at 5 fps (faces, FER, SER), find windows by the CAP-11 rule, choose 20 segments by profile (webcam: close faces with audio; sector: medium; security: any face, no audio), encode with PyAV (scale/crop/pad to the band, 10 fps, GOP 20, AAC 16 kHz mono), write `segments.md` -- prep.
- [x] `demo-videos/demo-cams.sh`, `demo-cams.json`, `ATTRIBUTION.md`, `.gitignore`, `docker-compose.yml` -- 20-stream `-c copy` loop with synthetic fallback, stream list, attribution, ignores -- demo.
- [x] `AGENTS.md` -- prep and demo commands -- docs.

**Acceptance Criteria:**
- Given the prep run in the GPU (or CPU) backend image, when it finishes, then 20 clips exist with profile geometry and `segments.md` lists the negative/positive windows per clip and the achieved median face sizes.
- Given the clips and `DEFAULT_STREAMS` from `demo-cams.json`, when `--profile gpu --profile demo-cams` runs, then 20 streams reach `running` and the grid shows them. The fps reached is recorded as the baseline for story 1.9.
- Given a clip listed with a negative window, when its stream plays, then at that time the stream detail shows the track's negative probability rising above 0.6 (checked once on the GPU profile).

## Verification

**Commands:**
- `cd backend && .venv/bin/pytest -q` -- expected: pass (prep unit tests included).
- Prep command from AGENTS.md -- expected: 20 clips + `segments.md`.
- `DEFAULT_STREAMS="$(cat demo-videos/demo-cams.json)" docker compose --profile gpu --profile demo-cams up -d` then `curl -s localhost/api/streams` -- expected: 20 running streams.

## Implementation Notes

- Sources (`demo-videos/sources.json`, all CC BY 3.0 on Wikimedia Commons): short films «Допрос», «Автобус» (TembAz), «Самоволка» (ВГИК diploma), «Время жить, время умирать», «С чистого листа»; interviews «Страна FM» (Аня Покров, Люся Чеботина) and «Интервью по средам» (Сергей Безруков). The 480p VP9 transcodes are used (825 MB in total, SHA-256 pinned); if a transcode changes, the original upload is used, verified by its Commons SHA-1. Faces are scaled down to the profile bands anyway, so 480p sources are enough.
- `backend/scripts/prep_demo_clips.py` runs inside the backend image (models on CUDA): it analysed 8 sources (≤ 15 min each, 5 fps, ~34k frames) in 13.5 min and encoded 20 clips in ~7 min. Analyses are cached in `demo-videos/cache/`, so re-planning takes only the encode time. Clips: `demo-videos/clips/cam01..cam20.mp4`, 85 MB, 75 s each, 10 fps, GOP 20, AAC 16 kHz mono on webcam/sector. Report: `demo-videos/segments.md`.
- Windows found (CAP-11 rule on fused probabilities): films are negative-heavy (e.g. «Время жить…» 97 negative / 1 positive, «Автобус» 70 / 0), interviews positive-heavy («Страна FM» 0 negative / 57 and 30 positive). The FER model reports the negative group (mostly sadness, anger) far more often than a human would; the report says so. The planner scores negative and positive windows the same, takes at most 3 segments per source, and keeps webcam segments to sources whose median face is 60–200 px. Final mix: 13 of 20 clips have a negative window, 14 a positive one, all 8 sources used. Measured median faces are all inside the bands: webcam 135–175 px, sector 72–87 px, security 29–43 px.
- Emulator: `demo-cams.sh` loops every prepared clip with `-re -stream_loop -1 -c copy`, and falls back to the 3 synthetic cameras when `clips/` is empty. `demo-cams.json` holds 20 streams (cam1–6 webcam, 7–14 sector, 15–20 security, 5 sites). The prep and demo commands are in AGENTS.md.
- Baseline for story 1.9 (GPU profile, RTX 3050, 20 streams): all 20 streams reached `running`, all decoded with NVDEC. The pipeline then collapsed: only 2/20 streams produced results within 90 s (~0.02 results/s), decode-to-result p50 11.2 s and p95 14.4 s, GPU util 6 %, backend ~1000 % CPU and 6 GB RSS, host RAM 13.6/15 GB used. The cause is the expected bottleneck: YuNet on CPU at 1920/2560 (69/164 ms per frame), plus full-size frames pickled into the worker's 256-deep inbound queue (gigabytes in flight). With 4 streams (2 webcam, 2 sector): 4.5 results/s each, decode-to-result p50 289 ms / p95 388 ms.
- AC3 check: with those 4 streams, every one (cam1, cam2, cam8, cam13 — each listed with negative windows) showed fused negative probability > 0.6 during the 80 s recording.
- The volume used for the check now holds the 20 demo streams with 16 disabled (to protect host memory). Re-enable them from Settings › Камеры once story 1.9 lands.
- Verified: `pytest -q` 177 passed (10 new prep tests); prep run end to end in the GPU image; 20-stream and 4-stream runs in Docker as above.

## Spec Change Log

## Review Triage Log
