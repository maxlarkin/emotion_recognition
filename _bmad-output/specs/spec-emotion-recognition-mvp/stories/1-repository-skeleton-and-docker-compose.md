---
title: 'Repository skeleton and Docker compose with CPU and GPU profiles'
type: 'feature'
created: '2026-10-01'
status: 'done'
baseline_commit: '1a462119e5bf40f2a770a1a0113842340b9283d6'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/SPEC.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/stack.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/interfaces.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The repository is empty. Every later story needs a runnable backend and frontend, a Docker setup that works offline on Linux and Windows, and model weights already inside the image.

**Approach:** Build three things:
- the layout from `stack.md`;
- a minimal FastAPI backend whose `/api/health` reports the device and which model files are present;
- a minimal React (Russian UI) page served by nginx that shows that health.

Docker compose runs CPU by default and CUDA under `--profile gpu`. All four model weights are downloaded during `docker build`.

## Boundaries & Constraints

**Always:**
- Commands are exactly `docker compose up` (CPU) and `docker compose --profile gpu up` (GPU). In both, nginx serves the app at `http://localhost:${APP_PORT:-80}` and proxies `/api` and `/ws` to the backend.
- Build-time downloads are pinned (version, tag or commit) and verified (checksum where the source offers one). Runtime makes zero network calls.
- Weights are stored in the image under `/models` (see Design Notes), and loaders are pointed there. Nothing is cached under `$HOME` at runtime.
- The GPU image uses `onnxruntime-gpu`, never both `onnxruntime` and `onnxruntime-gpu`. Use the same version line in both images.
- All UI strings are in Russian. Python ≥ 3.11.
- The `data` named volume is mounted at `/data`.

**Never:**
- Loading models into memory or running inference. That is story 1.2. Health checks file presence only.
- Live WebSocket logic, sessions, DB schema (stories 1.3 and 1.5). Only a `/ws/live` stub that accepts and closes is allowed, so the proxy can be tested.
- CDN links, remote fonts, runtime `pip`/`npm` installs.
- Passing the camera or microphone into containers.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| CPU start | `docker compose up` | `GET /api/health` → `{"status":"ok","device":"cpu","models":{"yunet":true,"fer":true,"vad":true,"ser":true}}`; page shows «Устройство: CPU» and 4 ✓ | N/A |
| GPU start | `docker compose --profile gpu up` on a host with the NVIDIA toolkit | `"device":"cuda"` and `"gpu_name"` present; page shows the GPU name | N/A |
| GPU image, no GPU visible | GPU container, CUDA unavailable | `"device":"cpu"`, `"warning":"cuda_unavailable"`; page shows the warning in Russian | does not crash |
| Missing weight | a file under `/models` removed | that model's flag is `false`; `"status":"degraded"`; HTTP 200 | page lists the missing model |
| Offline | network disabled after build | both profiles start and health is identical | N/A |

</frozen-after-approval>

## Code Map

- The repository is empty except for `README.md` and `_bmad*`. There is no existing code to reuse.
- `emotiefflib` 1.1.1 downloads to `~/.emotiefflib/<file>` from GitHub (`emotiefflib/utils.py`). Pre-fetch `enet_b0_8_va_mtl.onnx` at build time and keep it in the image.
- `gigaam`: PyPI 0.1.0 is stale (pins torch ≤ 2.5.1, onnxruntime 1.17). Install from GitHub at commit `7447938d791c4f3e643386ee22c33777004293a5` (0.2.0; pins `onnxruntime==1.23.*`, torch is an extra). Install it `--no-deps` and supply the deps yourself so onnxruntime is not duplicated. Weights come from `https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/emo.ckpt` (md5 `7ce76f9535cb254488985057c0d33006` in `gigaam/__init__.py`). Pass `download_root` to `load_model`.
- `silero-vad` 6.2.3 wheel bundles `silero_vad.onnx` (package `data/`). No download is needed.
- YuNet: `opencv-python-headless` latest is 5.0.x, which needs `face_detection_yunet_2026may.onnx` from `opencv/opencv_zoo` (Git LFS; fetch via the `media.githubusercontent.com` URL).
- `emotiefflib` requires `opencv-python`. Install it, then replace it with `opencv-python-headless` of the same version, or add `libgl1`. Pick one and record it.

## Tasks & Acceptance

**Execution:**
- [x] `.gitignore`, `.dockerignore`, `.env.example` -- ignore caches, `node_modules` and `data`; `.env.example` holds `APP_PORT=80` and `COMPOSE_PROFILES` -- reproducible builds.
- [x] `backend/pyproject.toml` (or `requirements*.txt`) -- pinned deps for the CPU and GPU variants -- one source of versions.
- [x] `backend/scripts/fetch_models.py` -- download and verify all weights into `/models`, fail the build on mismatch -- baked, offline weights.
- [x] `backend/Dockerfile` -- build arg `VARIANT=cpu|gpu`. CPU base is `python:3.11-slim`; GPU base is an `nvidia/cuda` 12.x cuDNN runtime image. Install torch (CPU or CUDA index), deps and gigaam, then run `fetch_models.py` -- single Dockerfile for both profiles.
- [x] `backend/app/main.py`, `backend/app/config.py`, `backend/app/api/health.py`, `backend/app/ws/live.py` -- FastAPI app, settings (models dir, data dir), health per the matrix, `/ws/live` stub -- contract from `interfaces.md`.
- [x] `backend/inference/__init__.py`, `backend/bench/__init__.py`, `backend/eval/__init__.py` -- empty packages fixing the layout -- later stories fill them.
- [x] `backend/tests/test_health.py` -- pytest for every matrix row that can be simulated (temp models dir, monkeypatched device) -- regression guard.
- [x] `frontend/` (Vite + React + TS) -- a page titled «Распознавание эмоций» that fetches `/api/health` and shows the device, GPU name, warning and model checklist in Russian -- proves the frontend → nginx → backend path.
- [x] `frontend/Dockerfile`, `frontend/nginx.conf` -- multi-stage node build to nginx; proxy `/api` and `/ws` (with upgrade headers) to `backend:8000` -- single entry point.
- [x] `docker-compose.yml` -- `frontend`, plus a CPU `backend` and a GPU backend under profile `gpu` (`gpus: all`), so exactly one backend answers at host `backend` in each command; volume `data` -- the commands in Always.
- [x] `README.md` -- run instructions in Russian for Linux and Windows (Docker Desktop, WSL2), NVIDIA toolkit prerequisite, offline note -- operator guide.

**Acceptance Criteria:**
- Given a clean checkout, when `docker compose up --build` runs, then `curl localhost/api/health` returns `device: cpu` with all four models `true`, and the page renders it in Russian.
- Given the host's RTX 3050 with NVIDIA toolkit, when `docker compose --profile gpu up --build` runs, then health returns `device: cuda`, and no CPU backend container is running.
- Given built images, when the network is disabled and either command runs, then the result is unchanged.
- Given the built frontend, when its `dist/` is grepped for `http://` or `https://`, then no external origins are found.

## Implementation Notes

- Environment: the agent shell predates the user's docker group membership; run Docker as `sg docker -c '<command>'`. nvidia runtime is registered (`docker info` Runtimes: nvidia). Host GPU: RTX 3050 Laptop 4 GB.
- Resume run (2026-10-01): reviewed all files against this spec; no code defects found. Fixed README dev instructions to use `uv venv --python 3.11` (host Python is 3.14). Recreated `backend/.venv` with Python 3.11.16 (it was 3.14).
- Verified: `pytest -q` → 11 passed (Python 3.11); `npm run build` ok, `dist/` contains only w3.org XML namespace URIs; `docker compose config --services` → `backend, frontend` and `--profile gpu config --services` → `backend-gpu, frontend` (flag replaces `COMPOSE_PROFILES=cpu`, so the fallback in Design Notes is not needed). Earlier run: CPU image built and served `/api/health` ok/cpu/all true.
- OpenCV choice recorded: emotiefflib installed `--no-deps`, `opencv-python-headless==5.0.0.93` used instead of `opencv-python` (no libGL).
- Not yet verified (Docker daemon was stopped and needs sudo to start): GPU image build + `--profile gpu` health (`cuda`, only GPU backend in `docker compose ps`), offline restart check, fresh `docker compose up --build` after this run, manual browser check.

- Docker verification (lead, 2026-10-01): host VPN `amn0` (MTU 1280) breaks pip TLS inside bridge-network builds; images built with `docker build --network host` (documented in AGENTS.md), then `compose up --no-build`. CPU: health ok/cpu/4 models; only `backend`+`frontend` running. GPU: image 17.6 GB, ort 1.23.2 with CUDA EP, torch 2.9.1+cu126; health ok/cuda/gpu_name RTX 3050 Laptop; only `backend-gpu`+`frontend` running. GPU image without `--gpus`: device cpu + warning cuda_unavailable. Offline: both images with `--network none` return identical health. README disk estimate corrected to ~25 GB for GPU.

## Spec Change Log

## Review Triage Log

Pass 1 (2026-10-01): layers blind-hunter, edge-case-hunter, verification-gap. Patches applied by implementer, re-verified by lead: pytest 17 passed; frontend build ok, dist only w3.org; CPU/GPU health ok; API process has no torch mapped; GPU image without GPU → cpu + cuda_unavailable. Note: `requirements-dev.txt` `-c requirements.txt` works with uv only (pip rejects constraints with extras for `uvicorn[standard]`); docs prescribe uv.

| # | Layer | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|---|
| 1 | blind, edge | `probe_cuda` initializes CUDA in the API process | medium | `torch.cuda.get_device_name` runs `_lazy_init` in the uvicorn process. A later `fork` worker hits "Cannot re-initialize CUDA". The VRAM part is false: GPU memory measured 175 MiB with the backend up and 172 MiB with it stopped. | patch |
| 2 | edge | Health reports cuda although the ORT CUDA EP is unavailable | low | Same root as #1. The probe checks torch only. | patch (with #1) |
| 3 | verif-gap, blind | `fetch_models.py` checksum paths untested | medium (gap) | Pre-verified by the layer. No test imports the script. | patch |
| 4 | blind, edge | `--only` uses `nargs="*"`, so an empty list fetches all | low | `if args.only and ...`: an empty list is falsy. Direct correction. | patch |
| 5 | blind, edge | `--proxy-headers` without `--forwarded-allow-ips` | low | Uvicorn trusts only 127.0.0.1 by default, and nginx is on another container IP. One-flag fix. | patch |
| 6 | verif-gap, blind, edge | `.gitignore` `data/` and `models/` are unanchored | low | `git check-ignore` matches nested dirs (shown by the layer). Direct correction. | patch |
| 7 | blind | `*.tar` from README `docker save` is not ignored | low | An 18 GB tar in the repo root could be committed. One line. | patch |
| 8 | edge | `.models/` is not ignored | false | `.gitignore` already lists `.models/`. | reject |
| 9 | blind | `frontend/package-lock.json` is missing from the diff | false | The lead excluded it from the review diff on purpose. The file exists and is not ignored (`git check-ignore`: none). | reject |
| 10 | blind | README `docker save` names `backend:gpu`, which may not exist; target `up` may try to build | low | `docker save` fails with "No such image" when the GPU image was never built. Doc fix. | patch |
| 11 | blind | `requirements-dev.txt` re-pins fastapi | low | The two pins drift independently. Fix: `-c requirements.txt`. | patch |
| 12 | blind, edge | Unknown warning code is shown as a raw key; payload unchecked; no fetch timeout | low | `WARNINGS[...] ?? \`Предупреждение: ${code}\`` shows English text. `health.models[k]` throws on a missing `models`. Small direct fixes. | patch |
| 13 | blind | `.env` duplicates `.env.example`, including the "Copy to .env" line | low | The header is wrong inside `.env`. One-line fix. | patch |
| 14 | blind | GPU image runs Python 3.12; docs say 3.11 "как в образе" | low | Ubuntu 24.04 base. The spec allows ≥3.11. The fix touches AGENTS.md (agent context). | defer |
| 15 | blind | GPU image duplicates CUDA/cuDNN (17.6 GB) | medium | The torch cu126 wheels bundle `nvidia-*` libraries on top of the cudnn-runtime base, which the spec prescribed. Slimming is not trivial. | defer |
| 16 | blind | No compose healthcheck / `depends_on` | low | nginx resolves lazily, so the stack works. Adding healthchecks is new surface. Rarely hit. | reject |
| 17 | blind | Unknown `APP_VARIANT` silently becomes cpu | low | Set only by the Dockerfile ENV. A user is unlikely to meet it. The guard adds a branch. | reject |
| 18 | blind | SER weights are checked with MD5 only | low | Upstream publishes only MD5 (`gigaam/__init__.py`), and the pinned CDN URL limits exposure. Adding SHA-256 is extra pinning. | reject |
| 19 | blind | AGENTS.md dev section suggests publishing backend port 8000 without a snippet; offline check leaves anonymous volumes (`rm -f` vs `rm -fv`) | low | Real. The fix edits an agent-context file. | defer |
| 20 | verif-gap | Frontend health rendering has no automated test | gap | Pre-verified: no test runner in `frontend/`. The layer's disposition is defer until the live-view story. | defer |
| 21 | verif-gap | Profile "exactly one backend" is checked manually only | gap | Pre-verified. There is no CI harness yet. | defer |
| 22 | edge | Switching profiles without `down` leaves both backends behind the alias | low | Real, but README "Переключение между CPU и GPU" instructs `down` first. A guard would need new startup logic. | reject |
| 23 | edge | GPU profile on a host without the NVIDIA toolkit never starts, so the fallback warning is unreachable | medium | `gpus: all` makes container creation fail. The matrix row "GPU container, CUDA unavailable" covers an in-container failure only. The hard-failure UX belongs to story 1.7 (fallback hardening). | defer |
| 24 | edge | `/data` not writable crash-loops | low | Only a named volume is used, and the image chowns `/data` to the app user. Unlikely. A guard adds a branch. | reject |
| 25 | edge | Truncated weight file shows health ok | false | Weights are SHA/MD5-verified at build, and health checks presence by spec ("Health reads presence only"). | reject |
| 26 | edge | `is_file()` raises on EACCES, so health returns 500 | low | `chmod -R a+rX /models` at build. Unlikely at runtime. Guard needed. | reject |
| 27 | edge | A `None` CUDA probe is cached for the process lifetime | low | Restarting the container re-probes. Rarely hit. Expiry logic needed. | reject |
| 28 | edge | `ws://` / protocol-relative URLs bypass the dist grep | low | The current bundle has none. Widening the guard is new logic. Rarely hit. | reject |
| 29 | edge | Target path is a directory, so `replace` raises IsADirectoryError | false | That is a loud build failure on an unreachable state (fresh `/models` in the image), which is acceptable behaviour. | reject |
| 30 | edge | Empty `MODELS_DIR`/`DATA_DIR` resolves to cwd | low | Set only by the Dockerfile ENV. Unlikely. | reject |


## Design Notes

Model layout: `/models/yunet/face_detection_yunet_2026may.onnx`, `/models/fer/enet_b0_8_va_mtl.onnx`, `/models/vad/silero_vad.onnx`, `/models/ser/emo.ckpt` (+ GigaAM config if needed). Health reads presence only.

Profile selection: compose services without `profiles` always start. Give the CPU backend `profiles: ["cpu"]`, the GPU backend `profiles: ["gpu"]` (both with network alias `backend`), and set `COMPOSE_PROFILES=cpu` in a committed `.env`. Then verify that `--profile gpu` replaces the env value rather than adding to it. If it adds, fall back to a single `backend` service whose image, build arg and device request come from an override file that `--profile gpu` cannot express. In that case, stop and report: the commands in Always would change.

## Verification

**Commands:**
- `cd backend && pytest -q` -- expected: all pass.
- `docker compose up -d --build && curl -s localhost/api/health` -- expected: CPU JSON per the matrix.
- `docker compose --profile gpu up -d --build && curl -s localhost/api/health && docker compose ps` -- expected: `cuda`, and only a GPU backend running.
- `grep -rE "https?://" frontend/dist` -- expected: no output, or only `localhost` / XML namespaces.

**Manual checks:**
- Open `http://localhost`: the Russian page shows the device and four ✓.
