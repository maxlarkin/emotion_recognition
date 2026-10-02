# Deferred work

- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/1-repository-skeleton-and-docker-compose.md`
  summary: GPU backend image is 17.6 GB because torch cu126 wheels bundle nvidia-* CUDA/cuDNN libraries on top of the cudnn-runtime base image.
  evidence: `docker images` shows backend:gpu 17.6 GB; spec prescribes the cudnn-runtime base, so slimming (cuda -base or slim Python + pip-provided libs) needs a spec change and re-verification of onnxruntime-gpu library loading.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/1-repository-skeleton-and-docker-compose.md`
  summary: GPU profile on a host without NVIDIA toolkit/driver fails at container creation (`gpus: all`), so the in-app cuda_unavailable fallback is never reached.
  evidence: compose device request fails before the process starts; matrix row only covers CUDA failing inside a running container. Belongs to story 1.7 (GPU→CPU fallback hardening).
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/1-repository-skeleton-and-docker-compose.md`
  summary: GPU image runs Python 3.12 (Ubuntu 24.04) while CPU image, local venv and AGENTS.md state 3.11.
  evidence: Dockerfile base-gpu installs distro python3; AGENTS.md says "3.11 (как в образе)". Either install 3.11 in the GPU image or correct AGENTS.md/README.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/1-repository-skeleton-and-docker-compose.md`
  summary: AGENTS.md dev section says "пробросить порт 8000" without a snippet, and its offline check uses `docker rm -f` which leaves anonymous /data volumes (use `rm -fv`).
  evidence: compose publishes no backend port; image declares VOLUME /data.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/1-repository-skeleton-and-docker-compose.md`
  summary: No automated test for frontend health rendering (Russian warning text, missing-model list).
  evidence: frontend has no test runner; add vitest + testing-library with the first story that adds real UI logic (1.3).
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/1-repository-skeleton-and-docker-compose.md`
  summary: "Exactly one backend per compose command" is verified manually only.
  evidence: add a script asserting `docker compose config --services` = {backend, frontend} and with `--profile gpu` = {backend-gpu, frontend} once a CI/check harness exists.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/2-inference-core-with-per-stage-benchmark.md`
  summary: interfaces.md shows the audio block `probs` with 4 keys (anger, sadness, neutral, happiness) while the inference core emits all 7 emotion keys (zeros for disgust, fear, surprise) per the story 1.2 matrix.
  evidence: AudioResult.probs from inference.types/worker carries 7 keys; WS stories (1.3/1.4, 1.6) must follow one shape — update interfaces.md via bmad-spec to 7 keys.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/2-inference-core-with-per-stage-benchmark.md`
  summary: stack.md fusion text ("classes without an audio counterpart use the face probability only; renormalise after fusion") reads differently from the implemented and matrix-pinned `0.7·face + 0.3·audio` (zeros on face-only classes).
  evidence: inference/fusion.py fuse(); story 1.2 matrix row "Speech window". Clarify stack.md via bmad-spec.
