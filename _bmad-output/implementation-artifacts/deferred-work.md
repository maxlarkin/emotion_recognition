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
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/3-live-webcam-analysis-with-overlay-and-latency-readout.md`
  summary: After WorkerDied the InferenceHub stays failed until the backend restarts; /api/health still reports ok, so a dead or still-loading worker is invisible until the user presses Start.
  evidence: inference_hub._drain sets FAILED and never restarts; health checks only model files. Restart policy or a hub-state signal (health contract change via bmad-spec) belongs to story 1.9 hardening.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/3-live-webcam-analysis-with-overlay-and-latency-readout.md`
  summary: frame_result proc_ms is asserted only as >= 0; no test ties it to receipt-to-send server time.
  evidence: test_ws_live.test_frame_with_face; a delayed fake-worker emit with proc_ms >= delay would pin it. Diagnostic field, CAP-1 uses client latency.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/3-live-webcam-analysis-with-overlay-and-latency-readout.md`
  summary: Client overlay clearing (faces=[] and the 1 s stale rule) has no automated test; only the manual browser check covers it.
  evidence: overlay.test.ts covers geometry and labels only; a drawOverlay test with a stub ctx plus a pure isStale() helper would close it.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/3-live-webcam-analysis-with-overlay-and-latency-readout.md`
  summary: The AGENTS.md offline-URL grep for frontend/dist excludes `localhost`, but the frontend Dockerfile guard does not, so a local check passed while the image build failed on UI text containing http://localhost.
  evidence: AGENTS.md Фронтенд section vs frontend/Dockerfile RUN npm run build guard; align AGENTS.md with the Dockerfile guard (agent-context edit).
