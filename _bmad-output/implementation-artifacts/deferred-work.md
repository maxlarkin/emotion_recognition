# Deferred work

- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/1-repository-skeleton-and-docker-compose.md`
  summary: GPU profile on a host without NVIDIA toolkit/driver fails at container creation (`gpus: all`), so the in-app cuda_unavailable fallback is never reached.
  evidence: compose device request fails before the process starts; matrix row only covers CUDA failing inside a running container. Belongs to story 1.9 (offline and cross-platform hardening, GPU→CPU fallback); numbered 1.7 before the CAP-10 re-breakdown.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/3-live-webcam-analysis-with-overlay-and-latency-readout.md`
  summary: After WorkerDied the InferenceHub stays failed until the backend restarts; /api/health still reports ok, so a dead or still-loading worker is invisible until the user presses Start.
  evidence: inference_hub._drain sets FAILED and never restarts; health checks only model files. Restart policy or a hub-state signal (health contract change via bmad-spec) belongs to story 1.9 hardening.
