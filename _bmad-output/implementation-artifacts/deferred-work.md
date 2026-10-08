# Deferred work

- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/1-repository-skeleton-and-docker-compose.md`
  summary: GPU profile on a host without NVIDIA toolkit/driver fails at container creation (`gpus: all`), so the in-app cuda_unavailable fallback is never reached.
  evidence: compose device request fails before the process starts; matrix row only covers CUDA failing inside a running container. Belongs to story 1.9 (offline and cross-platform hardening, GPU→CPU fallback); numbered 1.7 before the CAP-10 re-breakdown.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/3-live-webcam-analysis-with-overlay-and-latency-readout.md`
  summary: After WorkerDied the InferenceHub stays failed until the backend restarts; /api/health still reports ok, so a dead or still-loading worker is invisible until the user presses Start.
  evidence: inference_hub._drain sets FAILED and never restarts; health checks only model files. Restart policy or a hub-state signal (health contract change via bmad-spec) belongs to story 1.9 hardening.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/5-session-persistence-and-csv-json-export.md`
  summary: No test checks that the «Сессии» list refreshes when a live session starts or ends (LiveView `onSessionStart`/`onSessionEnd` → `refreshKey` → reload, plus the reload 1 s later).
  evidence: The frontend has no DOM or component-interaction harness (no jsdom or testing-library). `SessionsView.test.tsx` renders static markup only. For now the manual browser check covers it.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/5-session-persistence-and-csv-json-export.md`
  summary: The session store's write-queue cap (`QUEUE_MAX_FRAMES`, frame records dropped once the writer falls behind) is untested.
  evidence: No test fills the queue. Single-stream live never reaches the cap. Add the test before the 20-stream stories (1.6, 1.7) put real load on the writer.
