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
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/6-ui-shell-design-system-and-screens-on-mock-data.md`
  summary: interfaces.md does not say which clock file-session events use; the episode timeline needs media milliseconds, while the contract says server clock for all non-webcam sources.
  evidence: The story 1.6 UI treats event timestamps below 1e11 as media ms («запись мм:сс»). Decide via bmad-spec before story 1.10/1.11, e.g. "file-session event timestamps are media ms", and whether file incidents should count in the bell.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/6-ui-shell-design-system-and-screens-on-mock-data.md`
  summary: interfaces.md leaves the `/api/sessions/{id}/timeline` body unspecified ("full per-frame results plus audio windows").
  evidence: The story 1.6 player assumes `{frames:[{ts_ms,width,height,faces}], audio:[{window_end_ts,speech,probs,label}]}` (`frontend/src/api/types.ts` Timeline). Fix the shape in the spec before story 1.11.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/6-ui-shell-design-system-and-screens-on-mock-data.md`
  summary: The real `/ws/streams` client does not resync (refresh streams and incidents) after a reconnect and does not mark feeds stale while disconnected.
  evidence: `src/api/streamsSocket.ts` reconnects silently and `AppData` only reacts to messages. The path is unused while streams are mocked; fix it in story 1.7 when the real socket is switched on.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/6-ui-shell-design-system-and-screens-on-mock-data.md`
  summary: The `real.ts` adapters for streams, events, settings and files have no stubbed-fetch tests, and `toFileSession` maps unknown statuses to an empty label.
  evidence: Nothing reaches them while every `MOCKED` key is true. Each story that flips a key (1.7, 1.10, 1.11) adds `real.test.ts` cases for its endpoints first.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/6-ui-shell-design-system-and-screens-on-mock-data.md`
  summary: Uploads show no byte progress while the file is being sent; the queue card appears only after the POST returns.
  evidence: `real.uploadFile` uses `fetch`. Switch to XHR `upload.onprogress` with an «Загрузка N %» card state in story 1.11.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/6-ui-shell-design-system-and-screens-on-mock-data.md`
  summary: No interaction tests for toolbar menus (Esc, outside click), the unsaved-settings guard, or timeline/table/player sync.
  evidence: The frontend has no DOM harness (jsdom or testing-library); these were checked by hand in headless Chrome. Add a harness before UI behaviour grows further.
