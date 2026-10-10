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
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/7-rtsp-ingest-with-camera-profiles-and-stream-management.md`
  summary: No regression test covers the RTSP media-clock re-anchoring on a PTS jump.
  evidence: `sources/rtsp.run` re-anchors when media time moves more than 2 s, but the generated test clips cannot carry a PTS discontinuity (the muxer rejects non-monotonic timestamps). Feed a fake container or a mediamtx stream restart in story 1.8 to cover it.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/7-rtsp-ingest-with-camera-profiles-and-stream-management.md`
  summary: With 3 demo streams at 5 fps the worker delivers ~3.3–4 results/s per stream on GPU and ~2/s on CPU.
  evidence: All streams get the same rate, because the worker is bound by YuNet running on CPU at 1920/2560 detection sizes (backend at ~940 % CPU on the CPU image). This is the throughput work `stack.md` (Detection cost) assigns to story 1.8.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/8-twenty-camera-demo-set-and-throughput.md`
  summary: Throughput for the 20-stream mix (YuNet on the ORT CUDA EP, detection every Nth frame for large profiles, shared-memory frame transport, 10-minute before/after numbers) was split out of story 1.8.
  evidence: The user chose to split. The work is now story 1.9 «Throughput for the twenty-camera mix» in stories.yaml. Measured on the RTX 3050, YuNet CUDA takes 2.7/12/20.5 ms at 640/1920/2560, and the full mix with detection on every frame needs ~1.2 s of GPU per second.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/9-throughput-for-the-twenty-camera-mix.md`
  summary: The `CudaYuNet` decode (priors, `sqrt(cls·obj)`, NMS, clip) is only checked against OpenCV in the GPU image.
  evidence: `tests/test_detector_cuda.py` needs the CUDA EP. Run the same decode on the CPU EP against `cv2.FaceDetectorYN` so `pytest -m models` on any machine covers it.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/9-throughput-for-the-twenty-camera-mix.md`
  summary: The 10-min 20-stream run is slower than the 3-min runs (1.08 against 1.54–1.74 fps per stream).
  evidence: Per-round detection cost grew during the run. Candidates: GPU thermal throttling, VRAM fragmentation at the 4 GB limit, NVDEC contention. Record `nvidia-smi dmon -s pucm` over a 10-min run (story 1.14).
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/9-throughput-for-the-twenty-camera-mix.md`
  summary: During the story-1.9 measurements the backend container restarted once and one NVDEC open failed with a "generic error".
  evidence: The restart was not flagged as an OOM kill and the logs were lost on `down`. Keep `docker compose logs` before removing the stack in story 1.14 runs.
- source_spec: `_bmad-output/specs/spec-emotion-recognition-mvp/stories/9-throughput-for-the-twenty-camera-mix.md`
  summary: `bench/streams_ws.py` counts delivered `stream_result` messages, a lower bound of analysed fps when pushes are skipped.
  evidence: The server skips a push (not the analysis) when a stream's encoding falls behind. An analysed-frame counter in the stream row would give the exact rate.
