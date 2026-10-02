# Interfaces

These contracts are fixed between frontend and backend. All timestamps are in milliseconds. Emotion keys are always these seven: `anger, disgust, fear, happiness, sadness, surprise, neutral`.

## Live session — WebSocket `/ws/live`

**Client → server.** Each message is either binary or text.

| Kind | Format |
|---|---|
| Video frame | Binary: 1-byte type `0x01`, 8-byte float64 `capture_ts` (client clock, `performance.timeOrigin + performance.now()`), 4-byte uint32 `frame_id`, then the JPEG bytes. |
| Audio chunk | Binary: 1-byte type `0x02`, 8-byte float64 `capture_ts`, then PCM16 LE mono 16 kHz samples. |
| Control | Text JSON: `{"type":"start"}`, `{"type":"stop"}`, `{"type":"config","fusion":{"w_face":0.7,"w_audio":0.3}}` |

**Server → client.** Text JSON.

```json
{"type":"frame_result","frame_id":123,"capture_ts":1727780000123.4,"proc_ms":38.2,
 "faces":[{"track_id":1,"box":[x,y,w,h],"face":{"anger":0.02,...},"face_label":"happiness",
           "fused":{"anger":0.01,...},"fused_label":"happiness","confidence":0.81}],
 "audio":{"speech":true,"probs":{"anger":0.1,"sadness":0.05,"neutral":0.6,"happiness":0.25},"label":"neutral","window_end_ts":...}}
```

- Box coordinates are relative to the JPEG that was sent. The client scales them to the displayed video.
- When frames are dropped, the server answers only the latest frame. Dropped `frame_id`s get no reply.
- The client computes latency as `render_time − capture_ts` and shows p50 and p95.
- `{"type":"session","session_id":"..."}` is sent once after `start`. `{"type":"error","message":"..."}` is sent on failure.

## RTSP streams — WebSocket `/ws/streams`

This socket is for viewing only. The server pushes updates for every active stream, and one socket serves the whole grid.

**Server → client:**

| Kind | Format |
|---|---|
| Analysed frame | Binary: 1-byte type `0x11`, 2-byte uint16 `stream_id` length, then that many UTF-8 bytes of `stream_id`, 4-byte uint32 `frame_id`, then the JPEG bytes (longest side ≤ 640 px). |
| Frame result | Text JSON: `{"type":"stream_result","stream_id":"cam1","frame_id":123,"decode_ts":...,"emit_ts":...,"faces":[...same face objects as frame_result...],"audio":{...same audio block...},"fps":7.9}`. The binary frame with the same `frame_id` is sent just before it. |
| Stream status | Text JSON: `{"type":"stream_status","stream_id":"cam1","state":"connecting"|"running"|"reconnecting"|"stopped"|"error","message":"..."}` |
| System load | Text JSON, once per second: `{"type":"system","server_ts":..,"cpu_pct":..,"ram_used_mb":..,"ram_total_mb":..,"gpu_util_pct":..|null,"vram_used_mb":..|null,"vram_total_mb":..|null}` |

**Client → server.** Text JSON. `{"type":"subscribe","stream_ids":["cam1","cam2"]}` limits the subscription; with no subscribe message, the client receives all streams.

- Latency per stream is `client_render_time − decode_ts`. The server clock is used for `decode_ts`, so the client corrects it by the offset measured from `system` messages, which carry `server_ts`.
- Only the latest analysed frame per stream is sent. If the client falls behind, older frames are dropped.

## REST `/api`

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{"status":"ok","device":"cuda"|"cpu","models":{...loaded}}` |
| POST | `/api/files` | multipart upload; returns `{"session_id"}` and starts analysis |
| GET | `/api/streams` | list streams: `id, name, url (credentials masked), state, fps, session_id` |
| POST | `/api/streams` | `{"name":"Камера 1","url":"rtsp://..."}`; returns `{"id"}`, starts ingest and analysis, creates a session of kind `rtsp` |
| DELETE | `/api/streams/{id}` | stop the stream; its session becomes finished and exportable |
| GET | `/api/system` | same payload as the `system` WS message |
| GET | `/api/sessions` | list sessions: id, kind (`live`, `file` or `rtsp`), created_at, duration, status |
| GET | `/api/sessions/{id}` | status, progress 0–1, summary (dominant emotion share, faces seen, speech share) |
| GET | `/api/sessions/{id}/timeline` | full per-frame results (same face object as WS) plus audio windows |
| GET | `/api/sessions/{id}/media` | the uploaded file, for playback (file sessions only) |
| GET | `/api/sessions/{id}/export?format=csv|json` | download |
| DELETE | `/api/sessions/{id}` | remove session data |

File-session `status` takes one of these values: `queued`, `processing`, `done`, `failed`. The `notes` array may contain `no_audio_track` or `no_faces_detected`.

## Export row (CSV columns; JSON = array of same objects)

`session_id, stream_id, ts_ms, source(face|audio), track_id, box_x, box_y, box_w, box_h, p_anger, p_disgust, p_fear, p_happiness, p_sadness, p_surprise, p_neutral, face_label, audio_label, fused_label, speech`

- `stream_id` is empty for webcam and file sessions.
- Face rows carry box and face probabilities.
- Audio rows carry the audio probabilities, mapped to the 7 keys with zeros where there is no audio counterpart, plus `audio_label` and `speech`. Their box columns are empty.
