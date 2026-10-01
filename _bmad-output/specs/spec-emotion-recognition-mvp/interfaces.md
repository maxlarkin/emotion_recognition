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

## REST `/api`

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{"status":"ok","device":"cuda"|"cpu","models":{...loaded}}` |
| POST | `/api/files` | multipart upload; returns `{"session_id"}` and starts analysis |
| GET | `/api/sessions` | list sessions: id, kind (`live` or `file`), created_at, duration, status |
| GET | `/api/sessions/{id}` | status, progress 0–1, summary (dominant emotion share, faces seen, speech share) |
| GET | `/api/sessions/{id}/timeline` | full per-frame results (same face object as WS) plus audio windows |
| GET | `/api/sessions/{id}/media` | the uploaded file, for playback (file sessions only) |
| GET | `/api/sessions/{id}/export?format=csv|json` | download |
| DELETE | `/api/sessions/{id}` | remove session data |

File-session `status` takes one of these values: `queued`, `processing`, `done`, `failed`. The `notes` array may contain `no_audio_track` or `no_faces_detected`.

## Export row (CSV columns; JSON = array of same objects)

`session_id, ts_ms, source(face|audio), track_id, box_x, box_y, box_w, box_h, p_anger, p_disgust, p_fear, p_happiness, p_sadness, p_surprise, p_neutral, face_label, audio_label, fused_label, speech`

- Face rows carry box and face probabilities.
- Audio rows carry the audio probabilities, mapped to the 7 keys with zeros where there is no audio counterpart, plus `audio_label` and `speech`. Their box columns are empty.
