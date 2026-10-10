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
           "fused":{"anger":0.01,...},"fused_label":"happiness","confidence":0.81,
           "decision":"none","face_px":112}],
 "audio":{"speech":true,"probs":{"anger":0.1,"disgust":0.0,"fear":0.0,"happiness":0.25,"sadness":0.05,"surprise":0.0,"neutral":0.6},"label":"neutral","window_end_ts":...}}
```

- `decision` is one of `alert, watch, abstain, positive, none` (rules in `stack.md` Emotion events). `face_px` is the face box width in pixels of the frame the detector saw (detection size), not of the JPEG sent to the client.
- `audio.probs` always holds all 7 emotion keys. SER has 4 classes, so `disgust`, `fear` and `surprise` are always 0.
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
| Emotion event | Text JSON, sent for events from every source (RTSP, webcam, file): `{"type":"event","event":{...event object...}}` |
| System load | Text JSON, once per second: `{"type":"system","server_ts":..,"cpu_pct":..,"ram_used_mb":..,"ram_total_mb":..,"gpu_util_pct":..|null,"vram_used_mb":..|null,"vram_total_mb":..|null}` |

**Client → server.** Text JSON. `{"type":"subscribe","stream_ids":["cam1","cam2"]}` limits the subscription; with no subscribe message, the client receives all streams.

- In `stream_result`, `box` is in pixels of the analysed-frame JPEG sent just before it; `face_px` stays in pixels of the frame the detector saw (the stream's detection size).
- Latency per stream is `client_render_time − decode_ts`. The server clock is used for `decode_ts`, so the client corrects it by the offset measured from `system` messages, which carry `server_ts`.
- Only the latest analysed frame per stream is sent. If the client falls behind, older frames are dropped.

## REST `/api`

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | `{"status":"ok","device":"cuda"|"cpu","models":{...loaded}}` |
| POST | `/api/files` | multipart upload; returns `{"session_id"}` and starts analysis |
| GET | `/api/streams` | list streams: `id, name, site, url (credentials masked), camera_type, settings, state, fps, has_audio, source_resolution ([w,h] or null), decode ("nvdec"|"cpu"|null), session_id` |
| POST | `/api/streams` | `{"name":"Камера 1","site":"Тверская","url":"rtsp://...","camera_type":"webcam"|"sector"|"security","settings":{...}}`; `site` defaults to `""`, `camera_type` to `sector`, `settings` to defaults. Returns `{"id"}`, starts ingest and analysis (unless `settings.enabled` is false), creates a session of kind `rtsp` |
| PATCH | `/api/streams/{id}` | partial `{name, site, url, camera_type, settings}`; changes to `url`, `camera_type` or `settings` restart only this stream's source. Returns the stream object |
| POST | `/api/streams/{id}/reconnect` | reconnect now, resetting backoff |
| POST | `/api/streams/test` | `{"url"}` → `{"ok":true,"resolution":[w,h],"has_audio":true}` or `{"ok":false,"error":"..."}`; opens the source for at most 5 s |
| GET | `/api/streams/{id}/log?interval_s=10&limit=50` | interval journal of the stream's current session, newest first: `[{"from_ts","to_ts","clients","dominant_emotion","negative_prob","decision"}]`. `clients` = distinct tracks, `negative_prob` = mean negative-group probability, `decision` = worst face decision in the interval. `interval_s` defaults to the journal setting |
| DELETE | `/api/streams/{id}` | stop and remove the stream; its session becomes finished and exportable |
| GET | `/api/system` | same payload as the `system` WS message |
| GET | `/api/sessions?kind=` | list sessions: id, kind (`live`, `file` or `rtsp`), stream_id, created_at, duration, status; file sessions add `file_name`, `progress` |
| GET | `/api/sessions/{id}` | status, progress 0–1, notes, summary (dominant emotion share, faces seen, speech share, `usable_pct`); file sessions add `file_name`, `duration_ms`, `resolution` [w,h] |
| GET | `/api/sessions/{id}/timeline` | full per-frame results plus audio windows; body below (Session timeline) |
| GET | `/api/sessions/{id}/media` | the uploaded file, for playback (file sessions only) |
| GET | `/api/sessions/{id}/export?format=csv|json` | download |
| DELETE | `/api/sessions/{id}` | remove session data |
| GET | `/api/events?kind=negative|positive&state=new|confirmed|false_alarm&stream_id=&session_id=&from_ms=&to_ms=&limit=200&before_id=` | event objects, newest first |
| POST | `/api/events/{id}/ack` | `{"state":"confirmed"|"false_alarm"}` for a negative event; returns the event and broadcasts it again as an `event` message |
| GET | `/api/settings` | `{"events":{"negative_emotions":["anger","disgust","fear","sadness"],"negative_threshold":0.6,"positive_threshold":0.6,"min_duration_s":2,"cooldown_s":30},"journal":{"interval_s":10}}` |
| PUT | `/api/settings` | same object, validated against the ranges in `ui.md`; applies to new frames without restart |
| GET | `/api/events/export?format=csv|json` (same filters) | download the journal |

Stream `settings`: `{"enabled":true,"analysis_fps":3|5|8,"detection_size":"auto"|640|1280|1920|2560,"min_face_px":40|48|56|64}`; defaults `true, 5, "auto", 48`. `"auto"` means the camera type's detection size (`camera-profiles.md`). Webcam and file sessions use `min_face_px` 48.

Errors return HTTP 4xx/5xx with `{"detail":"..."}` (FastAPI default).

File-session `status` takes one of these values: `queued`, `processing`, `done`, `failed`. The `notes` array may contain `no_audio_track` or `no_faces_detected`.

## Session timeline

`GET /api/sessions/{id}/timeline` returns one JSON object:

```json
{"session_id":"...","kind":"file","duration_ms":412000,
 "frames":[{"ts_ms":1200,"width":640,"height":360,"faces":[...same face objects as frame_result...]}],
 "audio":[{"window_end_ts":3000,"speech":true,"probs":{...7 keys...},"label":"neutral"}]}
```

- `frames` and `audio` are sorted by time. A frame with no faces is kept (empty `faces`), so gaps without faces are visible.
- `width` and `height` are the size of the image the `box` coordinates refer to: the sent JPEG for webcam sessions, the analysed frame for RTSP and file sessions.
- `ts_ms` and `window_end_ts` use the session's clock: client `capture_ts` for webcam, server clock for RTSP, media milliseconds from the clip start for files.
- `audio` holds one entry per distinct window, with the same 7-key `probs` and silence rule as the WS audio block; it is empty for files with no audio track.
- `duration_ms` is the clip length for files and the session length otherwise.

## Export row (CSV columns; JSON = array of same objects)

`session_id, stream_id, ts_ms, source(face|audio), track_id, box_x, box_y, box_w, box_h, p_anger, p_disgust, p_fear, p_happiness, p_sadness, p_surprise, p_neutral, face_label, audio_label, fused_label, speech`

- `stream_id` is empty for webcam and file sessions.
- Face rows carry box and face probabilities.
- Audio rows carry the audio probabilities, mapped to the 7 keys with zeros where there is no audio counterpart, plus `audio_label` and `speech`. Their box columns are empty.

## Emotion event (CAP-11)

```json
{"event_id":"...","kind":"negative","session_id":"...","stream_id":"cam7","camera_type":"sector",
 "track_id":3,"start_ts":1727780000123.4,"emit_ts":1727780002301.0,"duration_ms":2100,
 "emotion":"anger","group_prob":0.74,"state":"new"}
```

- `state` is `new`, `confirmed` or `false_alarm` for negative events and null for positive ones. A new negative event starts as `new`.
- `kind` is `negative` or `positive`. `emotion` is the strongest emotion of the group over the held interval. `group_prob` is the mean group probability over that interval.
- `stream_id` and `camera_type` are null for webcam and file sessions. Timestamps use the server clock for RTSP streams, the client `capture_ts` for webcam sessions, and media milliseconds from the start of the clip for file sessions (the same clock as the file timeline and export `ts_ms`).
- Export columns (CSV; JSON = array of the same objects): `event_id, kind, session_id, stream_id, camera_type, track_id, start_ts, emit_ts, duration_ms, emotion, group_prob, state`.
