# Camera profiles

Three camera types from the use case. Each RTSP stream carries one `camera_type`, which sets its detection size. Face-size figures are estimates. CAP-7 and CAP-8 replace them with measured thresholds.

## Types

| `camera_type` | Real camera | Distance to client | Native resolution | Audio | Detection size (longest side) |
|---|---|---|---|---|---|
| `webcam` | webcam on the manager's PC, facing the client | 0.5–0.8 m | 1.2–1.5 MP (~1280×960) | usually yes | 640 |
| `sector` | high-resolution camera covering a room sector, 3–5 per large room | 2–3 m | 4K (3840×2160) | optional | 1920 |
| `security` | security dome camera | 4–8 m | ≤ 2K (2560×1440) | often none | 2560 |

The FER crop is taken from the frame at detection size. Tiles sent to the browser are downscaled to ≤ 640 px separately.

## Face size estimate

Formula: scene width `W = 2·d·tan(HFOV/2)`, pixel density `ppm = width_px / W`, face px `= 0.16 m · ppm`.

Assumed HFOV: webcam 70°, sector 90°, dome with 2.8 mm lens ~100°, dome with 6 mm lens ~50°.

| Type | Resolution (width) | Near distance | Far distance |
|---|---|---|---|
| webcam | 1280 | 0.5 m: ~290 px | 0.8 m: ~180 px |
| webcam | 640 | 0.5 m: ~145 px | 0.8 m: ~90 px |
| sector | 3840 | 2 m: ~155 px | 3 m: ~100 px |
| sector | 1920 | 2 m: ~75 px | 3 m: ~50 px |
| sector | 1280 | 2 m: ~50 px | 3 m: ~35 px |
| security, 2.8 mm | 2560 | 4 m: ~43 px | 8 m: ~21 px |
| security, 2.8 mm | 1920 | 4 m: ~32 px | 8 m: ~16 px |
| security, 6 mm | 2560 | 4 m: ~110 px | 8 m: ~55 px |

Reference (`research.md` §4.2, RAF-DB, 7 classes): 89% at 100 px, 88% at 50 px, 85% at 25 px, 77% at 12.5 px. Real-world data scores lower.

## Minimum resolution hypothesis (to confirm with CAP-7)

- **webcam:** 640×480 is enough; faces stay ≥ 90 px.
- **sector:** 1080p at ≤ 3 m (~50 px). 4K is recommended for ≥ 100 px.
- **security:** 2K with a 2.8 mm lens is at or below the reliable limit beyond ~4 m, and pitch often exceeds 30°. Lowering the resolution is not possible. Reliable recognition needs a narrower lens (≥ 6 mm), ROI streams, or a closer mount.

## 20-stream demo mix

| Type | Streams | Emulated stream | Faces in frame | Audio |
|---|---|---|---|---|
| webcam | 6 | 640×480, 10 fps | ≥ 90 px | yes |
| sector | 8 | 1920×1080, 10 fps (stands in for a 4K camera at 2–3 m after downscale) | 50–100 px | yes |
| security | 6 | 2560×1440, 10 fps; the clip is scaled down and padded so faces match 4–8 m | 25–45 px | no |

The mix is configurable. Each type's clips include moments of clear negative and positive emotion, so CAP-11 events can be shown.
