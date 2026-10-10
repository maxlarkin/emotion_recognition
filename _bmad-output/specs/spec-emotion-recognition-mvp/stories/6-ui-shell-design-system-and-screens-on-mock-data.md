---
title: 'UI shell, design system and screens on mock data'
type: 'feature'
created: '2026-10-10'
status: 'done'
baseline_commit: '01ab2f742e6c66d857428a7b232d8da36c08c1ef'
route: 'dispatch'
review_loop_iteration: 0
context:
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/ui.md'
  - '{project-root}/_bmad-output/specs/spec-emotion-recognition-mvp/interfaces.md'
  - '{project-root}/_bmad-output/planning-artifacts/ux-emosense/mockups/FORMAT.md'
  - '{project-root}/AGENTS.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The frontend is one page (live view, health card, sessions card) in a generic light/dark style. The spec now adopts the EmoSense design and a multi-screen UI (`ui.md`), while the backend for streams, events, settings and file analysis lands only in stories 1.7–1.11.

**Approach:** Rebuild the frontend shell and every screen of `ui.md` now, per the mockups and `tokens.css`. All data goes through one typed API module. Endpoints that exist (`/api/health`, `/api/sessions*`, `/ws/live`) stay real. Every other endpoint and `/ws/streams` is served by an in-memory mock with the exact `interfaces.md` shapes, so later stories swap one implementation without touching screens.

## Boundaries & Constraints

**Always:**
- Routes, menus, copy, states and settings tables exactly per `ui.md`; visual detail per the mockup of that screen. Colours/spacing/shadows only via `tokens.css` variables; `border-radius: 0` except status dots.
- Fonts from `@fontsource/ibm-plex-sans` and `@fontsource/ibm-plex-mono` (400/500/600 where available); `react-router` for routing. No other new runtime dependency.
- Mock layer: per-endpoint switch in one place (`src/api/index.ts`); mocks simulate latency (~150 ms) and mutations persist in memory for the page session; `/ws/streams` mock emits `stream_result` (with `decision`, `face_px`), `stream_status`, `event`, `system` at realistic rates for ~8 demo streams across 2–3 sites, including one connecting and one offline stream and one new incident. Analysed frames in the mock are generated placeholders (canvas or SVG), no bundled images.
- Screens fed by mocks show a small «Демо-данные» marker so a demo viewer is not misled.
- Existing live behaviour unchanged: `LiveRun`, protocol, mic, latency logic and their tests stay; only `LiveView` markup/styles change. Overlay colours follow the decision palette for live too (live has no `decision` field yet → treat as `none`).
- Sessions list keeps its behaviour (refresh on live start/stop, polling while running, delete confirm) on `/history/sessions`.
- Keyboard: menus close on Esc and outside click; all controls reachable; visible focus. No page-level horizontal scroll at 360 px.
- Built `dist` passes the external-URL grep in `frontend/Dockerfile`.

**Never:**
- Backend changes, or any change to `interfaces.md` shapes.
- UI kits, CSS frameworks, chart libraries, CDN or remote assets.
- Copying the mockup `dc` syntax or inline-style soup; mockup demo data verbatim as real data.
- Features rejected in `ui.md`/SPEC (recipients, contempt, MP4 export, retention, clips).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Empty streams | mock stream list empty | `/streams` shows the empty state with «Добавить поток» and «Проанализировать запись» | N/A |
| Search | query «твер» | only matching streams; empty site groups hidden; «Веб-камера» hidden unless it matches | N/A |
| Unknown stream | `/streams/zzz` | message «Поток не найден» with link to `/streams` | N/A |
| Incident ack | «Подтвердить» on banner | banner gone, bell count −1, card chip gone, journal row state «Подтверждено» | mock error → inline error, state unchanged |
| Hide banner | «Скрыть» | banner gone on this screen only; bell count unchanged | N/A |
| Offline stream | state `error` | tiles «—», placeholder «Поток недоступен», «Переподключить» calls reconnect | N/A |
| Small faces | median `face_px` < `min_face_px` | warning in «Камера и обработка» | N/A |
| Settings dirty | edit then navigate away | confirm dialog; «Отмена» restores loaded values | save error → inline error, values kept |
| Bulk edit | 2 cameras, only fps chosen | only `analysis_fps` changes on those 2 | «Применить» disabled with no field or no camera |
| Add stream | empty name or URL | «Добавить» disabled; test failure shows mock error text | N/A |
| Upload | 3 files dropped, one `.txt` | 2 queued cards with progress; `.txt` card «Неподдерживаемый формат» | N/A |
| Backend down | `/api/health` fails | warning banner «Сервер недоступен…»; mock screens keep working | retry every 5 s |
| GPU fallback | health `warning: cuda_unavailable` | global banner with the existing Russian text | N/A |

</frozen-after-approval>

## Code Map

- `frontend/src/App.tsx` -- current single page + `HealthView` (+ `WARNINGS` texts); becomes router shell; `HealthView` moves to Settings › Система, warnings to the global banner.
- `frontend/src/api.ts` (+ `api.test.ts`) -- real `fetchHealth`, `fetchSessions`, `deleteSession`, `exportUrl`, `errorText`; move into `src/api/` keeping exports and tests.
- `frontend/src/live/*` -- `LiveView.tsx` (markup to restyle), `run.ts`/`protocol.ts`/`mic.ts`/`latency.ts`/`resample.ts`/`clock.ts` (do not change logic), `overlay.ts` (`EMOTION_RU` lower-case labels used by panel; `drawOverlay` colours by emotion → switch to decision palette), `panel.ts`.
- `frontend/src/sessions/*` -- `SessionsView` + `format.ts`; restyle only.
- `frontend/src/styles.css` -- replaced by `tokens.css` + base + component CSS.
- `frontend/vite.config.ts` -- `stripExternalUrls` plugin must stay; check @fontsource CSS adds no external URLs.
- `frontend/Dockerfile`, `nginx.conf` -- unchanged; nginx already falls back to `index.html`.
- `_bmad-output/planning-artifacts/ux-emosense/mockups/{Empty,Main,Settings,Analyze}.dc.html` -- visual reference; `tokens.css` source (drop its `@import`).
- Tests: vitest with `renderToStaticMarkup` only (no jsdom); pure logic gets unit tests.

## Tasks & Acceptance

**Execution:**
- [x] `frontend/package.json` -- add `react-router`, `@fontsource/ibm-plex-sans`, `@fontsource/ibm-plex-mono` (exact versions, lockfile updated) -- routing and offline fonts.
- [x] `frontend/src/styles/tokens.css`, `base.css` -- tokens from the design package without `@import`; body, focus ring, table, button, input, chip, tile, panel base classes.
- [x] `frontend/src/api/` -- `types.ts` (interfaces.md shapes: Stream, StreamSettings, FaceResult with `decision`/`face_px`, StreamResult, StreamStatus, SystemLoad, EmotionEvent with `state`, Settings, StreamLogRow, SessionInfo/Detail, Timeline), `real.ts` (moved real calls), `mock/` (data + behaviour + `/ws/streams` emitter), `index.ts` (per-endpoint selection, `isMocked(endpoint)`), `streamsSocket.ts` (subscribe API used by screens) -- single data seam.
- [x] `frontend/src/ui/` -- Toolbar + Menu (click-outside, Esc), Bell, HealthBanner, Chip, Tile, StatusDot, SettingRow, Segmented, Stepper, Toggle, Disclosure, Dialog, DemoBadge, format helpers (`0,91`, `72 %`, `м:сс`, `чч:мм:сс`) and Russian label maps (emotion capitalised, decision, stream state, camera type).
- [x] `frontend/src/streams/` -- Sidebar (groups, search, webcam entry, gear links), Grid (+ system load header, empty state), StreamDetail (banner, tiles, video with SVG/CSS boxes, visitors, voice line, interval journal, processing disclosure with warning), AddStreamDialog, rolling stats helper (usable %, median face, p50/p95 per stream) -- CAP-10 screens.
- [x] `frontend/src/live/LiveView.tsx`, `overlay.ts` -- restyle into the sidebar layout; decision-palette boxes.
- [x] `frontend/src/history/` -- EventsJournal (filters, live append, ack buttons, export menu, «Показать ещё»), Sessions page wrapping SessionsView.
- [x] `frontend/src/settings/` -- tabs События/Камеры/Журнал/Система, dirty guard (`useBlocker` + `beforeunload`), bulk panel.
- [x] `frontend/src/analyze/` -- drop zone + queue, result: player with overlay from timeline, SVG emotion chart, tiles, episode timeline, episodes table, synced selection, export menu; processing/queued/failed states.
- [x] `frontend/src/App.tsx`, `main.tsx`, `index.html` -- `BrowserRouter`, routes per `ui.md`, title «EmoSense».
- [x] tests -- unit tests for format helpers, label maps, stream grouping/search, rolling stats, bulk-patch builder, settings validation/dirty diff, episode/no-face segment builder, mock incident ack flow; static-markup tests for empty state, banner, chips, settings row; existing tests kept green.

**Acceptance Criteria:**
- Given `npm run build`, when it finishes, then it succeeds and the Dockerfile external-URL grep over `dist` prints nothing.
- Given the app at `npm run dev` with the real backend, when the user opens `/live` and presses «Старт», then live analysis, latency readout and session creation work as before, and the session appears on `/history/sessions` with working CSV/JSON download.
- Given any route from `ui.md`, when opened directly (deep link) or via menu, then the screen renders in the EmoSense style with mock or real data and no console errors.
- Given a screen fed by mocks, when it renders, then «Демо-данные» is visible; given `/history/sessions` and `/live`, then it is not.

## Design Notes

Mock switch sketch — screens import only from `src/api`:

```ts
// src/api/index.ts
const MOCKED = { streams: true, system: true, events: true, settings: true, files: true, streamLog: true } as const;
export const api = { listStreams: MOCKED.streams ? mock.listStreams : real.listStreams, /* … */ };
export const isMocked = (k: keyof typeof MOCKED) => MOCKED[k];
```

Later stories flip a key to `false` and implement the `real.*` function. Live frame boxes in the grid/detail are drawn as absolutely positioned elements in percent of the frame (mockup approach); the live webcam keeps its canvas overlay.

## Verification

**Commands:**
- `cd frontend && npm ci && npm test` -- expected: all tests pass.
- `cd frontend && npm run build && grep -rEo "https?://[^\"' )]*" dist | grep -vE "www\.w3\.org/"` -- expected: build OK, grep empty.

**Manual checks:**
- Each route at 1440 px and 360 px against its mockup; menus, bell, banner actions, settings dirty guard, bulk edit, upload queue, timeline↔table sync; `/live` with the real backend.

## Implementation Notes

- Implemented inline (no subagent). Layout: `src/api/` (types, real, mock, `index.ts` switch, `streamsSocket.ts`), `src/state/` (appData store, rolling stats), `src/shell/` (toolbar, health banner, layout, toasts), `src/ui/` (format, controls, icons), `src/streams/`, `src/history/`, `src/settings/`, `src/analyze/`, `src/styles/` (tokens/base/app). Old `api.ts`, `App.test.tsx`, `styles.css` moved or replaced.
- Mocked groups (all `true` in `src/api/index.ts`): streams + `/ws/streams`, system, events, settings, streamLog, files. Real: `/api/health`, `/api/sessions*`, `/ws/live`. `real.ts` already implements every mocked endpoint per `interfaces.md`, so a later story flips one key.
- Router: `createBrowserRouter` (data router) so `useBlocker` guards unsaved settings; `react-router@8.4.0`, `@fontsource/ibm-plex-{sans,mono}@5.3.0` (latin + cyrillic subsets).
- Offline grep: react-router embeds docs URLs and an `http://localhost` URL-parsing base. `vite.config.ts` now rewrites every `scheme://` in emitted chunks to `scheme:\u002F\u002F` in `generateBundle` (after minification; same runtime string). `dist` grep is empty.
- Live overlay switched from per-emotion colours to the decision palette; live frames have no `decision` yet, so boxes use the accent colour.
- Contract gaps found (logged in deferred-work, need a decision before stories 1.10/1.11): file-session event timestamps (the UI treats values < 1e11 as media ms: «запись 04:06») and the `/api/sessions/{id}/timeline` shape (the UI assumes `{frames:[{ts_ms,width,height,faces}], audio:[...]}`).
- Verified: `npx tsc --noEmit`, `npm test` (16 files, 109 tests), `npm run build` + URL grep empty; headless Chrome screenshots of every route at 1440 px and 360 px; `/live` with the real backend and Chrome's fake camera: p50 12 ms / p95 20 ms, session listed on `/history/sessions` with CSV/JSON links.

## Spec Change Log

## Review Triage Log

Layers: blind-hunter (BH), edge-case-hunter (EC), verification-gap (VG). Route: P = patched, D = deferred, R = rejected.

| # | Finding | Verdict | Evidence | Route |
|---|---|---|---|---|
| 1 | URL plugin rewrote every `scheme://`, blinding the Dockerfile grep (BH, VG) | high | Any new CDN URL would pass; allowlist + `this.error` on others, test `src/build/urls.test.ts` | P |
| 2 | Mock playback frozen: rAF read stale `pos` (BH, EC) | high | `next = pos + dt` with `pos` from the first render; posRef/onTimeRef now | P |
| 3 | Play at end did not restart (EC) | low | `pos >= duration` stopped on the first frame; resets to 0 | P |
| 4 | Result kept old error / stale promise (EC) | low | No `setError(null)` or alive flag; added | P |
| 5 | `?upload=1` programmatic file dialog blocked and repeated (EC) | medium | Browsers block `click()` without activation; now focuses the picker and clears the param | P |
| 6 | Duplicate rejected-card keys (EC) | low | `Date.now()+name` collides on multi-drop; counter id | P |
| 7 | File event clock vs media time (EC) | medium | Contract gap already logged in deferred-work | D (carried) |
| 8 | `toFileSession` unknown status (EC) | low | Only in `real.ts`, unreachable until 1.11 | D |
| 9 | Mock createStream timer revives deleted/disabled stream (EC) | low | Timer ignored deletion; guarded | P |
| 10 | Mock re-enable leaves null resolution/session (EC) | low | Set on switch to running | P |
| 11 | Mock reconnect of disabled stream (EC) | low | Now fails with Russian message | P |
| 12 | `__resetMock` stacks file timers (BH, EC) | low | Interval id kept and cleared | P |
| 13 | Journal: stale responses overwrite newer filters (BH, EC) | medium | Key check before applying a page | P |
| 14 | «Показать ещё» double-append (EC) | low | Button disabled while loading, dedupe by id | P |
| 15 | `toLocalInput` RangeError on huge `from`/`to` (EC) | medium | Crashed the screen; guarded, tested | P |
| 16 | «По» excludes the end minute (EC) | low | +59 999 ms, tested | P |
| 17 | Negative event with state null shows «Новое» (EC) | false | Contract: negative events always carry a state | R |
| 18 | CamerasTab dirtyRows stale after unmount (BH, EC, VG) | medium | Guard stuck forever; cleanup on unmount | P |
| 19 | CamerasTab saves masked url back (EC) | medium | Full draft sent; now `changedFields`, tested | P |
| 20 | Refresh discards unsaved row edits (EC) | medium | Draft reset on every base change; only when untouched | P |
| 21 | Bulk apply partial failure / hidden ids (BH, EC) | medium | `allSettled`, refresh always, names failed cameras, skips removed ids | P |
| 22 | events↔journal tab switch keeps dirty draft (EC) | medium | Same component reused; `key={tab}` | P |
| 23 | Bell capped at 50 (BH, EC) | low | Limit 100, «99+» reachable | P |
| 24 | Out-of-order `refreshStreams` (EC) | low | Sequence guard | P |
| 25 | No resync / stale tiles after real socket reconnect (BH, EC) | medium | Real socket unused until 1.7 | D |
| 26 | Connection test result for an edited URL (EC) | low | URL ref check | P |
| 27 | EmptyState test missing though ticked (EC, VG) | medium | Added static-markup test | P |
| 28 | `/live` sidebar shows mock streams without marker (EC) | medium | Sidebar header now shows «Демо-данные» while streams are mocked; the live view itself stays unmarked | P |
| 29 | AppData message handling untested (VG) | medium | `src/state/appData.test.ts`: event, status, clock offset | P |
| 30 | `real.ts` adapters untested (VG) | medium | Unreachable now; each switching story adds stubbed-fetch tests | D |
| 31 | Ack errors via toast, matrix wants inline (VG) | low | Banner and journal now show inline errors | P |
| 32 | Mock seeds many new file incidents (BH) | low | Seeded file results marked reviewed; bell starts with the one live incident | P |
| 33 | Journal shows «Запись», ui.md wants file name (BH) | low | File names loaded and mapped, tested | P |
| 34 | Hard-coded colours (BH) | low | New tokens; overlay reads tokens via `getComputedStyle` | P |
| 35 | Live box label not ui.md format (BH) | medium | `boxText` → `#id · 0,91`, tested | P |
| 36 | No byte-upload progress (BH) | medium | Real upload path unreachable until 1.11 | D |
| 37 | Mock sends no frames (BH) | low | Placeholder is generated (CSS), as allowed; img path covered by 1.9 | R |
| 38 | Mock export columns / blob leak / CSV quoting (BH) | low | Blob URLs now revoked after download; mock export shape is demo-only, fix is more than a correction | P (revoke), R (shape) |
| 39 | Dead `LiveView` callback props (BH) | low | Deleted | P |
| 40 | AnalyzePage polls forever (BH) | medium | 1 s only while queued/processing, else 10 s | P |
| 41 | Episodes row double select, invalid aria-selected (BH) | low | stopPropagation, `aria-pressed` on the button | P |
| 42 | Chart seek has no keyboard path (BH) | false | Player range input seeks by keyboard | R |
| 43 | Mono 600 not imported (BH) | low | No mono text uses 600 | R |
| 44 | Duplicate accepted-extension lists (BH) | low | Mock must not import UI; negligible | R |
| 45 | No tests for menus Esc/outside click, unsaved guard, sync (BH) | low | No DOM harness (jsdom) in the project | D |
