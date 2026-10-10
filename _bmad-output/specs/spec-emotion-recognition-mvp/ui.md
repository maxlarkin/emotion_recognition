# UI contract

Decisions of record for the frontend. Visual reference: `../../planning-artifacts/ux-emosense/` (mockups `*.dc.html`, `tokens.css`, `mockups/FORMAT.md`). Mockups show layout, spacing, colours and copy. Where they disagree with this file, this file wins. Mockup data is fake.

## Design system

- Product name in the toolbar: «EmoSense». All UI text is Russian. Decimal comma (`0,91`), percent with a space (`72 %`).
- Colours, spacing, shadows and sizes come only from `tokens.css` custom properties. Dark flat theme, `border-radius: 0` everywhere except status dots.
- Shadows only on open dropdowns and active or expanded elements (`--shadow-dropdown`, `--shadow-active`).
- Fonts: IBM Plex Sans (text), IBM Plex Mono (numbers, times, resolutions, addresses), bundled from `@fontsource` packages. The `@import` of Google Fonts in `tokens.css` is not used.
- No UI kit, no native `<select>` in settings (segmented buttons instead). Real `<button>`/`<a>`/`<input>`+`<label>`, `aria-label` on icon buttons, `aria-expanded` on disclosures, `aria-pressed` on toggles, visible focus.
- Below ~900 px the sidebar stacks above the content and tables scroll inside their block. No page-level horizontal scroll at 360 px.

## Routes and menus

| Route | Screen | Capability |
|---|---|---|
| `/` | redirect to `/streams` | — |
| `/streams` | Sidebar + grid of all RTSP tiles; empty state when there are no streams (mockup Empty) | CAP-10 |
| `/streams/:id` | Sidebar + stream detail (mockup Main) | CAP-10, CAP-11 |
| `/live` | Sidebar + browser webcam session (existing live view restyled) | CAP-1, CAP-2 |
| `/history/events` | Event journal | CAP-11 |
| `/history/sessions` | Sessions list with CSV/JSON export and delete | CAP-4 |
| `/settings/:tab` | Settings, tabs `events`, `cameras`, `journal`, `system`; `?camera=:id` opens that camera expanded | CAP-10, CAP-11 |
| `/analyze`, `/analyze/:sessionId` | File analysis (mockup Analyze) | CAP-3 |

Toolbar (sticky, 44 px): logo → `/streams`; four click-open menus; on stream screens a bell with the count of new incidents.

| Menu | Items |
|---|---|
| Потоки | Все потоки · Веб-камера · Добавить поток |
| История | Журнал событий · Сеансы и экспорт |
| Настройки | События · Камеры · Журнал · Система |
| Анализ видео | Загрузить запись · Результаты |

Menus: click toggles, clicking another button switches, click-outside and Esc close, an item acts and closes. Health warnings (GPU fallback, models loading, inference failed) show as a banner under the toolbar on every screen.

## Shared states

**Stream status** (from `stream_status.state`): `running` → «В эфире» (`--status-live`); `connecting`, `reconnecting` → «Подключение» (`--status-connecting`); `stopped`, `error` → «Нет сигнала» (`--status-offline`).

**Face decision** (`decision` on the face object, computed by the backend, `stack.md` Emotion events):

| Code | Label | Meaning | Box |
|---|---|---|---|
| `alert` | Инцидент | the track has an open negative event | orange, solid |
| `watch` | Наблюдение | negative group ≥ threshold, not yet held for the minimum duration | accent, solid |
| `abstain` | Не уверена | face smaller than the stream's `min_face_px`, or track younger than 1 s | grey, dashed |
| `positive` | Позитив | happiness ≥ threshold | accent, solid |
| `none` | Норма | otherwise | accent, solid |

Box label: `#track · 0,91` (negative-group probability), `#track · ?` for `abstain`. Stream decision = worst face decision (alert > watch > abstain > positive > none).

**Emotion labels:** anger «Злость», disgust «Отвращение», fear «Страх», happiness «Радость», sadness «Грусть», surprise «Удивление», neutral «Нейтрально».

**No signal:** tiles show «—», video shows the placeholder, visitors «Нет данных», journal «Записей нет». Network and API errors appear inline or as a toast, never as a modal.

## Streams sidebar

- Header «Потоки» with «+» (opens the add-stream dialog). Search over stream name and site, case-insensitive, live; empty groups hidden.
- Group «Этот компьютер» with one entry «Веб-камера» → `/live`. Then RTSP streams grouped by `site` (empty site → «Без точки»), group title uppercase 12 px muted.
- Card: name, status dot + label, «Инцидент» chip when the stream has a new incident; gear link → `/settings/cameras?camera=:id`. Active card: `--bg-active`, accent border, inset 3 px accent, `--shadow-active`. Card with a new incident: `--alert-soft-border`.

## Grid (`/streams`)

- Header: «Все потоки», count, system load (CPU %, RAM, GPU %, VRAM used/total; GPU fields hidden when null).
- Responsive tile grid fitting 20 tiles on a 1920×1080 screen. Tile: analysed frame with boxes, name, status, worst decision chip, fps, p50/p95 latency, vocal emotion or «нет речи». Click → `/streams/:id`.

## Stream detail (`/streams/:id`)

Top to bottom:
1. Incident banner when the stream has a `new` incident not hidden on this screen: «Клиент #12 · негатив 0,91 · 1:20», buttons «Подтвердить» (→ confirmed), «Ложное» (→ false_alarm), «Скрыть» (local only, incident stays new).
2. Name (22 px) and outlined status badge.
3. Three tiles: «Клиентов» (active tracks), «Статус» (Инцидент / Наблюдение / Не уверена / Норма / Позитив, «—» without signal; alert styling on incident), «Пригодных кадров» (%).
4. Video (flex 4, 16:9): analysed frame, boxes, «LIVE» badge, fps and p50/p95 in a corner. Without signal: icon, «Подключение…» or «Поток недоступен», button «Переподключить».
   «Посетители» (flex 1, min 260 px), per track: «Клиент #id» + decision chip; 6 px negative bar (orange alert, `--abstain-bar` abstain, accent otherwise) + `0,91`; line «Лицо: Злость · Итог: Злость · м:сс». Below the tracks a voice line «Голос: Нейтрально» or «Голос: нет речи» («нет аудио» for streams without audio).
5. «Журнал»: Время `чч:мм:сс–мм:сс`, Клиентов, Преобладает, Негатив (orange for alert/watch), Статус chip. One row per journal interval, newest first, from `GET /api/streams/{id}/log`, refetched every interval.
6. «Камера и обработка» disclosure, collapsed by default, 4-column mono grid: Источник (resolution), Тип камеры, Поиск лиц (detection size), Частота анализа, Медиана лица, Пригодных кадров, Задержка p95, Интервал журнала. If median face < `min_face_px`: warning «Лица мельче N px: точность снижается. Камеру лучше разместить ближе к зоне обслуживания.» Link «Настройки камеры».

Usable frames and median face are computed client-side over the last 30 s of `stream_result` faces (`decision ≠ abstain` share; median `face_px`).

## Add-stream dialog

Fields: Название, Точка (free text with suggestions from existing sites), Адрес (`rtsp://…`), Тип камеры (segmented: Веб-камера / Секторная / Купольная = `webcam` / `sector` / `security`, default `sector`). «Проверить» calls `POST /api/streams/test` and shows resolution and audio presence or the error. «Добавить» creates the stream. Name and address are required.

## Live webcam (`/live`)

The existing live view (camera and mic start/stop, canvas overlay, latency p50/p95, audio and fused panels, session id) restyled to the design system, laid out like the stream detail: video + visitors column, no journal.

## Event journal (`/history/events`)

Filters: kind (Все / Негатив / Позитив), state (Все / Новые / Подтверждённые / Ложные), camera, period (from–to). Table: Время, Камера (site · name, or «Веб-камера» / file name), Клиент `#track`, Эмоция, Уверенность `0,74`, Длительность `м:сс`, Состояние. Negative rows highlighted; new incidents get «Подтвердить» / «Ложное» buttons. New events arrive live over `/ws/streams`. «Экспорт» menu: CSV, JSON (current filters). Pagination by «Показать ещё» (`before_id`).

## Sessions (`/history/sessions`)

The existing sessions list restyled: kind, start, duration, status, CSV/JSON download, delete with confirmation.

## Settings (`/settings/:tab`)

Vertical tab nav left (40 px rows, active `--bg-panel` + inset 2 px accent); content right; «Отмена» / «Сохранить» at the bottom for `events` and `journal`. Leaving with unsaved changes asks for confirmation. Each setting is one row «label — control» (min 48 px).

**События** (`GET/PUT /api/settings` → `events`):

| Row | Control | Default | Range |
|---|---|---|---|
| Негативные эмоции | multi-select chips (selected orange) | Злость, Грусть, Страх, Отвращение | subset, at least one |
| Порог негатива | stepper, step 0,05 | 0,60 | 0,50–0,90 |
| Порог позитива | stepper, step 0,05 | 0,60 | 0,50–0,90 |
| Эпизод не короче | stepper, step 1 s | 2 с | 1–30 с |
| Повтор на том же клиенте не чаще | stepper, step 10 s | 30 с | 10–300 с |

**Камеры:** search (name, site, address); row «Все камеры» with checkbox and selected count, «Настроить выбранные» (disabled with no selection). Bulk panel: same rows, nothing preselected, clicking the selected option clears it, only chosen fields change; «Отмена» / «Применить» (enabled with ≥ 1 field and ≥ 1 camera). Camera row: checkbox, «Название · точка», mono summary («Секторная · 5 к/с» or «выключен»), chevron. Expanded camera, saved per camera with its own «Сохранить»:

| Row | Options | Default |
|---|---|---|
| Анализ | toggle | on |
| Название, Точка, Адрес потока | text fields | — |
| Тип камеры | Веб-камера · Секторная · Купольная | Секторная |
| Размер кадра для поиска | Авто · 640 · 1280 · 1920 · 2560 | Авто (size of the camera type) |
| Частота анализа | 3 · 5 · 8 к/с | 5 к/с |
| Мин. размер лица | 40 · 48 · 56 · 64 px | 48 px |
| — | button «Удалить поток» with confirmation | — |

**Журнал:** «Интервал записи журнала» 5 · 10 · 30 · 60 с, default 10 с.

**Система:** device (CPU / GPU + name), model status list, warnings, live CPU/RAM/GPU/VRAM from `/api/system`. Read-only.

## File analysis (`/analyze`)

Sidebar «Записи»: drop zone (dashed, «Перетащите видео сюда», «Выбрать файлы», MP4, WEBM, MKV, AVI; drag-and-drop and multi-select; each file becomes a queued `file` session). Queue cards: file name (ellipsis), «мм:сс · разрешение», status right (Готово green, Обработка N % yellow, В очереди grey, Ошибка), 4 px progress bar while processing. Active card styled like a stream card. Errors for wrong format or too large are shown on the card.

Content for a finished file:
1. Header: file name, «мм:сс · разрешение · анализ 5 к/с», «Экспорт» menu: «Таблица кадров» CSV, «Кадры и события» JSON. Notes «Нет аудиодорожки» / «Лица не найдены» when present.
2. Player with synced box overlay (CAP-3) and the emotion-over-time chart under it; clicking the chart seeks.
3. Tiles: «Инцидентов» (with «из N эпизодов»), «Позитивных», «Пригодных кадров».
4. Timeline bar 36 px: grey segments without faces, episode segments (orange negative, `--positive-fg` positive), selected episode with white border; 5 time marks; legend.
5. Episodes table (events of the session): Время `мм:сс–мм:сс`, Клиент, Эмоция, Уверенность, Тип. Selection is synced between timeline, table and player position.

Processing or queued file: centre «Обработка: N %» or «Запись в очереди», progress bar, «Результат появится здесь после обработки». Failed: the error message.

## Data access

One typed API module wraps REST and WebSocket. Endpoints not yet built are served by an in-memory mock with the same shapes (`interfaces.md`), selected per endpoint, so screens are built ahead of their backend stories. Mock data never ships enabled in the Docker build once the endpoint exists.
