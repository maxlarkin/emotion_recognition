# EmoSense — контракт API (черновик)

REST + JSON. Обновления в реальном времени идут через WebSocket. Типы описаны в `SPEC.md`, §7. Это минимальный набор под экраны макета, его стоит сверить с бэкендом edge-пайплайна.

Даты в ISO 8601, вероятности 0–1, проценты 0–100.

## Потоки

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/sites` | Список точек `Site[]` |
| GET | `/api/streams?q=` | Список потоков `Stream[]` (поиск по камере и точке) |
| POST | `/api/streams` | Добавить поток `{ name, siteId, url }` → `Stream` |
| POST | `/api/streams/test` | Проверить подключение `{ url }` → `{ ok, resolution?, error? }` |
| PATCH | `/api/streams/:id` | Переименовать или изменить адрес |
| DELETE | `/api/streams/:id` | Удалить поток |
| POST | `/api/streams/:id/reconnect` | Переподключить |
| GET | `/api/streams/:id/tracks` | Текущие треки `Track[]` |
| GET | `/api/streams/:id/log?limit=50` | Журнал `LogEntry[]`, новые сверху |
| GET | `/api/streams/:id/video` | Живое видео (WebRTC / HLS / MJPEG — на выбор бэкенда) |

## Инциденты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/api/incidents?state=new&limit=20` | Для колокольчика `Incident[]` |
| POST | `/api/incidents/:id/confirm` | Подтвердить |
| POST | `/api/incidents/:id/false-alarm` | Отметить ложным (сигнал для калибровки) |

«Скрыть» — локальное состояние UI, запроса нет.

## Настройки

| Метод | Путь | Тело / ответ |
|---|---|---|
| GET / PUT | `/api/settings/notifications` | `NotificationSettings` |
| GET | `/api/recipients` | `Recipient[]` |
| POST | `/api/recipients` | `Omit<Recipient,'id'>` → `Recipient` |
| PATCH | `/api/recipients/:id` | частичное обновление (например, `{ enabled }`) |
| DELETE | `/api/recipients/:id` | — |
| GET | `/api/cameras?q=` | `CameraSettings[]` (+ имя, точка, адрес) |
| PATCH | `/api/cameras/:streamId` | частичное `CameraSettings` |
| PATCH | `/api/cameras/bulk` | `{ streamIds: string[], patch: Partial<CameraSettings> }` |
| GET / PUT | `/api/settings/log` | `LogSettings` |

## Анализ видео

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/api/video-jobs` | `multipart/form-data`: `file`, `params` (JSON) → `VideoJob` |
| GET | `/api/video-jobs` | Очередь и история `VideoJob[]` |
| GET | `/api/video-jobs/:id` | Задача с `result` |
| GET | `/api/video-jobs/:id/export?format=json\|csv\|mp4` | Файл результата |
| DELETE | `/api/video-jobs/:id` | Удалить задачу и файлы |

## WebSocket `/ws`

Подписка: `{ "type": "subscribe", "streamId": "cam2" }`. Без `streamId` приходят только общие события.

| Событие | Полезная нагрузка | Частота |
|---|---|---|
| `tracks` | `{ streamId, tracks: Track[] }` | 2–5 раз в секунду, только по подписке |
| `log` | `{ streamId, entry: LogEntry }` | раз в интервал журнала |
| `incident` | `{ incident: Incident }` | при появлении или смене состояния |
| `stream_status` | `{ streamId, status, currentDecision, usableFramesPct }` | при изменении |
| `job_progress` | `{ jobId, status, progressPct }` | во время обработки видео |

## Ошибки

`{ "error": { "code": "STREAM_UNREACHABLE", "message": "…" } }` с HTTP-статусом 4xx/5xx.
