# AGENTS.md — emotion_recognition

Инструкции для ИИ-агентов и разработчиков, работающих в этом репозитории.

## Проект

MVP распознавания эмоций в реальном времени по видео (лицо) и аудио (русская речь). Бэкенд — FastAPI, фронтенд — React + TypeScript + Vite. Запуск только через Docker, работа полностью офлайн. Целевое железо: ноутбук с NVIDIA GPU на 4 ГБ VRAM (RTX 3050 Laptop) и 16 ГБ RAM, ОС Fedora 44 KDE.

## Источники правды

Читать до начала любой задачи:

| Что | Где |
|---|---|
| Контракт продукта (CAP-1…CAP-9, ограничения, non-goals) | `_bmad-output/specs/spec-emotion-recognition-mvp/SPEC.md` |
| Модели, параметры обработки, раскладка репозитория | `_bmad-output/specs/spec-emotion-recognition-mvp/stack.md` |
| Протокол WebSocket, REST API, формат экспорта | `_bmad-output/specs/spec-emotion-recognition-mvp/interfaces.md` |
| Диаграммы архитектуры | `_bmad-output/specs/spec-emotion-recognition-mvp/architecture-diagrams.md` |
| Список и порядок stories | `_bmad-output/specs/spec-emotion-recognition-mvp/stories.yaml` |
| Спецификации отдельных stories | `_bmad-output/specs/spec-emotion-recognition-mvp/stories/<id>-*.md` |
| Статусы спринта | `_bmad-output/implementation-artifacts/sprint-status.yaml` |
| Исследование: обоснование выбора моделей, требования к медиа | `_bmad-output/planning-artifacts/research/technical-realtime-emotion-recognition-2026-10-01/research.md` |

- `SPEC.md` и `stories.yaml` руками не правятся. Их пересобирает `bmad-spec` из `.memlog.md` (в `stories.yaml` — после разбивки на stories).
- `epics.md` собран из `stories.yaml`. При изменении stories его нужно пересобрать.
- Блок `<frozen-after-approval>` в спецификации story меняет только человек.

## Обязательные правила

- **Офлайн в рантайме.** Никаких CDN, удалённых шрифтов, загрузок с model hub и `pip`/`npm install` во время работы. Веса моделей скачиваются при `docker build` в `/models` и проверяются по контрольной сумме.
- **Весь текст интерфейса — на русском.**
- **Веб-камера и микрофон захватываются в браузере** (`getUserMedia`). Устройства в контейнер не пробрасываются. RTSP-потоки (IP-камеры, эмулятор `demo-cams`) принимает бэкенд.
- **Инференс работает вне event loop FastAPI** — в отдельном процессе. Цепочку «источник → очередь → воркер → результаты» не ломать.
- **В образе ровно один пакет onnxruntime:** `onnxruntime` для CPU-образа или `onnxruntime-gpu` для GPU-образа.
- **Ключи эмоций всегда одни и те же:** `anger, disgust, fear, happiness, sadness, surprise, neutral`.
- **Протоколы в `interfaces.md` — фиксированный контракт.** Менять их можно только через обновление спецификации.
- **Бюджет памяти:** все модели вместе должны помещаться в 4 ГБ VRAM / 16 ГБ RAM. Режим «только CPU» обязан работать.

## Структура

```
backend/
  app/          FastAPI: config, api/ (health, files, sessions, export, streams, system), ws/ (live, streams)
  sources/      источники: webcam (WebSocket), rtsp (PyAV), file
  inference/    detector, tracker, fer, vad, ser, fusion, worker
  bench/        бенчмарк производительности (CAP-6)
  eval/         оценка точности (CAP-7)
  scripts/      fetch_models.py — загрузка и проверка весов при сборке
  tests/        pytest
  Dockerfile    один файл, аргумент VARIANT=cpu|gpu
frontend/       Vite + React + TS; Dockerfile (сборка в node, затем nginx); nginx.conf проксирует /api и /ws
demo-videos/    открытые ролики для эмулированных камер + ATTRIBUTION.md
docs/           media-requirements.md, scaling-plan.md, benchmark-report.md, evaluation-report.md
docker-compose.yml, .env (COMPOSE_PROFILES=cpu, APP_PORT=80)
```

## Особенности Fedora 44

**Docker.** Используется Docker CE, а не podman. Если в системе стоит `podman-docker`, удалите его: он подменяет команду `docker`.

```bash
sudo dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # затем перелогиниться
```

Если оболочка была открыта до добавления в группу, выполняйте команды так: `sg docker -c 'docker compose up'`. Другой вариант — `newgrp docker`.

**GPU в контейнерах:**

```bash
sudo dnf install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi   # проверка
```

**SELinux** работает в режиме Enforcing. Bind-mount каталогов хоста в контейнер делайте с меткой `:Z` (или `:z`), например `-v ./data:/data:Z`. Named volume (`data`) метки не требует.

**VPN и сборка образов.** Если включён VPN с MTU меньше 1500 (например, AmneziaVPN, интерфейс `amn0` с MTU 1280), `pip` внутри `docker compose build` зависает на TLS-рукопожатии (`Read timed out`). Варианты:
- выключить VPN на время сборки;
- собрать образ через сеть хоста, а потом запускать без пересборки:

```bash
docker build --network host --build-arg VARIANT=cpu -t emotion-recognition/backend:cpu backend
docker build --network host --build-arg VARIANT=gpu -t emotion-recognition/backend:gpu backend
docker compose up -d --no-build   # или: docker compose --profile gpu up -d --no-build
```

**Python на хосте — 3.14,** а проект требует 3.11 (как в образе). Локальное окружение создавайте через `uv`:

```bash
uv venv --python 3.11 backend/.venv
```

**Node:** системный Node 22 подходит для `frontend/`.

**Браузер** (Firefox или Chromium в KDE): `http://localhost` считается безопасным контекстом, поэтому доступ к камере и микрофону разрешается без HTTPS. Если камеру не видно, проверьте разрешение в браузере и что её не держит другое приложение: `fuser /dev/video0`.

**Порт 80** публикует демон Docker, sudo не нужен. Если порт занят, поменяйте `APP_PORT` в `.env`.

## Команды

### Запуск

```bash
docker compose up --build                     # CPU-профиль (по умолчанию из .env)
docker compose --profile gpu up --build       # GPU-профиль (CUDA)
docker compose up -d && docker compose logs -f backend   # в фоне + логи
docker compose ps                             # должен работать ровно один бэкенд
./scripts/check-compose.sh                    # то же по конфигурации: CPU и GPU профили дают ровно по одному бэкенду
docker compose down                           # остановить (volume data сохраняется)
docker compose --profile cpu --profile gpu down  # остановить оба профиля; делать перед сменой профиля
docker compose down -v                        # остановить и удалить данные (sessions, uploads)
```

Демо-камеры (профиль `demo-cams`: MediaMTX и эмулятор RTSP-камер `rtsp://mediamtx:8554/cam1..cam20`). Набор из 20 клипов (6 веб-камер, 8 секторных, 6 купольных) готовится один раз при включённой сети: скрипт скачивает открыто лицензированные источники из `demo-videos/sources.json`, проверяет контрольные суммы, размечает негатив и позитив моделями проекта и кодирует клипы под профили камер. Результат: `demo-videos/clips/` (в git не попадает) и отчёт `demo-videos/segments.md`. Без клипов эмулятор публикует три синтетические камеры.

```bash
docker run --rm --gpus all --network host --user "$(id -u):$(id -g)" \
  -v "$PWD/demo-videos:/demo:z" emotion-recognition/backend:gpu \
  python scripts/prep_demo_clips.py --demo /demo          # CPU-образ: без --gpus, тег :cpu
```

Подготовке нужно около 1 ГБ на диске (кэш перекодированных копий 480p; если Commons перекодирует файл, скачивается оригинал, до 3 ГБ). Она занимает около 20 минут на GPU и заметно дольше на CPU. Повторный запуск без сети берёт источники и разметку из `demo-videos/cache/`. Запуск демо. Профиль бэкенда указывайте явно: флаг `--profile` отменяет `COMPOSE_PROFILES` из `.env`. При пустом списке потоков `DEFAULT_STREAMS` добавляет камеры один раз:

```bash
DEFAULT_STREAMS="$(cat demo-videos/demo-cams.json)" docker compose --profile cpu --profile demo-cams up -d   # 20 потоков
docker compose --profile gpu --profile demo-cams up -d          # GPU; потоки уже сохранены в volume data
docker compose --profile cpu --profile gpu --profile demo-cams down
```

`DEFAULT_STREAMS` срабатывает только на пустом списке потоков. Если в volume `data` уже есть потоки (например, три синтетические камеры из story 1.7), удалите их в «Настройки → Камеры» или выполните `docker compose down -v`. Без подготовленных клипов эмулятор публикует только cam1..cam3, остальные 17 потоков из списка будут переподключаться. До story 1.9 весь набор из 20 потоков не держит нужную частоту: включайте 4–6 потоков одновременно.

Приложение: http://localhost. Проверка состояния:

```bash
curl -s localhost/api/health | python3 -m json.tool
```

Ожидаемый ответ: `"status":"ok"`, `"device":"cpu"` или `"cuda"`, все четыре модели `true`.

### Тесты

Бэкенд локально:

```bash
uv venv --python 3.11 backend/.venv
uv pip install --python backend/.venv -r backend/requirements-dev.txt
cd backend && .venv/bin/pytest -q
```

Тесты не входят в образ. Код, зависящий от моделей (stories 1.2+), прогоняйте в полном окружении: `uv pip install --python backend/.venv -r backend/requirements-cpu.txt && uv pip install --python backend/.venv --no-deps -r backend/requirements-nodeps.txt`.

Тесты с моделями (`-m models`) требуют весов в `MODELS_DIR` и около 3 ГБ RAM: один набор моделей занимает примерно 2 ГБ. Запускайте их с жёстким лимитом памяти, иначе при нехватке RAM система уходит в swap и зависает, а OOM killer не срабатывает:

```bash
cd backend && MODELS_DIR=$(realpath ../.models) systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 -- .venv/bin/pytest -q -m models
```

Перед запуском проверьте `docker ps`: работающий бэкенд держит ещё около 2 ГБ.

### Фронтенд

```bash
cd frontend && npm ci
npm run dev       # http://localhost:5173, проксирует /api и /ws на localhost:8000
npm run build     # tsc --noEmit + vite build → dist/
grep -rEo "https?://[^\"' )]*" dist | grep -vE "www\.w3\.org/"   # должно быть пусто (XML-namespace w3.org допустимы); та же проверка в frontend/Dockerfile, localhost тоже запрещён — в тексте UI пишите «localhost» без схемы
```

Бэкенд для `npm run dev` поднимается локально. Нужны полное окружение (см. «Тесты») и веса в `MODELS_DIR`:

```bash
cd backend && .venv/bin/python scripts/fetch_models.py --models-dir ../.models   # один раз
MODELS_DIR=$(realpath ../.models) DATA_DIR=/tmp/er-data .venv/bin/uvicorn app.main:app --reload --port 8000
```

Альтернатива — бэкенд из собранного образа, порт 8000 проброшен на хост (compose его не публикует):

```bash
docker run --rm --shm-size=1g -p 8000:8000 emotion-recognition/backend:cpu
```

### Проверка офлайн-режима

Без отключения сети хоста (бэкенд в контейнере вообще без сети):

```bash
docker run -d --name er-offline --network none --shm-size=1g emotion-recognition/backend:cpu
docker exec er-offline python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/api/health').read().decode())"
docker rm -fv er-offline   # -v: удалить и анонимный volume /data
```

Полная проверка с отключением сети:

```bash
docker compose up -d --build                        # собрать с сетью
nmcli networking off                                # отключить сеть (KDE: NetworkManager)
docker compose down && docker compose up -d && curl -s localhost/api/health
nmcli networking on
```

### Диагностика

```bash
nvidia-smi                                   # GPU хоста
docker compose exec backend-gpu nvidia-smi   # GPU в контейнере
nvidia-smi dmon -s um                        # загрузка GPU и памяти во время бенчмарка
docker stats                                 # CPU и RAM контейнеров
docker compose build --no-cache backend      # пересборка с повторной загрузкой весов
docker system df && docker builder prune     # место, занятое образами и кешем сборки
```

## Git

- Работа в ветках `feat/<story-id>-<slug>`, например `feat/1-1-skeleton-docker`. В `main` напрямую не коммитить.
- Сообщения коммитов — в стиле Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- Не коммитить: `.venv/`, `node_modules/`, `frontend/dist/`, `__pycache__/`, `.pytest_cache/`, веса моделей (`.models/`), данные из volume `data`.

## Процесс BMAD

- Следующая story — по `sprint-status.yaml`. Реализация через `bmad-build`, указав папку спецификации и id story.
- На stories с `done_checkpoint: true` агент останавливается и ждёт проверки человеком.
- Если story требует изменить контракт, сначала обновите спецификацию через `bmad-spec`, потом пишите код.
