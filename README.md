# Распознавание эмоций (MVP)

Прототип распознавания эмоций человека в реальном времени по видео и звуку.
Бэкенд — FastAPI, фронтенд — React (Vite), всё запускается в Docker и работает без интернета.

## Требования

- **Linux:** Docker Engine 24+ и плагин Docker Compose v2.30+.
- **Windows:** Docker Desktop с бэкендом WSL2 (Settings → General → «Use the WSL 2 based engine»).
- **Для GPU-профиля (необязательно):** видеокарта NVIDIA и актуальный драйвер.
  - Linux: установите [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    и выполните `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
  - Windows: достаточно драйвера NVIDIA для Windows с поддержкой WSL2. Docker Desktop пробрасывает GPU сам.
  - Проверка: `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi`.
- Свободное место на диске: ~5 ГБ для CPU-режима, ~25 ГБ для GPU-режима (образ GPU ~18 ГБ: CUDA, cuDNN, PyTorch).

Камера и микрофон захватываются в браузере. В контейнеры их пробрасывать не нужно.

## Запуск

Команды одинаковы для Linux (терминал) и Windows (PowerShell). Выполняйте их в корне репозитория.

| Режим | Команда |
|---|---|
| CPU (по умолчанию) | `docker compose up` |
| GPU (CUDA) | `docker compose --profile gpu up` |

Затем откройте <http://localhost>. На странице видны устройство (CPU или GPU), имя видеокарты и список моделей.

- При первом запуске образы собираются автоматически. Это требует интернета и занимает время: скачиваются зависимости и веса моделей.
  Чтобы пересобрать образы после изменений в коде, добавьте `--build`.
- Фоновый режим: добавьте `-d`. Остановка: `docker compose down` (или `docker compose --profile gpu down`).
- **Переключение между CPU и GPU:** сначала остановите текущий режим (`docker compose down` / `docker compose --profile gpu down`), затем запустите другой.
- Если образ GPU запущен, а видеокарта недоступна, система работает на CPU и показывает предупреждение.

### Настройки (`.env`)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `APP_PORT` | `80` | Порт на хосте, где открывается приложение (`http://localhost:${APP_PORT}`) |
| `COMPOSE_PROFILES` | `cpu` | Профиль для `docker compose up`. Флаг `--profile gpu` его заменяет |

Шаблон лежит в `.env.example`.

### Проверка

```bash
curl -s http://localhost/api/health
# {"status":"ok","device":"cpu","models":{"yunet":true,"fer":true,"vad":true,"ser":true}}
```

`status: "degraded"` означает, что в образе нет файла какой-то модели (её флаг равен `false`). Пересоберите образ.

## Работа без интернета

Интернет нужен только для сборки образов. Все веса моделей скачиваются и проверяются по контрольным суммам во время `docker build`
и хранятся в образе в `/models`. Во время работы система не обращается к сети, CDN и хабам моделей.

Перенос на машину без интернета:

```bash
# на машине с интернетом — вариант 1: только CPU
docker compose build
docker save -o emotion-images.tar emotion-recognition/frontend:local \
  emotion-recognition/backend:cpu

# на машине с интернетом — вариант 2: CPU и GPU
docker compose build
docker compose --profile gpu build
docker save -o emotion-images.tar emotion-recognition/frontend:local \
  emotion-recognition/backend:cpu emotion-recognition/backend:gpu

# на целевой машине (рядом с копией репозитория)
docker load -i emotion-images.tar
docker compose up --no-build                # или (вариант 2): docker compose --profile gpu up --no-build
```

Данные (SQLite, загруженные файлы) хранятся в именованном томе `data`, смонтированном в `/data`.

## Структура

```
backend/    FastAPI: app/ (api, ws), inference/, bench/, eval/, scripts/fetch_models.py, tests/
frontend/   React + TypeScript + Vite, nginx.conf (раздача SPA и прокси /api, /ws)
docker-compose.yml
```

## Разработка

```bash
# тесты бэкенда
uv venv --python 3.11 backend/.venv
uv pip install --python backend/.venv -r backend/requirements-dev.txt
cd backend && .venv/bin/pytest -q && cd ..
# фронтенд в режиме разработки (бэкенд на localhost:8000)
cd frontend && npm ci && npm run dev
```
