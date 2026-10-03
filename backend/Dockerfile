# syntax=docker/dockerfile:1
# Single Dockerfile for both backend variants.
#   docker build --build-arg VARIANT=cpu .   -> python:3.11-slim, CPU torch, onnxruntime
#   docker build --build-arg VARIANT=gpu .   -> CUDA 12.6 + cuDNN 9 runtime, CUDA torch, onnxruntime-gpu
# All model weights are downloaded and verified at build time and baked into /models.
ARG VARIANT=cpu

# ---------- CPU base ----------
FROM python:3.11.16-slim-bookworm AS base-cpu
RUN python -m venv /opt/venv

# ---------- GPU base ----------
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 AS base-gpu
# Ubuntu 24.04 ships Python 3.12 (>= 3.11 required).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

# ---------- final ----------
FROM base-${VARIANT} AS final
ARG VARIANT
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_VARIANT=${VARIANT} \
    MODELS_DIR=/models \
    DATA_DIR=/data \
    HF_HUB_OFFLINE=1 \
    TORCH_HOME=/models/torch

WORKDIR /app

# Python dependencies (variant-specific torch/onnxruntime, shared pins, then --no-deps packages).
COPY requirements.txt requirements-${VARIANT}.txt requirements-nodeps.txt ./
RUN pip install --no-cache-dir -r requirements-${VARIANT}.txt \
    && pip install --no-cache-dir --no-deps -r requirements-nodeps.txt \
    && python -c "import onnxruntime as ort, cv2, torch, torchaudio, emotiefflib, gigaam, silero_vad; \
print('ort', ort.__version__, ort.get_available_providers()); print('cv2', cv2.__version__); print('torch', torch.__version__)" \
    && if [ "$(pip list 2>/dev/null | grep -ciE '^onnxruntime(-gpu)? ')" != "1" ]; then echo 'exactly one onnxruntime package expected' >&2; exit 1; fi

# Weights: only the manifest and fetch script are copied first so code edits keep this layer cached.
COPY app/__init__.py app/config.py app/
COPY scripts/fetch_models.py scripts/
RUN python scripts/fetch_models.py --models-dir /models \
    && chmod -R a+rX /models

# Application code.
COPY app/ app/
COPY inference/ inference/
COPY sources/ sources/
COPY bench/ bench/
COPY eval/ eval/
COPY scripts/ scripts/

# Unprivileged runtime user. Library default caches point at the read-only baked weights,
# so nothing is downloaded or cached under $HOME at runtime.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data /home/app/.cache \
    && ln -s /models/fer /home/app/.emotiefflib \
    && ln -s /models/ser /home/app/.cache/gigaam \
    && chown -R app:app /data /home/app
USER app
ENV HOME=/home/app
VOLUME ["/data"]

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
