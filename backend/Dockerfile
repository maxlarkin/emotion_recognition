# syntax=docker/dockerfile:1
# Single Dockerfile for both backend variants.
#   docker build --build-arg VARIANT=cpu .   -> python:3.11-slim, CPU torch, onnxruntime
#   docker build --build-arg VARIANT=gpu .   -> python:3.11-slim, CUDA torch, onnxruntime-gpu
#     (CUDA 12.6 / cuDNN 9 user-space libraries come from the torch nvidia-* wheels; the driver from `gpus: all`)
# All model weights are downloaded and verified at build time and baked into /models.
ARG VARIANT=cpu

# ---------- CPU base ----------
FROM python:3.11.16-slim-bookworm AS base-cpu
RUN python -m venv /opt/venv

# ---------- GPU base ----------
# Same Python as the CPU image. No CUDA base image: torch cu126 already installs the CUDA/cuDNN
# libraries as nvidia-* wheels, and a cudnn-runtime base would ship them a second time.
FROM python:3.11.16-slim-bookworm AS base-gpu
RUN python -m venv /opt/venv
# onnxruntime-gpu loads the CUDA EP libraries from these wheel directories.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cufft/lib:/opt/venv/lib/python3.11/site-packages/nvidia/curand/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/opt/venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cusparse/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cusolver/lib
# Container toolkit: mount the driver libraries (libcuda, NVML) into this non-CUDA image.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

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

# RTSP decoding (PyAV with bundled FFmpeg) in its own layer: the torch layer above stays cached.
COPY requirements-stream.txt ./
RUN pip install --no-cache-dir -r requirements-stream.txt \
    && python -c "import av; from av.codec.hwaccel import hwdevices_available; print('av', av.__version__, hwdevices_available())"

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
