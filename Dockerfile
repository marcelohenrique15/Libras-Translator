FROM python:3.12.15-slim-bookworm

# Change the wheel index to select another CUDA version or a CPU-only image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG APP_UID=1004
ARG APP_GID=1030

# Runtime libraries used by OpenCV, MediaPipe and PyTorch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the selected PyTorch build before the project's remaining dependencies.
RUN python -m pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

WORKDIR /home/src/mhab/models

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH=/home/src/mhab/models/src \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/home/src/mhab/models/.cache/matplotlib \
    XDG_CACHE_HOME=/home/src/mhab/models/.cache \
    CUDA_CACHE_PATH=/home/src/mhab/models/.cache/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY pyproject.toml README.md ./
COPY src/ ./src/
# Keep the selected CUDA build when resolving the pyproject dependencies.
RUN python -m pip install --no-cache-dir . \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    && python -m pip check \
    && python -c "import cv2, mediapipe, matplotlib, numpy, optuna, pandas, scipy, torch, torchvision" \
    && python -m pip freeze > requirements-installed.txt

COPY configs/ ./configs/

# Match the NFS directory owner without modifying host users or permissions.
RUN groupadd --gid "${APP_GID}" libras \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
       --no-create-home --home-dir /home/src/mhab/models libras \
    && mkdir -p data runs weights .cache/matplotlib .cache/cuda \
    && chown -R libras:libras /home/src/mhab/models

USER libras
CMD ["/bin/bash"]
