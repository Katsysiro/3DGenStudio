# 3D Gen Studio video-to-motion service (:8401) -- MoCapAnything V2.
#
# The Docker counterpart of run_server.sh. Needs an NVIDIA GPU (nvidia-container-
# toolkit). The ~460 MB checkpoint and the per-rig bakes live in /data.
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# X/GL client libraries for bpy (the per-rig bake) and OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libegl1 libglib2.0-0 libx11-6 libxrender1 libxxf86vm1 libxfixes3 \
      libxi6 libxkbcommon0 libsm6 libice6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
RUN uv venv /opt/venv --python /usr/local/bin/python3.13

WORKDIR /app
COPY requirements.txt ./
RUN uv pip install --no-cache -r requirements.txt

# torch + torchvision from one index so their CUDA builds match; forced for the
# same reason as in the Kimodo image. cu126 needs host driver 560+, cu128 570+.
ARG TORCH_CUDA=cu126
RUN uv pip install --no-cache --reinstall-package torch --reinstall-package torchvision \
      torch torchvision \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

COPY . .
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ENV MOCAP_HOST=0.0.0.0 \
    MOCAP_PORT=8401 \
    MOCAP_DATA_DIR=/data \
    MOCAP_CKPT_DIR=/data/checkpoints \
    HF_HOME=/data/hf-cache

VOLUME ["/data"]
EXPOSE 8401
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8401/health', timeout=4)"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "mocap_server.py"]
