# 3D Gen Studio text-to-motion service (:8400) -- Kimodo.
#
# The Docker counterpart of run_server.sh. Needs an NVIDIA GPU (nvidia-container-
# toolkit). The text encoder (LLM2Vec on Llama-3-8B, ~16 GB of weights) runs in a
# subprocess on the CPU by default and needs about as much free RAM.
#
# Build context is this folder; the prebuilt MotionCorrection wheel comes from
# resources/wheels/motion_correction through the `wheels` additional context
# (see docker-compose.standalone.yml). Weights are downloaded into /data on
# first start, not baked into the image.
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Compiler + cmake + git only for the MotionCorrection source-build fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1
RUN uv venv /opt/venv --python /usr/local/bin/python3.13

WORKDIR /app
COPY requirements.txt ./
RUN uv pip install --no-cache -r requirements.txt

# torch AFTER requirements, forced: peft/accelerate already pulled a torch with
# the same version number, and uv would otherwise keep it. cu126 needs host
# driver 560+, cu128 570+, cu130 580+ (see `nvidia-smi`).
ARG TORCH_CUDA=cu126
RUN uv pip install --no-cache --reinstall-package torch torch \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

COPY . .
RUN SKIP_MOTION_CORRECTION_IN_SETUP=1 uv pip install --no-cache --no-deps -e .

# MotionCorrection (foot-skate cleanup, optional): the prebuilt wheel if it
# imports here, else a source build, else run without it -- as run_server.sh.
COPY --from=wheels . /tmp/wheels/
RUN set -u; ok=""; \
    for whl in /tmp/wheels/*linux_x86_64.whl; do \
      [ -f "$whl" ] || continue; \
      if uv pip install --no-cache --no-deps "$whl" && python -c "import motion_correction"; then ok=1; break; fi; \
      uv pip uninstall motion_correction || true; \
    done; \
    if [ -z "$ok" ]; then \
      uv pip install --no-cache --no-deps ./MotionCorrection \
        || echo "[warn] MotionCorrection not installed; foot-skate cleanup disabled."; \
    fi; \
    rm -rf /tmp/wheels

COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ENV KIMODO_HOST=0.0.0.0 \
    KIMODO_PORT=8400 \
    KIMODO_DATA_DIR=/data \
    KIMODO_CHECKPOINT_DIR=/data/checkpoints \
    KIMODO_CACHE_DIR=/data/cache/text-embeddings \
    KIMODO_LLAMA_BASE=NousResearch/Meta-Llama-3-8B-Instruct \
    HF_HOME=/data/hf-cache

VOLUME ["/data"]
EXPOSE 8400
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8400/health', timeout=4)"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "motion_server.py"]
