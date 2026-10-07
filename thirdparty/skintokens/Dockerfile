# 3D Gen Studio rigging service (:8300) -- SkinTokens / TokenRig Auto Rig.
#
# The Docker counterpart of run_server.sh. Needs an NVIDIA GPU passed through
# with nvidia-container-toolkit. torch and flash-attn come as an ABI-matched
# pair from flash_attention_linux.txt; TORCH_CUDA picks the row and must not be
# newer than the CUDA version `nvidia-smi` reports on the host:
#   cu126 -> driver 560+,  cu128 -> 570+,  cu130 -> 580+
# Model weights are NOT baked in: the entrypoint downloads them into /data on
# first start (a volume), so rebuilding the image never re-downloads them.
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# X/GL client libraries for bpy (the rig is written out through Blender).
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
COPY requirements.txt flash_attention_linux.txt ./
RUN uv pip install --no-cache -r requirements.txt

ARG TORCH_CUDA=cu126
# Same table run_server.sh reads: Torch;Cuda;Link;PyTorchLink. torch goes in
# AFTER requirements (lightning drags in its own torch) and is forced, or uv
# keeps the PyPI build because the version number already matches.
RUN set -eu; \
    case "$TORCH_CUDA" in cu126) key=12.6.0 ;; cu128) key=12.8.0 ;; cu130) key=13.0.0 ;; \
      *) echo "Unsupported TORCH_CUDA=$TORCH_CUDA (use cu126, cu128 or cu130)"; exit 1 ;; esac; \
    row="$(awk -F';' -v k="$key" 'NR>1 && $2==k {print; exit}' flash_attention_linux.txt)"; \
    [ -n "$row" ] || { echo "No flash-attn row for CUDA $key"; exit 1; }; \
    wheel="$(echo "$row" | cut -d';' -f3)"; \
    torchargs="$(echo "$row" | cut -d';' -f4 | sed -E 's/^pip3? install //')"; \
    echo "torch: $torchargs"; echo "flash-attn: $wheel"; \
    uv pip install --no-cache --reinstall-package torch --reinstall-package torchvision $torchargs; \
    uv pip install --no-cache "$wheel"; \
    python -c 'import torch, flash_attn; print("flash_attn", flash_attn.__version__, "/ torch", torch.__version__)' \
      || echo "[warn] flash-attn did not import at build time (no GPU here is expected; an ABI error is not)"

COPY . .
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

ENV RIGTOOLS_HOST=0.0.0.0 \
    RIGTOOLS_PORT=8300 \
    RIGTOOLS_DATA_DIR=/data \
    HF_HOME=/data/hf-cache

VOLUME ["/data"]
EXPOSE 8300
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8300/health', timeout=4)"

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "rig_server.py"]
