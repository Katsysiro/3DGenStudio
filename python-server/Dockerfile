# 3D Gen Studio mesh-tools service (:8200) -- Auto UV, Auto Retopo, repair,
# bake, LODs, collision hulls, segmentation, the tree generator, FBX export and
# headless mesh thumbnails (the last three through Blender's `bpy` module).
#
# The Docker counterpart of run.sh. Built by docker-compose.standalone.yml;
# see docs/DOCKER_STANDALONE.md. CPU-only by default: the GPU only speeds up two
# Auto Retopo stages, and on a single-GPU box that VRAM is better left to
# ComfyUI. Build with MESHTOOLS_GPU=1 to add Warp + CuPy anyway.
FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# X/GL client libraries bpy loads even when rendering headless.
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
COPY requirements.txt requirements-nvidia.txt ./
RUN uv pip install --no-cache -r requirements.txt

ARG MESHTOOLS_GPU=0
ARG CUPY_PACKAGE=cupy-cuda12x
RUN if [ "$MESHTOOLS_GPU" = "1" ]; then \
      uv pip install --no-cache -r requirements-nvidia.txt "$CUPY_PACKAGE"; \
    fi

COPY . .

ENV MESHTOOLS_HOST=0.0.0.0 \
    MESHTOOLS_PORT=8200 \
    MESHTOOLS_WORK_DIR=/tmp/meshtools

EXPOSE 8200
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8200/health', timeout=4)"

CMD ["python", "main.py"]
