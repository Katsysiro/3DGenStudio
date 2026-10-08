# ComfyUI для 3D Gen Studio — ваш исходный Dockerfile + всё, что нужно
# workflow из мастера установки 3D Gen Studio (setup/*.json).
#
# Блоки 1–3 — ваш образ без изменений (CUDA 12.6, Python 3.10, torch 2.8,
# SageAttention2, ваши ноды и фиксы под AMD A10).
# Блок 4 — новое: 13 недостающих паков нод, их Python-зависимости, flash-attn и
# CUDA-расширения Trellis2 (cumesh, o_voxel, flex_gemm, nvdiffrast), собранные
# из исходников под вашу связку Python 3.10 + torch 2.8 + cu126 — готовых
# колёс под неё у автора нет.
#
# Сборка долгая (CUDA-расширения на 4 ядрах — порядка часа). Подробности:
# docker/comfyui/README.md

# Базовый образ с CUDA 12.6.1 (в пару к PyTorch cu126)
# СТАЛО: Меняем runtime на devel (добавляет nvcc и заголовочные файлы CUDA)
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

# Установка системных зависимостей
# + xvfb/mesa: UltraTex рендерит через moderngl, которому на Linux нужен X-дисплей
#   (в контейнере его даёт виртуальный Xvfb, см. entrypoint.sh)
# + libeigen3-dev: заголовки Eigen для сборки cumesh/o_voxel
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    git-lfs \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    xvfb \
    libgl1-mesa-dri \
    libegl1 \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Клонируем официальный репозиторий ComfyUI.
# Здесь берётся master; конкретная версия закрепляется ближе к концу файла
# (ARG COMFYUI_VERSION), чтобы её смена не пересобирала весь образ.
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/comfyui && \
    git -C /workspace/comfyui checkout "$COMFYUI_REF"

WORKDIR /workspace/comfyui

# 🔥 PyTorch 2.8.0 + CUDA 12.6 — решает ошибку list[int] в comfy_kitchen (новый ComfyUI)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126

# Устанавливаем базовые зависимости ComfyUI
RUN pip3 install --no-cache-dir -r requirements.txt

# ====================================================================
# 🔥 SAGEATTENTION2 ДЛЯ CMP 90HX (Ampere, CC 8.7)
# ====================================================================

# Переменные окружения для компиляции под Ampere
ENV TORCH_CUDA_ARCH_LIST="8.6;8.7"
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=4

# Устанавливаем triton (обязателен для SageAttention)
RUN pip3 install --no-cache-dir "triton>=3.0.0" ninja

# Устанавливаем SageAttention2 с GitHub (официальный репозиторий thu-ml)
RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention && \
    cd /tmp/SageAttention && \
    pip3 install --no-cache-dir . --no-build-isolation && \
    cd /workspace && \
    rm -rf /tmp/SageAttention

# Проверка установки
RUN python3 -c "from sageattention import sageattn; import triton; print('✅ SageAttention2 + Triton', triton.__version__)" || echo "⚠️ SageAttention warning"

# ====================================================================

# ============================================
# УСТАНОВКА КАСТОМНЫХ НОД
# ============================================
WORKDIR /workspace/comfyui/custom_nodes

# 1. ComfyUI-Manager
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && pip3 install --no-cache-dir -r requirements.txt && cd ..

# 2. ComfyUI-Impact-Pack (🔥 ИСПРАВЛЕНО: полностью раскомментировано)
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && pip3 install --no-cache-dir -r requirements.txt && cd ..

# 3. ComfyUI-Inspire-Pack
RUN git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git && \
    cd ComfyUI-Inspire-Pack && pip3 install --no-cache-dir -r requirements.txt && cd ..

# 4. ComfyUI-KJNodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && pip3 install --no-cache-dir -r requirements.txt && cd ..

# 5. ComfyUI_essentials
RUN git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    cd ComfyUI_essentials && pip3 install --no-cache-dir -r requirements.txt && cd ..

# 6. ComfyUI-Crystools
RUN git clone https://github.com/crystian/comfyui-crystools.git && \
    cd comfyui-crystools && pip3 install --no-cache-dir -r requirements.txt && cd ..

RUN git clone https://github.com/willmiao/ComfyUI-Lora-Manager.git && \
    cd ComfyUI-Lora-Manager && pip3 install --no-cache-dir -r requirements.txt && cd ..

RUN git clone https://github.com/city96/ComfyUI-GGUF.git && \
    cd ComfyUI-GGUF && pip3 install --no-cache-dir -r requirements.txt && cd ..

RUN git clone https://github.com/PGCRT/CRT-Nodes.git && \
    cd CRT-Nodes && pip3 install --no-cache-dir -r requirements.txt && cd ..

RUN git clone https://github.com/jlucasmcrell/ComfyUI-H3-Multishot.git && \
    cd ComfyUI-H3-Multishot && \
        python3 apply_gguf_arch_patch.py && \
        cd ..

# ============================================
# 🔥 ФИКСЫ ДЛЯ AMD A10 (КРИТИЧЕСКИ ВАЖНО!)
# ============================================

# ============================================
# 🔥 УСТАНОВКА ВСЕХ ЗАВИСИМОСТЕЙ ДЛЯ CUSTOM NODES
# ============================================

# Зависимости для твоих локальных нод (приходят из тома ./custom_nodes)
RUN pip3 install --no-cache-dir \
    py-cpuinfo segment_anything dill pandas natsort matplotlib deepdiff \
    piexif ultralytics numexpr gguf blend_modes numba soundfile loguru \
    diffusers insightface

# 1. Откатываем NumPy до 1.x (без AVX2)
RUN pip3 install --no-cache-dir --force-reinstall "numpy<2.0"

# 2. Polars с поддержкой старых процессоров
RUN pip3 install --no-cache-dir --force-reinstall "polars[rtcompat]"

# 3. Удаляем kornia и kornia_rs полностью
RUN pip3 uninstall -y kornia kornia_rs || true
RUN rm -rf /usr/local/lib/python3.10/dist-packages/kornia_rs* || true
RUN rm -rf /usr/local/lib/python3.10/dist-packages/kornia* || true

# 4. 🔥 КРИТИЧЕСКИЙ ФИКС 2: Ставим kornia 0.6.12 (версия 0.7.3 вызовет краш на AMD A10!)
RUN pip3 install --no-cache-dir "kornia==0.6.12"

# 5. Дополнительные зависимости для нод из локальной папки
RUN pip3 install --no-cache-dir diffusers insightface

# ====================================================================
# 🧊 4. НОДЫ ДЛЯ 3D GEN STUDIO
# ====================================================================
#
# Всё, что ставится ниже, обязано НЕ сдвинуть torch/numpy/kornia, которые
# выше подобраны под вашу карту и процессор. Поэтому каждая установка идёт с
# этими ограничениями (constraints).
RUN printf '%s\n' \
      'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0' \
      'numpy<2.0' 'kornia==0.6.12' \
    > /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt

# ВАЖНО: ./custom_nodes у вас смонтирован томом, и том ПЕРЕКРЫВАЕТ всё, что
# склонировано в образ в /workspace/comfyui/custom_nodes. Поэтому новые ноды
# кладём в /opt/genstudio-nodes, а entrypoint.sh при старте копирует в том те,
# которых там ещё нет (существующие папки не трогает).
COPY genstudio-nodes.txt /opt/genstudio-nodes.txt
RUN set -eu; mkdir -p /opt/genstudio-nodes; cd /opt/genstudio-nodes; \
    grep -v '^#' /opt/genstudio-nodes.txt | while read -r name repo ref; do \
      [ -n "$name" ] || continue; \
      git init -q "$name"; \
      git -C "$name" fetch -q --depth 1 "https://github.com/$repo" "$ref"; \
      git -C "$name" checkout -q FETCH_HEAD; \
      echo "✅ $name @ $ref"; \
    done

# Python-зависимости каждого пака. Не валим сборку на одном паке — пишем в
# сводку, она печатается в конце сборки.
RUN set -u; cd /opt/genstudio-nodes; : > /opt/genstudio-install-report.txt; \
    for d in */; do d="${d%/}"; \
      if [ -f "$d/requirements.txt" ]; then \
        if pip3 install --no-cache-dir -r "$d/requirements.txt"; then \
          echo "OK    $d" >> /opt/genstudio-install-report.txt; \
        else \
          echo "FAIL  $d (pip install -r requirements.txt)" >> /opt/genstudio-install-report.txt; \
        fi; \
      fi; \
    done; cat /opt/genstudio-install-report.txt

# Trellis2/Hunyuan: зависимости, которых нет в их requirements.txt.
# utils3d — тот же коммит, что в lock-файле 3D Gen Studio.
RUN pip3 install --no-cache-dir \
      "utils3d @ https://github.com/EasternJournalist/utils3d/archive/94d1037aabbce32dea9c07a7c4849525817a1615.tar.gz" \
      xatlas pygltflib easydict plyfile zstandard

# flash-attn: Trellis2 по умолчанию работает через него (и для разреженного
# внимания другой вариант — только xformers). Официальное готовое колесо под
# Python 3.10 + torch 2.8 + CUDA 12 — без часовой компиляции.
RUN pip3 install --no-cache-dir \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"

# CUDA-расширения Trellis2 — из исходников (форки автора нод + NVlabs), под
# архитектуры из TORCH_CUDA_ARCH_LIST. Коммиты зафиксированы.
RUN set -eux; mkdir -p /tmp/ext; cd /tmp/ext; \
    fetch() { git init -q "$1"; git -C "$1" fetch -q --depth 1 "https://github.com/$2" "$3"; git -C "$1" checkout -q FETCH_HEAD; \
              git -C "$1" submodule update -q --init --recursive --depth 1 || true; }; \
    fetch nvdiffrast NVlabs/nvdiffrast 253ac4fcea7de5f396371124af597e6cc957bfae; \
    fetch CuMesh visualbruno/CuMesh d10e54c30ddd03d11472c1431693f985501c7966; \
    fetch FlexGEMM visualbruno/FlexGEMM db388bd17c34abc697792aa718dca780734c2783; \
    fetch TRELLIS.2 visualbruno/TRELLIS.2 65d1e13b4a92296036044df0633242bb9e95abf6; \
    # cubvh внутри CuMesh ищет Eigen в third_party/cubvh/third_party/eigen,
    # которого в репозитории нет, — подкладываем системный.
    mkdir -p CuMesh/third_party/cubvh/third_party; \
    ln -sfn /usr/include/eigen3 CuMesh/third_party/cubvh/third_party/eigen; \
    [ -e TRELLIS.2/o-voxel/third_party/eigen/Eigen ] || { rm -rf TRELLIS.2/o-voxel/third_party/eigen; ln -sfn /usr/include/eigen3 TRELLIS.2/o-voxel/third_party/eigen; }; \
    pip3 install --no-cache-dir --no-build-isolation ./nvdiffrast; \
    pip3 install --no-cache-dir --no-build-isolation ./CuMesh; \
    pip3 install --no-cache-dir --no-build-isolation ./FlexGEMM; \
    pip3 install --no-cache-dir --no-build-isolation ./TRELLIS.2/o-voxel; \
    cd /; rm -rf /tmp/ext

# Необязательное: PBR-рендер nvdiffrec (нужен только одному рендереру Trellis2).
RUN pip3 install --no-cache-dir --no-build-isolation \
      "git+https://github.com/JeffreyXiang/nvdiffrec.git@b296927cc7fd01c2ac1087c8065c4d7248f72da4" \
    || echo "⚠️ nvdiffrec_render не собрался — не критично"

# Сводка: каждый модуль импортируется в отдельном процессе, так что падение
# одного (в т.ч. «Illegal instruction» на CPU без AVX2) видно по имени и не
# валит сборку. Импорт CUDA-расширений без GPU при сборке может не пройти —
# это нормально; окончательная проверка — лог ComfyUI при старте.
RUN for m in numpy torch flash_attn sageattention cumesh o_voxel flex_gemm nvdiffrast.torch utils3d \
             meshlib.mrmeshpy pymeshlab open3d xatlas moderngl rembg trimesh diffusers transformers \
             onnxruntime cv2 gguf; do \
      if python3 -c "import $m" >/dev/null 2>&1; then echo "OK    import $m"; \
      else echo "FAIL  import $m (exit $?)"; fi; \
    done | tee -a /opt/genstudio-install-report.txt; \
    echo; echo "===== 3D Gen Studio: итог установки ====="; cat /opt/genstudio-install-report.txt

# ====================================================================
# 🧱 5. PIXELARTISTRY: WATERTIGHT & GAME-READY (ноды MostAadTech)
# ====================================================================
#
# Linux-аналог watertightMeshes_win_installer.bat: те же паки на тех же
# коммитах, Blender, workflow. Отличие одно — WTiVo: у него есть только
# Windows-бинарники, на Linux он не запускается (см. README.md).

# Blender — его запускают LODTailor, Bake Forger и LODsmith (headless).
# Официальная сборка для Linux; последний патч выбранной LTS-ветки берётся
# со страницы релизов. Ссылка /usr/local/bin/blender нужна, потому что в
# workflow путь к Blender по умолчанию — просто "blender".
ARG BLENDER_SERIES=4.5
RUN set -eux; apt-get update; apt-get install -y --no-install-recommends \
      curl ca-certificates xz-utils libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 libsm6 libice6 libx11-6; \
    rm -rf /var/lib/apt/lists/*; \
    base="https://download.blender.org/release/Blender${BLENDER_SERIES}"; \
    file="$(curl -fsSL "$base/" | grep -oE "blender-${BLENDER_SERIES}\.[0-9]+-linux-x64\.tar\.xz" | sort -V | tail -1)"; \
    [ -n "$file" ]; \
    curl -fsSL "$base/$file" | tar -xJ -C /opt; \
    mv /opt/blender-${BLENDER_SERIES}.*-linux-x64 /opt/blender; \
    ln -sf /opt/blender/blender /usr/local/bin/blender; \
    blender --background --factory-startup --python-expr "import bpy; print('Blender OK', bpy.app.version_string)" 2>&1 | grep "Blender OK"

# Паки нод — туда же, откуда entrypoint докладывает их в том custom_nodes.
COPY pixelartistry-nodes.txt /opt/pixelartistry-nodes.txt
RUN set -eu; cd /opt/genstudio-nodes; \
    grep -v '^#' /opt/pixelartistry-nodes.txt | while read -r name repo ref; do \
      [ -n "$name" ] || continue; \
      git init -q "$name"; \
      git -C "$name" fetch -q --depth 1 "https://github.com/$repo" "$ref"; \
      git -C "$name" checkout -q FETCH_HEAD; \
      echo "✅ $name @ $ref"; \
    done; \
    pip3 install --no-cache-dir \
      -r ComfyUI-Mesh-Quad-Reconstruct/requirements.txt \
      -r Trellis2-Mesh-Encoder/requirements.txt \
      -r WTiVo-FastMergeByDistance/requirements.txt

# Fast Merge: быстрый нативный режим. В репозитории лежит только Windows-DLL;
# собираем ту же C++-библиотеку под Linux и кладём под тем же именем —
# ctypes грузит её по пути, расширение файла ему безразлично. Без неё нода
# работает в медленном Python-режиме.
RUN set -eux; d=/opt/genstudio-nodes/WTiVo-FastMergeByDistance; \
    g++ -O3 -std=c++17 -shared -fPIC -fopenmp "$d/cpp/wtivo_fast_weld.cpp" -o "$d/native/wtivo_fast_weld_mt.dll"; \
    python3 -c "import sys; sys.path.insert(0, '$d'); import native; assert native.native_available(); print('FastMerge native OK, threads:', native.native_thread_count())"

# Workflow PixelArtistry. ./user у вас — том, поэтому их тоже раскладывает
# entrypoint (в user/default/workflows/PixelArtistry, если папки ещё нет).
ARG PIXELARTISTRY_REF=5b3a0e9d2c5783ed05ba45bbcc157b9509fb267b
RUN set -eu; git init -q /tmp/pa; \
    git -C /tmp/pa fetch -q --depth 1 https://github.com/pixelartistry/PixelArtistry-Watertight-Meshes "$PIXELARTISTRY_REF"; \
    git -C /tmp/pa checkout -q FETCH_HEAD; \
    mkdir -p /opt/pixelartistry-workflows; cp /tmp/pa/workflows/*.json /opt/pixelartistry-workflows/; \
    rm -rf /tmp/pa; ls /opt/pixelartistry-workflows

COPY pixelartistry-models.sh /opt/pixelartistry-models.sh

# Версия ComfyUI. Клон выше берёт master, а master ломает ноды без
# предупреждения: ComfyUI 0.39 (7 окт. 2026) стал передавать в слои
# аргументы input_act/act_weight, которых не знает ComfyUI-GGUF, и любой
# GGUF-энкодер/модель падает с "unexpected keyword argument 'input_act'".
# Поэтому здесь закрепляется версия, с которой тестирует 3D Gen Studio.
# Слой стоит в конце нарочно: смена версии пересобирает только его, а не
# CUDA-расширения выше. PIP_CONSTRAINT удерживает torch/numpy/kornia; строка
# kornia>=0.7.1 из requirements ComfyUI пропускается — у вас намеренно стоит
# 0.6.12 (фикс под AMD A10), с ней ComfyUI работал и раньше.
ARG COMFYUI_VERSION=v0.38.0
RUN git -C /workspace/comfyui fetch -q --tags origin || true; \
    git -C /workspace/comfyui checkout -q "$COMFYUI_VERSION" && \
    grep -viE '^\s*kornia' /workspace/comfyui/requirements.txt > /tmp/comfy-req.txt && \
    pip3 install --no-cache-dir -r /tmp/comfy-req.txt && \
    echo "ComfyUI: $(git -C /workspace/comfyui describe --tags --always)"

COPY entrypoint.sh /entrypoint.sh

WORKDIR /workspace/comfyui

EXPOSE 8188

# Команда запуска переопределена в docker-compose.yml через entrypoint.sh
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--lowvram", "--use-sage-attention"]
