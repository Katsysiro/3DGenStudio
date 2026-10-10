#!/bin/bash
set -e

echo "🔧 Применяем фиксы для AMD A10..."

# 1. Удаляем файл аудио-компрессора из crt-nodes (pedalboard вызывает Illegal instruction)
rm -f /workspace/comfyui/custom_nodes/crt-nodes/py/Audio_Compressor.py 2>/dev/null || true

# 2. Удаляем ComfyUI-nunchaku (требует nunchaku v1.0.0+, который несовместим с CUDA 12.1)
rm -rf /workspace/comfyui/custom_nodes/ComfyUI-nunchaku 2>/dev/null || true

# 3. Удаляем Jovimetrix и Jovi_Measure (требуют cozy_comfyui, который не установится)
rm -rf /workspace/comfyui/custom_nodes/Jovimetrix 2>/dev/null || true
rm -rf /workspace/comfyui/custom_nodes/Jovi_Measure 2>/dev/null || true

# 4. Удаляем PuLID_ComfyUI (требует insightface, который может вызвать проблемы)
rm -rf /workspace/comfyui/custom_nodes/PuLID_ComfyUI 2>/dev/null || true

# 5. Удаляем ComfyUI-layerdiffuse (требует diffusers)
rm -rf /workspace/comfyui/custom_nodes/ComfyUI-layerdiffuse 2>/dev/null || true

# 6. Удаляем любые проблемные kornia_rs файлы (на случай, если они появятся)
rm -rf /usr/local/lib/python3.10/dist-packages/kornia_rs* 2>/dev/null || true

echo "✅ Фиксы применены."

# 7. Ноды для 3D Gen Studio. ./custom_nodes — том, он перекрывает то, что лежит
#    в образе, поэтому ноды из /opt/genstudio-nodes копируются в него при
#    каждом старте — только те, которых там ещё нет. Уже существующие папки
#    (ваши версии) не трогаются. Чтобы обновить пак до версии из образа,
#    удалите его папку из ./custom_nodes и перезапустите контейнер.
if [ -d /opt/genstudio-nodes ]; then
  for src in /opt/genstudio-nodes/*/; do
    name="$(basename "$src")"
    if [ ! -e "/workspace/comfyui/custom_nodes/$name" ]; then
      echo "🧊 Добавляем ноду для 3D Gen Studio: $name"
      cp -a "$src" "/workspace/comfyui/custom_nodes/$name"
    fi
  done
fi

# 7a. Патчи к нодам (идемпотентны: уже применённые пропускаются).
if [ -f /opt/genstudio-patches/lodtailor_mesh_counts.py ]; then
  python3 /opt/genstudio-patches/lodtailor_mesh_counts.py \
    /workspace/comfyui/custom_nodes/LODTailor-The-Mesh-Trimmer-ComfyuiNode/__init__.py || true
fi

# 7б. Workflow PixelArtistry (./user — тоже том). Кладутся один раз: если
#     папка уже есть, ваши правки в них не перезаписываются.
WF_DIR=/workspace/comfyui/user/default/workflows/PixelArtistry
if [ -d /opt/pixelartistry-workflows ] && [ ! -e "$WF_DIR" ]; then
  echo "🧱 Добавляем workflow PixelArtistry"
  mkdir -p "$WF_DIR" && cp /opt/pixelartistry-workflows/*.json "$WF_DIR/"
fi

# 8. Виртуальный дисплей для UltraTex: moderngl на Linux создаёт OpenGL-контекст
#    только через X11. Xvfb даёт его без монитора.
if command -v Xvfb >/dev/null 2>&1 && [ -z "${DISPLAY:-}" ]; then
  Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
  export DISPLAY=:99
fi

echo "🚀 Запускаем ComfyUI..."
exec python3 /workspace/comfyui/main.py ${CLI_ARGS:---listen 0.0.0.0 --port 8188 --lowvram --use-sage-attention --disable-smart-memory --disable-pinned-memory}
