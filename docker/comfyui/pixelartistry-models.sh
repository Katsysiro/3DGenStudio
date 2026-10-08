#!/bin/bash
# Model files for the PixelArtistry workflows -- the list from
# watertightMeshes_win_installer.bat, step 7. Downloads only what is missing.
#
# Run it inside the ComfyUI container, so the files land in the models volume:
#   docker compose exec comfyui bash /opt/pixelartistry-models.sh
# or on the host with the models folder as the argument:
#   bash pixelartistry-models.sh /mnt/storage/comfy_data/models
set -u
MODELS="${1:-/workspace/comfyui/models}"
HF=https://huggingface.co

missing=0; failed=0
fetch() {  # folder file url [exact]
  # "exact": only this folder counts. For generic names (config.json,
  # model.safetensors, last.ckpt) a file found elsewhere is a different model.
  local dir="$MODELS/$1" target="$MODELS/$1/$2" exact="${4:-}"
  if [ -s "$target" ]; then echo "  [OK]   $1/$2"; return; fi
  # Elsewhere under models/ (a subfolder, or installed by the 3D Gen Studio
  # wizard)? The workflows name the file without a subfolder, so hard-link it
  # into place -- no extra disk space, and nothing is downloaded twice.
  local found=""
  [ -z "$exact" ] && found="$(find "$MODELS" -name "$2" -size +0 -print -quit 2>/dev/null)"
  if [ -n "$found" ]; then
    mkdir -p "$dir"; ln "$found" "$target" 2>/dev/null || cp "$found" "$target"
    echo "  [OK]   $1/$2 (reused ${found#$MODELS/})"; return
  fi
  missing=$((missing+1))
  echo "  [DL]   $1/$2"
  mkdir -p "$dir"
  if curl -fL --retry 3 -C - -o "$target.part" "$3"; then
    mv "$target.part" "$target"
  else
    echo "  [FAIL] $3"; failed=$((failed+1))
  fi
}

echo "Models folder: $MODELS"
fetch diffusion_models    trellis_2_int8_convrot.safetensors        "$HF/Comfy-Org/TRELLIS.2/resolve/main/diffusion_models/trellis_2_int8_convrot.safetensors"
fetch diffusion_models    pixal3d_int8_convrot.safetensors          "$HF/Comfy-Org/Pixal3D/resolve/main/diffusion_models/pixal3d_int8_convrot.safetensors"
fetch vae                 trellis_2_shape_vae_bf16.safetensors      "$HF/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_shape_vae_bf16.safetensors"
fetch vae                 trellis_2_texture_vae_bf16.safetensors    "$HF/Comfy-Org/Pixal3D/resolve/main/vae/trellis_2_texture_vae_bf16.safetensors"
fetch clip_vision         dino_v3_L_naf_fp32.safetensors            "$HF/Comfy-Org/Pixal3D/resolve/main/clip_vision/dino_v3_L_naf_fp32.safetensors"
fetch geometry_estimation moge_2_vitl_normal_fp16.safetensors       "$HF/Comfy-Org/MoGe/resolve/main/geometry_estimation/moge_2_vitl_normal_fp16.safetensors"
fetch background_removal  birefnet.safetensors                      "$HF/Comfy-Org/BiRefNet/resolve/main/background_removal/birefnet.safetensors"
# The Mesh Encoder looks for these in exactly this folder, so no "found
# elsewhere" shortcut -- link or download into Trellis2/encoders.
for f in shape_enc_next_dc_f16c32_fp16.safetensors shape_enc_next_dc_f16c32_fp16.json; do
  target="$MODELS/Trellis2/encoders/$f"
  if [ -s "$target" ]; then echo "  [OK]   Trellis2/encoders/$f"; continue; fi
  mkdir -p "$MODELS/Trellis2/encoders"
  found="$(find "$MODELS" -name "$f" -size +0 -not -path "*/Trellis2/encoders/*" -print -quit 2>/dev/null)"
  if [ -n "$found" ]; then
    ln "$found" "$target" 2>/dev/null || cp "$found" "$target"
    echo "  [OK]   Trellis2/encoders/$f (reused ${found#$MODELS/})"; continue
  fi
  missing=$((missing+1)); echo "  [DL]   Trellis2/encoders/$f"
  if curl -fL --retry 3 -C - -o "$target.part" "$HF/microsoft/TRELLIS.2-4B/resolve/main/ckpts/$f"; then mv "$target.part" "$target"
  else echo "  [FAIL] $f"; failed=$((failed+1)); fi
done


# DINOv3 for ComfyUI-Trellis2 / Trellis2-GGUF -- the same three files and the
# same folder as the Trellis2 installer .bat.
D=facebook/dinov3-vitl16-pretrain-lvd1689m
DINO=$HF/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m/resolve/main
fetch "$D" model.safetensors            "$DINO/model.safetensors" exact
fetch "$D" config.json                  "$DINO/config.json" exact
fetch "$D" preprocessor_config.json     "$DINO/preprocessor_config.json" exact

# SkinTokens (auto-rig node). The loader looks in models/skintoken/<repo path>;
# the LLM config is read from models/Qwen3-0.6B relative to ComfyUI's folder.
ST=$HF/VAST-AI/SkinTokens/resolve/main
fetch skintoken/experiments/skin_vae_2_10_32768                     last.ckpt     "$ST/experiments/skin_vae_2_10_32768/last.ckpt" exact
fetch skintoken/experiments/articulation_xl_quantization_256_token_4 grpo_1400.ckpt "$ST/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt" exact
QW=$HF/Qwen/Qwen3-0.6B/resolve/main
for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
  fetch Qwen3-0.6B "$f" "$QW/$f" exact
done

echo
if [ "$failed" -gt 0 ]; then echo "$failed download(s) failed -- run again to resume."; exit 1; fi
echo "Done: $missing file(s) downloaded, everything else was already there."
