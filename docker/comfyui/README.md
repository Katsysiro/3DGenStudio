# ComfyUI-образ для 3D Gen Studio

Это ваш Dockerfile для ComfyUI (CUDA 12.6, Python 3.10, torch 2.8, SageAttention2,
фиксы под AMD A10) плюс всё, что нужно workflow из мастера установки 3D Gen Studio.

## Каких нод не хватало

Workflow из `setup/` используют 16 паков. В вашем образе были 3
(ComfyUI-GGUF, ComfyUI-KJNodes, ComfyUI_essentials); 13 добавлены:

| Пак | Для чего в 3D Gen Studio |
| --- | --- |
| visualbruno/ComfyUI-Trellis2 | генерация и текстурирование мешей Trellis2/Pixal3D (33 ноды) |
| visualbruno/ComfyUI-UltraTex | текстурирование UltraTex |
| visualbruno/ComfyUI-Hunyuan3d-2-1 | меши Hunyuan 2.1 |
| visualbruno/ComfyUI-Hunyuan3DWrapper | меши Hunyuan 2.0 |
| visualbruno/ComfyUI-Marigold-v2 | нормали/albedo (MarigoldV2) |
| visualbruno/ComfyUI-Tools | обрезка по альфе, Trimesh → Mesh |
| visualbruno/comfyui-flux2fun-controlnet | ControlNet для Flux2 |
| 1038lab/ComfyUI-RMBG | удаление фона (RMBG, BiRefNet) |
| 1038lab/ComfyUI-QwenVL | описание картинок, Prompt Enhancer |
| pottokao-dotcom/ComfyUI-GGUF-Qwen3VL-TE | патч ComfyUI-GGUF для Qwen Image 2.1 GGUF |
| Fannovel16/comfyui_controlnet_aux | AIO_Preprocessor |
| aria1th/ComfyUI-LogicUtils | MultiplyNode |
| rgthree/rgthree-comfy | Display Any |

Остальные ~90 нод из workflow есть в самом ComfyUI (Trellis2 Native, MoGe,
SAM3, Qwen Image и т.д. — нужен свежий ComfyUI, автор тестирует с v0.38.0).

Коммиты паков зафиксированы теми же, с которыми тестирует 3D Gen Studio
(`genstudio-nodes.txt`, взято из `setup/comfyui.json`).

## Что ещё изменено

* **Версия ComfyUI закреплена на v0.38.0** (`ARG COMFYUI_VERSION`). На свежем
  master (0.39, 7 окт. 2026) ComfyUI-GGUF падает с
  `unexpected keyword argument 'input_act'`. Слой стоит в конце Dockerfile,
  поэтому смена версии пересобирает только его.
* **CUDA-расширения Trellis2** (cumesh, o_voxel, flex_gemm, nvdiffrast) собираются
  из исходников: у автора готовые Linux-колёса есть только для Python 3.12/3.13 и
  torch 2.7/2.9/2.11, под ваши Python 3.10 + torch 2.8 их нет.
* **flash-attn** — официальное готовое колесо под Python 3.10 + torch 2.8 + CUDA 12.
* **Xvfb** — UltraTex рендерит через OpenGL (moderngl), которому на Linux нужен
  X-дисплей; entrypoint поднимает виртуальный.
* **Ограничения pip** (`/opt/constraints.txt`): torch 2.8.0, numpy<2, kornia 0.6.12 —
  новые зависимости не могут сдвинуть ваши версии.
* **Том `./custom_nodes`.** Он перекрывает всё, что склонировано в образ в
  `custom_nodes` (это касается и Manager/Impact и др. из вашего исходного
  Dockerfile). Поэтому новые ноды лежат в образе в `/opt/genstudio-nodes`, а
  entrypoint при старте копирует в том те, которых там нет. Ваши существующие
  папки не перезаписываются.

## Как применить

Положите три файла рядом с вашим `docker-compose.yml` для ComfyUI (заменив
старые `Dockerfile` и `entrypoint.sh`):

```
Dockerfile
entrypoint.sh
genstudio-nodes.txt
```

`docker-compose.yml` менять не нужно. Затем:

```bash
docker compose build          # долго: CUDA-расширения на 4 ядрах ≈ час
docker compose up -d
docker compose logs -f comfyui
```

В конце сборки печатается **итог установки** — список `OK`/`FAIL` по пакам и
по импортам ключевых модулей. Если там есть `FAIL` — пришлите этот блок.
`FAIL import ... (exit 132)` означает «Illegal instruction», то есть пакет
собран под AVX2, которого нет у AMD A10.

При старте в логе ComfyUI не должно быть `IMPORT FAILED` у паков из таблицы выше.
