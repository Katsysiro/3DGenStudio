"""Patch LODTailor for ComfyUI's batched MESH (vertex_counts / face_counts).

LODTailor builds its output MESH by copying the input mesh and replacing the
geometry. Since ComfyUI 0.3x a MESH may also carry per-item vertex_counts /
face_counts (and per-vertex tangents); the copy kept the INPUT's counts, so a
mesh that grew in Blender was sliced to the old vertex count and saving failed
with "save_glb: face index out of range". WTiVo emits meshes without counts,
which is why the original workflow never hit it; Remesh Mesh (the WTiVo
replacement) emits them.

Idempotent: run it on every start; it does nothing if already applied or if
the upstream code changed shape. Usage: python3 lodtailor_mesh_counts.py <__init__.py>
"""
import sys
from pathlib import Path

MARKER = "# [3dgenstudio-patch] reset batch counts"
ANCHOR = '    if template is not None and hasattr(template, "vertex_normals"):\n'
PATCH = (
    f"    {MARKER}: the geometry is new, so the input's per-item counts and\n"
    "    # per-vertex tangents no longer describe it (single-item batch -> None).\n"
    "    if template is not None:\n"
    "        for _key in (\"vertex_counts\", \"face_counts\", \"tangents\"):\n"
    "            if hasattr(template, _key):\n"
    "                updates[_key] = None\n\n"
)

path = Path(sys.argv[1])
if not path.is_file():
    sys.exit(0)
src = path.read_text(encoding="utf-8")
if MARKER in src:
    print(f"[patch] LODTailor already patched: {path}")
elif src.count(ANCHOR) == 1:
    path.write_text(src.replace(ANCHOR, PATCH + ANCHOR), encoding="utf-8")
    print(f"[patch] LODTailor patched (vertex_counts/face_counts): {path}")
else:
    print(f"[patch] WARNING: LODTailor code changed, patch not applied: {path}")
