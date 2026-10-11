"""Let LODTailor Bake Forger use a chosen Cycles GPU backend.

Bake Forger tries OptiX first and CUDA second. On cards whose OptiX device is
listed but cannot actually run (the CMP 90HX mining card: "Failed to retain
CUDA context (Launch failed)" on the first bake), there is no way to pick CUDA
from the node. The patched bake script reads the backends to try, in order,
from BAKEFORGER_GPU_BACKENDS (comma-separated; this image sets "CUDA"). Unset,
the original order is used.

Idempotent. Usage: python3 bakeforger_gpu_backend.py <__init__.py>
"""
import sys
from pathlib import Path

MARKER = "BAKEFORGER_GPU_BACKENDS"
OLD = 'for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):'
NEW = ('for backend in tuple(b.strip().upper() for b in os.environ.get('
       '"BAKEFORGER_GPU_BACKENDS", "OPTIX,CUDA,HIP,METAL,ONEAPI").split(",") if b.strip()):')

path = Path(sys.argv[1])
if not path.is_file():
    sys.exit(0)
src = path.read_text(encoding="utf-8")
if MARKER in src:
    print(f"[patch] Bake Forger already patched: {path}")
elif src.count(OLD) == 1:
    path.write_text(src.replace(OLD, NEW), encoding="utf-8")
    print(f"[patch] Bake Forger patched (GPU backend from BAKEFORGER_GPU_BACKENDS): {path}")
else:
    print(f"[patch] WARNING: Bake Forger code changed, patch not applied: {path}")
