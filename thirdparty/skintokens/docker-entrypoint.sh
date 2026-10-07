#!/bin/sh
# Fetch the SkinTokens checkpoints + Qwen3-0.6B config into the /data volume
# (idempotent: present files are skipped), then start the service.
# RIGTOOLS_SKIP_MODEL=1 skips the check, e.g. on a machine with no internet.
set -e
if [ -z "${RIGTOOLS_SKIP_MODEL:-}" ]; then
  echo "[rigtools] checking model checkpoints in ${RIGTOOLS_DATA_DIR}..."
  python download.py --model --dir "${RIGTOOLS_DATA_DIR}" \
    || echo "[rigtools] [warn] model download failed; rigging will fail until it succeeds (restart to retry)."
fi
exec "$@"
