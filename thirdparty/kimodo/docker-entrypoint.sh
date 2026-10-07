#!/bin/sh
# Fetch the Kimodo checkpoint (1.1 GB) and, unless KIMODO_SKIP_ENCODER=1, the
# text encoder (~16 GB) into the /data volume. Both are idempotent and
# resumable; without the encoder here it is fetched on the first generation.
set -e
if [ -z "${KIMODO_SKIP_MODEL:-}" ]; then
  python download.py --model \
    || echo "[kimodo] [warn] checkpoint download failed; restart to retry."
fi
if [ -z "${KIMODO_SKIP_ENCODER:-}" ]; then
  python download.py --text-encoder \
    || echo "[kimodo] [warn] text-encoder download failed; it will retry on first use."
fi
exec "$@"
