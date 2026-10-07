#!/bin/sh
# Fetch the MoCapAnything checkpoint into /data (skipped when present).
set -e
if [ -z "${MOCAP_SKIP_MODEL:-}" ]; then
  python download.py || echo "[mocap] [warn] checkpoint download failed; restart to retry."
fi
exec "$@"
