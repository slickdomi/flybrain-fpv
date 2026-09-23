#!/bin/sh
# Dev server on http://localhost:5173 (PORT= to change it). Needs install.sh first.
# Open it in a browser with WebGPU: the fly's eyes are a compute shader, so there is no CPU fallback.
#   sh scripts/dev.sh
set -eu
cd "$(dirname "$0")/.."
PORT=${PORT:-5173}
docker run --rm -p "$PORT:$PORT" -v "$PWD:/app" -w /app/web node:22-alpine \
  npx vite --host 0.0.0.0 --port "$PORT"
