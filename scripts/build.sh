#!/bin/sh
# Production build into web/dist (static files, any host). Needs install.sh first.
#   sh scripts/build.sh
set -eu
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD:/app" -w /app node:22-alpine npm run build
