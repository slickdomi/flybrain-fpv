#!/bin/sh
# Typecheck the vendored library and the app. Needs install.sh first.
#   sh scripts/typecheck.sh
set -eu
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD:/app" -w /app node:22-alpine npm run typecheck
