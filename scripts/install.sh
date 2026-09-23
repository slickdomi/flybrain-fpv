#!/bin/sh
# Install the workspace in Docker. The .npmrc cooldown keeps npm from resolving anything published
# in the last two weeks. The first run writes package-lock.json; after that use `npm ci` semantics
# by keeping the lock committed.
#   sh scripts/install.sh
set -eu
cd "$(dirname "$0")/.."
docker run --rm -v "$PWD:/app" -w /app node:22-alpine npm install --no-audit --no-fund
