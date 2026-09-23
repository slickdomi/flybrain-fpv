#!/bin/sh
# One headless flight on the host GPU: prints a status line every 5 s, writes screenshots of every view
# and summary.json to .cache/smoke, and exits non-zero if the page logged any error.
#
# Reuses the flygames-gputest image from fly-games (Chromium + Mesa Vulkan). Build it there if missing:
#   sh ../fly-games/scripts/gpu-image.sh
#
#   sh scripts/play.sh            # 40 s on the GPU
#   sh scripts/play.sh 120        # longer
#   PRESS="2:10-15" sh scripts/play.sh 30    # hold stimulation key 2 from 10 s to 15 s
# Env: ADAPTER=swiftshader (no GPU, far slower), GPU_DEVICE, QUERY='?seed=2', VIEWPORT=1920x1080, PRESS, OUT,
# DIST (the build to serve, as seen inside the container: default /app/dist, e.g. /app/dist-graphics).
# Build first (scripts/build.sh): it serves web/dist.
set -eu
cd "$(dirname "$0")/.."
SECONDS_TO_RUN=${1:-40}
ADAPTER=${ADAPTER:-hardware}
OUT=${OUT:-.cache/smoke}
IMAGE=${IMAGE:-flygames-gputest}
mkdir -p "$OUT"
DEVICE=""
[ "$ADAPTER" = hardware ] && DEVICE="--device ${GPU_DEVICE:-/dev/dri/renderD128}"
# shellcheck disable=SC2086
docker run --rm $DEVICE --ipc=host -v "$PWD/web:/app:ro" -v "$PWD/$OUT:/out" \
  -e ADAPTER="$ADAPTER" -e SECONDS="$SECONDS_TO_RUN" -e QUERY="${QUERY:-}" -e VIEWPORT="${VIEWPORT:-}" -e PRESS="${PRESS:-}" \
  -e DIST="${DIST:-/app/dist}" \
  "$IMAGE" sh -c 'cp /app/scripts/smoke.mjs /runner/ && node /runner/smoke.mjs'
