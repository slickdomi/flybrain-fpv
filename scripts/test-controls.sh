#!/bin/sh
# Control tests: what each stimulation does to the drone, and whether designating food steers it.
# Headless GPU runs of ?bench= protocols (web/src/bench.ts), then web/scripts/check-controls.mjs grades them.
#
#   sh scripts/test-controls.sh                    # suites "open closed food", seeds "1 2": ~70 min
#   SUITES=food SEEDS="1 2 3" sh scripts/test-controls.sh   # the five-arm food-mode comparison, ~50 min
#   SUITES=open SEEDS=1 sh scripts/test-controls.sh
#   QUERY='&pitchref=0' OUT=.cache/tests-pitchref0 sh scripts/test-controls.sh
#   STIMS=1,2,paintL,paintR MVS=10,20 REPS=3 SUITES=open sh scripts/test-controls.sh
#
#   open    every stimulus with the drone held still: the brain's commands, uncontaminated by a changing view
#   closed  every stimulus while it flies: the turn and climb it actually makes
#   approach  obstacles: the drone flies straight at one tree set off to the side, steering off and nothing solid;
#           read with web/scripts/analyze-approach.mjs (check-controls.mjs does not grade it)
#   food    a truck designated as the target for SECS brain seconds, once per food mode (FOOD_MODES="none marker
#           paint both learn"): what each target drive adds over the fly on its own
#   survive the forest for SECS brain seconds, nothing to chase: how often it crashes
#
# Every run is in the game's default world (Forest mode: thick trunks and the looming turn-away).
# QUERY='&forest=0' runs the thin-trunk world instead.
#
# RESUMABLE: a run whose bench.json exists is skipped; delete its directory (or FRESH=1) to redo it.
# Build first (scripts/build.sh), and don't rebuild while this runs: every run serves web/dist.
# One GPU job at a time. Exit status is check-controls.mjs's: non-zero only for PLUMBING failures.
set -eu
cd "$(dirname "$0")/.."
SUITES=${SUITES:-"open closed food"}
SEEDS=${SEEDS:-"1 2"}
OUT=${OUT:-.cache/tests}
EXTRA=${QUERY:-}
SECS=${SECS:-180}
[ -n "${STIMS:-}" ] && EXTRA="$EXTRA&stims=$STIMS"
[ -n "${MVS:-}" ] && EXTRA="$EXTRA&mvs=$MVS"
[ -n "${REPS:-}" ] && EXTRA="$EXTRA&reps=$REPS"

run() { # name query wall-cap
  dir="$OUT/$1"
  if [ -f "$dir/bench.json" ] && [ -z "${FRESH:-}" ]; then
    echo "--- $1: done already, skipping"
    return
  fi
  echo "--- $1 ($2)"
  OUT="$dir" QUERY="$2" sh scripts/play.sh "$3" 2>&1 | grep -E 'test done|SUMMARY|LOAD FAILED|ERROR|error' || echo "  (run failed)"
}

# seeds outer, suites inner: drift in whatever else shares the GPU spreads over every suite
for seed in $SEEDS; do
  for suite in $SUITES; do
    case "$suite" in
      open) run "open/seed$seed" "?bench=stim&loop=open&seed=$seed$EXTRA" 1500 ;;
      closed) run "closed/seed$seed" "?bench=stim&loop=closed&seed=$seed$EXTRA" 1500 ;;
      food)
        # the same seeds and the same truck in every arm
        for mode in ${FOOD_MODES:-none marker paint both learn}; do
          run "food-$mode/seed$seed" "?bench=chase&secs=$SECS&food=$mode&seed=$seed$EXTRA" $((SECS * 2 + 120))
        done
        ;;
      approach) run "approach/seed$seed" "?bench=approach&cars=0&balloons=0&seed=$seed$EXTRA" 900 ;;
      survive) run "survive/seed$seed" "?bench=survive&cars=0&balloons=0&secs=$SECS&seed=$seed$EXTRA" $((SECS * 2 + 120)) ;;
      *) echo "unknown suite $suite (open | closed | food | approach | survive)"; exit 1 ;;
    esac
  done
done

docker run --rm -v "$PWD:/app" -w /app node:22-alpine node web/scripts/check-controls.mjs "$OUT"
