#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/output/long_orbit_campaign"
BIN="$OUT/orbit_interval_million_audit"
mkdir -p "$OUT"

clang++ -std=c++20 -O3 -DNDEBUG -Wall -Wextra -pedantic \
  "$ROOT/code/orbit_interval_million_audit.cpp" -o "$BIN"

CASES_PER_SEED="${CASES_PER_SEED:-2000000000}"
THREADS="${THREADS:-10}"
if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(20260712 20260713 20260717 314159265 271828182)
fi

for SEED in "${SEEDS[@]}"; do
  /usr/bin/time -p "$BIN" \
    --cases "$CASES_PER_SEED" \
    --threads "$THREADS" \
    --seed "$SEED" \
    --summary "$OUT/summary_seed_${SEED}.json" \
    --csv "$OUT/strata_seed_${SEED}.csv"
done
