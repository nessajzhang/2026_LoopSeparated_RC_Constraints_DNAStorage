#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/output/fixedk_smc_audit}"
CXX="${CXX:-c++}"
JOBS="${JOBS:-4}"
BIN="$OUT/smc_regime_grid_fast"
N_ALL="100,200,500,1000,2000,5000,10000,20000,50000,100000"

mkdir -p "$OUT"
"$CXX" -O3 -std=c++17 -pthread "$ROOT/analysis_tools/smc_regime_grid_fast.cpp" -o "$BIN"

BASE_K_VALUES=(0 1 2 5 10)

for k in "${BASE_K_VALUES[@]}"; do
  seed=$((2026070800 + 1000 * k))
  "$BIN" --m 2 --n-values "$N_ALL" --regimes fixed --fixed-k "$k" \
    --particles 2500 --reps 4 --jobs "$JOBS" --seed "$seed" \
    --out-dir "$OUT/base_k$k"
done

head -n 1 "$OUT/base_k0/smc_regime_m2_raw.csv" > "$OUT/base_raw.csv"
for k in "${BASE_K_VALUES[@]}"; do
  tail -n +2 "$OUT/base_k$k/smc_regime_m2_raw.csv" >> "$OUT/base_raw.csv"
done

"$BIN" --m 2 --n-values 100,200,500,1000,2000 --regimes fixed --fixed-k 20 \
  --particles 30000 --reps 8 --jobs "$JOBS" --seed 2026072920 \
  --out-dir "$OUT/k20_short"

"$BIN" --m 2 --n-values 5000,10000,20000,50000,100000 --regimes fixed --fixed-k 20 \
  --particles 12000 --reps 8 --jobs "$JOBS" --seed 2026072820 \
  --out-dir "$OUT/k20_long"

python3 "$ROOT/analysis_tools/curate_fixedk_smc_summary.py" \
  --base-raw "$OUT/base_raw.csv" \
  --k20-short-raw "$OUT/k20_short/smc_regime_m2_raw.csv" \
  --k20-long-raw "$OUT/k20_long/smc_regime_m2_raw.csv" \
  --out "$OUT/large_n_smc_m2_fixedk_extended_filtered_summary.csv"

python3 "$ROOT/analysis_tools/plot_large_n_smc_audit.py" \
  --out-prefix "$OUT/fig_large_n_smc_audit" \
  --m2 "$OUT/large_n_smc_m2_fixedk_extended_filtered_summary.csv" \
  --validation "$ROOT/analysis_tools/large_n_smc_m2_validation_against_exact.csv" \
  --six-panel

echo "Fixed-k SMC audit completed in $OUT"
