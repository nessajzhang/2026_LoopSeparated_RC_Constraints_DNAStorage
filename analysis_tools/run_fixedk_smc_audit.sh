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

# Build the small-n calibration table from source rather than relying on a
# pre-generated CSV.  Exact counts are computed through n=40 for k=0,1,2 and
# through n=10 for k=5; the latter supplies the additional short calibration
# point without making exact enumeration the dominant cost of this already
# large SMC audit.
CALIBRATION="$OUT/calibration"
mkdir -p "$CALIBRATION/exact" "$CALIBRATION/smc"
for k in 0 1 2; do
  python3 "$ROOT/analysis_tools/exact_counts.py" \
    --m 2 --k "$k" --n-max 40 \
    --out "$CALIBRATION/exact/counts_m2_k${k}.csv"
done
python3 "$ROOT/analysis_tools/exact_counts.py" \
  --m 2 --k 5 --n-max 10 \
  --out "$CALIBRATION/exact/counts_m2_k5.csv"

python3 "$ROOT/analysis_tools/large_n_smc_trend.py" \
  --m 2 --ks 0,1,2,5 --n-values 10,20,30,40 \
  --particles 4000 --reps 5 --jobs "$JOBS" --seed 20260707 \
  --out-dir "$CALIBRATION/smc" --exact-dir "$CALIBRATION/exact"

VALIDATION="$CALIBRATION/smc/smc_m2_validation_against_exact.csv"
if [[ ! -s "$VALIDATION" ]]; then
  echo "Calibration table was not generated: $VALIDATION" >&2
  exit 1
fi

python3 "$ROOT/analysis_tools/plot_large_n_smc_audit.py" \
  --out-prefix "$OUT/fig_large_n_smc_audit" \
  --m2 "$OUT/large_n_smc_m2_fixedk_extended_filtered_summary.csv" \
  --validation "$VALIDATION" \
  --six-panel

echo "Fixed-k SMC audit completed in $OUT"
