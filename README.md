# Loop-separated reverse-complement constraints

This repository contains computational tools for studying DNA words subject to loop-separated reverse-complement constraints. It is a code-only distribution: manuscripts, author metadata, generated figures, and experiment outputs are intentionally excluded.

## Contents

- `exact_counts.py`: exact finite-length enumeration for small parameters.
- `validate_enumerator.py`: independent brute-force checks of the enumerator.
- `smc_regime_grid_fast.cpp`: multithreaded sequential Monte Carlo estimator for large block lengths.
- `run_fixedk_smc_audit.sh`: reproducible fixed-threshold SMC sweep.
- `folding_grid.py` and `folding_validation.py`: seeded sequence generation and ViennaRNA evaluation.
- `folding_statistical_tests.py`: statistical comparisons for folding outputs.
- `plot_*.py`: figure-generation utilities.
- `check_*_sanity.py`: finite checks of combinatorial and matrix calculations.

## Quick checks

Run the exact-count validator:

```bash
cd analysis_tools
python3 validate_enumerator.py
python3 check_loop_separated_proof_sanity.py
python3 check_capacity_baseline_empirical.py
python3 check_matrix_growth_sanity.py
```

Compile a small SMC run:

```bash
c++ -O3 -std=c++17 -pthread analysis_tools/smc_regime_grid_fast.cpp -o /tmp/smc_regime_grid_fast
/tmp/smc_regime_grid_fast \
  --m 2 --n-values 100,500,1000 --regimes fixed --fixed-k 5 \
  --particles 1000 --reps 2 --jobs 2 --seed 20260708 \
  --out-dir /tmp/smc_example
```

For the folding pipeline, install the optional Python dependencies in `analysis_tools/requirements_folding.txt`. The scripts expose their full input and output options through `--help`.

## Reproducibility notes

All stochastic programs accept an explicit seed. Large SMC estimates should be inspected at replicate level; a finite particle population can collapse when the constraint becomes too restrictive. Folding calculations depend on the ViennaRNA version and parameter set reported in the generated tables.

No software license is granted at this stage.

