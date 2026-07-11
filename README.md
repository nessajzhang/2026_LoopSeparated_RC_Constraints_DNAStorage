# Loop-separated reverse-complement constraints

This repository contains computational tools for studying DNA words subject to loop-separated reverse-complement constraints. It is a code-only distribution; generated figures and experiment outputs are intentionally excluded.

## Contents

- `analysis_tools/exact_counts.py`: exact finite-length enumeration for small parameters.
- `analysis_tools/validate_enumerator.py`: independent brute-force checks of the enumerator.
- `analysis_tools/smc_regime_grid_fast.cpp`: multithreaded sequential Monte Carlo estimator for large block lengths.
- `analysis_tools/run_fixedk_smc_audit.sh`: reproducible fixed-threshold SMC sweep, including self-generated exact/SMC calibration data.
- `analysis_tools/folding_grid.py` and `analysis_tools/folding_validation.py`: seeded sequence generation and ViennaRNA evaluation.
- `analysis_tools/folding_statistical_tests.py`: statistical comparisons for folding outputs.
- `analysis_tools/plot_*.py`: figure-generation utilities.
- `analysis_tools/check_*_sanity.py`: finite checks of combinatorial and matrix calculations.

Generated CSV files and figures are intentionally not versioned. Reproduction
scripts write them under `output/` from the checked-in source code.

## Requirements

- Python 3.9 or newer.
- A C++17 compiler with thread support.
- Bash for the full fixed-threshold audit.

Install the Python plotting and statistical dependencies with
`python3 -m pip install -r analysis_tools/requirements.txt`. The optional
ViennaRNA pipeline additionally uses
`analysis_tools/requirements_folding.txt`.

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

All scripts expose their input and output options through `--help`.

Run the complete fixed-threshold SMC and calibration workflow with:

```bash
bash analysis_tools/run_fixedk_smc_audit.sh
```

This is a compute-intensive reproduction run. It creates all intermediate
tables and the six-panel audit figure under `output/fixedk_smc_audit`; no
pre-generated data are required.

## Reproducibility notes

All stochastic programs accept an explicit seed. Large SMC estimates should be inspected at replicate level; a finite particle population can collapse when the constraint becomes too restrictive. Folding calculations depend on the ViennaRNA version and parameter set reported in the generated tables.

No software license is granted at this stage.
