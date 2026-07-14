# Loop-separated reverse-complement constraints: reproducibility code

This code-only package contains no manuscript source, author biography,
affiliation, correspondence address, or unpublished prose. It reproduces the
proof-supporting computations for loop-separated reverse-complement constraints.

The current audited implementation is organized under `code/`,
`loop_alpha_audit/`, `tests/`, and `output/`. The `analysis_tools/` directory
retains the earlier finite-length, SMC, and folding-screen workflows so that
previous numerical artifacts remain reproducible.

## Environment

- Python 3.12
- `uv sync`
- A C++20 compiler with POSIX threads

## Routine validation

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## Exact and finite-block computations

```bash
uv run python code/run_exhaustive_consistency_audit.py
uv run python code/certify_dna_m2_spectral.py

clang++ -std=c++20 -O3 -DNDEBUG -Wall -Wextra -pedantic \
  code/paired_count_dna_m2.cpp -o output/paired_count_dna_m2
output/paired_count_dna_m2 output/paired_dna_m2_cpp_rates_r5000.csv 5000
uv run python code/check_paired_cpp_against_python.py \
  output/paired_dna_m2_cpp_rates_r5000.csv \
  output/paired_cpp_python_crosscheck_r5000.json

clang++ -std=c++20 -O3 -DNDEBUG -Wall -Wextra -pedantic \
  code/full_word_definition_audit.cpp -o output/full_word_definition_audit
output/full_word_definition_audit --extended --threads 10 \
  --summary output/full_word_definition_audit_extended_10t_summary.json \
  --csv output/full_word_definition_audit_extended_10t_jobs.csv
```

The extended profile enumerates actual words rather than occurrence sets. It
checks every available stem threshold and every finite loop threshold for the
specified binary, ternary, and quaternary jobs.

## Broad parameter and encoder computations

```bash
uv run python code/run_segmented_encoder_ultrascale.py
uv run python code/run_general_m_stress_audit.py
uv run python code/run_mkn_orbit_stress.py
uv run python code/make_scaling_evidence_figure.py
```

Given the row-level ViennaRNA CSV, the Watson--Crick versus wobble-pair
explanation can be reproduced with the following audit. It checks every
contiguous length-`m` subrun inside every qualifying predicted stem, rather
than checking only the maximal run as a whole.

```bash
uv run python code/audit_folding_wobble_runs.py ROW_LEVEL_FOLDING.csv \
  --assert-manuscript-counts --output output/folding_wobble_audit.json
```

## Ancillary SMC and folding workflows

The earlier large-block SMC, finite-length plotting, and ViennaRNA screen are
retained under `analysis_tools/`. Their additional dependencies are listed in
`analysis_tools/requirements.txt` and
`analysis_tools/requirements_folding.txt`. The frozen finite-length counts,
selected SMC replicate tables, curated SMC summary, and derived folding summary
are under `output/`. The exact finite-length and SMC figures can be regenerated
without rerunning the expensive experiments:

```bash
uv run python analysis_tools/plot_finite_length.py
uv run python analysis_tools/plot_large_n_smc_audit.py \
  --out-prefix output/figures/fig_large_n_smc_audit \
  --m2 output/smc/large_n_smc_m2_fixedk_summary.csv \
  --validation output/smc/large_n_smc_m2_validation_against_exact.csv \
  --six-panel
uv run python analysis_tools/plot_folding_comparison_tcom.py \
  --comparison output/folding/folding_comparison_len100_150_200_m6_7_s1000.csv \
  --out output/figures/fig_folding_validation_tcom.pdf
```

The selected SMC summary can also be rebuilt from the retained replicate rows:

```bash
uv run python analysis_tools/curate_fixedk_smc_summary.py \
  --base-raw output/smc/large_n_smc_m2_fixedk_selected_base_raw.csv \
  --k20-short-raw output/smc/large_n_smc_m2_k20_refine_short_raw.csv \
  --k20-long-raw output/smc/large_n_smc_m2_k20_refine_long_raw.csv \
  --out output/smc/large_n_smc_m2_fixedk_summary_rebuilt.csv
```

The full fixed-threshold SMC workflow is started with:

```bash
bash analysis_tools/run_fixedk_smc_audit.sh
```

The 1000-sequence-per-group ViennaRNA grid reported by the accompanying study
used seed `20260629` and requires ViennaRNA 2.7.2. Install the additional
dependency into the project environment and regenerate the row-level grid with:

```bash
uv pip install --python .venv/bin/python \
  -r analysis_tools/requirements_folding.txt
uv run python analysis_tools/folding_grid.py \
  --lengths 100,150,200 --m-values 6,7 --k-values 0,3,4,5 \
  --samples 1000 --seed 20260629 \
  --gc-low 0.40 --gc-high 0.60 --max-homopolymer 4 \
  --out output/folding_grid_1000_rows.csv \
  --summary-out output/folding_grid_1000_summary.csv
uv run python analysis_tools/summarize_folding_comparison.py \
  --rows output/folding_grid_1000_rows.csv --m-values 6,7 \
  --out output/folding_comparison_1000.csv
uv run python analysis_tools/plot_folding_comparison_tcom.py \
  --comparison output/folding_comparison_1000.csv \
  --out output/fig_folding_validation_tcom.pdf
```

These scripts accept explicit seeds and write generated products under
`output/`. The row-level folding table is intentionally not committed because
it is large; the deterministic seed and complete command are recorded above.

## Optional long occurrence-position campaign

The following command is intentionally expensive. It stress-tests the geometric
interval lemma; it is not a capacity computation.

```bash
CASES_PER_SEED=2000000000 THREADS=10 \
  bash code/run_long_orbit_campaign.sh \
  20260712 20260713 20260717 314159265 271828182
uv run python code/aggregate_long_orbit_campaign.py
```

The aggregator accepts either uncompressed `strata_seed_*.csv` files from a new
run or the compressed `strata_seed_*.csv.gz` files retained in this release.

## Evidence levels

- The DNA `m=2` spectral maximum is an exact rational/symbolic certificate over
  all 65,536 subsets.
- The complete paired-endpoint state space is propagated numerically through
  `r=5000`; an independent exact-integer implementation cross-checks nine
  checkpoints through `r=12`.
- The broad-`m` construction audit includes 217 exact finite-length cells over
  15 stem thresholds through `m=128`, eight linear-threshold ratios, and two
  block lengths; certified asymptotic cells extend through `m=1024`.
- General-`m` tables are certified structured lower bounds. For `m>2`, they are
  not global computations of the spectral maximum.
- The large `(m,k,n)` rows and occurrence-position campaign stress proof
  geometry. They are not uniform language samples and not capacity estimates.
- Numerical checks can reveal implementation or indexing errors but do not
  replace the universal proofs.

All retained outputs have deterministic parameters or seeds. `MANIFEST.sha256`
records the public code, tests, and frozen machine-readable artifacts.
