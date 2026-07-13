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
uv run ruff check code loop_alpha_audit tests
uv run ruff format --check code loop_alpha_audit tests
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
explanation can be reproduced with:

```bash
uv run python code/audit_folding_wobble_runs.py ROW_LEVEL_FOLDING.csv \
  --assert-manuscript-counts --output output/folding_wobble_audit.json
```

## Ancillary SMC and folding workflows

The earlier large-block SMC, finite-length plotting, and ViennaRNA screen are
retained under `analysis_tools/`. Their additional dependencies are listed in
`analysis_tools/requirements.txt` and
`analysis_tools/requirements_folding.txt`. The fixed-threshold SMC workflow is
started with:

```bash
bash analysis_tools/run_fixedk_smc_audit.sh
```

These scripts accept explicit seeds and write generated products under
`output/`.

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
- The paired endpoint language is propagated completely through `r=5000` and
  cross-checked against an exact-integer Python implementation.
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
