# Loop-separated reverse-complement constraints

Executable constrained-sequence mappings, exact spectral certificates, diagnostic
storage-link simulations and computational structure predictions.

The current implementation is the **2026-10-27 code-only release**. The earlier
`318a65699470a29e37332ca5fa0bdb03578d7c7d` commit remains available in history.
No manuscript, review correspondence, author-contact information or institutional
account data is distributed in this repository.

## Environment

Python 3.12 and `uv` are required. Install all computational extras before
running the full test suite.

```sh
uv sync --frozen --all-extras
uv run --frozen pytest -q
```

`loop_alpha_audit/` contains the mapping, framing, receiving, channel and
spectral implementations. Executable drivers are in `code/`; frozen
computational grids are in `configs/revision/`. Configuration labels describe
experiment versions, not editorial correspondence.

## Exact spectral checks

```sh
# Independent integer scan of all 65,536 DNA dinucleotide subsets.
uv run --frozen python code/check_m2_full_domain.py --output output/new_m2_audit

# Produce and independently check the m=3 finite certificate.
uv run --frozen python code/certify_m3_analytic.py --output-dir output/new_m3_audit

# Check the included m=3 certificate without third-party imports.
uv run --frozen python -I code/check_m3_analytic_certificate.py \
  --certificate certificates/m3/certificate.json
```

Choose new output directories. The m=3 optimum is justified by an analytic
inequality and its equality witness. The finite checker verifies exact
identities and the supplied witness; it is not enumeration of all `2^64`
subsets and is not a proof-assistant formalization. Earlier incomplete search
code is retained as incomplete-search tooling, not as the global certificate.

## Actual mappings and receiving

`prepared_codec.py` implements endpoint mappings and their binary interface.
`short_loop_codec.py` supplies the separate short-cutoff mappings. An endpoint
mapper requiring a high separation ratio is not used as a short-cutoff codec.
The storage link adds addresses, CRC checks, observed-read reconstruction and
outer Reed--Solomon recovery. Inverse enumeration is not error correction.

```sh
# Validate a small diagnostic configuration without launching a simulation.
uv run --frozen python code/run_ids_benchmark.py \
  --config configs/revision/ids_pilot_v1.json \
  --out output/ids_pilot --validate-only

# Run the bounded pilot explicitly, with four worker processes.
uv run --frozen python code/run_ids_benchmark.py \
  --config configs/revision/ids_pilot_v1.json \
  --out output/ids_pilot --jobs 4

# Inspect the measured-cost driver interface.
uv run --frozen python code/benchmark_codecs.py --help
```

Large formal configurations enforce source/configuration freezes. Do not remove
those checks or reinterpret a failed/capped encoding as a successful trial.
The receiving interface accepts observations and public parameters, not sender
payloads, original strands or edit histories. Public sequencing-profile drivers
require their explicitly supplied external simulator/source snapshot; they do
not silently download data or submit cloud jobs.

The schema and generator wrappers in `reproducibility/profiles/` include the
bounded source-profile interface tests. Their two small historical CSV fixtures
retain an observed source-pool overdraw rather than correcting it away.
The wrappers do not include the external simulator, institutional runtime or
account settings.

## Source tables and reproducibility

`source_data/20261027/` contains frozen communication, structure-prediction,
sequencing-profile and mapper-cost tables. It preserves failed trials,
non-evaluable quantities and null comparisons. The controlled IDS comparisons
do not establish a FER benefit or equivalence for RC screening. Structure
predictions are in silico and are not measured biochemical or channel-error
improvements.

`reproducibility/analysis/` contains the byte-identical statistical builders
used for these tables. Full intermediate inputs are distributed separately in
the computational data archive; they are not present in this repository. To
rebuild tables from an extracted frozen archive while preserving its
root-relative provenance layout:

```sh
uv run --frozen python reproducibility/reproduce_tables.py \
  --bundle /path/to/extracted/computational_bundle \
  --out /path/to/new_table_rebuild --jobs 4
```

The wrapper checks source identity before executing the builders inside that
bundle. This reconstructs tables from saved observations, not all original
simulations. The larger full-raw archives are a separate layer; a checksum
receipt alone does not provide their contents.

Plotting source is under `reproducibility/figures_src/`. For example:

```sh
uv run --frozen python reproducibility/figures_src/plot_communication_results.py \
  --tables source_data/20261027/communication \
  --qa-scripts reproducibility/figures_src/qa_portable \
  --out output/new_communication_plot
uv run --frozen python reproducibility/figures_src/plot_physical_results.py \
  --tables source_data/20261027/physical \
  --qa-scripts reproducibility/figures_src/qa_portable \
  --out output/new_physical_plot
```

The optional two-row vector schematic additionally requires ReportLab and
PyMuPDF. No rendered manuscript figures or manuscript documents are included.

## Earlier workflows

Existing `analysis_tools/`, `output/` and `figures/source_data/` retain earlier
finite-length, SMC and folding computations. Their saved experiments are
distinct from the new source-table release and must not be pooled as independent
samples. The exact-certificate row digest is distinct from its compressed-file
digest; use the explicitly named digest field when checking either.

`MANIFEST.sha256` and `PUBLIC_RELEASE.json` record the current public file
snapshot. They exclude runtime caches and self-referential manifest files.
