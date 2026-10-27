#!/usr/bin/env python3
"""Compare deployment models on identical preselected words/files only."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--details", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(
        (ROOT / "data/physical_revision_v1/run001/config.json").read_text()
    )
    by_word = defaultdict(dict)
    with gzip.open(a.details / "fold_details.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["arm"] != "deployment":
                continue
            assert row["model_id"] not in by_word[row["row_id"]]
            by_word[row["row_id"]][row["model_id"]] = row
    common = {key: value for key, value in by_word.items() if len(value) == 10}
    expected = (
        len(config["m_values"])
        * len(config["sensitivity_k"])
        * len(config["methods"])
        * len(config["deployment_seeds"])
        * config["sensitivity_files_per_seed_cell_method"]
        * len(config["ensemble_strand_indices"])
    )
    assert len(common) == expected == 1152
    grouped = defaultdict(list)
    for models in common.values():
        assert len(models) == 10
        for model, row in models.items():
            assert row["n"] == 150 and row["k"] in config["sensitivity_k"]
            assert row["file_id"] < 2 and row["ensemble_paired_fraction"] is not None
            key = (
                row["m"],
                row["k"],
                row["method"],
                model,
                row["seed"],
                row["file_id"],
            )
            grouped[key].append(row)
    files = []
    for key, rows in sorted(grouped.items()):
        assert len(rows) == 8
        files.append(
            dict(zip(("m", "k", "method", "model", "seed", "file_id"), key))
            | {
                "n": 150,
                "preselected_strands": 8,
                "stem_fraction": statistics.fmean(r["has_stem_ge_m"] for r in rows),
                "MFE_mean": statistics.fmean(r["mfe"] for r in rows),
                "ensemble_paired_fraction_mean": statistics.fmean(
                    r["ensemble_paired_fraction"] for r in rows
                ),
                "GT_pair_probability_mass_mean": statistics.fmean(
                    r["GT_pair_probability_mass"] for r in rows
                ),
            }
        )
    summaries = defaultdict(list)
    for row in files:
        summaries[tuple(row[k] for k in ("m", "k", "method", "model"))].append(row)
    cells = []
    for key, rows in sorted(summaries.items()):
        assert len(rows) == 6 and len({(r["seed"], r["file_id"]) for r in rows}) == 6
        cells.append(
            dict(zip(("m", "k", "method", "model"), key))
            | {
                "n": 150,
                "N_files": 6,
                "strands_per_file": 8,
                "mean_file_stem_fraction": statistics.fmean(
                    r["stem_fraction"] for r in rows
                ),
                "mean_file_MFE": statistics.fmean(r["MFE_mean"] for r in rows),
                "mean_file_ensemble_paired_fraction": statistics.fmean(
                    r["ensemble_paired_fraction_mean"] for r in rows
                ),
                "shared_file_keys_across_models": True,
                "descriptive_not_independent_model_replicates": True,
            }
        )
    assert len(files) == 1440 and len(cells) == 240
    for name, value in (("common_file_values", files), ("common_cells", cells)):
        (a.out / f"{name}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
        with (a.out / f"{name}.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, sorted(value[0]))
            writer.writeheader()
            writer.writerows(value)
    result = {
        "status": "ALL_TEN_MODELS_USE_IDENTICAL_PRESELECTED_DEPLOYMENT_WORDS",
        "word_ids": 1152,
        "file_model_rows": 1440,
        "model_cells": 240,
        "files_per_cell": 6,
        "strands_per_file": 8,
        "new_folding_calls": 0,
        "source_detail_sha256": json.loads((a.details / "report.json").read_text())[
            "fold_details_sha256"
        ],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (a.out / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
