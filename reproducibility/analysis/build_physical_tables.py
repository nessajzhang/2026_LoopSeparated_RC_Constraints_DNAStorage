#!/usr/bin/env python3
"""Complete matched, actual-message, sensitivity and boundary tables.

Primary intervals use the predeclared 24-cell conservative paired-discordance
family. Deployment summaries use files, not strands, as units. Sensitivities
are restricted to identical preselected pair IDs across all 10 models.
These are computational predictions, never causal or measured FER effects.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = "T37_S1.021_default"
METRICS = (
    "has_stem_ge_m",
    "mfe",
    "ensemble_paired_fraction",
    "GT_pair_probability_mass",
)


def read(path):
    return json.loads(path.read_text())


def exact(events, n, alpha=0.05):
    ci = binomtest(events, n).proportion_ci(1 - alpha, method="exact")
    return [float(ci.low), float(ci.high)]


def dump(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(text)


def save(out, name, rows):
    dump(out / f"{name}.json", rows)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (out / f"{name}.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (dict, list)) else v
                for k, v in r.items()
            }
            for r in rows
        )


def pair_summary(pairs, alpha=0.05):
    target, control = [p["target"] for p in pairs], [p["control"] for p in pairs]
    n = len(pairs)
    a = sum(
        c["has_stem_ge_m"] and not t["has_stem_ge_m"] for t, c in zip(target, control)
    )
    b = sum(
        t["has_stem_ge_m"] and not c["has_stem_ge_m"] for t, c in zip(target, control)
    )
    lo_a, hi_a = exact(a, n, alpha / 2)
    lo_b, hi_b = exact(b, n, alpha / 2)
    result = {
        "N_pairs": n,
        "control_only_stem": a,
        "target_only_stem": b,
        "target_stem_fraction": statistics.fmean(t["has_stem_ge_m"] for t in target),
        "control_stem_fraction": statistics.fmean(c["has_stem_ge_m"] for c in control),
        "difference_pp": 100 * (a - b) / n,
        "CI_pp": [100 * max(-1.0, lo_a - hi_b), 100 * min(1.0, hi_a - lo_b)],
        "alpha": alpha,
        "direction": "control-minus-target predicted incidence;not causal",
        "GC_mean_difference_exact": statistics.fmean(
            c["GC_fraction"] - t["GC_fraction"] for t, c in zip(target, control)
        ),
    }
    assert result["GC_mean_difference_exact"] == 0
    for metric in METRICS[1:]:
        differences = [
            c[metric] - t[metric]
            for t, c in zip(target, control)
            if c[metric] is not None and t[metric] is not None
        ]
        result[f"N_{metric}_pairs"] = len(differences)
        result[f"mean_control_minus_target_{metric}"] = (
            statistics.fmean(differences) if differences else None
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--details", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.details = a.details.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    extraction = read(a.details / "report.json")
    assert extraction["rows"] == 230424 and extraction["new_native_folding_calls"] == 0
    mechanism = defaultdict(dict)
    deployment = defaultdict(list)
    high, witnesses = defaultdict(list), []
    count, unique_primary = 0, set()
    alpha_config = read(ROOT / "data/physical_revision_v1/run001/config.json")["alphas"]
    with gzip.open(a.details / "fold_details.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            assert row["status"] == "OK"
            count += 1
            if row["model_id"] == PRIMARY:
                assert row["row_id"] not in unique_primary
                unique_primary.add(row["row_id"])
            if row["arm"] == "mechanism":
                key = row["n"], row["m"], row["k"], row["model_id"], row["pair_id"]
                assert (
                    row["role"] in ("target", "control")
                    and row["role"] not in mechanism[key]
                )
                mechanism[key][row["role"]] = row
            elif row["arm"] == "deployment":
                key = (
                    row["n"],
                    row["m"],
                    row["k"],
                    row["model_id"],
                    row["method"],
                    row["seed"],
                    row["file_id"],
                )
                deployment[key].append(row)
            elif row["arm"] == "high_alpha":
                nominal = [x for x in alpha_config if math.floor(x * row["n"]) == row["k"]]
                assert len(nominal) == 1
                key = row["n"], row["m"], nominal[0], row["model_id"]
                high[key].append(row)
            else:
                assert row["arm"] == "witness"
                witnesses.append(row)
    assert count == 230424 and len(unique_primary) == 205068
    grouped = defaultdict(list)
    pair_models = defaultdict(set)
    for (n, m, k, model, pair_id), pair in mechanism.items():
        assert set(pair) == {"target", "control"}
        grouped[n, m, k, model].append(pair)
        pair_models[n, m, k, pair_id].add(model)
    primary = []
    for key, pairs in sorted(grouped.items()):
        if key[-1] == PRIMARY:
            assert len(pairs) == 1152
            primary.append(
                dict(zip(("n", "m", "k", "model"), key))
                | pair_summary(pairs, 0.05 / 24)
            )
    assert len(primary) == 24
    frozen = read(ROOT / "data/physical_revision_v1/run001/SUMMARY.json")[
        "mechanism_cells"
    ]
    original = {(r["n"], r["m"], r["k"]): r for r in frozen}
    for row in primary:
        source = original[row["n"], row["m"], row["k"]]
        assert (
            abs(
                source["paired_discordance_conservative_interval"]["difference_pp"]
                - row["difference_pp"]
            )
            < 1e-12
        )
        assert max(abs(x - y) for x, y in zip(source["CI_pp"], row["CI_pp"])) < 1e-10
    sensitivity = []
    for key, pairs in sorted(grouped.items()):
        n, m, k, model = key
        common = [
            pair
            for pair in pairs
            if len(pair_models[n, m, k, pair["target"]["pair_id"]]) == 10
        ]
        if common:
            assert len(common) == 96
            sensitivity.append(
                dict(zip(("n", "m", "k", "model"), key)) | pair_summary(common)
            )
    frozen_config = read(ROOT / "data/physical_revision_v1/run001/config.json")
    assert len(sensitivity) == 10 * len(frozen_config["m_values"]) * len(frozen_config["sensitivity_k"])
    deployment_rows = []
    for key, rows in sorted(deployment.items()):
        n, m, k, model, method, seed, file_id = key
        expected = 63 if model == PRIMARY else 8
        assert len(rows) == expected
        deployment_rows.append(
            {
                "n": n,
                "m": m,
                "k": k,
                "model": model,
                "method": method,
                "seed": seed,
                "file_id": file_id,
                "strands_folded": len(rows),
                "uncertainty_unit": "file",
                "stem_fraction": statistics.fmean(r["has_stem_ge_m"] for r in rows),
                "mean_MFE": statistics.fmean(r["mfe"] for r in rows),
                "WC_m_subrun_fraction": statistics.fmean(
                    r["has_WC_m_subrun"] for r in rows
                ),
                "GT_m_subrun_fraction": statistics.fmean(
                    r["has_GT_m_subrun"] for r in rows
                ),
                "constraint_valid_fraction": statistics.fmean(
                    r["full_constraint_valid"] for r in rows
                ),
                "mean_ensemble_paired_fraction": statistics.fmean(
                    r["ensemble_paired_fraction"]
                    for r in rows
                    if r["ensemble_paired_fraction"] is not None
                ),
                "ensemble_strands": sum(
                    r["ensemble_paired_fraction"] is not None for r in rows
                ),
            }
        )
    deploy_groups = defaultdict(list)
    for r in deployment_rows:
        deploy_groups[r["n"], r["m"], r["k"], r["model"], r["method"]].append(r)
    deployment_summary = []
    for key, rows in sorted(deploy_groups.items()):
        is_primary = key[3] == PRIMARY
        assert len(rows) == (24 if is_primary else 6)
        values = sorted(r["stem_fraction"] for r in rows)
        deployment_summary.append(
            dict(zip(("n", "m", "k", "model", "method"), key))
            | {
                "N_independent_files": len(rows),
                "strands_per_file": rows[0]["strands_folded"],
                "mean_file_stem_fraction": statistics.fmean(values),
                "median_file_stem_fraction": statistics.median(values),
                "min_file_stem_fraction": min(values),
                "max_file_stem_fraction": max(values),
                "constraint_valid_fraction": statistics.fmean(
                    r["constraint_valid_fraction"] for r in rows
                ),
                "mean_file_ensemble_paired_fraction": statistics.fmean(
                    r["mean_ensemble_paired_fraction"] for r in rows
                ),
                "intervals_not_strand_binomial": True,
            }
        )
    high_rows = []
    for key, rows in sorted(high.items()):
        assert len(rows) == 384 and all(r["full_constraint_valid"] for r in rows)
        result = dict(zip(("n", "m", "alpha", "model"), key)) | {
            "N_encoded_words": 384,
            "binary_message_distribution": "uniform bits;first2^K;not uniform M",
            "k": rows[0]["k"],
            "finite_k_over_n": rows[0]["k"] / rows[0]["n"],
        }
        for metric in (
            "has_stem_ge_m",
            "has_WC_m_subrun",
            "has_GT_m_subrun",
            "has_qualifying_true_hairpin_span_3to12",
            "has_WC_qualifying_true_hairpin_span_3to12",
        ):
            events = sum(r[metric] for r in rows)
            result[f"{metric}_events"] = events
            result[f"{metric}_fraction"] = events / 384
            result[f"{metric}_CI95"] = exact(events, 384)
        result["WC_m_subruns_at_or_above_k"] = sum(
            r["WC_m_subruns_at_or_above_k"] for r in rows
        )
        assert result["WC_m_subruns_at_or_above_k"] == 0
        high_rows.append(result)
    assert len(high_rows) == 24 and len(witnesses) == 24
    for name, rows in (
        ("physical_primary_matched", primary),
        ("physical_sensitivity_common_pairs", sensitivity),
        ("physical_deployment_file_values", deployment_rows),
        ("physical_deployment_summary", deployment_summary),
        ("physical_high_alpha", high_rows),
        ("physical_witnesses", witnesses),
    ):
        save(a.out, name, rows)
    summary = {
        "status": "ALL_PHYSICAL_ARMS_AGGREGATED_NOT_FINAL_REVIEW",
        "fold_records": count,
        "unique_primary_input_words": len(unique_primary),
        "primary_cells": 24,
        "primary_complete_pairs": 27648,
        "primary_cap_failures": 0,
        "primary_positive_adjusted_CI_cells": sum(r["CI_pp"][0] > 0 for r in primary),
        "primary_difference_range_pp": [
            min(r["difference_pp"] for r in primary),
            max(r["difference_pp"] for r in primary),
        ],
        "sensitivity_common_pair_cells": len(sensitivity),
        "pairs_per_sensitivity_cell": 96,
        "deployment_primary_files": sum(r["model"] == PRIMARY for r in deployment_rows),
        "high_alpha_encoded_words": 4608,
        "witness_words": 12,
        "all_interpretations_are_computational_not_causal_or_FER": True,
    }
    dump(a.out / "SUMMARY.json", summary)
    dump(
        a.out / "input_manifest.json",
        {
            str(a.details.relative_to(ROOT) / "fold_details.jsonl.gz"): extraction[
                "fold_details_sha256"
            ],
            str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
