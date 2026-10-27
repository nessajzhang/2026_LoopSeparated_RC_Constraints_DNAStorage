#!/usr/bin/env python3
"""Build file-unit tables solely from independently audited frozen rows.

No simulator, producer summarizer, receiver or mapper is imported. The
predeclared primary comparison is nonce-minus-GC/H file failure. Other
intervals remain descriptive, with all failures in their denominators.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode()


def read(path):
    return json.loads(path.read_text())


def exact(events, n, alpha=0.05):
    interval = binomtest(events, n).proportion_ci(1 - alpha, method="exact")
    return [float(interval.low), float(interval.high)]


def paired(left, right, alpha):
    assert set(left) == set(right) and left
    a = sum(left[k] and not right[k] for k in left)
    b = sum(right[k] and not left[k] for k in left)
    n = len(left)
    lo_a, hi_a = exact(a, n, alpha / 2)
    lo_b, hi_b = exact(b, n, alpha / 2)
    return {
        "N": n,
        "left_only_failures": a,
        "right_only_failures": b,
        "difference": (a - b) / n,
        "CI": [max(-1.0, lo_a - hi_b), min(1.0, hi_a - lo_b)],
        "alpha": alpha,
        "direction": "nonce-minus-GC/H file failure",
    }


def dump(path, value):
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(encoded)


def csv_rows(path, rows):
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (list, dict)) else v
                for k, v in r.items()
            }
            for r in rows
        )


def analyze(config, audited, run_name, sources):
    report = read(audited / "report.json")
    assert report["status"] == "PASS_FULL_RAW_AUDIT_NOT_FINAL_MANUSCRIPT_REVIEW"
    assert not report["errors"] and not report["partial_case_test"]
    assert report["config_sha256"] == hashlib.sha256(canonical(config)).hexdigest()
    sources.append(audited / "report.json")
    cell_definitions = {
        hashlib.sha256(canonical(g)).hexdigest()[:16]: g for g in config["grid"]
    }
    expected_keys = {
        (s, f) for s in config["seeds"] for f in range(config["files_per_seed"])
    }
    all_rows, cells, contrasts = {}, [], []
    for cid, grid in cell_definitions.items():
        for method in config["methods"]:
            path = audited / f"{cid}_{method}" / "file_rows.json"
            sources.append(path)
            rows = read(path)
            keyed = {(r["seed"], r["file_id"]): r for r in rows}
            assert len(keyed) == len(rows) and set(keyed) == expected_keys
            assert all(r["cell_id"] == cid and r["mapper"] == method for r in rows)
            all_rows[cid, method] = keyed
            n = len(rows)
            failures = sum(not r["file_correct"] for r in rows)
            ci = exact(failures, n)
            total = Counter(
                {
                    name: sum(r.get(name, 0) for r in rows)
                    for name in (
                        "frames_attempted",
                        "frame_erasures",
                        "frame_undetected_errors",
                        "file_undetected_error",
                        "returned_payload_bit_errors",
                        "returned_payload_bits",
                        "terminal_mapper_failures",
                        "primary_mapper_failures",
                        "fallback_used",
                        "total_synthesized_nt",
                        "planned_synthesized_nt",
                        "total_sequenced_nt",
                        "assignment_correct_evaluator_only",
                        "assignment_wrong_evaluator_only",
                        "assignment_rejected",
                        "read_count",
                        "realized_insertions",
                        "realized_deletions",
                        "realized_substitutions",
                        "user_bits",
                    )
                }
            )
            recovered_bits = sum(r["user_bits"] * r["file_correct"] for r in rows)
            times = sorted(r["receive_total_ns"] / 1e6 for r in rows)
            cells.append(
                {
                    "run": run_name,
                    "config_sha256": report["config_sha256"],
                    "cell_id": cid,
                    **config["protocol"],
                    **grid,
                    "method": method,
                    "N_files": n,
                    "file_failures": failures,
                    "FER": failures / n,
                    "FER_CI_low": ci[0],
                    "FER_CI_high": ci[1],
                    "zero_failure_one_sided_upper95": -math.expm1(math.log(0.05) / n)
                    if failures == 0
                    else None,
                    **dict(total),
                    "frame_erasure_fraction_descriptive": total["frame_erasures"]
                    / total["frames_attempted"],
                    "R_net_planned": total["user_bits"]
                    / total["planned_synthesized_nt"],
                    "R_net_actual": total["user_bits"] / total["total_synthesized_nt"]
                    if total["total_synthesized_nt"]
                    else None,
                    "goodput_planned": recovered_bits / total["planned_synthesized_nt"],
                    "goodput_actual": recovered_bits / total["total_synthesized_nt"]
                    if total["total_synthesized_nt"]
                    else None,
                    "conditional_returned_frame_BER": total[
                        "returned_payload_bit_errors"
                    ]
                    / total["returned_payload_bits"]
                    if total["returned_payload_bits"]
                    else None,
                    "erasure_aware_bit_loss_mean": statistics.fmean(
                        r["erasure_aware_bit_loss_score"] for r in rows
                    ),
                    "receive_file_median_ms_nonexclusive": statistics.median(times),
                    "receive_file_P95_ms_nonexclusive": times[math.ceil(0.95 * n) - 1],
                    "uncertainty_unit": "file;not strand or read",
                }
            )
        if {"nonce", "gc_h_nonce"} <= set(config["methods"]):
            alpha = (
                0.05 / config["primary_comparison_family_size"]
                if grid["primary"]
                else 0.05
            )
            contrasts.append(
                {
                    "run": run_name,
                    "cell_id": cid,
                    **config["protocol"],
                    **grid,
                    **paired(
                        {
                            k: not r["file_correct"]
                            for k, r in all_rows[cid, "nonce"].items()
                        },
                        {
                            k: not r["file_correct"]
                            for k, r in all_rows[cid, "gc_h_nonce"].items()
                        },
                        alpha,
                    ),
                }
            )
    assert sum(r["N_files"] for r in cells) == report["counts"]["file_evaluations"]
    return cells, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve()]
    config_path = ROOT / "data/ids_main_20261023/attempt-001/config.json"
    sources.append(config_path)
    primary, primary_pairs = analyze(
        read(config_path), ROOT / "data/ids_main_20261023/raw_audit001", "main", sources
    )
    secondary_root = (
        ROOT
        / "data/tianhe_secondary_20261023/independent_received_1804783/output/secondary_independent_audit_1804783"
    )
    execution = read(secondary_root / "EXECUTION.json")
    assert execution["audited_runs"] == 19 and all(
        r["returncode"] == 0 for r in execution["rows"]
    )
    sources.append(secondary_root / "EXECUTION.json")
    secondary, secondary_pairs = [], []
    for row in execution["rows"]:
        path = ROOT / "code/configs/revision" / row["configuration"]
        sources.append(path)
        cells, pairs = analyze(
            read(path),
            secondary_root / f"{row['index']:02d}",
            row["configuration"],
            sources,
        )
        secondary.extend(cells)
        secondary_pairs.extend(pairs)
    assert sum(r["N_files"] for r in primary) == 72000
    assert sum(r["N_files"] for r in secondary) == 32640
    assert len([r for r in primary_pairs if r["primary"]]) == 48
    for name, rows in (
        ("ids_main_cells", primary),
        ("ids_secondary_cells", secondary),
        ("ids_main_paired", primary_pairs),
        ("ids_secondary_paired", secondary_pairs),
    ):
        dump(args.out / f"{name}.json", rows)
        csv_rows(args.out / f"{name}.csv", rows)
    summary = {
        "status": "TABLES_FROM_FULL_INDEPENDENT_RAW_AUDITS_NOT_FINAL_REVIEW",
        "main_cells": len(primary),
        "secondary_cells": len(secondary),
        "file_method_condition_evaluations": 104640,
        "main_primary_contrasts": 48,
        "main_primary_CI_exclude_zero": sum(
            (r["CI"][0] > 0 or r["CI"][1] < 0) for r in primary_pairs if r["primary"]
        ),
        "main_file_failures": sum(r["file_failures"] for r in primary),
        "secondary_file_failures": sum(r["file_failures"] for r in secondary),
        "main_noiseless_evaluations": sum(
            r["N_files"] for r in primary if r["error_scale"] == 0
        ),
        "main_noiseless_failures": sum(
            r["file_failures"] for r in primary if r["error_scale"] == 0
        ),
        "main_max_abs_primary_paired_difference": max(
            abs(r["difference"]) for r in primary_pairs if r["primary"]
        ),
        "main_returned_frame_bit_errors": sum(
            r["returned_payload_bit_errors"] for r in primary
        ),
        "all_mapper_terminal_failures": sum(
            r["terminal_mapper_failures"] for r in primary + secondary
        ),
        "cell_independence_not_required_for_family_adjustment": True,
        "cross_configuration_shared_streams_not_pooled_as_independent": True,
        "script_does_not_rerun_experiments": True,
    }
    dump(args.out / "SUMMARY.json", summary)
    dump(
        args.out / "input_manifest.json",
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(sources))
        },
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
