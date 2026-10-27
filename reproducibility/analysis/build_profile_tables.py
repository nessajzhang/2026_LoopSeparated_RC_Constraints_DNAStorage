#!/usr/bin/env python3
"""Summarize the frozen 96-user source-backed readout arm, not pooled N768."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def ci(events, n, alpha=0.05):
    value = binomtest(events, n).proportion_ci(1 - alpha, method="exact")
    return [float(value.low), float(value.high)]


def dump(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as stream:
        stream.write(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.metadata = a.metadata.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    rows = read(a.metadata / "file_rows.json")
    reports = read(a.metadata / "independent_audits.json")
    receipt = read(a.metadata / "EXECUTION_RECEIPT.json")
    assert (
        receipt["independent_users"] == 96
        and receipt["file_method_budget_evaluations"] == 768
    )
    assert len(rows) == 768 and len(reports) == 96
    independent = {}
    science_hashes = {
        "driver": hashlib.sha256(
            (ROOT / "code/code/run_profile_formal_v2.py").read_bytes()
        ).hexdigest(),
        "worker": hashlib.sha256(
            (ROOT / "literature/profiles/p3_formal_v2/generate_worker.py").read_bytes()
        ).hexdigest(),
        "schema": hashlib.sha256(
            (ROOT / "literature/profiles/p3_formal_v2/schema.py").read_bytes()
        ).hexdigest(),
    }
    for report in reports:
        assert (
            report["status"]
            == "PASS_INDEPENDENT_P3_V2_RAW_NOT_FINAL_PACKAGE_ACCEPTANCE"
        )
        assert report["implementation_sha256"] == science_hashes
        assert report["read_origin"] is report["assignment_accuracy"] is None
        for row in report["rows"]:
            key = report["index"], row["method"], row["read_pairs"]
            assert key not in independent
            independent[key] = row
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["index"], row["method"], row["read_pairs"]
        assert key not in seen and key in independent
        seen.add(key)
        reference = independent[key]
        for field in (
            "method",
            "mode",
            "read_pairs",
            "seed",
            "file_id",
            "file_correct",
            "frame_erasures",
        ):
            assert row[field] == reference[field]
        assert (
            row["read_origin"]
            is row["true_edit_script"]
            is row["assignment_accuracy"]
            is None
        )
        assert row["non_evaluable_reason"] == "NON_EVALUABLE_NO_LINEAGE_CERTIFICATE"
        assert (
            row["undetected_frame_payload_errors"] == 0
            and row["undetected_file_error"] is False
        )
        assert row["generator_status"] == reference["generator_status"] == "COMPLETE"
        g = row["generator"]
        # The first flag is the policy "never repair source overdraw", not an
        # observed event. The actual event field is a per-template dictionary.
        assert g["overdraw_not_corrected"] is True
        assert g["selected_template_overdraw"] == {}
        assert g["actual_R1"] == g["actual_R2"] == row["read_pairs"]
        assert row["accounting"]["full_template_net_rate_planned"] == 9616 / 17136
        assert row["accounting"]["planned_molecule_weighted_nt"] == 8773632
        assert (
            row["accounting"]["actual_sequenced_nt_both_mates"]
            == 300 * row["read_pairs"]
        )
        groups[row["method"], row["read_pairs"]].append(row)
    assert len(seen) == 768 and set(seen) == set(independent)
    tables = []
    for (method, budget), values in sorted(groups.items()):
        assert len(values) == 96 and {r["index"] for r in values} == set(range(96))
        failed = sum(r["file_failed"] for r in values)
        tables.append(
            {
                "method": method,
                "requested_read_pairs": budget,
                "N_independent_files": 96,
                "file_failures": failed,
                "FER": failed / 96,
                "FER_CI95": ci(failed, 96),
                "zero_failure_one_sided_upper95": -math.expm1(math.log(0.05) / 96)
                if failed == 0
                else None,
                "frame_erasures": sum(r["frame_erasures"] for r in values),
                "frame_denominator_descriptive": 96 * 63,
                "undetected_frame_errors": sum(
                    r["undetected_frame_payload_errors"] for r in values
                ),
                "body_design_net_rate": 9616 / 9450,
                "full_template_design_net_rate": 9616 / 17136,
                "full_template_design_goodput": 9616 * (96 - failed) / (17136 * 96),
                "initial_copies_per_template": 512,
                "initial_molecule_weighted_nt_per_file": 8773632,
                "actual_both_mate_sequenced_nt_per_file": budget * 300,
                "receiver_R1_only": True,
                "origin_assignment_accuracy_edit_script": None,
            }
        )
    comparisons = []
    for budget in (192, 315):
        left = {r["index"]: r["file_failed"] for r in groups["nonce", budget]}
        right = {r["index"]: r["file_failed"] for r in groups["gc_h_nonce", budget]}
        a10 = sum(left[k] and not right[k] for k in left)
        a01 = sum(right[k] and not left[k] for k in left)
        lo_a, hi_a = ci(a10, 96, 0.025 / 2)
        lo_b, hi_b = ci(a01, 96, 0.025 / 2)
        comparisons.append(
            {
                "read_pairs": budget,
                "N_files": 96,
                "alpha": 0.025,
                "left_only_failures": a10,
                "right_only_failures": a01,
                "difference_nonce_minus_GCH": (a10 - a01) / 96,
                "CI": [max(-1.0, lo_a - hi_b), min(1.0, hi_a - lo_b)],
                "same_users_across_budgets_no_pooled_independence": True,
            }
        )
    dump(a.out / "profile_cells.json", tables)
    dump(a.out / "profile_paired.json", comparisons)
    summary = {
        "status": "CLOUD_INDEPENDENT_RAW_AUDITS_AND_LOCAL_METADATA_RECONCILIATION_PASS",
        "local_complete_raw_replay_still_required": True,
        "independent_users": 96,
        "file_method_budget_evaluations": 768,
        "file_failures": sum(t["file_failures"] for t in tables),
        "source_profile_failures": 0,
        "source_template_overdraws": 0,
        "all_actual_read_pair_budgets_match": True,
        "allocated_core_seconds_formal": receipt["allocated_core_seconds_formal"],
        "unit_RMB_rate": None,
        "not_wetlab_or_entire_storage_channel": True,
    }
    dump(a.out / "SUMMARY.json", summary)
    dump(
        a.out / "input_manifest.json",
        {
            str((a.metadata / name).relative_to(ROOT)): hashlib.sha256(
                (a.metadata / name).read_bytes()
            ).hexdigest()
            for name in (
                "file_rows.json",
                "independent_audits.json",
                "EXECUTION_RECEIPT.json",
            )
        },
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
