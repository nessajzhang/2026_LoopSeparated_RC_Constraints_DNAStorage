#!/usr/bin/env python3
"""Cross-check independent C++ paired counts against exact Python integers."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.paired import paired_count_dynamic  # noqa: E402


DEFAULT_CSV_PATH = ROOT / "output" / "paired_dna_m2_cpp_rates.csv"
DEFAULT_SUMMARY_PATH = ROOT / "output" / "paired_cpp_python_crosscheck.json"


def main() -> None:
    csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else DEFAULT_CSV_PATH
    summary_path = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else DEFAULT_SUMMARY_PATH
    if len(sys.argv) > 3:
        raise SystemExit("usage: check_paired_cpp_against_python.py [INPUT.csv] [SUMMARY.json]")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    checked: list[dict[str, object]] = []
    for row in rows:
        r = int(row["r"])
        if r > 12:
            continue
        exact = paired_count_dynamic(4, r, 2, (3, 2, 1, 0))
        expected = math.log2(exact)
        observed = float(row["log2_F"])
        error = abs(expected - observed)
        if error > 5e-13:
            raise AssertionError((r, expected, observed, error))
        checked.append(
            {
                "r": r,
                "exact_integer_bit_length": exact.bit_length(),
                "abs_log2_error": error,
            }
        )

    excesses = [float(row["normalized_excess"]) for row in rows if int(row["r"]) > 0]
    if any(excess < -2e-14 for excess in excesses):
        raise AssertionError("finite paired rate fell below the certified spectral exponent")
    tail = [float(row["normalized_excess"]) for row in rows if int(row["r"]) >= 20]
    if any(later >= earlier for earlier, later in zip(tail, tail[1:])):
        raise AssertionError("reported tail excess is not strictly decreasing")

    summary = {
        "status": "CROSS_VALIDATED",
        "cpp_rows": len(rows),
        "python_exact_rows": len(checked),
        "python_exact_checks": checked,
        "max_abs_log2_error": max(item["abs_log2_error"] for item in checked),
        "final_r": int(rows[-1]["r"]),
        "final_states": int(rows[-1]["states"]),
        "final_normalized_excess": float(rows[-1]["normalized_excess"]),
        "notes": [
            "C++ uses a packed seen-mask state and long-double weights.",
            "Python uses tuple/frozenset states and exact integer weights.",
            "The asymptotic value comes from the independently certified subset spectrum.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
