#!/usr/bin/env python3
"""Validate the frozen C++ and independent Python orbit-audit outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
CPP_SUMMARY = OUTPUT / "orbit_interval_million_audit_summary.json"
CPP_STRATA = OUTPUT / "orbit_interval_million_audit_strata.csv"
PYTHON_SUMMARY = OUTPUT / "orbit_interval_python_crosscheck_summary.json"
VALIDATION = OUTPUT / "orbit_interval_audit_validation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cpp = json.loads(CPP_SUMMARY.read_text(encoding="utf-8"))
    python = json.loads(PYTHON_SUMMARY.read_text(encoding="utf-8"))
    totals = {
        "cases": 0,
        "passed": 0,
        "valid_cases": 0,
        "invalid_cases_with_witness": 0,
        "atomic_assertions": 0,
        "tight_pair_cases": 0,
        "tight_self_cases": 0,
    }
    max_pair_ratio = 0.0
    max_self_ratio = 0.0
    row_count = 0
    with CPP_STRATA.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row_count += 1
            totals["cases"] += int(row["cases"])
            totals["passed"] += int(row["passed"])
            totals["valid_cases"] += int(row["valid"])
            totals["invalid_cases_with_witness"] += int(row["invalid"])
            totals["atomic_assertions"] += int(row["assertions"])
            totals["tight_pair_cases"] += int(row["tight_pair"])
            totals["tight_self_cases"] += int(row["tight_self"])
            max_pair_ratio = max(max_pair_ratio, float(row["max_pair_cover_ratio"]))
            max_self_ratio = max(max_self_ratio, float(row["max_self_cover_ratio"]))

    assert cpp["status"] == "PASS"
    assert cpp["failures"] == 0
    assert cpp["cases"] >= 5_800_000
    assert row_count == cpp["nonempty_strata"]
    for key, value in totals.items():
        assert value == cpp[key], (key, value, cpp[key])
    assert cpp["cases"] == cpp["valid_cases"] + cpp["invalid_cases_with_witness"]
    assert max_pair_ratio == cpp["maximum_pair_cover_ratio"] == 1.0
    assert max_self_ratio == cpp["maximum_self_cover_ratio"] == 1.0
    assert cpp["tight_pair_cases"] > 0
    assert cpp["tight_self_cases"] > 0

    assert python["status"] == "PASS"
    assert python["implementation"] == "independent_python"
    assert python["cases"] >= 500_000
    assert python["cases"] == python["valid_cases"] + python["invalid_cases_with_witness"]

    result = {
        "status": "PASS",
        "cpp_cases": cpp["cases"],
        "python_cases": python["cases"],
        "combined_property_cases": cpp["cases"] + python["cases"],
        "cpp_atomic_assertions": cpp["atomic_assertions"],
        "python_atomic_assertions": python["atomic_assertions"],
        "nonempty_cpp_strata": row_count,
        "tight_pair_bound_attained": True,
        "tight_self_bound_attained": True,
        "sha256": {
            CPP_SUMMARY.name: sha256(CPP_SUMMARY),
            CPP_STRATA.name: sha256(CPP_STRATA),
            PYTHON_SUMMARY.name: sha256(PYTHON_SUMMARY),
        },
        "evidence_boundary": (
            "The C++ and Python audits validate the occurrence-position geometry. "
            "They do not replace the proof or estimate constrained-language capacity."
        ),
    }
    VALIDATION.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
