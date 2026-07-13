"""Validate the extended full-word audit and its independent regressions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
JOBS = (
    ("binary_swap_extended", 2, 24),
    ("binary_identity_extended", 2, 24),
    ("ternary_fixed_point_extended", 3, 14),
    ("dna_watson_crick_extended", 4, 12),
    ("quaternary_identity_extended", 4, 12),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    summary_path = OUTPUT / "full_word_definition_audit_extended_10t_summary.json"
    jobs_path = OUTPUT / "full_word_definition_audit_extended_10t_jobs.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with jobs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected_words = sum(q**n for _, q, n in JOBS)
    expected_stem_cells = sum((q**n) * n for _, q, n in JOBS)
    expected_parameter_cells = sum((q**n) * n * (n + 1) for _, q, n in JOBS)
    checks = {
        "status_pass": summary["status"] == "PASS",
        "mode_extended": summary["mode"] == "extended",
        "job_count": summary["jobs"] == len(JOBS) == len(rows),
        "word_total": summary["words"] == expected_words,
        "stem_cell_total": summary["stem_cells"] == expected_stem_cells,
        "parameter_cell_total": summary["word_parameter_cells"] == expected_parameter_cells,
        "reduction_equalities_complete": summary["exact_gap_equalities"] == expected_stem_cells,
        "orbit_cover_complete": (
            summary["orbit_cover_cells"] == summary["avoiding_parameter_cells"]
        ),
        "zero_failures": summary["failures"] == 0 and not summary["first_failure"],
        "old_full_csv_byte_identical": (
            (OUTPUT / "full_word_definition_audit_10t_jobs.csv").read_bytes()
            == (OUTPUT / "full_word_definition_audit_optimized_full_jobs.csv").read_bytes()
        ),
        "quick_ubsan_csv_byte_identical": (
            (OUTPUT / "full_word_definition_audit_quick_optimized_jobs.csv").read_bytes()
            == (OUTPUT / "full_word_definition_audit_ubsan_current_jobs.csv").read_bytes()
        ),
    }

    expected_by_name = {name: (q, n) for name, q, n in JOBS}
    for row in rows:
        name = row["job"]
        q, n = expected_by_name[name]
        word_count = q**n
        checks[f"{name}_shape"] = (
            int(row["q"]) == q
            and int(row["n"]) == n
            and int(row["words"]) == word_count
            and int(row["stem_cells"]) == word_count * n
            and int(row["word_parameter_cells"]) == word_count * n * (n + 1)
            and int(row["failures"]) == 0
        )

    files = (
        ROOT / "code" / "full_word_definition_audit.cpp",
        summary_path,
        jobs_path,
        OUTPUT / "full_word_definition_audit_optimized_full_jobs.csv",
        OUTPUT / "full_word_definition_audit_ubsan_current_jobs.csv",
    )
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected": {
            "words": expected_words,
            "stem_cells": expected_stem_cells,
            "word_parameter_cells": expected_parameter_cells,
        },
        "observed": {
            "avoiding_parameter_cells": summary["avoiding_parameter_cells"],
            "orbit_cover_cells": summary["orbit_cover_cells"],
            "atomic_assertions": summary["atomic_assertions"],
            "digest_xor": summary["digest_xor"],
            "elapsed_seconds": summary["elapsed_seconds"],
        },
        "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in files},
        "evidence_boundary": (
            "This validates frozen totals, regressions, and file identity. Exhaustive finite "
            "computations support but do not replace the proof."
        ),
    }
    output_path = OUTPUT / "full_word_definition_audit_extended_validation.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
