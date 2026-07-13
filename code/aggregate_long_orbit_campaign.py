#!/usr/bin/env python3
"""Aggregate and validate the five-seed long orbit campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "output" / "long_orbit_campaign"
SEEDS = (20260712, 20260713, 20260717, 314159265, 271828182)
CASES_PER_SEED = 2_000_000_000
OUTPUT = CAMPAIGN / "campaign_summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summaries: list[dict[str, object]] = []
    source_sha256: dict[str, str] = {}
    for seed in SEEDS:
        summary_path = CAMPAIGN / f"summary_seed_{seed}.json"
        strata_path = CAMPAIGN / f"strata_seed_{seed}.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["status"] == "PASS"
        assert summary["failures"] == 0
        assert summary["cases"] == CASES_PER_SEED
        assert summary["passed"] == CASES_PER_SEED
        assert summary["seed"] == seed
        assert summary["maximum_pair_cover_ratio"] == 1.0
        assert summary["maximum_self_cover_ratio"] == 1.0
        assert summary["nonempty_strata"] == 12 * 16 * 27 * 18
        assert int(summary["tight_pair_cases"]) > 0
        assert int(summary["tight_self_cases"]) > 0
        assert int(summary["valid_cases"]) + int(summary["invalid_cases_with_witness"]) == int(
            summary["cases"]
        )
        source_sha256[summary_path.name] = sha256(summary_path)
        source_sha256[strata_path.name] = sha256(strata_path)
        summaries.append(summary)

    result = {
        "status": "PASS",
        "seeds": list(SEEDS),
        "seed_count": len(SEEDS),
        "cases_per_seed": CASES_PER_SEED,
        "threads_by_seed": {
            str(seed): int(summary["threads"]) for seed, summary in zip(SEEDS, summaries)
        },
        "total_cases": sum(int(item["cases"]) for item in summaries),
        "total_passed": sum(int(item["passed"]) for item in summaries),
        "total_valid_cases": sum(int(item["valid_cases"]) for item in summaries),
        "total_invalid_cases_with_witness": sum(
            int(item["invalid_cases_with_witness"]) for item in summaries
        ),
        "total_atomic_assertions": sum(int(item["atomic_assertions"]) for item in summaries),
        "total_tight_pair_cases": sum(int(item["tight_pair_cases"]) for item in summaries),
        "total_tight_self_cases": sum(int(item["tight_self_cases"]) for item in summaries),
        "total_elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in summaries),
        "m_min": min(int(item["m_min"]) for item in summaries),
        "m_max": max(int(item["m_max"]) for item in summaries),
        "m_value_count": 27,
        "maximum_start_positions": max(int(item["maximum_start_positions"]) for item in summaries),
        "start_domain_count": 18,
        "schedule_count": int(summaries[0]["schedule_count"]),
        "case_mode_count": int(summaries[0]["case_mode_count"]),
        "source_sha256": source_sha256,
        "evidence_boundary": (
            "The campaign repeatedly stress-tests the occurrence-position lemma. "
            "It is not a capacity computation and does not replace proof."
        ),
    }
    assert result["total_cases"] == 10_000_000_000
    assert result["total_passed"] == result["total_cases"]
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
