#!/usr/bin/env python3
"""Run frozen, file-level diagnostic IDS experiments; never provide truth to receiving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_alpha_audit.benchmark_revision import run_benchmark, validate_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reviewed-freeze-sha256")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.validate_only:
        cells = validate_config(config)
        print(json.dumps({"validated_cells": len(cells), "actual_experiment_run": False}))
        return 0
    result = run_benchmark(
        config,
        args.out,
        jobs=args.jobs,
        resume=args.resume,
        reviewed_freeze=args.reviewed_freeze_sha256,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "phase", "planned_files", "observed_files_in_complete_cases")
            }
        )
    )
    return 0 if result["status"] == "COMPLETE_UNVERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
