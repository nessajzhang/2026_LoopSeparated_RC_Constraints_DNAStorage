"""Replay byte-matched table builders in an explicitly supplied frozen data bundle.

The public tree contains the algorithms and derived tables, not full raw runs.
Executing inside the supplied bundle preserves its relative provenance paths.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PUBLIC = Path(__file__).resolve().parent
BUILDERS = {
    "communication": "build_communication_tables.py",
    "physical": "build_physical_tables.py",
    "deployment_sensitivity": "build_deployment_model_sensitivity.py",
    "profile": "build_profile_tables.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root = args.bundle.resolve(strict=True)
    output = args.out.resolve()
    if not (root / "PACKAGE_MANIFEST.json").is_file():
        raise ValueError("Expected the extracted frozen computational bundle")
    for name in BUILDERS.values():
        local = PUBLIC / "analysis" / name
        frozen = root / "revision/analysis" / name
        if not frozen.resolve().is_relative_to(root) or sha(local) != sha(frozen):
            raise ValueError(f"Bundle/public analysis source mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    env = os.environ | {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                        "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def execute(item):
        label, name = item
        argv = [sys.executable, str(root / "revision/analysis" / name)]
        if label in {"physical", "deployment_sensitivity"}:
            argv += ["--details", str(root / "data/analysis_physical_details_20261023/attempt-001")]
        if label == "profile":
            argv += ["--metadata", str(root / "data/profile_formal_20261023/metadata_received")]
        argv += ["--out", str(output / label)]
        result = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True)
        (output / "logs" / f"{label}.stdout.log").write_text(result.stdout)
        (output / "logs" / f"{label}.stderr.log").write_text(result.stderr)
        return {"builder": label, "exit_code": result.returncode,
                "source_sha256": sha(PUBLIC / "analysis" / name)}

    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, 4))) as pool:
        results = list(pool.map(execute, BUILDERS.items()))
    report = {"status": "PASS" if not any(r["exit_code"] for r in results) else "FAIL",
              "commands": results, "raw_simulations_rerun": False,
              "source_identity_verified": True}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
