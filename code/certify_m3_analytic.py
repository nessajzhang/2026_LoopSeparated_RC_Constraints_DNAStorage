"""Create an immutable analytic m=3 evidence attempt and run its independent checker.

Example from the code checkout:
  uv run --no-sync python code/certify_m3_analytic.py --output-dir /path/to/new-attempt

The global bound is the supplied human-readable analytic theorem. This tool
does not enumerate the DNA domain or formalize that theorem in a proof assistant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.spectral_m3_analytic import (  # noqa: E402
    ANALYTIC_PROOF,
    analytic_certificate,
)


def _write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True, indent=2) + "\n")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources():
    paths = (
        "loop_alpha_audit/spectral_m3_analytic.py",
        "code/certify_m3_analytic.py",
        "code/check_m3_analytic_certificate.py",
        "tests/test_m3_analytic.py",
        "docs/m3_analytic.md",
        "pyproject.toml",
        "uv.lock",
    )
    return {path: _sha256(ROOT / path) for path in paths}


def produce_attempt(output_directory):
    """Generate a new directory; existing attempts are never replaced."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    wall_start, cpu_start = time.monotonic(), time.process_time()
    source_hashes = _sources()
    config = {
        "q": 4,
        "m": 3,
        "arithmetic": "exact Python integers and fractions.Fraction",
        "global_method": "human-readable analytic proof; not DNA-domain enumeration",
        "binary_sanity_masks": 256,
        "binary_collatz_vector_steps": 64,
        "random_sampling": False,
        "seed": None,
        "cpu_processes_in_parallel": 1,
        "external_dependencies_in_checker": [],
        "proof_assistant_formalization": False,
        "proof_sha256": hashlib.sha256(ANALYTIC_PROOF.encode()).hexdigest(),
    }
    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": list(platform.python_build()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
    }
    _write_json(output_directory / "config.json", config)
    _write_json(output_directory / "environment.json", environment)
    _write_json(output_directory / "source_hashes.json", source_hashes)
    with (output_directory / "analytic_proof.txt").open("x", encoding="utf-8") as stream:
        stream.write(ANALYTIC_PROOF)
    certificate = analytic_certificate()
    certificate_path = output_directory / "certificate.json"
    _write_json(certificate_path, certificate)
    checker = ROOT / "code/check_m3_analytic_certificate.py"
    command = [
        sys.executable,
        "-I",
        str(checker),
        "--certificate",
        str(certificate_path),
        "--output",
        str(output_directory / "checker_result.json"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
    with (output_directory / "independent_checker.log").open("x", encoding="utf-8") as stream:
        stream.write(f"returncode={completed.returncode}\n")
        stream.write(completed.stdout)
        if completed.stderr:
            stream.write("stderr:\n" + completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            "independent checker refused the attempt; retain logs and use a new path"
        )
    if _sources() != source_hashes:
        raise RuntimeError("source changed during production; retain this attempt as unaccepted")
    result = json.loads((output_directory / "checker_result.json").read_text())
    own_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    rss_scale = 1 if sys.platform == "darwin" else 1024
    summary = {
        "task_state": "IMPLEMENTED_UNVERIFIED",
        "checker_status": result["status"],
        "global_product_lower": 9,
        "global_product_upper": 9,
        "conclusion_basis": "human-readable analytic theorem plus exact integer witness",
        "human_proof_review_is_required": True,
        "finite_evidence_is_not_formal_proof": True,
        "legacy_search_artifacts_modified": False,
        "wall_seconds": time.monotonic() - wall_start,
        "parent_cpu_seconds": time.process_time() - cpu_start,
        "checker_cpu_seconds": child_usage.ru_utime + child_usage.ru_stime,
        "peak_rss_bytes": max(own_usage.ru_maxrss, child_usage.ru_maxrss) * rss_scale,
        "certificate_sha256": _sha256(certificate_path),
        "config_sha256": _sha256(output_directory / "config.json"),
        "environment_sha256": _sha256(output_directory / "environment.json"),
        "source_hashes_sha256": _sha256(output_directory / "source_hashes.json"),
    }
    _write_json(output_directory / "summary.json", summary)
    with (output_directory / "run.log").open("x", encoding="utf-8") as stream:
        stream.write("Analytic certificate produced and isolated stdlib checker executed.\n")
        stream.write("No DNA-domain enumeration, floating pruning, or formal theorem proof.\n")
        stream.write(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    # The manifest covers all prior outputs, not itself. Nothing is appended later.
    manifest = {
        path.name: _sha256(path) for path in sorted(output_directory.iterdir()) if path.is_file()
    }
    _write_json(output_directory / "output_hashes.json", manifest)
    return {**summary, "output_hashes_sha256": _sha256(output_directory / "output_hashes.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(produce_attempt(args.output_dir), sort_keys=True, indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"analytic production refused: {error}\n")


if __name__ == "__main__":
    main()
