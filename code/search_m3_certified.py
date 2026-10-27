"""Bounded, checkpointable exact reference search (all 64 edge bits for m=3)."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_alpha_audit.spectral_m3 import extend_search, initial_search_certificate  # noqa: E402


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "independent_m3_checker", Path(__file__).with_name("check_m3_certificate.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_certificate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-nodes", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--exponent", type=int, default=4)
    parser.add_argument("--vector-steps", type=int, default=8)
    parser.add_argument("--threshold", default="2253/250")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if (
        not 0 <= args.seconds <= 120
        or args.max_nodes < 0
        or not 1 <= args.exponent <= 128
        or not 0 <= args.vector_steps <= 128
    ):
        parser.error("reference local pilot: seconds<=120, nodes>=0, exponents/steps<=128")
    check = load_checker()
    config = {
        "q": 4,
        "m": 3,
        "max_nodes": args.max_nodes,
        "seconds": args.seconds,
        "exponent": args.exponent,
        "vector_steps": args.vector_steps,
        "prune_threshold": args.threshold,
        "cpu_limit": 1,
        "memory_limit_bytes": 4 * 1024**3,
        "compute": "unmetered existing local CPU; no cloud",
        "producer": "exact Python reference",
    }
    if args.config:
        requested = json.loads(args.config.read_text())
        for key in ("q", "m", "cpu_limit", "memory_limit_bytes"):
            if key in requested and requested[key] != config[key]:
                parser.error(f"config {key} differs from reference pilot contract")
        config["declared_config"] = requested
    certificate = (
        json.loads(args.resume.read_text())
        if args.resume
        else initial_search_certificate(threshold=Fraction(args.threshold))
    )
    if args.resume:
        check(certificate)  # no trust in an unchecked producer checkpoint
    args.output.parent.mkdir(parents=True, exist_ok=True)
    limit = config["memory_limit_bytes"]
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard == resource.RLIM_INFINITY or hard >= limit:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
            config["address_space_cap_enforced"] = True
        except (ValueError, OSError) as error:
            # Darwin may report an infinite RLIMIT_AS yet reject setting it.
            # Keep the refusal auditable; the bounded node budget and RSS
            # guard in the reference loop still limit this small local pilot.
            config["address_space_cap_enforced"] = False
            config["address_space_cap_error"] = str(error)
    certificate = extend_search(
        certificate,
        max_nodes=args.max_nodes,
        max_seconds=args.seconds,
        exponent=args.exponent,
        steps=args.vector_steps,
    )
    certificate["run_config"] = config
    # Atomic replacement is safe for checkpoint recovery, including a SIGINT.
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(certificate, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, args.output)
    result = check(certificate)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result.update(
        {
            "stats": certificate["stats"],
            "max_rss_bytes": usage.ru_maxrss,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "certificate_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "resource_note": "macOS ru_maxrss is bytes; one CPU; 3 GiB measured-RSS stop guard; RLIMIT_AS enforcement recorded in run_config",
        }
    )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
