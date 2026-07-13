#!/usr/bin/env python3
"""Exact high-alpha encoder rates on an ultra-large deterministic grid."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.encoder import SegmentedDNAEncoder  # noqa: E402


CSV_OUTPUT = ROOT / "output" / "segmented_encoder_ultrascale_rates.csv"
JSON_OUTPUT = ROOT / "output" / "segmented_encoder_ultrascale_summary.json"
ALPHAS = (0.5, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
LENGTHS = (1_000, 10_000, 100_000, 1_000_000)


def log2_integer(value: int) -> float:
    if value <= 0:
        raise ValueError("log argument must be positive")
    shift = max(0, value.bit_length() - 53)
    return math.log2(value >> shift) + shift


def asymptotic_capacity(alpha: float) -> float:
    lambda_log = math.log2(3 * (1 + math.sqrt(13)) / 2)
    return (2 * alpha - 1) * 2 + (1 - alpha) * lambda_log


def main() -> None:
    rows: list[dict[str, object]] = []
    for alpha in ALPHAS:
        capacity = asymptotic_capacity(alpha)
        previous_error = math.inf
        for n in LENGTHS:
            k = math.floor(alpha * n)
            encoder = SegmentedDNAEncoder(n=n, k=k)
            right_count = encoder.right_count
            log2_size = (
                encoder.endpoint_length * math.log2(3)
                + 2 * encoder.free_length
                + log2_integer(right_count)
            )
            rate = log2_size / n
            error = rate - capacity
            if error < -1e-12:
                raise AssertionError(("finite_rate_below_limit", alpha, n, error))
            if error >= previous_error:
                raise AssertionError(("nondecreasing_error", alpha, n, error, previous_error))
            previous_error = error
            rows.append(
                {
                    "alpha": alpha,
                    "n": n,
                    "k": k,
                    "endpoint_length": encoder.endpoint_length,
                    "free_length": encoder.free_length,
                    "right_count_bit_length": right_count.bit_length(),
                    "finite_block_rate": f"{rate:.15f}",
                    "asymptotic_capacity": f"{capacity:.15f}",
                    "finite_minus_asymptotic": f"{error:.15e}",
                    "n_times_error": f"{n * error:.12f}",
                }
            )

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(CSV_OUTPUT.read_bytes()).hexdigest()

    terminal = [row for row in rows if row["n"] == max(LENGTHS)]
    summary = {
        "status": "COMPUTE_CERTIFIED",
        "alpha_values": ALPHAS,
        "lengths": LENGTHS,
        "cells": len(rows),
        "largest_n": max(LENGTHS),
        "max_terminal_abs_error": max(
            abs(float(row["finite_minus_asymptotic"])) for row in terminal
        ),
        "terminal_n_times_error_range": [
            min(float(row["n_times_error"]) for row in terminal),
            max(float(row["n_times_error"]) for row in terminal),
        ],
        "counting": "exact integer binary matrix power; logarithms only for reporting",
        "csv_sha256": digest,
        "csv": CSV_OUTPUT.name,
    }
    JSON_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
