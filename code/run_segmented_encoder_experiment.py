#!/usr/bin/env python3
"""Generate exact finite-block rates for the certified segmented encoder."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.encoder import SegmentedDNAEncoder  # noqa: E402
from loop_alpha_audit.model import is_avoiding_m_windows  # noqa: E402


CSV_OUTPUT = ROOT / "output" / "segmented_encoder_finite_block_rates.csv"
JSON_OUTPUT = ROOT / "output" / "segmented_encoder_experiment_summary.json"
ALPHAS = (0.5, 0.6, 0.75, 0.9)
LENGTHS = (40, 80, 160, 320, 640, 1000, 2000, 5000, 10000)
SEED = 20260712


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
    rng = random.Random(SEED)
    random_words_checked = 0

    for alpha in ALPHAS:
        capacity = asymptotic_capacity(alpha)
        for n in LENGTHS:
            k = math.floor(alpha * n)
            encoder = SegmentedDNAEncoder(n=n, k=k)
            log2_size = (
                encoder.endpoint_length * math.log2(3)
                + 2 * encoder.free_length
                + log2_integer(encoder.right_count)
            )
            rate = log2_size / n
            rows.append(
                {
                    "alpha": alpha,
                    "n": n,
                    "k": k,
                    "D": encoder.distance,
                    "endpoint_length": encoder.endpoint_length,
                    "free_length": encoder.free_length,
                    "log2_codebook_size": f"{log2_size:.12f}",
                    "finite_block_rate": f"{rate:.12f}",
                    "asymptotic_capacity": f"{capacity:.12f}",
                    "finite_minus_asymptotic": f"{rate - capacity:.12f}",
                    "right_count_bit_length": encoder.right_count.bit_length(),
                }
            )

            if n <= 160:
                for _ in range(8):
                    index = rng.randrange(encoder.codebook_size)
                    word = encoder.encode_index(index)
                    if encoder.decode_index(word) != index:
                        raise AssertionError(("rank_roundtrip", alpha, n, index))
                    if not is_avoiding_m_windows(word, 2, k, (3, 2, 1, 0)):
                        raise AssertionError(("constraint", alpha, n, index))
                    random_words_checked += 1

    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    max_terminal_error = max(
        abs(float(row["finite_minus_asymptotic"])) for row in rows if row["n"] == max(LENGTHS)
    )
    summary = {
        "status": "CROSS_VALIDATED",
        "seed": SEED,
        "alpha_values": ALPHAS,
        "lengths": LENGTHS,
        "cells": len(rows),
        "random_rank_constraint_checks": random_words_checked,
        "max_abs_finite_minus_asymptotic_at_n_10000": max_terminal_error,
        "counting": "exact integer SFT path counts; logarithms only for reporting rates",
        "csv": CSV_OUTPUT.name,
    }
    JSON_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
