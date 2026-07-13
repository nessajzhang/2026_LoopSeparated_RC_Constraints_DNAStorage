#!/usr/bin/env python3
"""Wide-m stress audit for the certified structured high-alpha construction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.structured_general_m import (  # noqa: E402
    right_endpoint_count,
    right_endpoint_radius_mp,
)


M_VALUES = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
ALPHAS = (0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
VALIDATION_ALPHAS = (0.5, 0.75, 0.9, 0.99)
# Exact integer finite-length counts are more expensive than the asymptotic
# recurrence-root calculation, but they remain practical through m=128 on the
# present audit grid.  Keeping these values separate from M_VALUES preserves
# the evidence boundary: m<=128 below is exact finite-block counting, whereas
# all m<=1024 receive a certified asymptotic construction and word-level checks.
EXACT_M_VALUES = (2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128)
EXACT_ALPHAS = ALPHAS
EXACT_LENGTHS = (1_000, 10_000)
SEED = 20260712

ASYMPTOTIC_CSV = ROOT / "output" / "general_m_structured_asymptotic.csv"
VALIDATION_CSV = ROOT / "output" / "general_m_constructed_word_validation.csv"
EXACT_CSV = ROOT / "output" / "general_m_exact_finite_rates.csv"
SUMMARY_JSON = ROOT / "output" / "general_m_stress_audit_summary.json"


def log2_integer(value: int) -> float:
    shift = max(0, value.bit_length() - 53)
    return math.log2(value >> shift) + shift


def max_non_a_run(word: bytes) -> int:
    largest = 0
    current = 0
    for symbol in word:
        current = 0 if symbol == 0 else current + 1
        largest = max(largest, current)
    return largest


def make_structured_word(n: int, k: int, m: int, rng: random.Random) -> bytes:
    distance = m + k
    endpoint = n - distance
    free = 2 * distance - n
    if endpoint < m - 1 or free < 0:
        raise ValueError("inactive endpoint decomposition")
    left = bytes(rng.randrange(3) for _ in range(endpoint))
    middle = bytes(rng.randrange(4) for _ in range(free))
    right_symbols: list[int] = []
    non_a_run = 0
    for _ in range(endpoint):
        symbol = 0 if non_a_run == m - 1 else rng.randrange(4)
        right_symbols.append(symbol)
        non_a_run = 0 if symbol == 0 else non_a_run + 1
    return left + middle + bytes(right_symbols)


def reduced_scan(word: bytes, m: int, k: int) -> bool:
    distance = m + k
    n_windows = max(0, len(word) - m + 1)
    mature: set[bytes] = set()
    for later in range(n_windows):
        earlier = later - distance
        if earlier >= 0:
            mature.add(word[earlier : earlier + m])
        current = word[later : later + m]
        reverse_complement = bytes(3 - symbol for symbol in reversed(current))
        if reverse_complement in mature:
            return False
    return True


def sampled_cross_endpoint_check(
    word: bytes, n: int, k: int, m: int, rng: random.Random, samples: int
) -> bool:
    distance = m + k
    endpoint = n - distance
    r = endpoint - m + 1
    if r <= 0:
        return True
    for _ in range(samples):
        left_start = rng.randrange(r)
        right_relative = rng.randrange(left_start, r)
        right_start = distance + right_relative
        left = word[left_start : left_start + m]
        right = word[right_start : right_start + m]
        if left == bytes(3 - symbol for symbol in reversed(right)):
            return False
    return True


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ROOT.joinpath("output").mkdir(parents=True, exist_ok=True)
    asymptotic_rows: list[dict[str, object]] = []
    radii: dict[int, mp.mpf] = {}
    for m in M_VALUES:
        radius = right_endpoint_radius_mp(m, decimal_places=80)
        radii[m] = radius
        with mp.workdps(max(80, math.ceil(m * math.log10(4 / 3)) + 30)):
            gap = mp.mpf(4) - radius
            lambda_lower = 3 * radius
            for alpha in ALPHAS:
                capacity = 2 * (2 * alpha - 1) + (1 - alpha) * mp.log(lambda_lower, 2)
                asymptotic_rows.append(
                    {
                        "m": m,
                        "alpha": alpha,
                        "right_radius": mp.nstr(radius, 40),
                        "four_minus_radius": mp.nstr(gap, 12),
                        "lambda_lower": mp.nstr(lambda_lower, 40),
                        "capacity_lower_bits_per_nt": mp.nstr(capacity, 18),
                    }
                )

    rng = random.Random(SEED)
    validation_rows: list[dict[str, object]] = []
    full_scans = 0
    sampled_pairs = 0
    for m in M_VALUES:
        for alpha in VALIDATION_ALPHAS:
            small_n = max(10_000, math.ceil(4 * m / (1 - alpha)))
            for n in sorted({small_n, 1_000_000}):
                k = math.floor(alpha * n)
                endpoint = n - m - k
                if endpoint < m - 1:
                    continue
                word = make_structured_word(n, k, m, rng)
                free = 2 * (m + k) - n
                left = word[:endpoint]
                right = word[endpoint + free :]
                invariant = all(symbol < 3 for symbol in left) and max_non_a_run(right) < m
                if not invariant:
                    raise AssertionError(("construction_invariant", m, alpha, n))
                complete_scan = n * m <= 2_000_000
                if complete_scan:
                    if not reduced_scan(word, m, k):
                        raise AssertionError(("complete_reduced_scan", m, alpha, n))
                    full_scans += 1
                samples = 5_000
                if not sampled_cross_endpoint_check(word, n, k, m, rng, samples):
                    raise AssertionError(("sampled_cross_endpoint", m, alpha, n))
                sampled_pairs += samples
                validation_rows.append(
                    {
                        "m": m,
                        "alpha": alpha,
                        "n": n,
                        "k": k,
                        "endpoint_length": endpoint,
                        "free_length": free,
                        "complete_reduced_scan": int(complete_scan),
                        "sampled_cross_pairs": samples,
                        "max_right_non_a_run": max_non_a_run(right),
                        "status": "PASS",
                    }
                )

    exact_rows: list[dict[str, object]] = []
    for m in EXACT_M_VALUES:
        radius = float(radii[m])
        lambda_lower_log = math.log2(3 * radius)
        for alpha in EXACT_ALPHAS:
            for n in EXACT_LENGTHS:
                k = math.floor(alpha * n)
                endpoint = n - m - k
                free = 2 * (m + k) - n
                if endpoint < m - 1 or free < 0:
                    continue
                right_count = right_endpoint_count(endpoint, m)
                log_size = endpoint * math.log2(3) + 2 * free + log2_integer(right_count)
                asymptotic = 2 * (2 * alpha - 1) + (1 - alpha) * lambda_lower_log
                exact_rows.append(
                    {
                        "m": m,
                        "alpha": alpha,
                        "n": n,
                        "k": k,
                        "endpoint_length": endpoint,
                        "right_count_bit_length": right_count.bit_length(),
                        "finite_rate": f"{log_size / n:.15f}",
                        "asymptotic_lower_rate": f"{asymptotic:.15f}",
                        "finite_minus_asymptotic": f"{log_size / n - asymptotic:.15e}",
                    }
                )

    digests = {
        ASYMPTOTIC_CSV.name: write_csv(ASYMPTOTIC_CSV, asymptotic_rows),
        VALIDATION_CSV.name: write_csv(VALIDATION_CSV, validation_rows),
        EXACT_CSV.name: write_csv(EXACT_CSV, exact_rows),
    }
    summary = {
        "status": "CROSS_VALIDATED",
        "m_values": M_VALUES,
        "m_max": max(M_VALUES),
        "alpha_values": ALPHAS,
        "asymptotic_cells": len(asymptotic_rows),
        "constructed_words_checked": len(validation_rows),
        "maximum_constructed_n": max(int(row["n"]) for row in validation_rows),
        "complete_reduced_scans": full_scans,
        "sampled_cross_endpoint_pairs": sampled_pairs,
        "exact_finite_rate_cells": len(exact_rows),
        "exact_finite_m_values": EXACT_M_VALUES,
        "exact_finite_alpha_values": EXACT_ALPHAS,
        "exact_finite_lengths": EXACT_LENGTHS,
        "seed": SEED,
        "evidence_boundary": (
            "The structured construction gives certified lower bounds for every reported m. "
            "It is not a global maximization of Lambda for m>2."
        ),
        "sha256": digests,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
