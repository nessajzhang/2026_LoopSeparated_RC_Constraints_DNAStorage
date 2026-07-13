#!/usr/bin/env python3
"""Broad (m,k,n) stress audit of avoidance and observed orbit clustering."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "output" / "mkn_orbit_stress.csv"
SUMMARY_JSON = ROOT / "output" / "mkn_orbit_stress_summary.json"
M_VALUES = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32)
N_VALUES = (1_000, 10_000, 100_000)
REPLICATES = 2
SEED = 20260712


def reverse_complement(block: bytes) -> bytes:
    return bytes(3 - symbol for symbol in reversed(block))


def threshold_grid(n: int) -> tuple[tuple[str, int], ...]:
    candidates = (
        ("zero", 0),
        ("fixed4", 4),
        ("sqrt", math.floor(math.sqrt(n))),
        ("n_0.60", math.floor(n**0.60)),
        ("n_0.75", math.floor(n**0.75)),
        ("n_log_n", math.floor(n / math.log(n))),
        ("linear_0.10", math.floor(0.10 * n)),
        ("linear_0.50", math.floor(0.50 * n)),
        ("linear_0.90", math.floor(0.90 * n)),
        ("near_vacuous", max(0, n - 2)),
    )
    seen: set[int] = set()
    result: list[tuple[str, int]] = []
    for label, value in candidates:
        value = min(max(0, value), max(0, n - 1))
        if value not in seen:
            result.append((label, value))
            seen.add(value)
    return tuple(result)


def greedy_avoiding_word(n: int, m: int, k: int, rng: random.Random) -> bytes:
    safe = bytes(rng.randrange(2) for _ in range(n))
    if n < m:
        return safe
    distance = m + k
    local_length = min(n, max(m, min(distance - 1, n // 4)))
    for _attempt in range(40):
        candidate = bytearray(safe)
        lo = rng.randrange(0, n - local_length + 1)
        edits = max(1, local_length // 8)
        for _ in range(edits):
            candidate[lo + rng.randrange(local_length)] = rng.randrange(4)
        result = bytes(candidate)
        if reduced_scan(result, m, k):
            return result
    return safe


def reduced_scan(word: bytes, m: int, k: int) -> bool:
    distance = m + k
    mature: set[bytes] = set()
    n_windows = max(0, len(word) - m + 1)
    for later in range(n_windows):
        earlier = later - distance
        if earlier >= 0:
            mature.add(word[earlier : earlier + m])
        if reverse_complement(word[later : later + m]) in mature:
            return False
    return True


def observed_orbit_cover_metrics(word: bytes, m: int, k: int) -> tuple[int, int, int]:
    starts: dict[bytes, list[int]] = defaultdict(list)
    for start in range(max(0, len(word) - m + 1)):
        starts[word[start : start + m]].append(start)
    distance = m + k
    processed: set[bytes] = set()
    intervals: list[tuple[int, int]] = []
    for block in sorted(starts):
        if block in processed:
            continue
        partner = reverse_complement(block)
        processed.add(block)
        processed.add(partner)
        if block == partner:
            lo = min(starts[block])
            hi = max(starts[block])
            if hi - lo + 1 > distance:
                raise AssertionError(("self_rc_span", m, k, lo, hi))
            intervals.append((lo, hi))
            continue
        if partner not in starts:
            continue
        selected, noncore = sorted((block, partner))
        anchor = min(starts[selected])
        lo = max(0, anchor - distance + 1)
        hi = min(len(word) - m, anchor + distance - 1)
        if any(not lo <= start <= hi for start in starts[noncore]):
            raise AssertionError(("paired_orbit_span", m, k, selected, noncore))
        intervals.append((lo, hi))

    marked: set[int] = set()
    for lo, hi in intervals:
        marked.update(range(lo, min(len(word), hi + m)))
    components = 0
    previous = -2
    for coordinate in sorted(marked):
        if coordinate != previous + 1:
            components += 1
        previous = coordinate
    return len(intervals), len(marked), components


def main() -> None:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    for m in M_VALUES:
        for n in N_VALUES:
            for schedule, k in threshold_grid(n):
                for replicate in range(REPLICATES):
                    word = greedy_avoiding_word(n, m, k, rng)
                    if not reduced_scan(word, m, k):
                        raise AssertionError(("reduced_scan", m, n, k, replicate))
                    interval_count, marked_count, marked_components = observed_orbit_cover_metrics(
                        word, m, k
                    )
                    rows.append(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "schedule": schedule,
                            "replicate": replicate,
                            "distance": m + k,
                            "observed_orbit_intervals": interval_count,
                            "marked_coordinates": marked_count,
                            "marked_fraction": f"{marked_count / n:.12f}",
                            "marked_components": marked_components,
                            "status": "PASS",
                        }
                    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(OUTPUT_CSV.read_bytes()).hexdigest()
    summary = {
        "status": "CROSS_VALIDATED",
        "m_values": M_VALUES,
        "n_values": N_VALUES,
        "schedules": sorted({str(row["schedule"]) for row in rows}),
        "replicates": REPLICATES,
        "cells": len(rows),
        "maximum_k": max(int(row["k"]) for row in rows),
        "maximum_observed_intervals": max(int(row["observed_orbit_intervals"]) for row in rows),
        "maximum_marked_fraction": max(float(row["marked_fraction"]) for row in rows),
        "seed": SEED,
        "csv_sha256": digest,
        "csv": OUTPUT_CSV.name,
        "evidence_boundary": (
            "These are generator and orbit-cover stress tests, not uniform samples from the full language "
            "and not capacity estimates."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
