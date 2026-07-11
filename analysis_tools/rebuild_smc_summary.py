#!/usr/bin/env python3
"""Rebuild Python SMC summary rows from archived replicate-level CSV data."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


CLASSICAL_C2 = 1.167987028946
FIELDS = [
    "m",
    "k",
    "n",
    "reps",
    "particles",
    "mean_rate_est_bits_per_nt",
    "sd_rate",
    "sem_rate",
    "excess_over_capacity",
    "capacity_reference",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capacity-reference", type=float, default=CLASSICAL_C2)
    args = parser.parse_args()

    with args.raw.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        rate = float(row["rate_est_bits_per_nt"])
        if not math.isfinite(rate):
            raise ValueError(f"non-finite SMC rate in selected raw table: {row}")
        key = (int(row["m"]), int(row["k"]), int(row["n"]), int(row["particles"]))
        groups[key].append(rate)

    output: list[dict[str, str]] = []
    for (m, k, n, particles), values in sorted(groups.items()):
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        sem = sd / math.sqrt(len(values))
        output.append(
            {
                "m": str(m),
                "k": str(k),
                "n": str(n),
                "reps": str(len(values)),
                "particles": str(particles),
                "mean_rate_est_bits_per_nt": f"{mean:.12f}",
                "sd_rate": f"{sd:.12f}",
                "sem_rate": f"{sem:.12f}",
                "excess_over_capacity": f"{mean - args.capacity_reference:.12f}",
                "capacity_reference": f"{args.capacity_reference:.12f}",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} SMC summary rows to {args.out}")


if __name__ == "__main__":
    main()
