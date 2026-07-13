#!/usr/bin/env python3
"""Build a fixed-k SMC summary from prespecified replicate tables.

The selected base C++ archive uses 2,500 particles for k in {0,1,2,5,10}.
Low-particle k=20 and k=50 pilot rows are diagnostic history and are not
included.  The k=20 series instead uses two prespecified higher-particle runs:
30,000 particles through n=2,000 and 12,000 particles from n=5,000 onward.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


CLASSICAL_C2 = 1.167987028946
SUMMARY_FIELDS = [
    "m",
    "regime",
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "m",
        "regime",
        "k",
        "n",
        "rep",
        "particles",
        "rate_est_bits_per_nt",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} is not a compatible C++ SMC raw table")
    return rows


def summarize(rows: list[dict[str, str]], capacity: float) -> list[dict[str, str]]:
    grouped: dict[tuple[int, str, int, int, int], list[float]] = defaultdict(list)
    for row in rows:
        rate = float(row["rate_est_bits_per_nt"])
        if not math.isfinite(rate):
            raise ValueError(
                "selected SMC evidence contains a non-finite replicate: "
                f"k={row['k']}, n={row['n']}, rep={row['rep']}"
            )
        key = (
            int(row["m"]),
            row["regime"],
            int(row["k"]),
            int(row["n"]),
            int(row["particles"]),
        )
        grouped[key].append(rate)

    output: list[dict[str, str]] = []
    for (m, regime, k, n, particles), values in sorted(grouped.items()):
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        sem = sd / math.sqrt(len(values))
        output.append(
            {
                "m": str(m),
                "regime": regime,
                "k": str(k),
                "n": str(n),
                "reps": str(len(values)),
                "particles": str(particles),
                "mean_rate_est_bits_per_nt": f"{mean:.12f}",
                "sd_rate": f"{sd:.12f}",
                "sem_rate": f"{sem:.12f}",
                "excess_over_capacity": f"{mean - capacity:.12f}",
                "capacity_reference": f"{capacity:.12f}",
            }
        )
    return output


def select_rows(
    base: list[dict[str, str]],
    k20_short: list[dict[str, str]],
    k20_long: list[dict[str, str]],
) -> list[dict[str, str]]:
    selected = [row for row in base if int(row["k"]) in {0, 1, 2, 5, 10}]
    selected.extend(
        row
        for row in k20_short
        if int(row["k"]) == 20 and int(row["n"]) in {100, 200, 500, 1000, 2000}
    )
    selected.extend(
        row
        for row in k20_long
        if int(row["k"]) == 20 and int(row["n"]) in {5000, 10000, 20000, 50000, 100000}
    )
    return selected


def validate_grid(summary: list[dict[str, str]]) -> None:
    expected_k = {0, 1, 2, 5, 10, 20}
    expected_n = {100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000}
    observed = {(int(row["k"]), int(row["n"])) for row in summary}
    expected = {(k, n) for k in expected_k for n in expected_n}
    if observed != expected:
        raise ValueError(
            f"selected fixed-k grid mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    for row in summary:
        if int(row["m"]) != 2 or row["regime"] != "fixed":
            raise ValueError(f"unexpected selected fixed-k regime row: {row}")
        if int(row["k"]) == 20:
            n = int(row["n"])
            expected_particles = 30000 if n <= 2000 else 12000
            if int(row["particles"]) != expected_particles or int(row["reps"]) != 8:
                raise ValueError(f"unexpected refined k=20 design at n={n}: {row}")
        elif int(row["particles"]) != 2500 or int(row["reps"]) != 4:
            raise ValueError(f"unexpected selected base design: {row}")


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-raw", type=Path, required=True)
    parser.add_argument("--k20-short-raw", type=Path, required=True)
    parser.add_argument("--k20-long-raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--unfiltered-out",
        type=Path,
        default=None,
        help="Optional full summary of the base raw grid using the same capacity reference.",
    )
    parser.add_argument("--capacity-reference", type=float, default=CLASSICAL_C2)
    args = parser.parse_args()

    base_rows = read_rows(args.base_raw)
    rows = select_rows(
        base_rows,
        read_rows(args.k20_short_raw),
        read_rows(args.k20_long_raw),
    )
    summary = summarize(rows, args.capacity_reference)
    validate_grid(summary)
    write_rows(args.out, summary)
    print(f"wrote {len(summary)} selected fixed-k rows to {args.out}")
    if args.unfiltered_out is not None:
        unfiltered = summarize(base_rows, args.capacity_reference)
        write_rows(args.unfiltered_out, unfiltered)
        print(f"wrote {len(unfiltered)} unfiltered base-grid rows to {args.unfiltered_out}")


if __name__ == "__main__":
    main()
