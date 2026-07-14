"""Run a grid of ViennaRNA DNA folding validations.

The output is a single row-level CSV and a summary CSV.  The experiment is
intended as a reproducible pre-benchmark for the biology-facing revision, not
as a final wet-lab substitute.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

from folding_validation import (
    ENERGY_MODEL,
    fold_sequence,
    gc_fraction,
    greedy_mk_sequence,
    is_mk_ssa,
    random_screened_sequence,
)


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["group"], row["length"], row["m"], row["k"])
        groups.setdefault(key, []).append(row)

    out: list[dict[str, object]] = []
    for (group, length, m, k), subset in sorted(groups.items(), key=lambda item: str(item[0])):
        mfes = [float(r["mfe_kcal_mol"]) for r in subset]
        pairs = [float(r["mfe_pair_count"]) for r in subset]
        stems = [float(r["mfe_max_stem"]) for r in subset]
        out.append(
            {
                "group": group,
                "length": length,
                "m": m,
                "k": k,
                "samples": len(subset),
                "mean_mfe_kcal_mol": f"{mean(mfes):.4f}",
                "sd_mfe_kcal_mol": f"{stdev(mfes):.4f}",
                "mean_pair_count": f"{mean(pairs):.4f}",
                "sd_pair_count": f"{stdev(pairs):.4f}",
                "mean_mfe_max_stem": f"{mean(stems):.4f}",
                "sd_mfe_max_stem": f"{stdev(stems):.4f}",
                "frac_max_stem_ge_m": f"{sum(1 for r in subset if m not in ('', None) and int(r['mfe_max_stem']) >= int(m)) / len(subset):.4f}"
                if m not in ("", None)
                else "",
                "energy_model": ENERGY_MODEL,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="100,150,200")
    parser.add_argument("--m-values", default="6,7")
    parser.add_argument("--k-values", default="0,3,4,5")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--gc-low", type=float, default=0.40)
    parser.add_argument("--gc-high", type=float, default=0.60)
    parser.add_argument("--max-homopolymer", type=int, default=4)
    parser.add_argument("--max-restarts", type=int, default=20000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    lengths = parse_ints(args.lengths)
    m_values = parse_ints(args.m_values)
    k_values = parse_ints(args.k_values)
    rng = random.Random(args.seed)

    rows: list[dict[str, object]] = []
    for length in lengths:
        for sample in range(args.samples):
            seq = random_screened_sequence(
                rng,
                length,
                args.gc_low,
                args.gc_high,
                args.max_homopolymer,
                args.max_restarts,
            )
            structure, mfe, pair_count, max_stem = fold_sequence(seq)
            rows.append(
                {
                    "group": "random_gc_homopolymer",
                    "length": length,
                    "m": "",
                    "k": "",
                    "sample": sample,
                    "gc": f"{gc_fraction(seq):.4f}",
                    "mfe_kcal_mol": f"{mfe:.4f}",
                    "mfe_pair_count": pair_count,
                    "mfe_max_stem": max_stem,
                    "energy_model": ENERGY_MODEL,
                    "mk_ssa_valid": "",
                    "sequence": seq,
                    "mfe_structure": structure,
                }
            )

        for m in m_values:
            for k in k_values:
                for sample in range(args.samples):
                    seq = greedy_mk_sequence(
                        rng,
                        length,
                        m,
                        k,
                        args.gc_low,
                        args.gc_high,
                        args.max_homopolymer,
                        args.max_restarts,
                    )
                    structure, mfe, pair_count, max_stem = fold_sequence(seq)
                    rows.append(
                        {
                            "group": "mk_ssa_gc_homopolymer",
                            "length": length,
                            "m": m,
                            "k": k,
                            "sample": sample,
                            "gc": f"{gc_fraction(seq):.4f}",
                            "mfe_kcal_mol": f"{mfe:.4f}",
                            "mfe_pair_count": pair_count,
                            "mfe_max_stem": max_stem,
                            "energy_model": ENERGY_MODEL,
                            "mk_ssa_valid": is_mk_ssa(seq, m, k),
                            "sequence": seq,
                            "mfe_structure": structure,
                        }
                    )
                print(f"finished length={length} m={m} k={k}", flush=True)

    write_csv(args.out, rows)
    summary = summarize(rows)
    write_csv(args.summary_out, summary)
    print(f"wrote rows: {args.out}")
    print(f"wrote summary: {args.summary_out}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
