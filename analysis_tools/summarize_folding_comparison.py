"""Create an interpretation-friendly summary of folding validation results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """Normal-approximation 95% CI for a mean."""
    if len(values) < 2:
        return values[0], values[0]
    mu = mean(values)
    half_width = 1.96 * stdev(values) / math.sqrt(len(values))
    return mu - half_width, mu + half_width


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--m-values", default="6,7")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.rows)
    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    lengths = sorted({int(r["length"]) for r in rows})
    output: list[dict[str, object]] = []

    for length in lengths:
        random_subset = [r for r in rows if r["group"] == "random_gc_homopolymer" and int(r["length"]) == length]
        for m in m_values:
            for subset_name, subset in [("random_gc_homopolymer", random_subset)]:
                mfes = [float(r["mfe_kcal_mol"]) for r in subset]
                stems = [int(r["mfe_max_stem"]) for r in subset]
                mfe_low, mfe_high = mean_ci95(mfes)
                stem_values = [float(x) for x in stems]
                stem_low, stem_high = mean_ci95(stem_values)
                risk_count = sum(1 for x in stems if x >= m)
                risk_low, risk_high = wilson_ci(risk_count, len(stems))
                output.append(
                    {
                        "group": subset_name,
                        "length": length,
                        "m": m,
                        "k": "",
                        "samples": len(subset),
                        "mean_mfe_kcal_mol": f"{mean(mfes):.4f}",
                        "sd_mfe_kcal_mol": f"{stdev(mfes):.4f}",
                        "ci95_mfe_low": f"{mfe_low:.4f}",
                        "ci95_mfe_high": f"{mfe_high:.4f}",
                        "mean_mfe_max_stem": f"{mean(stem_values):.4f}",
                        "sd_mfe_max_stem": f"{stdev(stem_values):.4f}",
                        "ci95_max_stem_low": f"{stem_low:.4f}",
                        "ci95_max_stem_high": f"{stem_high:.4f}",
                        "risk_count_max_stem_ge_m": risk_count,
                        "frac_max_stem_ge_m": f"{risk_count / len(stems):.4f}",
                        "ci95_frac_low": f"{risk_low:.4f}",
                        "ci95_frac_high": f"{risk_high:.4f}",
                        "delta_frac_vs_random": "",
                    }
                )

            mk_rows = [
                r
                for r in rows
                if r["group"] == "mk_ssa_gc_homopolymer"
                and int(r["length"]) == length
                and int(r["m"]) == m
            ]
            for k in sorted({int(r["k"]) for r in mk_rows}):
                subset = [r for r in mk_rows if int(r["k"]) == k]
                mfes = [float(r["mfe_kcal_mol"]) for r in subset]
                stems = [int(r["mfe_max_stem"]) for r in subset]
                stem_values = [float(x) for x in stems]
                mfe_low, mfe_high = mean_ci95(mfes)
                stem_low, stem_high = mean_ci95(stem_values)
                risk_count = sum(1 for x in stems if x >= m)
                risk_low, risk_high = wilson_ci(risk_count, len(stems))
                random_stems = [int(r["mfe_max_stem"]) for r in random_subset]
                random_risk = sum(1 for x in random_stems if x >= m) / len(random_stems)
                current_risk = risk_count / len(stems)
                output.append(
                    {
                        "group": "mk_ssa_gc_homopolymer",
                        "length": length,
                        "m": m,
                        "k": k,
                        "samples": len(subset),
                        "mean_mfe_kcal_mol": f"{mean(mfes):.4f}",
                        "sd_mfe_kcal_mol": f"{stdev(mfes):.4f}",
                        "ci95_mfe_low": f"{mfe_low:.4f}",
                        "ci95_mfe_high": f"{mfe_high:.4f}",
                        "mean_mfe_max_stem": f"{mean(stem_values):.4f}",
                        "sd_mfe_max_stem": f"{stdev(stem_values):.4f}",
                        "ci95_max_stem_low": f"{stem_low:.4f}",
                        "ci95_max_stem_high": f"{stem_high:.4f}",
                        "risk_count_max_stem_ge_m": risk_count,
                        "frac_max_stem_ge_m": f"{current_risk:.4f}",
                        "ci95_frac_low": f"{risk_low:.4f}",
                        "ci95_frac_high": f"{risk_high:.4f}",
                        "delta_frac_vs_random": f"{current_risk - random_risk:.4f}",
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)
    print(args.out)


if __name__ == "__main__":
    main()
