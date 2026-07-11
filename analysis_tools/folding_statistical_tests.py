"""Statistical comparison table for the folding-validation experiment.

The binary endpoint is ``max MFE stem length >= m``. The row-level checks are:

* bootstrap confidence intervals for the risk difference;
* one-sided Fisher exact tests for lower SSA risk than the random baseline;
* one-sided Mann--Whitney tests for shorter SSA MFE stems;
* Cliff's delta for the MFE max-stem length.

The tests are descriptive validation aids.  They are not used in the capacity
proof and should not be interpreted as a thermodynamic theorem.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def bootstrap_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    statistic: str,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[float, float]:
    """Return percentile CI for stat(x)-stat(y)."""
    if len(x) == 0 or len(y) == 0:
        return math.nan, math.nan
    x_idx = rng.integers(0, len(x), size=(resamples, len(x)))
    y_idx = rng.integers(0, len(y), size=(resamples, len(y)))
    if statistic == "mean":
        diffs = np.mean(x[x_idx], axis=1) - np.mean(y[y_idx], axis=1)
    elif statistic == "proportion":
        diffs = np.mean(x[x_idx], axis=1) - np.mean(y[y_idx], axis=1)
    else:
        raise ValueError(f"unknown statistic: {statistic}")
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(low), float(high)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Return Cliff's delta = P(x>y)-P(x<y). Negative means x tends lower."""
    gt = 0
    lt = 0
    for value in x:
        gt += int(np.sum(value > y))
        lt += int(np.sum(value < y))
    total = len(x) * len(y)
    return (gt - lt) / total if total else math.nan


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm--Bonferroni adjusted p-values, returned in original order."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * n
    running = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        value = min(1.0, (n - rank + 1) * p)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def format_float(value: float, digits: int = 6) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--m-values", default="6,7")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260628)
    args = parser.parse_args()

    rows = load_rows(args.rows)
    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    lengths = sorted({int(r["length"]) for r in rows})
    rng = np.random.default_rng(args.seed)

    out_rows: list[dict[str, object]] = []
    fisher_p: list[float] = []
    mann_p: list[float] = []

    for length in lengths:
        random_rows = [
            r for r in rows if r["group"] == "random_gc_homopolymer" and int(r["length"]) == length
        ]
        random_max_stem = np.array([int(r["mfe_max_stem"]) for r in random_rows], dtype=float)
        random_mfe = np.array([float(r["mfe_kcal_mol"]) for r in random_rows], dtype=float)

        for m in m_values:
            random_risk = (random_max_stem >= m).astype(float)
            mk_rows = [
                r
                for r in rows
                if r["group"] == "mk_ssa_gc_homopolymer"
                and int(r["length"]) == length
                and int(r["m"]) == m
            ]
            for k in sorted({int(r["k"]) for r in mk_rows}):
                ssa_rows = [r for r in mk_rows if int(r["k"]) == k]
                ssa_max_stem = np.array([int(r["mfe_max_stem"]) for r in ssa_rows], dtype=float)
                ssa_mfe = np.array([float(r["mfe_kcal_mol"]) for r in ssa_rows], dtype=float)
                ssa_risk = (ssa_max_stem >= m).astype(float)

                ssa_success = int(np.sum(ssa_risk))
                random_success = int(np.sum(random_risk))
                ssa_total = len(ssa_risk)
                random_total = len(random_risk)
                _, fisher_less = fisher_exact(
                    [
                        [ssa_success, ssa_total - ssa_success],
                        [random_success, random_total - random_success],
                    ],
                    alternative="less",
                )
                mw_less = float(
                    mannwhitneyu(ssa_max_stem, random_max_stem, alternative="less").pvalue
                )

                risk_delta_low, risk_delta_high = bootstrap_diff_ci(
                    ssa_risk,
                    random_risk,
                    statistic="proportion",
                    rng=rng,
                    resamples=args.bootstrap_resamples,
                )
                stem_delta_low, stem_delta_high = bootstrap_diff_ci(
                    ssa_max_stem,
                    random_max_stem,
                    statistic="mean",
                    rng=rng,
                    resamples=args.bootstrap_resamples,
                )

                fisher_p.append(float(fisher_less))
                mann_p.append(mw_less)
                random_risk_mean = mean(random_risk)
                ssa_risk_mean = mean(ssa_risk)
                row: dict[str, object] = {
                    "length": length,
                    "m": m,
                    "k": k,
                    "samples_random": random_total,
                    "samples_ssa": ssa_total,
                    "random_risk": random_risk_mean,
                    "ssa_risk": ssa_risk_mean,
                    "risk_delta_ssa_minus_random": ssa_risk_mean - random_risk_mean,
                    "risk_reduction_fraction": random_risk_mean - ssa_risk_mean,
                    "bootstrap_ci95_risk_delta_low": risk_delta_low,
                    "bootstrap_ci95_risk_delta_high": risk_delta_high,
                    "fisher_less_p": float(fisher_less),
                    "random_mean_max_stem": mean(random_max_stem),
                    "ssa_mean_max_stem": mean(ssa_max_stem),
                    "mean_max_stem_delta_ssa_minus_random": mean(ssa_max_stem) - mean(random_max_stem),
                    "bootstrap_ci95_mean_max_stem_delta_low": stem_delta_low,
                    "bootstrap_ci95_mean_max_stem_delta_high": stem_delta_high,
                    "mannwhitney_max_stem_less_p": mw_less,
                    "cliffs_delta_max_stem": cliffs_delta(ssa_max_stem, random_max_stem),
                    "random_mean_mfe_kcal_mol": mean(random_mfe),
                    "ssa_mean_mfe_kcal_mol": mean(ssa_mfe),
                    "mean_mfe_delta_ssa_minus_random": mean(ssa_mfe) - mean(random_mfe),
                }
                out_rows.append(row)

    fisher_adj = holm_adjust(fisher_p)
    mann_adj = holm_adjust(mann_p)
    for row, f_adj, m_adj in zip(out_rows, fisher_adj, mann_adj):
        row["fisher_less_p_holm"] = f_adj
        row["mannwhitney_max_stem_less_p_holm"] = m_adj
        row["risk_delta_ci_excludes_zero"] = (
            float(row["bootstrap_ci95_risk_delta_high"]) < 0
        )
        row["max_stem_delta_ci_excludes_zero"] = (
            float(row["bootstrap_ci95_mean_max_stem_delta_high"]) < 0
        )
        row["primary_support_after_holm_0p05"] = (
            float(row["risk_delta_ssa_minus_random"]) < 0
            and f_adj < 0.05
            and float(row["mean_max_stem_delta_ssa_minus_random"]) < 0
            and m_adj < 0.05
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys())
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in out_rows:
            formatted = {
                key: format_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted)

    supported = sum(1 for row in out_rows if row["primary_support_after_holm_0p05"])
    print(f"wrote {len(out_rows)} comparison rows to {args.out}")
    print(f"primary support after Holm correction: {supported}/{len(out_rows)}")


if __name__ == "__main__":
    main()
