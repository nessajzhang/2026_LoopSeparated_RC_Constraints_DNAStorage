#!/usr/bin/env python3
"""Independent integer recurrences and exact-branch visualization, not throughput."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def right_count(length, m):
    """Count words in which every m-window contains A, with short boundary."""
    if length < m:
        return 4**length
    matrix = [[3**j for j in range(m)]] + [
        [int(j == i) for j in range(m)] for i in range(m - 1)
    ]
    vector = [[4**j] for j in range(m - 1, -1, -1)]
    exponent = length - m + 1
    while exponent:
        if exponent & 1:
            vector = multiply(matrix, vector)
        exponent >>= 1
        if exponent:
            matrix = multiply(matrix, matrix)
    return vector[0][0]


def codebook(n, m, alpha):
    k = n * alpha.numerator // alpha.denominator
    if n < 2 * m + k:
        return 4**n, k
    ell, free = n - m - k, 2 * (m + k) - n
    assert free >= 0
    # The all-window left language is vacuous below m, unlike ACG^ell.
    left = 3**ell if ell >= m else 4**ell
    return left * 4**free * right_count(ell, m), k


def capacity(m, alpha):
    with localcontext() as context:
        context.prec = 80
        a = Decimal(alpha.numerator) / alpha.denominator
        product = Decimal(3) * (1 + Decimal(13).sqrt()) / 2 if m == 2 else Decimal(9)
        return (4 * a - 2) + (1 - a) * product.ln() / Decimal(2).ln()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qa-scripts", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "code"))
    from loop_alpha_audit.endpoint_general import EndpointMapper
    from itertools import product

    checks = 0
    for m in (2, 3):
        for n in (8, 12, 20, 100):
            for alpha in (Fraction(1, 2), Fraction(3, 4), Fraction(9, 10), Fraction(1)):
                count, k = codebook(n, m, alpha)
                prepared = EndpointMapper(
                    n, m, k, frozenset(product(range(3), repeat=m))
                )
                assert count == prepared.codebook_size
                checks += 1
    rows = []
    alphas = [
        Fraction(1, 2),
        Fraction(3, 5),
        Fraction(3, 4),
        Fraction(9, 10),
        Fraction(1),
    ]
    for m in (2, 3):
        for n in (100, 1000, 10000, 100000, 1000000):
            for alpha in alphas:
                count, k = codebook(n, m, alpha)
                K = count.bit_length() - 1
                with localcontext() as context:
                    context.prec = 80
                    c = capacity(m, alpha)
                    excess = Decimal(K) / n - c
                rows.append(
                    {
                        "m": m,
                        "n": n,
                        "alpha": str(alpha),
                        "k": k,
                        "binary_bits": K,
                        "binary_rate": K / n,
                        "capacity": str(c),
                        "rate_excess": str(excess),
                        "codebook_sha256_big_endian": hashlib.sha256(
                            count.to_bytes((count.bit_length() + 7) // 8, "big")
                        ).hexdigest(),
                    }
                )
    with (args.out / "exact_counts.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sys.path.insert(0, str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.3,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.29, top=0.92, wspace=0.50)
    colors = {2: "#2C6088", 3: "#39888D"}
    curve_data = {}
    for m in (2, 3):
        ratios = [Fraction(50 + j, 100) for j in range(51)]
        curve = [float(capacity(m, a)) for a in ratios]
        axes[0].plot(
            [float(a) for a in ratios],
            curve,
            color=colors[m],
            label=f"Exact capacity, m = {m}",
        )
        curve_data[str(m)] = curve
        for n, marker in ((100, "s"), (10000, "o")):
            chosen = [r for r in rows if r["m"] == m and r["n"] == n]
            axes[0].plot(
                [float(Fraction(r["alpha"])) for r in chosen],
                [r["binary_rate"] for r in chosen],
                linestyle="none",
                marker=marker,
                color=colors[m],
                markerfacecolor="white" if n == 100 else colors[m],
                markeredgewidth=0.7,
                zorder=3,
            )
        for alpha, style, marker in (("1/2", "-", "o"), ("3/4", "--", "s")):
            chosen = [r for r in rows if r["m"] == m and r["alpha"] == alpha]
            values = [float(r["rate_excess"]) for r in chosen]
            assert min(values) > 0
            axes[1].plot(
                [r["n"] for r in chosen],
                values,
                style,
                marker=marker,
                color=colors[m],
                label=f"m = {m}, α = {float(Fraction(alpha)):g}",
            )
    axes[0].set(
        xlim=(0.49, 1.01),
        ylim=(1.36, 2.035),
        xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        xlabel="Linear cutoff ratio, α",
        ylabel="Input rate (bits/nt)",
    )
    axes[0].legend(loc="lower right", bbox_to_anchor=(1.02, 0.06), handlelength=1.6)
    axes[0].text(
        0.03,
        0.95,
        "□ n = 100     ● n = 10,000",
        transform=axes[0].transAxes,
        va="top",
        fontsize=7.3,
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Block length, n (nt)",
        ylabel="Codebook-rate excess (bits/nt)",
        xticks=[100, 1000, 10000, 100000, 1000000],
        xticklabels=["100", "1,000", "10,000", "100,000", "1,000,000"],
        yticks=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
        yticklabels=["0.1", "0.01", "0.001", "0.0001", "0.00001", "0.000001"],
    )
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend(loc="upper right", ncol=1, handlelength=1.8, fontsize=7.3)
    for axis, title in zip(
        axes, ("(a) Certified exact branches", "(b) Exact binary-codebook convergence")
    ):
        axis.grid(axis="y", color="#DFDFDF", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(length=2.5, width=0.65)
        axis.text(
            0.5,
            -0.39,
            title,
            ha="center",
            va="top",
            transform=axis.transAxes,
            fontsize=8,
        )
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=args.out / "alignment.json",
        overlay_svg=args.out / "alignment.svg",
        panel_ids=["a", "b"],
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    for extension in ("pdf", "svg", "png"):
        fig.savefig(args.out / f"exact_branches.{extension}", dpi=600)
    plt.close(fig)
    report = {
        "status": "EXACT_INTEGER_COUNTS_NOT_MAPPING_THROUGHPUT",
        "integer_rows": len(rows),
        "small_prepared_crosschecks": checks,
        "binary_rate": "floor(log2 M)/n, not full finite language",
        "curve_is_proved_formula_not_fit": True,
        "new_stochastic_samples": 0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(
            (args.out / "exact_counts.csv").read_bytes()
        ).hexdigest(),
        "curve_data": curve_data,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "curve_data"}))


if __name__ == "__main__":
    main()
