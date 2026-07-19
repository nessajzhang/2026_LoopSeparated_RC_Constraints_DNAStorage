"""Create the single-column predicted-incidence figure used in the TCOM draft.

The figure is application-level evidence only.  It visualizes the row-level
ViennaRNA screen summaries and keeps the sampler / thermodynamic limitations in
the caption rather than presenting the plot as proof of the capacity theorem.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SERIES = [
    ("0", r"$k=0$", "#2f5277", "o", -3.6),
    ("3", r"$k=3$", "#5a9a9a", "s", -1.2),
    ("4", r"$k=4$", "#c8a85a", "^", 1.2),
    ("5", r"$k=5$", "#b85c70", "D", 3.6),
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def sample_label(rows: list[dict[str, str]]) -> str:
    values = sorted({int(r["samples"]) for r in rows if r.get("samples")})
    if len(values) == 1:
        return f"n={values[0]} per group"
    return "mixed n"


def matched_rows(rows: list[dict[str, str]], key: str, m: int) -> list[dict[str, str]]:
    out = [
        r
        for r in rows
        if r["group"] == "mk_ssa_gc_homopolymer" and int(r["m"]) == m and r["k"] == key
    ]
    return sorted(out, key=lambda r: int(r["length"]))


def random_by_length(rows: list[dict[str, str]], m: int) -> dict[int, dict[str, str]]:
    return {
        int(r["length"]): r
        for r in rows
        if r["group"] == "random_gc_homopolymer" and int(r["m"]) == m
    }


def reduction_arrays(
    random_rows: dict[int, dict[str, str]],
    items: list[dict[str, str]],
    *,
    z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([int(r["length"]) for r in items], dtype=float)
    reduction = []
    lo = []
    hi = []
    for r in items:
        rr = random_rows[int(r["length"])]
        p0 = float(rr["frac_max_stem_ge_m"])
        p1 = float(r["frac_max_stem_ge_m"])
        n0 = int(rr["samples"])
        n1 = int(r["samples"])
        x0 = int(rr["risk_count_max_stem_ge_m"])
        x1 = int(r["risk_count_max_stem_ge_m"])
        l0, u0 = wilson_interval(x0, n0, z=z)
        l1, u1 = wilson_interval(x1, n1, z=z)
        delta = 100.0 * (p0 - p1)
        # Newcombe's score interval for a difference of two independent
        # proportions, with the per-comparison confidence level supplied by a
        # Bonferroni adjustment over the complete plotted family.
        lower = 100.0 * ((p0 - p1) - math.sqrt((p0 - l0) ** 2 + (u1 - p1) ** 2))
        upper = 100.0 * ((p0 - p1) + math.sqrt((u0 - p0) ** 2 + (p1 - l1) ** 2))
        reduction.append(delta)
        lo.append(lower)
        hi.append(upper)
    y = np.asarray(reduction, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return x, y, lo, hi


def wilson_interval(successes: int, total: int, *, z: float) -> tuple[float, float]:
    """Wilson score interval for one binomial proportion."""
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.8,
            "axes.linewidth": 0.75,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def save_all(
    fig: mpl.figure.Figure,
    out_pdf: Path,
    *,
    png: bool,
    svg: bool,
    tiff: bool,
) -> None:
    out_png = out_pdf.with_suffix(".png")
    out_svg = out_pdf.with_suffix(".svg")
    out_tiff = out_pdf.with_suffix(".tiff")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(out_pdf)
    if png:
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        print(out_png)
    if svg:
        fig.savefig(out_svg, bbox_inches="tight")
        print(out_svg)
    if tiff:
        fig.savefig(out_tiff, dpi=600, bbox_inches="tight")
        print(out_tiff)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        help="Write the familywise interval audit as JSON.",
    )
    parser.add_argument("--png", action="store_true", help="Also write a PNG next to --out.")
    parser.add_argument("--svg", action="store_true", help="Also write an SVG next to --out.")
    parser.add_argument(
        "--tiff", action="store_true", help="Also write a high-resolution TIFF next to --out."
    )
    args = parser.parse_args()

    configure_matplotlib()
    rows = load_rows(args.comparison)
    m_values = sorted({int(r["m"]) for r in rows})
    comparison_count = sum(len(matched_rows(rows, key, m)) for m in m_values for key, *_ in SERIES)
    familywise_alpha = 0.05
    z = NormalDist().inv_cdf(1.0 - familywise_alpha / (2.0 * comparison_count))

    fig, axes = plt.subplots(
        len(m_values),
        1,
        figsize=(3.45, 4.20),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    if len(m_values) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.17, top=0.86, hspace=0.56)

    handles = []
    labels = []
    interval_rows: list[dict[str, float | int]] = []
    for ax, m, panel in zip(axes, m_values, ("a", "b", "c", "d")):
        baseline = random_by_length(rows, m)
        for key, label, color, marker, offset in SERIES:
            items = matched_rows(rows, key, m)
            if not items:
                continue
            x, y, lo, hi = reduction_arrays(baseline, items, z=z)
            interval_rows.extend(
                {
                    "length": int(length),
                    "m": m,
                    "k": int(key),
                    "reduction_percentage_points": float(point),
                    "lower_percentage_points": float(lower),
                    "upper_percentage_points": float(upper),
                }
                for length, point, lower, upper in zip(x, y, lo, hi)
            )
            xp = x + offset
            err = np.vstack([y - lo, hi - y])
            ax.errorbar(
                xp,
                y,
                yerr=err,
                color=color,
                marker=marker,
                markersize=3.1,
                linewidth=0.9,
                elinewidth=0.75,
                capsize=1.8,
                linestyle="none",
                label=label,
            )
            if ax is axes[0]:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        marker=marker,
                        linestyle="none",
                        markersize=3.1,
                    )
                )
                labels.append(label)

        ax.axhline(0, color="#555555", linewidth=0.65, linestyle=(0, (2.5, 2.0)))
        ax.set_ylim(-2, 43)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_yticklabels(["0", "10", "20", "30", "40"])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.55)
        ax.set_xticks([100, 150, 200])
        ax.set_xlim(92, 208)
        panel_title_y = -0.24 if ax is not axes[-1] else -0.40
        ax.text(
            0.5,
            panel_title_y,
            rf"({panel}) Stem threshold $m = {m}$",
            transform=ax.transAxes,
            fontsize=7.4,
            ha="center",
            va="top",
            clip_on=False,
        )
        ax.text(
            0.98,
            0.08,
            "positive = lower predicted incidence\nunder constrained generator",
            transform=ax.transAxes,
            fontsize=6.2,
            ha="right",
            color="#555555",
        )

    axes[-1].set_xlabel("Sequence length (nt)")
    fig.text(
        0.02,
        0.48,
        "Predicted stem-incidence difference\n(filter-matched baseline - constrained; pp)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.4,
    )
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.985),
        handlelength=1.55,
        columnspacing=0.8,
    )

    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "two-sided Bonferroni-Newcombe score intervals",
            "familywise_confidence": 0.95,
            "comparison_count": comparison_count,
            "per_comparison_alpha": familywise_alpha / comparison_count,
            "per_comparison_z": z,
            "all_lower_bounds_positive": all(
                row["lower_percentage_points"] > 0 for row in interval_rows
            ),
            "minimum_lower_percentage_points": min(
                row["lower_percentage_points"] for row in interval_rows
            ),
            "maximum_upper_percentage_points": max(
                row["upper_percentage_points"] for row in interval_rows
            ),
            "intervals": interval_rows,
        }
        args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(args.summary)

    save_all(fig, args.out, png=args.png, svg=args.svg, tiff=args.tiff)


if __name__ == "__main__":
    main()
