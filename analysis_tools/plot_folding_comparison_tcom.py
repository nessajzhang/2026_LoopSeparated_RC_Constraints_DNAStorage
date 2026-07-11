"""Create the single-column folding-risk figure used in the TCOM/TIT draft.

The figure is application-level evidence only.  It visualizes the row-level
ViennaRNA screen summaries and keeps the sampler / thermodynamic limitations in
the caption rather than presenting the plot as proof of the capacity theorem.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
        # Independent-binomial CI for the difference in proportions.  The plot
        # is a diagnostic screen, so this is used only to show sampling scale.
        se = np.sqrt(p0 * (1.0 - p0) / n0 + p1 * (1.0 - p1) / n1)
        delta = 100.0 * (p0 - p1)
        half_width = 100.0 * 1.96 * se
        reduction.append(delta)
        lo.append(delta - half_width)
        hi.append(delta + half_width)
    y = np.asarray(reduction, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return x, y, lo, hi


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
    parser.add_argument("--png", action="store_true", help="Also write a PNG next to --out.")
    parser.add_argument("--svg", action="store_true", help="Also write an SVG next to --out.")
    parser.add_argument("--tiff", action="store_true", help="Also write a high-resolution TIFF next to --out.")
    args = parser.parse_args()

    configure_matplotlib()
    rows = load_rows(args.comparison)
    m_values = sorted({int(r["m"]) for r in rows})

    fig, axes = plt.subplots(
        len(m_values),
        1,
        figsize=(3.45, 3.70),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    if len(m_values) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.13, top=0.84, hspace=0.24)

    handles = []
    labels = []
    for ax, m, panel in zip(axes, m_values, ("a", "b", "c", "d")):
        baseline = random_by_length(rows, m)
        for key, label, color, marker, offset in SERIES:
            items = matched_rows(rows, key, m)
            if not items:
                continue
            x, y, lo, hi = reduction_arrays(baseline, items)
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
        ax.set_ylim(-2, 40)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_yticklabels(["0", "10", "20", "30", "40"])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.45, alpha=0.55)
        ax.set_xticks([100, 150, 200])
        ax.set_xlim(92, 208)
        ax.text(-0.145, 1.03, panel, transform=ax.transAxes, fontsize=8.2, fontweight="bold", va="bottom")
        ax.text(0.02, 0.90, rf"$m={m}$", transform=ax.transAxes, fontsize=7.8, fontweight="bold")
        ax.text(
            0.98,
            0.08,
            "positive = lower risk",
            transform=ax.transAxes,
            fontsize=6.2,
            ha="right",
            color="#555555",
        )

    axes[-1].set_xlabel("Sequence length (nt)")
    fig.text(
        0.02,
        0.48,
        "Risk reduction vs random (percentage points)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.4,
    )
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.54, 0.985), handlelength=1.55, columnspacing=0.8)

    save_all(fig, args.out, png=args.png, svg=args.svg, tiff=args.tiff)


if __name__ == "__main__":
    main()
