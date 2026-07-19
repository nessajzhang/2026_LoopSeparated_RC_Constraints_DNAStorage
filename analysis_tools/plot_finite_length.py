"""Plot exact finite-block rates and the bias of short-window slope fitting."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "output" / "finite_length"
OUT_DIR = ROOT / "output" / "figures"
OUT_PNG = OUT_DIR / "fig_finite_length_rate.png"
OUT_PDF = OUT_DIR / "fig_finite_length_rate.pdf"
OUT_SVG = OUT_DIR / "fig_finite_length_rate.svg"
OUT_TIFF = OUT_DIR / "fig_finite_length_rate.tiff"
CLASSICAL_C2 = 1.167987028946


def read_counts(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA / f"counts_m2_k{k}.csv"
    ns: list[int] = []
    counts: list[int] = []
    rates: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n = int(row["n"])
            if n == 0:
                continue
            ns.append(n)
            counts.append(int(row["N"]))
            rates.append(float(row["finite_block_rate_bits_per_nt"]))
    return np.asarray(ns), np.asarray(counts, dtype=object), np.asarray(rates)


def slope_capacity(
    ns: np.ndarray, counts: np.ndarray, width: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    endpoints: list[int] = []
    estimates: list[float] = []
    log_fraction = np.asarray(
        [math.log(int(count)) - int(n) * math.log(4.0) for n, count in zip(ns, counts)]
    )
    for end in range(width - 1, len(ns)):
        x = ns[end - width + 1 : end + 1]
        y = log_fraction[end - width + 1 : end + 1]
        slope = np.polyfit(x, y, 1)[0]
        endpoints.append(int(ns[end]))
        estimates.append(2.0 + slope / math.log(2.0))
    return np.asarray(endpoints), np.asarray(estimates)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 6.9,
    color: str = "#222222",
    weight: str | None = None,
) -> FancyBboxPatch:
    """Draw a compact rounded box in axes-data coordinates."""

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=0.75,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        linespacing=1.08,
    )
    return patch


def add_arrow(
    ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = "#6d7280"
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.0,
            linewidth=0.75,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def add_panel_subtitle(ax: plt.Axes, label: str, title: str) -> None:
    """Place a compact panel label and title below an axes."""

    ax.text(
        0.5,
        -0.36,
        f"({label}) {title}",
        transform=ax.transAxes,
        fontsize=7.7,
        ha="center",
        va="top",
        clip_on=False,
    )


def plot_rule_schematic(ax: plt.Axes) -> None:
    """Draw the delayed reverse-complement rule in the same style as the plots."""

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ink = "#2b2b2b"
    muted = "#5b6370"
    pale = "#d7dde3"
    tick = "#cbd2d9"
    blue = "#2f5277"
    teal = "#5a9a9a"
    gold = "#c8a85a"
    wine = "#8c3550"

    # Sequence line and the delayed reverse-complement exclusion.
    yseq = 0.715
    ax.plot([0.09, 0.96], [yseq, yseq], color=pale, lw=0.75, solid_capstyle="round", zorder=1)
    for idx, xpos in enumerate(np.linspace(0.13, 0.92, 24), start=1):
        half = 0.020 if idx % 5 == 0 else 0.014
        ax.plot([xpos, xpos], [yseq - half / 2, yseq + half / 2], color=tick, lw=0.55, zorder=1)
    ax.text(0.045, yseq, r"$x$", fontsize=8.0, ha="center", va="center", color=ink)

    left_box = Rectangle(
        (0.215, yseq - 0.048), 0.165, 0.096, facecolor="#f2f6fa", edgecolor=blue, lw=0.9
    )
    right_box = Rectangle(
        (0.710, yseq - 0.048), 0.190, 0.096, facecolor="#fbf3f5", edgecolor=wine, lw=0.9
    )
    ax.add_patch(left_box)
    ax.add_patch(right_box)
    ax.text(0.2975, yseq, r"$u$", fontsize=8.6, ha="center", va="center", color=ink)
    ax.text(0.805, yseq, r"$\overline{u}^{\,R}$", fontsize=8.4, ha="center", va="center", color=ink)

    ax.add_patch(
        FancyArrowPatch(
            (0.410, 0.825),
            (0.690, 0.825),
            arrowstyle="<->",
            mutation_scale=8.0,
            connectionstyle="arc3,rad=-0.09",
            color=muted,
            lw=0.75,
        )
    )
    ax.text(0.550, 0.885, r"gap $\geq k$", fontsize=7.2, ha="center", va="center", color=muted)
    ax.text(
        0.550,
        0.610,
        r"forbidden only after the separation gap reaches $k$",
        fontsize=6.8,
        ha="center",
        va="center",
        color=muted,
    )

    # Capacity accounting: finite-type core plus a sublinear exception budget.
    ax.plot([0.07, 0.95], [0.495, 0.495], color=pale, lw=0.7)
    ax.text(
        0.075,
        0.435,
        "capacity accounting",
        fontsize=7.7,
        fontweight="bold",
        ha="left",
        va="center",
        color=ink,
    )

    eq_y = 0.315
    ax.text(
        0.235, eq_y, r"$\mathcal{L}_{m,k_n}$", fontsize=9.0, ha="center", va="center", color=ink
    )
    ax.text(0.362, eq_y, r"$\lesssim$", fontsize=8.4, ha="center", va="center", color=muted)
    ax.text(
        0.505,
        eq_y,
        r"$\mathcal{L}_{m}^{\mathrm{RC}}$",
        fontsize=9.0,
        ha="center",
        va="center",
        color=ink,
    )
    ax.text(0.640, eq_y, r"$\times$", fontsize=8.0, ha="center", va="center", color=muted)
    ax.text(
        0.790,
        eq_y,
        r"$\mathcal{E}(E_n^{\mathrm{start}})$",
        fontsize=8.5,
        ha="center",
        va="center",
        color=ink,
    )

    ax.plot([0.445, 0.565], [0.250, 0.250], color=teal, lw=0.85)
    ax.plot([0.445, 0.445], [0.250, 0.270], color=teal, lw=0.85)
    ax.plot([0.565, 0.565], [0.250, 0.270], color=teal, lw=0.85)
    ax.text(
        0.505,
        0.200,
        "RC-free\nfinite-type core",
        fontsize=5.7,
        ha="center",
        va="center",
        color=muted,
        linespacing=0.92,
    )

    ax.plot([0.715, 0.865], [0.250, 0.250], color=gold, lw=0.85)
    ax.plot([0.715, 0.715], [0.250, 0.270], color=gold, lw=0.85)
    ax.plot([0.865, 0.865], [0.250, 0.270], color=gold, lw=0.85)
    ax.text(
        0.790,
        0.200,
        "exceptional\nwindows",
        fontsize=5.7,
        ha="center",
        va="center",
        color=muted,
        linespacing=0.92,
    )

    ax.annotate(
        "",
        xy=(0.550, 0.135),
        xytext=(0.550, 0.160),
        arrowprops=dict(arrowstyle="-|>", lw=0.65, color=muted),
    )
    ax.text(
        0.550,
        0.090,
        r"$E_n^{\mathrm{start}}=(2(m+k_n)-1)q^m$",
        fontsize=7.15,
        ha="center",
        va="center",
        color=ink,
    )
    ax.text(
        0.550,
        0.020,
        r"$k_n=o(n)\Rightarrow R_{\theta,n}(m,k_n)\to c_{\theta,m}$",
        fontsize=7.15,
        ha="center",
        va="center",
        color=ink,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", action="store_true", help="Also write the optional SVG export.")
    parser.add_argument(
        "--tiff", action="store_true", help="Also write the optional high-resolution TIFF export."
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.75,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "figure.dpi": 220,
            # Keep text as embedded TrueType fonts in the PDF output.
            # This avoids Type 3 fonts, which are often rejected by
            # journal/IEEE PDF preflight systems.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.25, 3.15), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.22, 1.0, 1.0))
    axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    fig.subplots_adjust(left=0.045, right=0.987, bottom=0.235, top=0.765, wspace=0.34)

    colors = {0: "#2f5277", 1: "#5a9a9a", 2: "#c8a85a", 3: "#d98b5f", 5: "#b85c70"}
    markers = {0: "o", 1: "s", 2: "^", 3: "D", 5: "v"}
    handles = []
    labels = []
    plot_rule_schematic(axes[0])
    for k in (0, 1, 2, 3, 5):
        ns, counts, rates = read_counts(k)
        (line,) = axes[1].plot(
            ns,
            rates,
            color=colors[k],
            marker=markers[k],
            markevery=max(1, len(ns) // 8),
            markersize=2.8,
            linewidth=1.25,
            label=rf"$k={k}$",
        )
        handles.append(line)
        labels.append(rf"$k={k}$")
    cap_line = axes[1].axhline(
        CLASSICAL_C2,
        color="#2b2b2b",
        linestyle=(0, (4, 2)),
        linewidth=0.9,
        label=rf"$c_2\approx{CLASSICAL_C2:.4f}$",
    )
    handles.append(cap_line)
    labels.append(rf"$c_2\approx{CLASSICAL_C2:.4f}$")
    axes[1].set_xlabel("Block length $n$")
    axes[1].set_ylabel(r"Finite-block rate $R_n(2,k)$ [bits/nt]")
    axes[1].set_xlim(left=1)
    axes[1].set_ylim(1.1, 2.03)
    axes[1].grid(axis="y", alpha=0.18, linewidth=0.45)
    axes[1].text(
        39.4, CLASSICAL_C2 + 0.018, r"$c_2$", ha="right", va="bottom", fontsize=7.0, color="#2b2b2b"
    )

    ns, counts, _ = read_counts(0)
    ends, estimates = slope_capacity(ns, counts, width=8)
    axes[2].plot(ends, estimates, color="#8c3550", marker="o", markersize=2.8, linewidth=1.35)
    axes[2].axhline(CLASSICAL_C2, color="#2b2b2b", linestyle=(0, (4, 2)), linewidth=0.9)
    axes[2].set_xlabel(r"Window endpoint $N$ of 8-point OLS fit")
    axes[2].set_ylabel("Slope-based estimate [bits/nt]")
    axes[2].set_xlim(left=8)
    axes[2].set_ylim(1.1, 2.03)
    axes[2].grid(axis="y", alpha=0.18, linewidth=0.45)
    axes[2].text(
        39.4,
        CLASSICAL_C2 + 0.018,
        r"baseline $c_2$",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#2b2b2b",
    )
    axes[2].annotate(
        "finite-window\noverestimate",
        xy=(15, estimates[np.where(ends == 15)[0][0]]),
        xytext=(22, 1.70),
        arrowprops=dict(arrowstyle="-", color="#7a7a7a", lw=0.6),
        fontsize=7.0,
        color="#4f4f4f",
        ha="left",
        va="center",
    )

    for label, title, ax in zip(
        ("a", "b", "c"),
        ("Delayed RC screening rule", "Exact finite-block rates", "Eight-point slope-fit bias"),
        axes,
    ):
        add_panel_subtitle(ax, label, title)

    fig.legend(
        handles,
        labels,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.982),
        frameon=False,
        handlelength=1.8,
    )

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)
    if args.svg:
        fig.savefig(OUT_SVG, bbox_inches="tight")
        print(OUT_SVG)
    if args.tiff:
        fig.savefig(OUT_TIFF, dpi=600, bbox_inches="tight")
        print(OUT_TIFF)


if __name__ == "__main__":
    main()
