#!/usr/bin/env python3
"""Create the submission figure for the scaling-law evidence chain.

The script reads only frozen audit tables.  It does not estimate, smooth, or
drop observations.  Analytic curves are evaluated from the formulas proved in
the manuscript and are styled differently from finite-block data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"
SOURCE_DATA = FIGURES / "source_data"
STEM = FIGURES / "fig_scaling_evidence"

INPUTS = {
    "encoder": OUTPUT / "segmented_encoder_ultrascale_rates.csv",
    "paired": OUTPUT / "paired_dna_m2_cpp_rates_r5000.csv",
    "broad_m": OUTPUT / "general_m_structured_asymptotic.csv",
    "finite_m": OUTPUT / "general_m_exact_finite_rates.csv",
    "encoder_summary": OUTPUT / "segmented_encoder_ultrascale_summary.json",
    "broad_m_summary": OUTPUT / "general_m_stress_audit_summary.json",
    "spectral_summary": OUTPUT / "dna_m2_spectral_certificate_summary.json",
}

WIDTH_MM = 178.0
HEIGHT_MM = 132.0
INCH_PER_MM = 1 / 25.4

BLUE = "#315E86"
BLUE_MID = "#6388A7"
BLUE_LIGHT = "#9DB5C8"
ORANGE = "#B86832"
ORANGE_LIGHT = "#D6A47D"
INK = "#202020"
MID = "#666666"
GRID = "#D9D9D9"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.8,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "axes.linewidth": 0.65,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.fontsize": 6.1,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )


def quiet_grid(ax: mpl.axes.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.45, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)


def draw_panel_a(ax: mpl.axes.Axes, encoder_rows: list[dict[str, str]]) -> None:
    log_lambda = math.log2(3 * (1 + math.sqrt(13)) / 2)
    alpha = np.linspace(0.5, 0.9995, 500)
    capacity = 2 * (2 * alpha - 1) + (1 - alpha) * log_lambda

    ax.plot(alpha, capacity, color=BLUE, linewidth=1.8, zorder=3)

    styles = {
        1_000: dict(marker="s", facecolors="white", edgecolors=ORANGE, linewidths=0.7),
        1_000_000: dict(marker="o", facecolors=ORANGE, edgecolors="white", linewidths=0.35),
    }
    labels = {1_000: r"encoder, $n=10^3$", 1_000_000: r"encoder, $n=10^6$"}
    for n in (1_000, 1_000_000):
        rows = sorted(
            (row for row in encoder_rows if int(row["n"]) == n),
            key=lambda row: float(row["alpha"]),
        )
        ax.scatter(
            [float(row["alpha"]) for row in rows],
            [float(row["finite_block_rate"]) for row in rows],
            s=18,
            label=labels[n],
            zorder=4,
            **styles[n],
        )

    ax.text(
        0.73,
        1.77,
        "exact capacity",
        ha="center",
        va="center",
        color=BLUE,
        fontsize=6.4,
    )
    ax.set_xlim(0.49, 1.005)
    ax.set_ylim(1.36, 2.025)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([1.4, 1.6, 1.8, 2.0])
    ax.set_xlabel(r"linear threshold ratio, $\alpha$")
    ax.set_ylabel("rate (bits/nt)")
    ax.legend(loc="lower right", borderaxespad=0.2, handletextpad=0.4)
    quiet_grid(ax)
    panel_label(ax, "a")


def draw_panel_b(ax: mpl.axes.Axes, paired_rows: list[dict[str, str]]) -> None:
    rows = [row for row in paired_rows if int(row["r"]) >= 20]
    r_values = np.array([int(row["r"]) for row in rows])
    rates = np.array([float(row["log2_F_over_r"]) for row in rows])
    limit = float(rows[0]["log2_lambda"])
    terminal_excess = float(rows[-1]["normalized_excess"])
    maximum_r = int(r_values[-1])

    ax.plot(r_values, rates, color=BLUE, marker="o", markersize=2.4, markeredgewidth=0)
    ax.axhline(limit, color=INK, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.text(
        1_450,
        limit + 0.005,
        r"certified $\log_2\Lambda_2$",
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=INK,
    )
    ax.annotate(
        f"excess = {terminal_excess:.4f}",
        xy=(r_values[-1], rates[-1]),
        xytext=(3_500, 2.91),
        fontsize=6.0,
        color=ORANGE,
        arrowprops=dict(arrowstyle="-", linewidth=0.6, color=ORANGE),
    )
    ax.set_xlim(0, maximum_r * 1.025)
    ax.set_ylim(2.77, 3.56)
    ax.set_xticks([0, 1_000, 2_000, 3_000, 4_000, 5_000])
    ax.set_yticks([2.8, 3.0, 3.2, 3.4])
    ax.set_xlabel(r"endpoint-window length, $r$")
    ax.set_ylabel(r"$r^{-1}\log_2 F_2(r)$")
    quiet_grid(ax)
    panel_label(ax, "b")


def draw_panel_c(ax: mpl.axes.Axes, broad_rows: list[dict[str, str]]) -> None:
    alphas = (0.5, 0.6, 0.8, 0.95)
    colors = (BLUE, BLUE_MID, ORANGE_LIGHT, ORANGE)
    for alpha, color in zip(alphas, colors):
        rows = sorted(
            (row for row in broad_rows if math.isclose(float(row["alpha"]), alpha)),
            key=lambda row: int(row["m"]),
        )
        m_values = np.array([int(row["m"]) for row in rows])
        bounds = np.array([float(row["capacity_lower_bits_per_nt"]) for row in rows])
        ax.plot(
            m_values,
            bounds,
            color=color,
            marker="o",
            markersize=2.5,
            markeredgewidth=0,
            label=rf"$\alpha={alpha:g}$",
        )
        limit = 2 * (2 * alpha - 1) + (1 - alpha) * math.log2(12)
        ax.hlines(limit, 2, 1024, color=color, linewidth=0.55, linestyle=(0, (2, 2)), alpha=0.7)

    ax.set_xscale("log", base=2)
    ax.set_xlim(1.8, 1250)
    ax.set_ylim(1.36, 2.01)
    ax.set_xticks([2, 4, 8, 16, 64, 256, 1024])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks([1.4, 1.6, 1.8, 2.0])
    ax.set_xlabel(r"stem threshold, $m$")
    ax.set_ylabel("certified rate lower bound (bits/nt)")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6)
    quiet_grid(ax)
    panel_label(ax, "c")


def draw_panel_d(ax: mpl.axes.Axes, finite_rows: list[dict[str, str]]) -> None:
    alphas = (0.5, 0.7, 0.9, 0.95)
    colors = (BLUE, BLUE_MID, ORANGE_LIGHT, ORANGE)
    for alpha, color in zip(alphas, colors):
        current = sorted(
            (
                row
                for row in finite_rows
                if int(row["n"]) == 10_000 and math.isclose(float(row["alpha"]), alpha)
            ),
            key=lambda row: int(row["m"]),
        )
        ax.plot(
            [int(row["m"]) for row in current],
            [float(row["finite_rate"]) for row in current],
            color=color,
            marker="o",
            markersize=2.8,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0,
            label=rf"$\alpha={alpha:g}$",
        )
        ax.plot(
            [int(row["m"]) for row in current],
            [float(row["asymptotic_lower_rate"]) for row in current],
            color=color,
            linestyle=(0, (2, 2)),
            linewidth=0.7,
            alpha=0.8,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlim(1.8, 150)
    ax.set_ylim(1.36, 2.01)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_yticks([1.4, 1.6, 1.8, 2.0])
    ax.set_xlabel(r"stem threshold, $m$")
    ax.set_ylabel("exact finite structured rate (bits/nt)")
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.5)
    quiet_grid(ax)
    panel_label(ax, "d")


def main() -> None:
    for name, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing required input {name}: {path}")

    setup_style()
    FIGURES.mkdir(exist_ok=True)
    SOURCE_DATA.mkdir(exist_ok=True)

    encoder_rows = read_csv(INPUTS["encoder"])
    paired_rows = read_csv(INPUTS["paired"])
    broad_rows = read_csv(INPUTS["broad_m"])
    finite_rows = read_csv(INPUTS["finite_m"])

    exact_m_values = sorted({int(row["m"]) for row in finite_rows})
    exact_alpha_values = sorted({float(row["alpha"]) for row in finite_rows})
    exact_lengths = sorted({int(row["n"]) for row in finite_rows})
    if not exact_m_values or exact_m_values[-1] < 128:
        raise AssertionError("finite-m source grid has not been expanded through m=128")
    if not exact_alpha_values or exact_alpha_values[-1] < 0.99:
        raise AssertionError("finite-m source grid has not been expanded through alpha=0.99")
    if max(int(row["m"]) for row in broad_rows) < 1024:
        raise AssertionError("broad-m source grid has not been expanded through m=1024")
    if max(int(row["n"]) for row in encoder_rows) < 1_000_000:
        raise AssertionError("encoder source grid does not reach n=10^6")
    if max(int(row["r"]) for row in paired_rows) < 5000:
        raise AssertionError("paired-language source grid does not reach r=5000")

    fig = plt.figure(
        figsize=(WIDTH_MM * INCH_PER_MM, HEIGHT_MM * INCH_PER_MM),
        layout="constrained",
    )
    grid = fig.add_gridspec(2, 2, width_ratios=(1.08, 0.92), hspace=0.12, wspace=0.08)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    draw_panel_a(axes[0], encoder_rows)
    draw_panel_b(axes[1], paired_rows)
    draw_panel_c(axes[2], broad_rows)
    draw_panel_d(axes[3], finite_rows)

    fig.savefig(STEM.with_suffix(".svg"))
    fig.savefig(STEM.with_suffix(".pdf"))
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600)
    fig.savefig(STEM.with_suffix(".png"), dpi=300)
    plt.close(fig)

    manifest = {
        "status": "GENERATED_FROM_FROZEN_AUDIT_TABLES",
        "backend": "Python/matplotlib",
        "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
        "exports": {
            suffix: sha256(STEM.with_suffix(suffix)) for suffix in (".svg", ".pdf", ".tiff", ".png")
        },
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()},
        "checks": {
            "m_asymptotic_max": max(int(row["m"]) for row in broad_rows),
            "m_exact_finite_max": max(exact_m_values),
            "alpha_exact_finite_values": exact_alpha_values,
            "n_exact_finite_values": exact_lengths,
            "exact_finite_cells": len(finite_rows),
            "n_encoder_max": max(int(row["n"]) for row in encoder_rows),
            "r_paired_max": max(int(row["r"]) for row in paired_rows),
            "panel_a_alpha_domain": [0.5, 1.0],
            "open_interval_plotted": False,
            "m_gt_2_global_optimality_claimed": False,
            "smoothing_or_manual_deletion": False,
        },
    }
    (SOURCE_DATA / "fig_scaling_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
