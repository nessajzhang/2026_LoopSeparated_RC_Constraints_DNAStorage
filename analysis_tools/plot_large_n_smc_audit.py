#!/usr/bin/env python3
"""Publication-style visualization for large-n SMC capacity audits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_mid": "#3775BA",
    "blue_soft": "#9DB9D8",
    "teal": "#42949E",
    "violet": "#7B5EA7",
    "rose": "#B85C70",
    "gold": "#C8A85A",
    "neutral_dark": "#3F3F3F",
    "neutral_mid": "#777777",
    "neutral_light": "#D9D9D9",
    "green": "#2E9E44",
    "red": "#B64342",
}

CLASSICAL_C2 = 1.167987028946


FIXED_K_COLORS = {
    0: PALETTE["blue_main"],
    1: PALETTE["blue_mid"],
    2: PALETTE["teal"],
    5: PALETTE["gold"],
    10: PALETTE["violet"],
    20: PALETTE["rose"],
}


REGIME_COLORS = {
    "fixed": PALETTE["blue_main"],
    "sqrt": PALETTE["teal"],
    "n067": PALETTE["violet"],
    "n2over3": PALETTE["violet"],
    "n075": PALETTE["gold"],
    "nlog": PALETTE["rose"],
    "n_over_log": PALETTE["rose"],
    "n09": PALETTE["neutral_dark"],
    "linear0.01": PALETTE["red"],
    "linear0.05": PALETTE["rose"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_summaries(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sources: dict[tuple[int, int, int], Path] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"summary input does not exist: {path}")
        for row in read_csv(path):
            key = (int(row["m"]), int(row["k"]), int(row["n"]))
            if key in sources:
                raise ValueError(
                    "duplicate (m,k,n) summary key "
                    f"{key} in {sources[key]} and {path}; curate one authoritative row before plotting"
                )
            sources[key] = path
            rows.append(row)
    rows.sort(key=lambda r: (int(r["m"]), int(r["k"]), int(r["n"])))
    return rows


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.75,
            "legend.fontsize": 6.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.dpi": 220,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8.6,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def plot_fixed_k_panel(ax: plt.Axes, rows: list[dict[str, str]], m: int, capacity: float) -> None:
    subset = [r for r in rows if int(r["m"]) == m]
    for k in sorted({int(r["k"]) for r in subset}):
        data = sorted([r for r in subset if int(r["k"]) == k], key=lambda r: int(r["n"]))
        ns = np.array([int(r["n"]) for r in data], dtype=float)
        rates = np.array([as_float(r, "mean_rate_est_bits_per_nt") for r in data], dtype=float)
        sem = np.array([as_float(r, "sem_rate") for r in data], dtype=float)
        color = FIXED_K_COLORS.get(k, PALETTE["neutral_dark"])
        ax.plot(ns, rates, color=color, lw=1.25, marker="o", ms=2.6, label=rf"$k={k}$")
        ax.fill_between(ns, rates - 1.96 * sem, rates + 1.96 * sem, color=color, alpha=0.12, lw=0)

    ax.axhline(capacity, color=PALETTE["neutral_dark"], lw=0.9, ls=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_xlabel("Block length $n$")
    ax.set_ylabel("SMC rate estimate [bits/nt]")
    ax.set_title(rf"Fixed-$k$ convergence, $m={m}$")
    ax.grid(axis="y", alpha=0.18, lw=0.45)
    ax.text(
        0.97,
        0.06,
        rf"$c_{m}\approx {capacity:.6f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=PALETTE["neutral_dark"],
    )


def plot_excess_panel(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    targets = [1000, 10000]
    ks = sorted({int(r["k"]) for r in rows if int(r["m"]) == 2})
    width = 0.36
    x = np.arange(len(ks))
    for idx, n in enumerate(targets):
        vals = []
        sems = []
        for k in ks:
            matches = [
                r for r in rows if int(r["m"]) == 2 and int(r["k"]) == k and int(r["n"]) == n
            ]
            if matches:
                vals.append(as_float(matches[0], "excess_over_capacity"))
                sems.append(as_float(matches[0], "sem_rate"))
            else:
                vals.append(np.nan)
                sems.append(0.0)
        color = PALETTE["blue_mid"] if n == 1000 else PALETTE["rose"]
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            yerr=sems,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            capsize=2.2,
            label=rf"$n={n}$",
        )
    ax.axhline(0, color=PALETTE["neutral_dark"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Loop threshold $k$")
    ax.set_ylabel(r"Excess over $c_2$ [bits/nt]")
    ax.set_title("Residual finite-block excess")
    ax.grid(axis="y", alpha=0.18, lw=0.45)


def plot_aggregate_decay_panel(
    ax: plt.Axes, rows: list[dict[str, str]], capacity: float = CLASSICAL_C2
) -> None:
    target_ks = {0, 1, 2, 5, 10, 20}
    subset = [r for r in rows if int(r["m"]) == 2 and int(r["k"]) in target_ks]
    ns = []
    for n in sorted({int(r["n"]) for r in subset}):
        if {int(r["k"]) for r in subset if int(r["n"]) == n} == target_ks:
            ns.append(n)
    mean_excess = []
    spread = []
    sem_envelope = []
    for n in ns:
        data = [r for r in subset if int(r["n"]) == n]
        rates = np.array([as_float(r, "mean_rate_est_bits_per_nt") for r in data], dtype=float)
        sems = np.array([as_float(r, "sem_rate") for r in data], dtype=float)
        mean_excess.append(float(np.mean(rates - capacity)))
        spread.append(float(np.max(rates) - np.min(rates)))
        sem_envelope.append(float(np.sqrt(np.sum(sems**2)) / len(sems)))
    ns_arr = np.array(ns, dtype=float)
    mean_excess_arr = np.array(mean_excess)
    spread_arr = np.array(spread)
    sem_arr = np.array(sem_envelope)
    ax.plot(
        ns_arr,
        mean_excess_arr,
        color=PALETTE["blue_main"],
        lw=1.35,
        marker="o",
        ms=2.6,
        label=r"mean excess over $c_2$",
    )
    ax.fill_between(
        ns_arr,
        mean_excess_arr - 1.96 * sem_arr,
        mean_excess_arr + 1.96 * sem_arr,
        color=PALETTE["blue_main"],
        alpha=0.12,
        lw=0,
    )
    ax.plot(
        ns_arr,
        spread_arr,
        color=PALETTE["rose"],
        lw=1.25,
        marker="s",
        ms=2.4,
        label=r"between-$k$ spread",
    )
    ax.axhline(0, color=PALETTE["neutral_dark"], lw=0.8, ls=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block length $n$")
    ax.set_ylabel("Rate difference [bits/nt]")
    ax.set_title(r"Decay of finite-block elevation, $m=2$")
    ax.grid(axis="y", alpha=0.18, lw=0.45)


def plot_excess_heatmap_panel(
    ax: plt.Axes, rows: list[dict[str, str]], capacity: float = CLASSICAL_C2
) -> None:
    subset = [r for r in rows if int(r["m"]) == 2 and int(r["k"]) in {0, 1, 2, 5, 10, 20}]
    ks = sorted({int(r["k"]) for r in subset})
    ns = sorted({int(r["n"]) for r in subset})
    grid = np.full((len(ks), len(ns)), np.nan, dtype=float)
    lookup = {
        (int(r["k"]), int(r["n"])): max(as_float(r, "excess_over_capacity"), 1e-5) for r in subset
    }
    for i, k in enumerate(ks):
        for j, n in enumerate(ns):
            if (k, n) in lookup:
                grid[i, j] = lookup[(k, n)]

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        norm=LogNorm(vmin=1e-4, vmax=max(0.5, np.nanmax(grid))),
    )
    ax.set_xticks(np.arange(len(ns)))
    ax.set_xticklabels(
        [
            rf"$10^{{{int(np.log10(n))}}}$" if n in {100, 1000, 10000, 100000} else str(n)
            for n in ns
        ],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("Block length $n$")
    ax.set_ylabel("Loop threshold $k$")
    ax.set_title(r"Excess-rate map above $c_2$")
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.ax.tick_params(labelsize=6.2, length=2)
    cbar.set_label("bits/nt", fontsize=6.6)


def plot_terminal_excess_panel(ax: plt.Axes, rows: list[dict[str, str]], n: int = 100000) -> None:
    subset = sorted(
        [
            r
            for r in rows
            if int(r["m"]) == 2 and int(r["n"]) == n and int(r["k"]) in {0, 1, 2, 5, 10, 20}
        ],
        key=lambda r: int(r["k"]),
    )
    ks = [int(r["k"]) for r in subset]
    vals = np.array([1000.0 * as_float(r, "excess_over_capacity") for r in subset])
    sems = np.array([1000.0 * as_float(r, "sem_rate") for r in subset])
    colors = [FIXED_K_COLORS.get(k, PALETTE["neutral_dark"]) for k in ks]
    x = np.arange(len(ks))
    ax.bar(x, vals, yerr=1.96 * sems, color=colors, edgecolor="white", linewidth=0.4, capsize=2.4)
    ax.axhline(0, color=PALETTE["neutral_dark"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Loop threshold $k$")
    ax.set_ylabel(r"Excess at $n=10^5$ [milli-bits/nt]")
    ax.set_title(r"Fixed-$k$ rates nearly coalesce at $10^5$")
    ax.grid(axis="y", alpha=0.18, lw=0.45)


def plot_regime_panel(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    if not rows:
        ax.text(0.5, 0.5, "Regime grid pending", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    labels = {
        "fixed": r"fixed $k=20$",
        "sqrt": r"$k_n=\lfloor\sqrt{n}\rfloor$",
        "n067": r"$k_n=\lfloor n^{2/3}\rfloor$",
        "n2over3": r"$k_n=\lfloor n^{2/3}\rfloor$",
        "n075": r"$k_n=\lfloor n^{0.75}\rfloor$",
        "nlog": r"$k_n=\lfloor n/\ln n\rfloor$",
        "n_over_log": r"$k_n=\lfloor n/\ln n\rfloor$",
        "n09": r"$k_n=\lfloor n^{0.9}\rfloor$",
        "linear0.01": r"$k_n=\lfloor0.01n\rfloor$",
        "linear0.05": r"$k_n=\lfloor0.05n\rfloor$",
    }
    for regime in labels:
        data = sorted([r for r in rows if r["regime"] == regime], key=lambda r: int(r["n"]))
        if not data:
            continue
        ns = np.array([int(r["n"]) for r in data], dtype=float)
        excess = np.array([as_float(r, "excess_over_capacity") for r in data], dtype=float)
        sem = np.array([as_float(r, "sem_rate") for r in data], dtype=float)
        color = REGIME_COLORS.get(regime, PALETTE["neutral_dark"])
        ax.plot(ns, excess, color=color, lw=1.25, marker="o", ms=2.6, label=labels[regime])
        ax.fill_between(ns, excess - 1.96 * sem, excess + 1.96 * sem, color=color, alpha=0.12, lw=0)
    ax.axhline(0, color=PALETTE["neutral_dark"], lw=0.8, ls=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_xlabel("Final block length $n$")
    ax.set_ylabel(r"Excess over $c_2$ [bits/nt]")
    ax.set_title("Threshold-growth regimes")
    ax.grid(axis="y", alpha=0.18, lw=0.45)


def threshold_k(regime: str, n: np.ndarray) -> np.ndarray:
    if regime == "sqrt":
        return np.floor(np.sqrt(n))
    if regime in {"n067", "n2over3"}:
        return np.floor(n ** (2.0 / 3.0))
    if regime == "n075":
        return np.floor(n**0.75)
    if regime in {"nlog", "n_over_log"}:
        return np.floor(n / np.log(np.maximum(n, 3.0)))
    if regime == "n09":
        return np.floor(n**0.90)
    if regime.startswith("linear"):
        alpha = float(regime.replace("linear", ""))
        return np.floor(alpha * n)
    raise ValueError(f"unknown threshold regime: {regime}")


def plot_threshold_scale_panel(ax: plt.Axes) -> None:
    regimes = ["sqrt", "n067", "n075", "nlog", "n09", "linear0.01"]
    labels = {
        "sqrt": r"$\lfloor\sqrt{n}\rfloor$",
        "n067": r"$\lfloor n^{2/3}\rfloor$",
        "n075": r"$\lfloor n^{0.75}\rfloor$",
        "nlog": r"$\lfloor n/\ln n\rfloor$",
        "n09": r"$\lfloor n^{0.9}\rfloor$",
        "linear0.01": r"$\lfloor0.01n\rfloor$",
    }
    ns = np.logspace(2, 12, 520)
    q = 4
    m = 2
    for regime in regimes:
        k_vals = threshold_k(regime, ns)
        budget_ratio = (2.0 * (m + k_vals) - 1.0) * (q**m) / ns
        color = REGIME_COLORS.get(regime, PALETTE["neutral_dark"])
        style = "--" if regime.startswith("linear") else "-"
        ax.plot(ns, budget_ratio, color=color, lw=1.25, ls=style, label=labels[regime])

    ax.axhline(0.5, color=PALETTE["neutral_light"], lw=0.75, ls=(0, (3, 2)))
    ax.axhline(0.1, color=PALETTE["neutral_light"], lw=0.75, ls=(0, (3, 2)))
    ax.axhline(0.01, color=PALETTE["neutral_mid"], lw=0.75, ls=(0, (3, 2)))
    ax.axvline(1e5, color=PALETTE["neutral_light"], lw=0.75, ls=(0, (3, 2)))
    ax.axvline(1e6, color=PALETTE["neutral_light"], lw=0.75, ls=(0, (3, 2)))
    ax.text(
        1.05e6,
        18.0,
        r"$n=10^6$",
        rotation=90,
        va="top",
        ha="left",
        fontsize=6.4,
        color=PALETTE["neutral_mid"],
    )
    ax.text(
        1.2e2,
        0.55,
        r"$B_n/n=0.5$",
        va="bottom",
        ha="left",
        fontsize=6.2,
        color=PALETTE["neutral_mid"],
    )
    ax.text(
        1.2e2,
        0.011,
        r"$0.01$",
        va="bottom",
        ha="left",
        fontsize=6.2,
        color=PALETTE["neutral_mid"],
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 20.0)
    ax.set_xlabel("Block length $n$")
    ax.set_ylabel(r"Proof budget ratio $B_n/n$")
    ax.set_title(r"Analytic proof-budget scale (no SMC)")
    ax.grid(axis="y", alpha=0.18, lw=0.45)


def plot_calibration_panel(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    if not rows:
        ax.text(
            0.5, 0.5, "Calibration data pending", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
        return
    subset = [r for r in rows if int(r["n"]) <= 40]
    for k in sorted({int(r["k"]) for r in subset}):
        data = sorted([r for r in subset if int(r["k"]) == k], key=lambda r: int(r["n"]))
        ns = np.array([int(r["n"]) for r in data], dtype=float)
        delta = np.array([float(r["smc_minus_exact"]) for r in data], dtype=float)
        sd = np.array([float(r["sd_rate"]) for r in data], dtype=float)
        color = FIXED_K_COLORS.get(k, PALETTE["neutral_dark"])
        ax.plot(ns, delta, color=color, lw=1.15, marker="o", ms=2.4, label=rf"$k={k}$")
        ax.fill_between(ns, delta - sd, delta + sd, color=color, alpha=0.10, lw=0)
    ax.axhline(0, color=PALETTE["neutral_dark"], lw=0.8, ls=(0, (4, 2)))
    ax.set_xlabel("Exact-enumeration length $n$")
    ax.set_ylabel("SMC $-$ exact [bits/nt]")
    ax.set_title("SMC calibration against exact counts")
    ax.grid(axis="y", alpha=0.18, lw=0.45)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--m2", nargs="+", type=Path, required=True)
    ap.add_argument(
        "--m3",
        nargs="*",
        type=Path,
        default=[],
        help="Optional m=3 summaries retained for backward-compatible calls.",
    )
    ap.add_argument("--regime", type=Path, default=None)
    ap.add_argument("--validation", nargs="*", type=Path, default=[])
    ap.add_argument(
        "--threshold-scale",
        action="store_true",
        help="Use panel d for k_n/n scaling curves instead of SMC regime-rate estimates.",
    )
    ap.add_argument(
        "--six-panel",
        action="store_true",
        help="Create an expanded six-panel audit figure with heatmap, terminal excess, calibration, and threshold-ratio panels.",
    )
    ap.add_argument(
        "--tiff",
        action="store_true",
        help="Also write the optional high-resolution TIFF export (about 70 MB).",
    )
    args = ap.parse_args()

    apply_style()
    m2_rows = merge_summaries(args.m2)
    regime_rows = read_csv(args.regime) if args.regime and args.regime.exists() else []
    validation_rows = merge_summaries(args.validation) if args.validation else []

    if args.six_panel:
        fig = plt.figure(figsize=(7.25, 7.1))
        gs = fig.add_gridspec(
            3, 2, left=0.080, right=0.985, bottom=0.085, top=0.925, hspace=0.58, wspace=0.32
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])
        ax_e = fig.add_subplot(gs[2, 0])
        ax_f = fig.add_subplot(gs[2, 1])

        plot_fixed_k_panel(ax_a, m2_rows, m=2, capacity=CLASSICAL_C2)
        plot_aggregate_decay_panel(ax_b, m2_rows, capacity=CLASSICAL_C2)
        plot_excess_heatmap_panel(ax_c, m2_rows, capacity=CLASSICAL_C2)
        plot_terminal_excess_panel(ax_d, m2_rows, n=100000)
        plot_calibration_panel(ax_e, validation_rows)
        plot_threshold_scale_panel(ax_f)

        for ax, label in zip((ax_a, ax_b, ax_c, ax_d, ax_e, ax_f), "abcdef"):
            add_panel_label(ax, label)
        ax_a.legend(ncol=3, loc="upper right", handlelength=1.4, columnspacing=0.8, fontsize=6.0)
        ax_b.legend(ncol=1, loc="upper right", handlelength=1.4, columnspacing=0.8)
        ax_e.legend(ncol=2, loc="upper right", handlelength=1.2, fontsize=6.1)
        ax_f.legend(loc="upper right", fontsize=5.7, handlelength=1.15)
    else:
        fig = plt.figure(figsize=(7.25, 5.6))
        gs = fig.add_gridspec(
            2, 2, left=0.085, right=0.985, bottom=0.12, top=0.90, hspace=0.50, wspace=0.30
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        plot_fixed_k_panel(ax_a, m2_rows, m=2, capacity=CLASSICAL_C2)
        plot_aggregate_decay_panel(ax_b, m2_rows, capacity=CLASSICAL_C2)
        plot_excess_panel(ax_c, m2_rows)
        if args.threshold_scale:
            plot_threshold_scale_panel(ax_d)
        elif regime_rows:
            plot_regime_panel(ax_d, regime_rows)
        else:
            plot_calibration_panel(ax_d, validation_rows)

        add_panel_label(ax_a, "a")
        add_panel_label(ax_b, "b")
        add_panel_label(ax_c, "c")
        add_panel_label(ax_d, "d")
        ax_a.legend(ncol=3, loc="upper right", handlelength=1.4, columnspacing=0.8)
        ax_b.legend(ncol=2, loc="upper right", handlelength=1.4, columnspacing=0.8)
        ax_c.legend(loc="upper right", handlelength=1.2)
        ax_d.legend(loc="upper right", fontsize=6.1, handlelength=1.3)

    fig.suptitle(
        "Large-block numerical audit of loop-separated capacity stability",
        y=0.982,
        fontsize=8.6,
        fontweight="bold",
    )

    out = args.out_prefix
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    print(out.with_suffix(".pdf"))
    print(out.with_suffix(".svg"))
    print(out.with_suffix(".png"))
    if args.tiff:
        fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
        print(out.with_suffix(".tiff"))


if __name__ == "__main__":
    main()
