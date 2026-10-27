#!/usr/bin/env python3
"""Plot complete computational comparisons with bounded interpretation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--qa-scripts", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    names = [
        "physical_primary_matched",
        "physical_deployment_file_values",
        "physical_sensitivity_common_pairs",
        "physical_high_alpha",
    ]
    paths = [a.tables / f"{n}.json" for n in names]
    primary, files, sensitivity, high = [json.loads(p.read_text()) for p in paths]
    assert len(primary) == 24 and all(r["N_pairs"] == 1152 for r in primary)
    blue, teal, orange, gray = "#2C6088", "#39888D", "#AC653D", "#777777"
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.3,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.05,
            "lines.markersize": 3.6,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.10, 5.20))
    fig.subplots_adjust(
        left=0.105, right=0.98, bottom=0.18, top=0.94, wspace=0.38, hspace=0.98
    )
    plotted = {}
    axis = axes[0, 0]
    for m, color, marker, offset in ((6, orange, "s", -0.12), (7, blue, "o", 0.12)):
        rows = sorted(
            [r for r in primary if r["m"] == m], key=lambda r: (r["n"], r["k"])
        )
        y = np.array([r["difference_pp"] for r in rows])
        low = np.array([r["CI_pp"][0] for r in rows])
        hi = np.array([r["CI_pp"][1] for r in rows])
        axis.errorbar(
            np.arange(12) + offset,
            y,
            yerr=[y - low, hi - y],
            fmt=marker,
            color=color,
            ms=3.2,
            capsize=1.3,
            elinewidth=0.65,
            label=f"m = {m}",
        )
        plotted[f"a_m{m}"] = rows
    for x in (3.5, 7.5):
        axis.axvline(x, color="#DDDDDD", lw=0.55)
    for x, n in ((1.5, 100), (5.5, 150), (9.5, 200)):
        axis.text(x, 46, f"n = {n}", ha="center", va="center", fontsize=7.4)
    axis.axhline(0, color="#333333", lw=0.7, ls="--")
    axis.set(
        xlim=(-0.55, 11.55),
        ylim=(-1, 49),
        xticks=np.arange(12),
        xticklabels=["0", "3", "4", "5"] * 3,
        xlabel="Short separation cutoff, k",
        ylabel="Predicted incidence difference (pp)",
        yticks=[0, 10, 20, 30, 40],
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, 1.20),
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )

    axis = axes[0, 1]
    methods = ["short_core", "nonce", "gc_h_nonce", "two_bit"]
    colors = [blue, orange, teal, gray]
    selected = []
    for i, (method, color) in enumerate(zip(methods, colors), 1):
        rows = sorted(
            [
                r
                for r in files
                if r["n"] == 150
                and r["m"] == 7
                and r["k"] == 3
                and r["model"] == "T37_S1.021_default"
                and r["method"] == method
            ],
            key=lambda r: (r["seed"], r["file_id"]),
        )
        assert len(rows) == 24 and all(r["strands_folded"] == 63 for r in rows)
        selected.extend(rows)
        values = np.array([100 * r["stem_fraction"] for r in rows])
        axis.boxplot(
            [values],
            positions=[i],
            widths=0.47,
            whis=(0, 100),
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 0.85},
            medianprops={"color": color, "linewidth": 1.1},
            whiskerprops={"color": color, "linewidth": 0.7},
            capprops={"color": color, "linewidth": 0.7},
        )
        axis.scatter(
            i + np.linspace(-0.13, 0.13, len(rows)),
            values,
            s=7,
            color=color,
            alpha=0.45,
            edgecolors="none",
            zorder=3,
        )
    plotted["b_all_file_means"] = selected
    axis.set(
        xlim=(0.5, 4.5),
        ylim=(-2, 70),
        xticks=range(1, 5),
        xticklabels=["Core", "RC nonce", "GC/H", "Two-bit"],
        ylabel="Predicted stem incidence / file (%)",
        xlabel="Complete DNA generator",
    )

    axis = axes[1, 0]
    settings = [
        ("T25_S1.021", "25°C\n1.021 M"),
        ("T37_S1.021", "37°C\n1.021 M"),
        ("T50_S1.021", "50°C\n1.021 M"),
        ("T37_S0.05", "37°C\n0.05 M"),
        ("T37_S0.15", "37°C\n0.15 M"),
    ]
    for kind, color, marker in (("default", orange, "s"), ("WC", blue, "o")):
        rows = []
        for setting, _ in settings:
            found = [
                r
                for r in sensitivity
                if r["n"] == 150
                and r["m"] == 7
                and r["k"] == 3
                and r["model"] == setting + "_" + kind
            ]
            assert (
                len(found) == 1
                and found[0]["N_pairs"] == 96
                and found[0]["N_ensemble_paired_fraction_pairs"] == 96
            )
            rows.append(found[0])
        axis.plot(
            range(5),
            [
                100 * r["mean_control_minus_target_ensemble_paired_fraction"]
                for r in rows
            ],
            marker=marker,
            color=color,
            label="Default pairs" if kind == "default" else "WC only",
        )
        plotted[f"c_{kind}"] = rows
    axis.axhline(0, color="#333333", lw=0.7, ls="--")
    axis.set(
        xlim=(-0.25, 4.25),
        xticks=range(5),
        xticklabels=[s[1] for s in settings],
        xlabel="Effective temperature / salt setting",
        ylabel="Mean paired-fraction difference (pp)",
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.23),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    axis = axes[1, 1]
    for m, color in ((6, orange), (7, blue)):
        for alpha, ls, marker in ((0.5, "-", "o"), (0.75, "--", "s")):
            rows = sorted(
                [
                    r
                    for r in high
                    if r["m"] == m
                    and r["alpha"] == alpha
                    and r["model"] == "T37_S1.021_WC"
                ],
                key=lambda r: r["n"],
            )
            assert len(rows) == 3 and all(
                r["N_encoded_words"] == 384 and r["WC_m_subruns_at_or_above_k"] == 0
                for r in rows
            )
            y = np.array([100 * r["has_WC_m_subrun_fraction"] for r in rows])
            lo = np.array([100 * r["has_WC_m_subrun_CI95"][0] for r in rows])
            hi = np.array([100 * r["has_WC_m_subrun_CI95"][1] for r in rows])
            axis.errorbar(
                [r["n"] for r in rows],
                y,
                yerr=[y - lo, hi - y],
                color=color,
                ls=ls,
                marker=marker,
                capsize=1.5,
                elinewidth=0.65,
                label=f"m={m}, α={alpha:g}",
            )
            plotted[f"d_m{m}_a{alpha}"] = rows
    axis.set(
        xlim=(90, 210),
        xticks=[100, 150, 200],
        xlabel="Block length, n (nt)",
        ylabel="WC stem incidence, length ≥ m (%)",
        ylim=(0, 80),
    )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.25),
        ncol=2,
        columnspacing=0.65,
        handletextpad=0.35,
        fontsize=7.2,
    )
    for axis, title in zip(
        axes.flat,
        [
            "(a) Dinucleotide / endpoint / H matching",
            "(b) Encoded files, n=150, m=7, k=3",
            "(c) Ensemble sensitivity, common 96 pairs",
            "(d) High-α words still admit short-gap stems",
        ],
    ):
        axis.grid(axis="y", color="#DFDFDF", lw=0.45)
        axis.set_axisbelow(True)
        axis.text(
            0.5,
            -0.49,
            title,
            ha="center",
            va="top",
            transform=axis.transAxes,
            fontsize=8,
        )
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=a.out / "panel_alignment.json",
        overlay_svg=a.out / "panel_alignment.svg",
        panel_ids=["a", "b", "c", "d"],
        strict=True,
    )
    for extension in ("pdf", "svg", "png"):
        fig.savefig(a.out / f"physical_results.{extension}", dpi=300)
    (a.out / "plotted_data.json").write_text(
        json.dumps(plotted, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "size_in": [7.10, 5.20],
        "predictions_not_causal_or_FER": True,
    }
    (a.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
