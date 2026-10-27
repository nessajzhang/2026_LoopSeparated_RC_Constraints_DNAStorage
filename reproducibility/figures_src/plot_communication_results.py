#!/usr/bin/env python3
"""Plot measured file errors/costs and all preregistered null contrasts."""

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

    paths = [
        a.tables / f"{n}.json"
        for n in ("ids_main_cells", "ids_secondary_cells", "ids_main_paired")
    ]
    main_rows, secondary, pairs = [json.loads(p.read_text()) for p in paths]
    assert sum(r["N_files"] for r in main_rows) == 72000
    assert sum(r["N_files"] for r in secondary) == 32640
    styles = {
        "short_core": ("Guaranteed core", "#2C6088", "o", "-"),
        "nonce": ("RC nonce", "#AC653D", "s", "--"),
        "gc_h_nonce": ("GC/H nonce", "#39888D", "^", "-."),
        "two_bit": ("Two-bit", "#777777", "D", ":"),
    }
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.05,
            "lines.markersize": 3.8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.10, 4.95))
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.17, top=0.89, wspace=0.35, hspace=0.86
    )
    plotted = {}

    def show(axis, rows, xkey, method, tag):
        rows = sorted(rows, key=lambda r: r[xkey])
        assert rows
        name, color, marker, ls = styles[method]
        y = np.array([r["FER"] for r in rows])
        low = np.array([r["FER_CI_low"] for r in rows])
        high = np.array([r["FER_CI_high"] for r in rows])
        axis.errorbar(
            [r[xkey] for r in rows],
            y,
            yerr=[y - low, high - y],
            label=name,
            color=color,
            marker=marker,
            ls=ls,
            markerfacecolor="white",
            markeredgewidth=0.9,
            capsize=1.8,
            elinewidth=0.65,
        )
        plotted[f"{tag}_{method}"] = [
            {
                k: r[k]
                for k in (
                    "run",
                    "cell_id",
                    "method",
                    "N_files",
                    "file_failures",
                    xkey,
                    "FER",
                    "FER_CI_low",
                    "FER_CI_high",
                )
            }
            for r in rows
        ]

    for method in styles:
        show(
            axes[0, 0],
            [
                r
                for r in main_rows
                if r["method"] == method
                and r["channel"] == "mixed"
                and r["coverage"] == 1
                and r["error_scale"] > 0
            ],
            "error_scale",
            method,
            "a",
        )
        show(
            axes[0, 1],
            [
                r
                for r in main_rows
                if r["method"] == method
                and r["channel"] == "mixed"
                and r["error_scale"] == 0.01
            ],
            "coverage",
            method,
            "b",
        )
        show(
            axes[1, 0],
            [
                r
                for r in main_rows + secondary
                if r["method"] == method
                and r["channel"] == "mixed"
                and r["error_scale"] == 0.005
                and r["coverage"] == 1
                and (r["run"] == "main" or r["run"].startswith("ids_outer_"))
            ],
            "R_net_planned",
            method,
            "c",
        )
    axes[0, 0].set(
        xscale="log",
        xlim=(0.000075, 0.0135),
        xticks=[0.0001, 0.001, 0.005, 0.01],
        xlabel="Diagnostic error scale, e",
        ylabel="File error rate",
    )
    # Decimal tick labels retain the same log coordinates without tiny
    # mathtext superscripts at the final double-column print size.
    axes[0, 0].set_xticklabels(["0.0001", "0.001", "0.005", "0.01"])
    axes[0, 1].set(
        xlim=(0.6, 10.4),
        xticks=[1, 3, 5, 10],
        xlabel="Fixed reads per sent strand",
        ylabel="File error rate",
    )
    axes[1, 0].set(
        xlim=(0.84, 1.195),
        xticks=[0.87, 1.02, 1.09, 1.17],
        xlabel="Planned useful rate (bits/nt)",
        ylabel="File error rate",
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set(ylim=(-0.035, 1.035), yticks=[0, 0.25, 0.5, 0.75, 1])
        axis.grid(axis="y", color="#DFDFDF", linewidth=0.45)
        axis.set_axisbelow(True)
    channels = ["substitution", "indel", "mixed"]
    pairs = [r for r in pairs if r["primary"]]
    pairs.sort(
        key=lambda r: (channels.index(r["channel"]), r["error_scale"], r["coverage"])
    )
    assert len(pairs) == 48 and all(r["CI"][0] <= 0 <= r["CI"][1] for r in pairs)
    axis = axes[1, 1]
    for j, (channel, color) in enumerate(
        zip(channels, ["#2C6088", "#39888D", "#AC653D"])
    ):
        rows = pairs[j * 16 : (j + 1) * 16]
        x = np.arange(j * 16 + 1, (j + 1) * 16 + 1)
        y = np.array([100 * r["difference"] for r in rows])
        low = np.array([100 * r["CI"][0] for r in rows])
        high = np.array([100 * r["CI"][1] for r in rows])
        axis.errorbar(
            x,
            y,
            yerr=[y - low, high - y],
            fmt="o",
            ms=2.1,
            color=color,
            elinewidth=0.55,
            capsize=0,
        )
        if j < 2:
            axis.axvline((j + 1) * 16 + 0.5, color="#DDDDDD", lw=0.5)
    axis.axhline(0, color="#333333", lw=0.75, ls="--", zorder=0)
    axis.set(
        xlim=(0, 49),
        xticks=[8.5, 24.5, 40.5],
        xticklabels=["Substitution", "Indel", "Mixed"],
        xlabel="All 48 prespecified comparisons",
        ylabel="Paired FER difference (pp)",
    )
    axis.grid(axis="y", color="#DFDFDF", linewidth=0.45)
    axis.set_axisbelow(True)
    plotted["d_all48"] = pairs
    for axis, title in zip(
        axes.flat,
        [
            "(a) Mixed channel, one read / strand",
            "(b) Read redundancy at e = 0.01",
            "(c) Outer redundancy at e = 0.005",
            "(d) RC nonce minus GC/H nonce",
        ],
    ):
        axis.text(
            0.5,
            -0.42,
            title,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.53, 0.995),
        columnspacing=1.4,
        handlelength=2,
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
        fig.savefig(a.out / f"communication_results.{extension}", dpi=300)
    (a.out / "plotted_data.json").write_text(
        json.dumps(plotted, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "size_in": [7.10, 4.95],
        "uncertainty": "files;pointwise CP in a-c;predeclared family in d",
        "no_interpolated_or_replaced_zero_observations": True,
    }
    (a.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
