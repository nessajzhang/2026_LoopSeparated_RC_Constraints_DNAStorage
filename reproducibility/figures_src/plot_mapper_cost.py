#!/usr/bin/env python3
"""Render frozen mapper cost data; no simulated or interpolated observations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--qa-scripts", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    a.out.mkdir(parents=True, exist_ok=False)
    with a.input.open(newline="") as h:
        data = list(csv.DictReader(h))
    for r in data:
        for name in ("index", "n", "m", "k", "K", "all_inputs", "all_encode_failures",
                     "retained_count_table_bytes"):
            r[name] = int(r[name])
        r["nonce_budget"] = int(r["nonce_budget"]) if r["nonce_budget"] else None
        for name in ("batch10000_pooled_encode_median_ns", "batch10000_pooled_encode_P95_ns"):
            r[name] = float(r[name])
    assert len(data) == 58 and sum(r["all_inputs"] for r in data) == 3190290
    assert sum(r["all_encode_failures"] for r in data) == 10040
    blue, teal, orange, gray = "#2C6088", "#39888D", "#AC653D", "#777777"
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
        "lines.linewidth": 1.15, "lines.markersize": 3.4, "legend.frameon": False,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
    })
    fig, ax = plt.subplots(2, 2, figsize=(7.10, 4.85))
    fig.subplots_adjust(left=.10, right=.98, bottom=.17, top=.89, wspace=.43, hspace=.92)
    plotted = {}

    def choose(kind, **conditions):
        rows = [r for r in data if r["kind"] == kind and
                all(r[k] == v for k, v in conditions.items())]
        return sorted(rows, key=lambda r: r["n"])

    def timed(axis, rows, label, color, marker):
        assert rows, f"missing plotted group: {label}"
        x = [r["n"] for r in rows]
        med = np.array([r["batch10000_pooled_encode_median_ns"] / 1000 for r in rows])
        hi = np.array([r["batch10000_pooled_encode_P95_ns"] / 1000 for r in rows])
        assert np.all(med > 0) and np.all(hi >= med)
        axis.errorbar(x, med, yerr=[np.zeros_like(med), hi-med], color=color,
                      marker=marker, capsize=2.1, elinewidth=.8, label=label)

    for kind, name, color, marker in (
        ("endpoint_m2", "Specialized m = 2", blue, "o"),
        ("endpoint_m3", "Generic m = 3", teal, "s"),
    ):
        rows = [r for r in choose(kind) if r["k"] == r["n"] // 2]
        timed(ax[0, 0], rows, name, color, marker)
        plotted["a_" + kind] = [r["index"] for r in rows]
    ax[0, 0].set(xlim=(82, 318), xticks=[100,150,200,300], yscale="log",
                 xlabel="Block length, n (nt)", ylabel="Encode latency (µs / word)")
    ax[0, 0].legend(loc="upper left", bbox_to_anchor=(-.02,1.30), ncol=1, borderaxespad=0)

    for kind, name, color, marker in (
        ("short_core", "Guaranteed core", blue, "o"),
        ("nonce", "RC nonce", orange, "s"),
        ("gc_h_nonce", "GC/H nonce", teal, "^"),
        ("two_bit", "Two-bit", gray, "D"),
    ):
        rows = choose(kind, m=7)
        if kind in ("nonce", "gc_h_nonce"):
            rows = [r for r in rows if r["nonce_budget"] == 64]
        timed(ax[0, 1], rows, name, color, marker)
        plotted["b_" + kind] = [r["index"] for r in rows]
    ax[0, 1].set(xlim=(90,210), xticks=[100,150,200], yscale="log",
                 xlabel="Total word length, n (nt)", ylabel="Encode latency (µs / word)")
    ax[0, 1].legend(loc="upper left", bbox_to_anchor=(-.02,1.32), ncol=2,
                    columnspacing=.9, handletextpad=.4, borderaxespad=0)

    rows = sorted(choose("nonce", n=200, m=6), key=lambda r: r["nonce_budget"])
    assert len(rows) == 2
    for r in rows:
        x, n, f = r["nonce_budget"], r["all_inputs"], r["all_encode_failures"]
        y = 100 * f / n
        low = 100 * beta.ppf(.025, f, n-f+1) if f else 0.
        high = 100 * beta.ppf(.975, f+1, n-f) if f < n else 100.
        ax[1, 0].errorbar([x], [y], yerr=[[y-low],[high-y]], fmt="o",
                          color=orange, capsize=3, elinewidth=1.)
        ax[1, 0].text(x, high*1.35, f"{f:,} / {n:,}", ha="center", va="bottom", fontsize=7.5)
    ax[1, 0].set(xlim=(24,296), xticks=[64,256], ylim=(.04,50), yscale="log",
                 xlabel="Candidate budget, B", ylabel="Encoding failures (%)")
    ax[1, 0].text(0,1.12, "n = 200, m = 6", transform=ax[1, 0].transAxes,
                   va="bottom", fontsize=7.5)
    plotted["c_nonce_n200_m6"] = [r["index"] for r in rows]

    for kind, name, color, marker in (
        ("endpoint_m2", "Endpoint m = 2", blue, "o"),
        ("endpoint_m3", "Endpoint m = 3", teal, "s"),
        ("short_core", "Core m = 7", gray, "^"),
    ):
        rows = choose(kind, m=7) if kind == "short_core" else [
            r for r in choose(kind) if r["k"] == r["n"] // 2]
        ax[1, 1].plot([r["n"] for r in rows],
                       [r["retained_count_table_bytes"]/1024 for r in rows],
                       marker=marker, color=color, label=name)
        plotted["d_" + kind] = [r["index"] for r in rows]
    ax[1, 1].set(xlim=(82,318), xticks=[100,150,200,300], yscale="log",
                 xlabel="Block length, n (nt)", ylabel="Count-table objects (KiB)")
    ax[1, 1].legend(loc="upper left", bbox_to_anchor=(-.02,1.32),
                    fontsize=7.2, ncol=2, columnspacing=.8, handletextpad=.4,
                    borderaxespad=0)

    titles = ["(a) Noiseless endpoint mapping", "(b) Short-threshold mapping",
              "(c) Finite-candidate failure boundary", "(d) Suffix-count table storage"]
    for axis, title in zip(ax.flat, titles):
        axis.grid(axis="y", color="#DEDEDE", linewidth=.5)
        axis.set_axisbelow(True)
        axis.tick_params(direction="out", length=2.5)
        axis.text(.5, -.46, title, ha="center", va="top",
                  transform=axis.transAxes, fontsize=8)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=a.out/"alignment.json",
        overlay_svg=a.out/"alignment.svg", panel_ids=["a","b","c","d"],
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(a.out/"mapper_cost.pdf")
    fig.savefig(a.out/"mapper_cost.svg")
    fig.savefig(a.out/"mapper_cost.png", dpi=600)
    fig.savefig(a.out/"mapper_cost.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    (a.out/"source_and_selection.json").write_text(json.dumps({
        "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
        "input_rows":58, "panel_row_indices":plotted,
        "all_rows_retained_in_SI":True, "latency":"P50 point to P95 whisker, not CI",
        "failure_CI":"Pointwise95% Clopper-Pearson, all55005 inputs/cell",
        "labels_below":"Explicit user convention; axes-alignment gate applies to plot areas.",
        "interpretation":"Mapper input only, not net user rate or file FER; uncontrolled background.",
    }, indent=2)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
