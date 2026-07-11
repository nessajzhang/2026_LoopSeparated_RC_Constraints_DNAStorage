"""Compute ViennaRNA ensemble metrics for existing folding-validation rows.

The calculation reuses sequence rows from ``folding_grid.py`` and adds
partition-function and base-pair-probability summaries without resampling.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

GROUP_FIELDS = ("group", "length", "m", "k")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def mean(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else math.nan


def stdev(values: list[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return 0.0
    mu = mean(clean)
    return math.sqrt(sum((x - mu) ** 2 for x in clean) / (len(clean) - 1))


def expected_pair_count_from_bpp(bpp: tuple[tuple[float, ...], ...], n: int) -> float:
    """Return sum_{i<j} p_ij from ViennaRNA's 1-indexed bpp matrix."""
    total = 0.0
    for i in range(1, min(len(bpp), n + 1)):
        row = bpp[i]
        for j in range(i + 1, n + 1):
            if j < len(row):
                total += float(row[j])
    return total


def ensemble_metrics(seq: str) -> dict[str, object]:
    # Import the folding backend lazily so stored ensemble rows can be
    # re-summarized without requiring ViennaRNA at import time.
    from folding_validation import ENERGY_MODEL, RNA, mfe_stem_summary

    fc = RNA.fold_compound(seq)
    mfe_structure, mfe = fc.mfe()
    if hasattr(fc, "exp_params_rescale"):
        fc.exp_params_rescale(float(mfe))
    pf_result = fc.pf()
    if isinstance(pf_result, (list, tuple)) and len(pf_result) >= 2:
        pf_structure = str(pf_result[0])
        ensemble_free_energy = float(pf_result[1])
    else:
        pf_structure = ""
        ensemble_free_energy = safe_float(pf_result)

    bpp = fc.bpp()
    expected_pair_count = expected_pair_count_from_bpp(bpp, len(seq))
    centroid_structure, centroid_distance = fc.centroid()
    mean_bp_distance = fc.mean_bp_distance()
    ensemble_defect_mfe = fc.ensemble_defect(mfe_structure)
    mfe_pair_count, mfe_max_stem = mfe_stem_summary(mfe_structure)
    centroid_pair_count, centroid_max_stem = mfe_stem_summary(str(centroid_structure))

    return {
        "ensemble_energy_model": ENERGY_MODEL,
        "ensemble_mfe_structure": mfe_structure,
        "ensemble_mfe_kcal_mol": f"{float(mfe):.4f}",
        "ensemble_pf_structure": pf_structure,
        "ensemble_free_energy_kcal_mol": f"{ensemble_free_energy:.4f}",
        "ensemble_expected_pair_count": f"{expected_pair_count:.4f}",
        "ensemble_expected_pair_fraction": f"{expected_pair_count / len(seq):.6f}",
        "ensemble_mean_bp_distance": f"{float(mean_bp_distance):.4f}",
        "ensemble_centroid_structure": centroid_structure,
        "ensemble_centroid_distance": f"{float(centroid_distance):.4f}",
        "ensemble_defect_mfe": f"{float(ensemble_defect_mfe):.4f}",
        "ensemble_mfe_pair_count": mfe_pair_count,
        "ensemble_mfe_max_stem": mfe_max_stem,
        "ensemble_centroid_pair_count": centroid_pair_count,
        "ensemble_centroid_max_stem": centroid_max_stem,
    }


def select_rows(rows: list[dict[str, str]], max_per_group: int | None) -> list[dict[str, str]]:
    if max_per_group is None:
        return rows
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    selected: list[dict[str, str]] = []
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        if counts[key] < max_per_group:
            selected.append(row)
            counts[key] += 1
    return selected


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups[key].append(row)

    out: list[dict[str, object]] = []
    metric_fields = [
        "ensemble_free_energy_kcal_mol",
        "ensemble_expected_pair_count",
        "ensemble_expected_pair_fraction",
        "ensemble_mean_bp_distance",
        "ensemble_centroid_distance",
        "ensemble_defect_mfe",
        "ensemble_mfe_pair_count",
        "ensemble_mfe_max_stem",
        "ensemble_centroid_pair_count",
        "ensemble_centroid_max_stem",
    ]
    for key, subset in sorted(groups.items(), key=lambda item: str(item[0])):
        row: dict[str, object] = {
            "group": key[0],
            "length": key[1],
            "m": key[2],
            "k": key[3],
            "samples": len(subset),
            "energy_model": subset[0].get("ensemble_energy_model", ""),
        }
        for field in metric_fields:
            values = [safe_float(r.get(field, "")) for r in subset]
            row[f"mean_{field}"] = f"{mean(values):.6f}"
            row[f"sd_{field}"] = f"{stdev(values):.6f}"
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument(
        "--summary-only-from",
        type=Path,
        default=None,
        help="Recompute a summary from stored ensemble rows without running ViennaRNA.",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=10,
        help="Limit rows per (group,length,m,k). Use 0 for all rows.",
    )
    args = parser.parse_args()

    if args.summary_only_from is not None:
        with args.summary_only_from.open("r", encoding="utf-8", newline="") as fh:
            stored_rows = list(csv.DictReader(fh))
        write_csv(args.summary_out, summarize(stored_rows))
        print(f"wrote ensemble summary: {args.summary_out}")
        return
    if args.infile is None or args.out is None:
        parser.error("--in and --out are required unless --summary-only-from is used")

    with args.infile.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    max_per_group = None if args.max_per_group == 0 else args.max_per_group
    selected = select_rows(rows, max_per_group)

    out_rows: list[dict[str, object]] = []
    for idx, row in enumerate(selected, start=1):
        seq = row["sequence"]
        metrics = ensemble_metrics(seq)
        out_rows.append({**row, **metrics})
        if idx % 25 == 0:
            print(f"processed {idx}/{len(selected)} rows", flush=True)

    write_csv(args.out, out_rows)
    summary = summarize(out_rows)
    write_csv(args.summary_out, summary)
    print(f"wrote ensemble rows: {args.out}")
    print(f"wrote ensemble summary: {args.summary_out}")


if __name__ == "__main__":
    main()
