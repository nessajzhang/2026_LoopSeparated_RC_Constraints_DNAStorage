"""Compare ViennaRNA ensemble summaries against the random baseline.

The ensemble-metric extension writes one summary row for each
``(group,length,m,k)`` cell. Random baseline rows are grouped only by length,
with blank ``m`` and ``k`` fields. This script converts that wide summary into
a long-format baseline-comparison table so the direction and size of each
ensemble metric can be inspected without hand-parsing the CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


BASELINE_GROUP = "random_gc_homopolymer"


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def metric_name_from_mean_field(field: str) -> str:
    if not field.startswith("mean_"):
        raise ValueError(f"not a mean field: {field}")
    return field[len("mean_") :]


def format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def build_comparison(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("no rows in ensemble summary")

    metric_fields = [
        field
        for field in rows[0]
        if field.startswith("mean_ensemble_")
    ]
    if not metric_fields:
        raise ValueError("no mean_ensemble_* fields found")

    baselines: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("group") == BASELINE_GROUP:
            length = row.get("length", "")
            baselines[length] = row

    out: list[dict[str, object]] = []
    for row in rows:
        if row.get("group") == BASELINE_GROUP:
            continue
        length = row.get("length", "")
        baseline = baselines.get(length)
        if baseline is None:
            raise ValueError(f"missing random baseline for length={length}")
        for field in metric_fields:
            metric = metric_name_from_mean_field(field)
            target = safe_float(row.get(field, ""))
            base = safe_float(baseline.get(field, ""))
            delta = target - base
            rel = delta / abs(base) if base not in (0.0, -0.0) and not math.isnan(base) else math.nan
            out.append(
                {
                    "group": row.get("group", ""),
                    "length": length,
                    "m": row.get("m", ""),
                    "k": row.get("k", ""),
                    "samples": row.get("samples", ""),
                    "baseline_group": BASELINE_GROUP,
                    "baseline_samples": baseline.get("samples", ""),
                    "energy_model": row.get("energy_model", baseline.get("energy_model", "")),
                    "metric": metric,
                    "target_mean": format_float(target),
                    "baseline_mean": format_float(base),
                    "delta_vs_random": format_float(delta),
                    "relative_delta_vs_random": format_float(rel),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out_rows = build_comparison(rows)
    write_csv(args.out, out_rows)
    print(f"wrote ensemble comparison rows: {args.out}")
    print(f"rows: {len(out_rows)}")


if __name__ == "__main__":
    main()
