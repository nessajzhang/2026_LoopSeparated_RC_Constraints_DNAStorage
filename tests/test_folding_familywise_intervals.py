from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from analysis_tools.plot_folding_comparison_tcom import (
    SERIES,
    load_rows,
    matched_rows,
    random_by_length,
    reduction_arrays,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "output/folding/folding_comparison_len100_150_200_m6_7_s1000.csv"


def test_familywise_bonferroni_newcombe_intervals_exclude_zero() -> None:
    rows = load_rows(SUMMARY)
    m_values = sorted({int(row["m"]) for row in rows})
    comparison_count = sum(len(matched_rows(rows, key, m)) for m in m_values for key, *_ in SERIES)
    assert comparison_count == 24
    z = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * comparison_count))

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for m in m_values:
        baseline = random_by_length(rows, m)
        for key, *_ in SERIES:
            items = matched_rows(rows, key, m)
            _, _, lower, upper = reduction_arrays(baseline, items, z=z)
            lower_bounds.extend(np.asarray(lower, dtype=float).tolist())
            upper_bounds.extend(np.asarray(upper, dtype=float).tolist())

    assert len(lower_bounds) == comparison_count
    assert min(lower_bounds) > 0.0
    assert math.isclose(min(lower_bounds), 2.5454453037575444, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(max(upper_bounds), 40.42590075428377, rel_tol=0.0, abs_tol=1e-12)
