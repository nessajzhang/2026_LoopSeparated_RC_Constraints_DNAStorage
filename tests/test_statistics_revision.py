from __future__ import annotations

import math

import pytest
from scipy.stats import binomtest

from loop_alpha_audit.statistics_revision import (
    clopper_pearson,
    paired_failure_difference,
    summarize_independent_files,
    zero_event_upper,
)


def test_exact_interval_matches_independent_scipy_binomtest() -> None:
    for n in (1, 3, 10, 100, 300):
        for k in {0, 1, n // 2, n - 1, n}:
            expected = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
            low, high = clopper_pearson(k, n)
            assert abs(low - expected.low) < 1e-10
            assert abs(high - expected.high) < 1e-10
    assert math.isclose(zero_event_upper(300), 1 - 0.05 ** (1 / 300))
    assert zero_event_upper(300) > 0.009


def test_paired_interval_counts_and_zero_event_uncertainty() -> None:
    left = [True, False, True, False]
    right = [False, True, True, False]
    result = paired_failure_difference(left, right)
    assert result["left_only_failures"] == result["right_only_failures"] == 1
    assert result["difference"] == 0
    all_zero = paired_failure_difference([False] * 300, [False] * 300)
    assert all_zero["confidence_interval"][0] < 0 < all_zero["confidence_interval"][1]
    assert paired_failure_difference([True] * 10, [False] * 10)["difference"] == 1
    with pytest.raises(ValueError):
        paired_failure_difference([1], [0])


def test_full_denominators_retain_erasures_and_failures() -> None:
    row = {
        "seed": 1,
        "file_id": 0,
        "file_correct": False,
        "user_bits": 100,
        "total_synthesized_nt": 1000,
        "total_sequenced_nt": 3000,
        "frames_attempted": 7,
        "frame_correct": 4,
        "frame_failures": 3,
        "frame_erasures": 2,
        "frame_undetected_errors": 1,
    }
    summary = summarize_independent_files([row], expected_keys=[(1, 0)])
    assert summary["file_FER"] == 1 and summary["goodput"] == 0
    assert summary["frame_count"] == 7 and summary["frame_failures"] == 3
    assert summary["R_net"] == 0.1
    with pytest.raises(ValueError):
        summarize_independent_files([row, row], expected_keys=[(1, 0)])
    with pytest.raises(ValueError):
        summarize_independent_files([row | {"frame_erasures": 0}], expected_keys=[(1, 0)])
    with pytest.raises(ValueError, match="missing=1"):
        summarize_independent_files([row], expected_keys=[(1, 0), (1, 1)])
    with pytest.raises(ValueError, match="extra=1"):
        summarize_independent_files([row], expected_keys=[(2, 0)])
    with pytest.raises(ValueError, match="unique"):
        summarize_independent_files([row], expected_keys=[(1, 0), (1, 0)])
    for field in ("frames_attempted", "frame_failures", "user_bits", "total_sequenced_nt"):
        for invalid in (-1, 0.5, True):
            with pytest.raises(ValueError, match="nonnegative integer"):
                summarize_independent_files([row | {field: invalid}], expected_keys=[(1, 0)])
