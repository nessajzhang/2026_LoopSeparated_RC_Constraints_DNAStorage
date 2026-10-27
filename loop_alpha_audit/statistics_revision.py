"""Declared uncertainty calculations for independent simulated files.

The binomial unit is a complete independently generated file, not an individual
read or one of the correlated RS strands. Paired differences use discordant
file outcomes. Their conservative simultaneous interval does not assume the
two discordant-event estimates, methods or grid cells are independent.
"""

from __future__ import annotations

import math
from typing import Sequence

from scipy.stats import beta


def clopper_pearson(events: int, trials: int, *, alpha: float = 0.05) -> tuple[float, float]:
    if (
        type(events) is not int
        or type(trials) is not int
        or not 0 <= events <= trials
        or trials < 1
    ):
        raise ValueError("require integer 0 <= events <= trials and trials >= 1")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    low = 0.0 if events == 0 else float(beta.ppf(alpha / 2, events, trials - events + 1))
    high = 1.0 if events == trials else float(beta.ppf(1 - alpha / 2, events + 1, trials - events))
    return low, high


def zero_event_upper(trials: int, *, alpha: float = 0.05) -> float:
    """One-sided exact upper confidence limit; zero observations is not zero risk."""
    if type(trials) is not int or trials < 1 or not 0 < alpha < 1:
        raise ValueError("invalid zero-event confidence parameters")
    return -math.expm1(math.log(alpha) / trials)


def paired_failure_difference(
    left: Sequence[bool],
    right: Sequence[bool],
    *,
    alpha: float = 0.05,
) -> dict[str, float | int | list[float] | str]:
    """Estimate P(left fails)-P(right fails), using paired independent files.

    The two disjoint discordant events each receive a (1-alpha/2) exact
    marginal interval. Their Cartesian difference contains the difference
    with probability at least 1-alpha by the union bound. This is conservative,
    valid without within-pair independence, and non-degenerate for zero events.
    """
    if len(left) != len(right) or not left:
        raise ValueError("nonempty paired outcome vectors must have equal length")
    if any(type(value) is not bool for value in (*left, *right)):
        raise ValueError("paired outcomes must be explicit Boolean failure indicators")
    n = len(left)
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    lo_l, hi_l = clopper_pearson(left_only, n, alpha=alpha / 2)
    lo_r, hi_r = clopper_pearson(right_only, n, alpha=alpha / 2)
    return {
        "independent_files": n,
        "left_only_failures": left_only,
        "right_only_failures": right_only,
        "difference": (left_only - right_only) / n,
        "confidence_interval": [max(-1.0, lo_l - hi_r), min(1.0, hi_l - lo_r)],
        "alpha": alpha,
        "method": "Bonferroni union of exact marginal discordant-event intervals",
        "direction": "negative means lower file FER for left; not a causal constraint effect",
    }


def summarize_independent_files(
    rows: Sequence[dict],
    *,
    expected_keys: Sequence[tuple[int, int]],
) -> dict:
    """Keep failures and full denominators; reject incomplete or duplicate rows."""
    if not rows:
        raise ValueError("cannot summarize an empty cell")
    keys = [(row["seed"], row["file_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate independent file key")
    expected = list(expected_keys)
    if (
        not expected
        or len(expected) != len(set(expected))
        or any(
            not isinstance(key, tuple)
            or len(key) != 2
            or any(type(value) is not int or value < 0 for value in key)
            for key in expected
        )
    ):
        raise ValueError("expected planned file keys must be unique nonnegative integer pairs")
    if set(keys) != set(expected):
        missing, extra = set(expected) - set(keys), set(keys) - set(expected)
        raise ValueError(
            f"planned file keys do not reconcile: missing={len(missing)}, extra={len(extra)}"
        )
    for row in rows:
        if type(row["file_correct"]) is not bool:
            raise ValueError("file_correct must be explicitly evaluated for every attempted file")
        for field in (
            "seed",
            "file_id",
            "user_bits",
            "total_synthesized_nt",
            "total_sequenced_nt",
            "frames_attempted",
            "frame_correct",
            "frame_failures",
            "frame_erasures",
            "frame_undetected_errors",
        ):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if row["frames_attempted"] < 1:
            raise ValueError("attempted files must retain their positive planned frame count")
        if row["frames_attempted"] != row["frame_correct"] + row["frame_failures"]:
            raise ValueError("frame denominator reconciliation failed")
        if row["frame_failures"] != row["frame_erasures"] + row["frame_undetected_errors"]:
            raise ValueError("failure/erasure/undetected reconciliation failed")
    n = len(rows)
    failures = sum(not row["file_correct"] for row in rows)
    total_bits = sum(row["user_bits"] for row in rows)
    recovered_bits = sum(row["user_bits"] for row in rows if row["file_correct"])
    total_nt = sum(row["total_synthesized_nt"] for row in rows)
    frames = sum(row["frames_attempted"] for row in rows)
    frame_errors = sum(row["frame_failures"] for row in rows)
    any_frame_error = sum(row["frame_failures"] > 0 for row in rows)
    return {
        "independent_files": n,
        "planned_files": len(expected),
        "file_failures": failures,
        "file_FER": failures / n,
        "file_FER_exact_95_CI": list(clopper_pearson(failures, n)),
        "file_FER_one_sided_95_upper_if_zero": zero_event_upper(n) if failures == 0 else None,
        "frame_count": frames,
        "frame_failures": frame_errors,
        "frame_FER_point_estimate": frame_errors / frames if frames else None,
        "files_with_any_frame_error": any_frame_error,
        "any_frame_error_file_probability_exact_95_CI": list(clopper_pearson(any_frame_error, n)),
        "frame_uncertainty_scope": "strands are clustered within files; no independent-strand binomial CI",
        "user_bits_attempted": total_bits,
        "user_bits_correctly_recovered": recovered_bits,
        "total_synthesized_nt": total_nt,
        "R_net": total_bits / total_nt if total_nt else None,
        "goodput": recovered_bits / total_nt if total_nt else None,
        "total_sequenced_nt": sum(row["total_sequenced_nt"] for row in rows),
        "all_attempts_retained": True,
    }
