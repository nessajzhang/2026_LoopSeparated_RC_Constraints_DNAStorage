#!/usr/bin/env python3
"""Run counted exhaustive checks across proof and counting definitions."""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.model import (  # noqa: E402
    construct_orbit_interval_cover,
    covered_start_set,
    is_avoiding_full,
    is_avoiding_m_windows,
    is_maximal_rc_free,
    marked_letter_set,
    rc_orbits,
    windows,
)
from loop_alpha_audit.paired import (  # noqa: E402
    brute_force_language_count,
    decomposed_language_count,
    decomposition_parameters,
    paired_count_bruteforce,
    paired_count_dynamic,
)


OUTPUT = ROOT / "output" / "exhaustive_consistency_audit.json"


def main() -> None:
    stem_comparisons = 0
    cover_words = 0
    decomposition_cells = 0
    paired_cells = 0

    stem_cases = (
        ((1, 0), 1, 7),
        ((1, 0), 2, 7),
        ((0, 1), 1, 7),
        ((0, 1), 2, 7),
        ((1, 0, 2), 1, 7),
        ((1, 0, 2), 2, 7),
    )
    for theta, m, n_max in stem_cases:
        q = len(theta)
        for n in range(n_max + 1):
            for k in range(n + 1):
                for word in product(range(q), repeat=n):
                    full = is_avoiding_full(word, m, k, theta)
                    reduced = is_avoiding_m_windows(word, m, k, theta)
                    if full != reduced:
                        raise AssertionError(("stem_reduction", theta, m, n, k, word))
                    stem_comparisons += 1

    cover_cases = (
        ((1, 0), 1, 8),
        ((1, 0), 2, 8),
        ((0, 1), 1, 8),
        ((0, 1), 2, 8),
        ((1, 0, 2), 1, 7),
        ((1, 0, 2), 2, 7),
        ((3, 2, 1, 0), 1, 6),
        ((3, 2, 1, 0), 2, 6),
    )
    for theta, m, n_max in cover_cases:
        q = len(theta)
        orbit_count = len(rc_orbits(q, m, theta))
        for n in range(n_max + 1):
            for k in range(n + 1):
                distance = m + k
                for word in product(range(q), repeat=n):
                    if not is_avoiding_m_windows(word, m, k, theta):
                        continue
                    core, covers = construct_orbit_interval_cover(word, m, k, theta)
                    if not is_maximal_rc_free(core, q, m, theta):
                        raise AssertionError(("maximal_core", theta, m, n, k, word))
                    if len(covers) > orbit_count:
                        raise AssertionError(("too_many_intervals", theta, m, n, k, word))
                    if any(c.start_length > 2 * distance - 1 for c in covers):
                        raise AssertionError(("interval_too_long", theta, m, n, k, word))
                    covered = covered_start_set(covers)
                    for start, block in enumerate(windows(word, m)):
                        if block not in core and start not in covered:
                            raise AssertionError(("uncovered_start", theta, m, n, k, word))
                    marked = marked_letter_set(covers, m, n)
                    if len(marked) > orbit_count * (2 * distance + m - 2):
                        raise AssertionError(("marked_budget", theta, m, n, k, word))
                    cover_words += 1

    decomposition_cases = (
        ((1, 0), 1, 10),
        ((1, 0), 2, 10),
        ((0, 1), 1, 10),
        ((0, 1), 2, 10),
        ((1, 0, 2), 1, 8),
        ((1, 0, 2), 2, 8),
        ((3, 2, 1, 0), 1, 7),
        ((3, 2, 1, 0), 2, 7),
    )
    for theta, m, n_max in decomposition_cases:
        q = len(theta)
        for n in range(max(0, 2 * m - 2), n_max + 1):
            for k in range(n + 1):
                _, _, r, free = decomposition_parameters(n, m, k)
                if r < 0 or free < 0:
                    continue
                full = brute_force_language_count(q, n, m, k, theta, original_definition=True)
                reduced = brute_force_language_count(q, n, m, k, theta, original_definition=False)
                decomposed = decomposed_language_count(q, n, m, k, theta)
                if not full == reduced == decomposed:
                    raise AssertionError(
                        ("decomposition", theta, m, n, k, full, reduced, decomposed)
                    )
                decomposition_cells += 1

    for theta in ((1, 0), (0, 1), (1, 0, 2), (3, 2, 1, 0)):
        q = len(theta)
        for m in (1, 2):
            max_r = 4 if q == 2 else 2
            for r in range(max_r + 1):
                brute = paired_count_bruteforce(q, r, m, theta)
                dynamic = paired_count_dynamic(q, r, m, theta)
                if brute != dynamic:
                    raise AssertionError(("paired", theta, m, r, brute, dynamic))
                paired_cells += 1

    result = {
        "status": "CROSS_VALIDATED",
        "stem_reduction_word_parameter_comparisons": stem_comparisons,
        "avoiding_words_with_orbit_cover_checked": cover_words,
        "full_reduced_decomposed_count_cells": decomposition_cells,
        "paired_bruteforce_dynamic_cells": paired_cells,
        "notes": [
            "All comparisons use exact integer counts.",
            "The original all-stem definition is evaluated independently.",
            "The paired dynamic program does not enumerate full words.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
