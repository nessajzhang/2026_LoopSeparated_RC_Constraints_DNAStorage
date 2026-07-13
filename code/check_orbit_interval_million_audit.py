#!/usr/bin/env python3
"""Independent Python audit of the occurrence-position interval property."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "orbit_interval_python_crosscheck_summary.json"
M_VALUES = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    12,
    16,
    24,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
)
START_COUNTS = (0, 1, 2, 8, 32, 128, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
SEED = 20260713


@dataclass(frozen=True)
class Case:
    m: int
    starts: int
    k: int
    singleton: bool
    selected: tuple[int, ...]
    noncore: tuple[int, ...]

    @property
    def distance(self) -> int:
        return self.m + self.k


def threshold(n: int, schedule: int) -> int:
    candidates = (
        0,
        1,
        4,
        20,
        math.floor(math.log2(max(1, n))),
        math.floor(math.sqrt(max(1, n))),
        math.floor(max(1, n) ** 0.60),
        math.floor(max(1, n) ** 0.75),
        math.floor(max(1, n) / max(1.0, math.log(max(1, n)))),
        math.floor(0.01 * n),
        math.floor(0.10 * n),
        math.floor(0.25 * n),
        math.floor(0.50 * n),
        math.floor(0.90 * n),
        math.floor(0.99 * n),
        max(0, n - 1),
    )
    return min(max(0, candidates[schedule]), max(0, n - 1))


def unique_random_positions(rng: random.Random, starts: int, maximum: int = 24) -> tuple[int, ...]:
    if starts <= 0:
        return ()
    return tuple(sorted({rng.randrange(starts) for _ in range(rng.randint(1, maximum))}))


def interval_positions(rng: random.Random, lo: int, hi: int) -> tuple[int, ...]:
    if hi < lo:
        return ()
    values = {lo, hi}
    values.update(rng.randint(lo, hi) for _ in range(rng.randint(0, 22)))
    return tuple(sorted(values))


def generate_case(rng: random.Random, case_id: int) -> Case:
    m = M_VALUES[case_id % len(M_VALUES)]
    starts = START_COUNTS[(case_id // len(M_VALUES)) % len(START_COUNTS)]
    n = max(0, m - 1) if starts == 0 else starts + m - 1
    k = threshold(n, (case_id // (len(M_VALUES) * len(START_COUNTS))) % 16)
    distance = m + k
    mode = rng.randrange(12)
    singleton = mode in {3, 4, 5}
    if starts == 0:
        return Case(m, starts, k, singleton, (), ())
    last = starts - 1

    if mode in {0, 3}:
        a = unique_random_positions(rng, starts)
        b = () if singleton else unique_random_positions(rng, starts)
    elif mode == 1:
        p = rng.randrange(starts)
        a = (p,)
        b = interval_positions(rng, max(0, p - distance + 1), min(last, p + distance - 1))
    elif mode == 2:
        a, b = ((0,), (distance,)) if distance < starts else ((0,), (last,))
    elif mode == 4:
        lo = rng.randrange(starts)
        a = interval_positions(rng, lo, min(last, lo + distance - 1))
        b = ()
    elif mode == 5:
        a = (0, distance) if distance < starts else (0, last)
        b = ()
    elif mode == 6:
        a, b = unique_random_positions(rng, starts), ()
    elif mode == 7:
        a = b = (0, last)
    elif mode == 8:
        width = min(last, distance - 1)
        base = rng.randrange(starts - width)
        a = interval_positions(rng, base, base + width)
        b = interval_positions(rng, base, base + width)
    elif mode in {9, 10}:
        p = 0 if mode == 9 else last
        a = (p,)
        b = interval_positions(rng, max(0, p - distance + 1), min(last, p + distance - 1))
    else:
        width = min(last, distance - 1)
        base = rng.randrange(starts - width)
        a = (base, base + width)
        b = interval_positions(rng, base, base + width)
    return Case(m, starts, k, singleton, tuple(sorted(set(a))), tuple(sorted(set(b))))


def direct_valid(case: Case) -> bool:
    if case.singleton:
        return all(
            abs(left - right) < case.distance
            for index, left in enumerate(case.selected)
            for right in case.selected[index + 1 :]
        )
    return all(
        abs(left - right) < case.distance for left in case.selected for right in case.noncore
    )


def independent_extrema_valid(case: Case) -> bool:
    if case.singleton:
        return len(case.selected) < 2 or max(case.selected) - min(case.selected) < case.distance
    if not case.selected or not case.noncore:
        return True
    four_corners = (
        abs(min(case.selected) - min(case.noncore)),
        abs(min(case.selected) - max(case.noncore)),
        abs(max(case.selected) - min(case.noncore)),
        abs(max(case.selected) - max(case.noncore)),
    )
    return max(four_corners) < case.distance


def audit(case: Case) -> tuple[bool, int]:
    valid = direct_valid(case)
    assert valid == independent_extrema_valid(case)
    assertions = 1
    if valid and case.singleton and case.selected:
        span = max(case.selected) - min(case.selected) + 1
        assert span <= case.distance
        assert span + case.m - 1 <= case.distance + case.m - 1
        assertions += 2
    elif valid and case.selected and case.noncore:
        span = max(case.noncore) - min(case.noncore) + 1
        assert span <= 2 * case.distance - 1
        assert span + case.m - 1 <= 2 * case.distance + case.m - 2
        assertions += 2
        for anchor in case.selected:
            assert min(case.noncore) >= anchor - case.distance + 1
            assert max(case.noncore) <= anchor + case.distance - 1
            assertions += 2
    elif not valid:
        if case.singleton:
            witness = any(
                abs(left - right) >= case.distance
                for index, left in enumerate(case.selected)
                for right in case.selected[index + 1 :]
            )
        else:
            witness = any(
                abs(left - right) >= case.distance
                for left in case.selected
                for right in case.noncore
            )
        assert witness
        assertions += 1
    return valid, assertions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    valid_cases = 0
    assertions = 0
    digest = hashlib.sha256()
    for case_id in range(args.cases):
        case = generate_case(rng, case_id)
        valid, case_assertions = audit(case)
        valid_cases += int(valid)
        assertions += case_assertions
        digest.update(repr(case).encode("utf-8"))

    result = {
        "status": "PASS",
        "implementation": "independent_python",
        "cases": args.cases,
        "valid_cases": valid_cases,
        "invalid_cases_with_witness": args.cases - valid_cases,
        "atomic_assertions": assertions,
        "seed": args.seed,
        "case_stream_sha256": digest.hexdigest(),
        "m_min": min(M_VALUES),
        "m_max": max(M_VALUES),
        "maximum_start_positions": max(START_COUNTS),
        "evidence_boundary": (
            "This implementation uses a separate RNG and extrema formulation. "
            "It cross-checks the geometric property, not capacity."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
