"""Reference definitions and the orbit-oriented interval cover.

This module deliberately favors transparent definitions over speed.  Optimized
counting code is kept separate so that tests can compare independent paths.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

Symbol = int
Word = tuple[Symbol, ...]
Involution = tuple[Symbol, ...]


@dataclass(frozen=True)
class OrbitCover:
    """One standardized cover interval for an RC orbit.

    Start coordinates are zero-based and inclusive.  ``selected`` is the core
    representative; it is ``None`` for a self-RC orbit.  ``noncore`` is the
    word whose occurrences are covered.
    """

    orbit: tuple[Word, ...]
    selected: Word | None
    noncore: Word
    start_lo: int
    start_hi: int

    @property
    def start_length(self) -> int:
        return self.start_hi - self.start_lo + 1


def validate_involution(theta: Sequence[int]) -> Involution:
    result = tuple(theta)
    q = len(result)
    if q < 2 or sorted(result) != list(range(q)):
        raise ValueError("theta must be a permutation of range(q), q>=2")
    if any(result[result[a]] != a for a in range(q)):
        raise ValueError("theta must be an involution")
    return result


def reverse_complement(word: Sequence[int], theta: Sequence[int]) -> Word:
    involution = validate_involution(theta)
    return tuple(involution[symbol] for symbol in reversed(word))


def windows(word: Sequence[int], length: int) -> tuple[Word, ...]:
    if length < 1:
        raise ValueError("window length must be positive")
    x = tuple(word)
    return tuple(x[i : i + length] for i in range(max(0, len(x) - length + 1)))


def is_avoiding_m_windows(word: Sequence[int], m: int, k: int, theta: Sequence[int]) -> bool:
    """Check the reduced length-m definition directly."""

    if m < 1 or k < 0:
        raise ValueError("require m>=1 and k>=0")
    involution = validate_involution(theta)
    blocks = windows(word, m)
    distance = m + k
    for i, left in enumerate(blocks):
        for j in range(i + distance, len(blocks)):
            if left == reverse_complement(blocks[j], involution):
                return False
    return True


def is_avoiding_full(word: Sequence[int], m: int, k: int, theta: Sequence[int]) -> bool:
    """Check the original all-stem-length definition without reductions."""

    if m < 1 or k < 0:
        raise ValueError("require m>=1 and k>=0")
    involution = validate_involution(theta)
    x = tuple(word)
    n = len(x)
    for stem in range(m, n + 1):
        blocks = windows(x, stem)
        min_start_separation = stem + k
        for i, left in enumerate(blocks):
            for j in range(i + min_start_separation, len(blocks)):
                if left == reverse_complement(blocks[j], involution):
                    return False
    return True


def rc_orbits(q: int, m: int, theta: Sequence[int]) -> tuple[tuple[Word, ...], ...]:
    involution = validate_involution(theta)
    if q != len(involution) or m < 1:
        raise ValueError("q and m are inconsistent")
    unseen = set(product(range(q), repeat=m))
    result: list[tuple[Word, ...]] = []
    while unseen:
        first = min(unseen)
        partner = reverse_complement(first, involution)
        orbit = tuple(sorted({first, partner}))
        result.append(orbit)
        unseen.difference_update(orbit)
    return tuple(result)


def occurrence_starts(word: Sequence[int], m: int) -> dict[Word, tuple[int, ...]]:
    positions: dict[Word, list[int]] = defaultdict(list)
    for start, block in enumerate(windows(word, m)):
        positions[block].append(start)
    return {block: tuple(starts) for block, starts in positions.items()}


def construct_orbit_interval_cover(
    word: Sequence[int], m: int, k: int, theta: Sequence[int]
) -> tuple[frozenset[Word], tuple[OrbitCover, ...]]:
    """Construct the deterministic maximal core and standardized cover.

    The caller is responsible for checking that ``word`` is avoiding.  A
    ``ValueError`` is raised if the interval conclusion fails, which makes the
    function useful as an executable assertion in exhaustive tests.
    """

    if m < 1 or k < 0:
        raise ValueError("require m>=1 and k>=0")
    involution = validate_involution(theta)
    x = tuple(word)
    q = len(involution)
    starts = occurrence_starts(x, m)
    n_windows = max(0, len(x) - m + 1)
    distance = m + k
    core: set[Word] = set()
    covers: list[OrbitCover] = []

    for orbit in rc_orbits(q, m, involution):
        if len(orbit) == 1:
            self_rc = orbit[0]
            occurrences = starts.get(self_rc, ())
            if not occurrences:
                continue
            lo = min(occurrences)
            hi = min(n_windows - 1, lo + distance - 1)
            if any(not (lo <= start <= hi) for start in occurrences):
                raise ValueError("self-RC occurrences do not fit the required interval")
            covers.append(OrbitCover(orbit, None, self_rc, lo, hi))
            continue

        left, right = orbit
        left_occurs = bool(starts.get(left))
        right_occurs = bool(starts.get(right))
        if left_occurs and not right_occurs:
            selected, noncore = left, right
        elif right_occurs and not left_occurs:
            selected, noncore = right, left
        else:
            # Deterministic choice when both or neither occur.
            selected, noncore = left, right
        core.add(selected)

        noncore_occurrences = starts.get(noncore, ())
        if not noncore_occurrences:
            continue
        selected_occurrences = starts.get(selected, ())
        if not selected_occurrences:
            raise ValueError("a noncore word occurs without the selected RC anchor")
        anchor = min(selected_occurrences)
        lo = max(0, anchor - distance + 1)
        hi = min(n_windows - 1, anchor + distance - 1)
        if any(not (lo <= start <= hi) for start in noncore_occurrences):
            raise ValueError("noncore occurrences do not fit the required interval")
        covers.append(OrbitCover(orbit, selected, noncore, lo, hi))

    return frozenset(core), tuple(covers)


def is_maximal_rc_free(core: Iterable[Word], q: int, m: int, theta: Sequence[int]) -> bool:
    chosen = set(core)
    for block in chosen:
        if reverse_complement(block, theta) in chosen:
            return False
    for orbit in rc_orbits(q, m, theta):
        expected = 0 if len(orbit) == 1 else 1
        if len(chosen.intersection(orbit)) != expected:
            return False
    return True


def covered_start_set(covers: Iterable[OrbitCover]) -> frozenset[int]:
    return frozenset(
        start for cover in covers for start in range(cover.start_lo, cover.start_hi + 1)
    )


def marked_letter_set(covers: Iterable[OrbitCover], m: int, n: int) -> frozenset[int]:
    return frozenset(
        coordinate
        for cover in covers
        for coordinate in range(cover.start_lo, min(n, cover.start_hi + m))
    )
