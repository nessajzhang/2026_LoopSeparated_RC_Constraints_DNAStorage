"""Exact endpoint decomposition and independent paired-count algorithms."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Sequence

from .model import (
    Involution,
    Word,
    is_avoiding_full,
    is_avoiding_m_windows,
    reverse_complement,
    validate_involution,
    windows,
)


def decomposition_parameters(n: int, m: int, k: int) -> tuple[int, int, int, int]:
    if n < 0 or m < 1 or k < 0:
        raise ValueError("require n>=0, m>=1, and k>=0")
    n_windows = max(0, n - m + 1)
    distance = m + k
    paired_length = n_windows - distance
    free_length = 2 * distance - n
    return n_windows, distance, paired_length, free_length


def brute_force_language_count(
    q: int,
    n: int,
    m: int,
    k: int,
    theta: Sequence[int],
    *,
    original_definition: bool = False,
) -> int:
    involution = validate_involution(theta)
    if q != len(involution):
        raise ValueError("q and theta are inconsistent")
    predicate = is_avoiding_full if original_definition else is_avoiding_m_windows
    return sum(predicate(word, m, k, involution) for word in product(range(q), repeat=n))


def endpoint_pair_is_valid(
    left: Sequence[int], right: Sequence[int], m: int, theta: Sequence[int]
) -> bool:
    involution = validate_involution(theta)
    left_windows = windows(left, m)
    right_windows = windows(right, m)
    if len(left_windows) != len(right_windows):
        raise ValueError("endpoint words must induce the same number of windows")
    for t, right_window in enumerate(right_windows):
        right_rc = reverse_complement(right_window, involution)
        if any(left_windows[i] == right_rc for i in range(t + 1)):
            return False
    return True


def paired_count_bruteforce(q: int, r: int, m: int, theta: Sequence[int]) -> int:
    involution = validate_involution(theta)
    if q != len(involution) or r < 0 or m < 1:
        raise ValueError("invalid q, r, m, or theta")
    endpoint_length = r + m - 1
    endpoint_words = tuple(product(range(q), repeat=endpoint_length))
    return sum(
        endpoint_pair_is_valid(left, right, m, involution)
        for left in endpoint_words
        for right in endpoint_words
    )


def paired_count_dynamic(q: int, r: int, m: int, theta: Sequence[int]) -> int:
    """Count endpoint pairs by a seen-left-set dynamic program.

    This implementation is independent of the full-word brute-force counter.
    A state contains the two `(m-1)` suffixes and the exact set of left
    windows seen so far.
    """

    involution: Involution = validate_involution(theta)
    if q != len(involution) or r < 0 or m < 1:
        raise ValueError("invalid q, r, m, or theta")
    suffixes = tuple(product(range(q), repeat=m - 1))
    states: dict[tuple[Word, Word, frozenset[Word]], int] = {
        (left, right, frozenset()): 1 for left in suffixes for right in suffixes
    }

    for _ in range(r):
        next_states: dict[tuple[Word, Word, frozenset[Word]], int] = defaultdict(int)
        for (left_suffix, right_suffix, seen), multiplicity in states.items():
            for left_symbol in range(q):
                left_window = left_suffix + (left_symbol,)
                updated_seen = seen.union((left_window,))
                new_left_suffix = left_window[1:] if m > 1 else ()
                for right_symbol in range(q):
                    right_window = right_suffix + (right_symbol,)
                    if reverse_complement(right_window, involution) in updated_seen:
                        continue
                    new_right_suffix = right_window[1:] if m > 1 else ()
                    key = (new_left_suffix, new_right_suffix, updated_seen)
                    next_states[key] += multiplicity
        states = next_states
    return sum(states.values())


def decomposed_language_count(
    q: int,
    n: int,
    m: int,
    k: int,
    theta: Sequence[int],
    *,
    paired_method: str = "dynamic",
) -> int:
    _, _, r, free_length = decomposition_parameters(n, m, k)
    if r < 0 or free_length < 0:
        raise ValueError("the exact endpoint decomposition requires r>=0 and 2D>=n")
    if paired_method == "dynamic":
        paired = paired_count_dynamic(q, r, m, theta)
    elif paired_method == "bruteforce":
        paired = paired_count_bruteforce(q, r, m, theta)
    else:
        raise ValueError(f"unknown paired method: {paired_method}")
    return (q**free_length) * paired
