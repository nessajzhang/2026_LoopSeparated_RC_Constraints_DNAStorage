#!/usr/bin/env python3
"""Finite brute-force sanity checks for the loop-separated RC proof.

This script does not prove the asymptotic theorems.  It exhaustively checks
the finite combinatorial reductions that the proof relies on for small
alphabets and block lengths:

1. the original length-at-least-m definition is equivalent to the length-m
   reduction;
2. the frequent-window set is reverse-complement-free at threshold 2(m+k);
3. every admissible word is covered by a maximal RC-free core plus a bounded
   exceptional-window budget.

The parameter ranges are deliberately small so the checks remain exhaustive.
"""

from __future__ import annotations

from collections import Counter
from itertools import product
import random


def rc(word: tuple[int, ...], theta: dict[int, int]) -> tuple[int, ...]:
    return tuple(theta[x] for x in reversed(word))


def is_avoiding_original(
    word: tuple[int, ...], q: int, theta: dict[int, int], m: int, k: int
) -> bool:
    n = len(word)
    for t in range(m, n + 1):
        for i in range(0, n - t + 1):
            left = word[i : i + t]
            for j in range(i + 1, n - t + 1):
                if j - i - t >= k and left == rc(word[j : j + t], theta):
                    return False
    return True


def is_avoiding_reduced(
    word: tuple[int, ...], q: int, theta: dict[int, int], m: int, k: int
) -> bool:
    n = len(word)
    d = m + k
    for i in range(0, n - m + 1):
        left = word[i : i + m]
        for j in range(i + d, n - m + 1):
            if left == rc(word[j : j + m], theta):
                return False
    return True


def is_rc_free(words: set[tuple[int, ...]], theta: dict[int, int]) -> bool:
    for u in words:
        for v in words:
            if u == rc(v, theta):
                return False
    return True


def extend_to_maximal_rc_free(
    seed: set[tuple[int, ...]], universe: list[tuple[int, ...]], theta: dict[int, int]
) -> set[tuple[int, ...]]:
    s = set(seed)
    assert is_rc_free(s, theta)
    changed = True
    while changed:
        changed = False
        for u in universe:
            if u not in s and is_rc_free(s | {u}, theta):
                s.add(u)
                changed = True
    return s


def window_counter(word: tuple[int, ...], m: int) -> Counter[tuple[int, ...]]:
    return Counter(word[i : i + m] for i in range(0, len(word) - m + 1))


def check_case(name: str, q: int, theta: dict[int, int], max_n: int, max_m: int, max_k: int) -> dict[str, int]:
    stats = {
        "words": 0,
        "avoiding_words": 0,
        "reduction_checks": 0,
        "frequent_checks": 0,
        "cover_checks": 0,
    }
    for n in range(1, max_n + 1):
        for m in range(1, min(max_m, n) + 1):
            universe = list(product(range(q), repeat=m))
            for k in range(0, max_k + 1):
                d = m + k
                budget = (2 * d - 1) * (q**m)
                for word in product(range(q), repeat=n):
                    stats["words"] += 1
                    orig = is_avoiding_original(word, q, theta, m, k)
                    red = is_avoiding_reduced(word, q, theta, m, k)
                    stats["reduction_checks"] += 1
                    if orig != red:
                        raise AssertionError(
                            f"{name}: reduction mismatch word={word} n={n} m={m} k={k} "
                            f"original={orig} reduced={red}"
                        )
                    if not orig:
                        continue
                    stats["avoiding_words"] += 1
                    counts = window_counter(word, m)
                    frequent = {u for u, c in counts.items() if c >= 2 * d}
                    stats["frequent_checks"] += 1
                    if not is_rc_free(frequent, theta):
                        raise AssertionError(
                            f"{name}: frequent set is not RC-free word={word} n={n} m={m} k={k} "
                            f"frequent={frequent}"
                        )
                    core = extend_to_maximal_rc_free(frequent, universe, theta)
                    outside_occurrences = sum(c for u, c in counts.items() if u not in core)
                    stats["cover_checks"] += 1
                    if outside_occurrences > budget:
                        raise AssertionError(
                            f"{name}: bounded-exception budget exceeded word={word} n={n} m={m} k={k} "
                            f"outside={outside_occurrences} budget={budget}"
                        )
    return stats


def check_one_word(
    name: str,
    word: tuple[int, ...],
    q: int,
    theta: dict[int, int],
    m: int,
    k: int,
) -> None:
    universe = list(product(range(q), repeat=m))
    d = m + k
    budget = (2 * d - 1) * (q**m)
    orig = is_avoiding_original(word, q, theta, m, k)
    red = is_avoiding_reduced(word, q, theta, m, k)
    if orig != red:
        raise AssertionError(
            f"{name}: random reduction mismatch word={word} m={m} k={k} "
            f"original={orig} reduced={red}"
        )
    if not orig:
        return
    counts = window_counter(word, m)
    frequent = {u for u, c in counts.items() if c >= 2 * d}
    if not is_rc_free(frequent, theta):
        raise AssertionError(
            f"{name}: random frequent set is not RC-free word={word} m={m} k={k} "
            f"frequent={frequent}"
        )
    core = extend_to_maximal_rc_free(frequent, universe, theta)
    outside_occurrences = sum(c for u, c in counts.items() if u not in core)
    if outside_occurrences > budget:
        raise AssertionError(
            f"{name}: random bounded-exception budget exceeded word={word} m={m} k={k} "
            f"outside={outside_occurrences} budget={budget}"
        )


def random_stress_case(
    name: str,
    q: int,
    theta: dict[int, int],
    *,
    samples: int,
    max_n: int,
    max_m: int,
    max_k: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    stats = Counter()
    for _ in range(samples):
        n = rng.randint(1, max_n)
        m = rng.randint(1, min(max_m, n))
        k = rng.randint(0, max_k)
        word = tuple(rng.randrange(q) for _ in range(n))
        check_one_word(name, word, q, theta, m, k)
        stats["samples"] += 1
    return dict(stats)


def main() -> None:
    cases = [
        # The identity and swap cases stress self-RC and non-self-RC behavior.
        ("binary_identity", 2, {0: 0, 1: 1}, 8, 3, 3),
        ("binary_swap", 2, {0: 1, 1: 0}, 8, 3, 3),
        # The ternary case mixes one fixed point with one complementary pair.
        ("ternary_one_fixed", 3, {0: 0, 1: 2, 2: 1}, 7, 3, 2),
        # The DNA-pairing case is kept small enough for a fast laptop gate.
        ("dna_pairing", 4, {0: 3, 3: 0, 1: 2, 2: 1}, 6, 2, 2),
    ]
    total = Counter()
    for name, q, theta, max_n, max_m, max_k in cases:
        stats = check_case(name, q, theta, max_n, max_m, max_k)
        total.update(stats)
        print(
            f"PASS {name}: words={stats['words']} avoiding={stats['avoiding_words']} "
            f"reduction={stats['reduction_checks']} frequent={stats['frequent_checks']} "
            f"cover={stats['cover_checks']}",
            flush=True,
        )
    stress_cases = [
        ("stress_binary_swap", 2, {0: 1, 1: 0}, 3000, 18, 5, 6, 2026070101),
        ("stress_ternary_one_fixed", 3, {0: 0, 1: 2, 2: 1}, 3000, 15, 4, 5, 2026070102),
        ("stress_dna_pairing", 4, {0: 3, 3: 0, 1: 2, 2: 1}, 3000, 13, 3, 5, 2026070103),
    ]
    for name, q, theta, samples, max_n, max_m, max_k, seed in stress_cases:
        stats = random_stress_case(
            name,
            q,
            theta,
            samples=samples,
            max_n=max_n,
            max_m=max_m,
            max_k=max_k,
            seed=seed,
        )
        total.update(stats)
        print(
            f"PASS {name}: samples={stats['samples']} seed={seed} "
            f"max_n={max_n} max_m={max_m} max_k={max_k}",
            flush=True,
        )
    print(
        "LOOP_SEPARATED_PROOF_SANITY_PASS: "
        f"words={total['words']} avoiding={total['avoiding_words']} "
        f"reduction={total['reduction_checks']} frequent={total['frequent_checks']} "
        f"cover={total['cover_checks']} random_samples={total['samples']}"
    )


if __name__ == "__main__":
    main()
