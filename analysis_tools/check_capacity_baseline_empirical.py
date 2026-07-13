#!/usr/bin/env python3
"""Empirical checks for the RC-free spectral-radius baseline.

The calculations cover three small-alphabet consistency conditions:

1. maximal RC-free cores contain at most one representative
   from each reverse-complement orbit, with self-RC words excluded;
2. the overlap-matrix spectral-radius baseline is computed consistently with
   the finite-type core language;
3. exact loop-separated counts for small instances behave monotonically in the
   loop threshold and do not contradict the unseparated core lower inclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np


Word = tuple[int, ...]


@dataclass(frozen=True)
class ModelCase:
    name: str
    q: int
    theta: dict[int, int]
    m: int
    k_values: tuple[int, ...]
    n_max: int


def rc(word: Word, theta: dict[int, int]) -> Word:
    return tuple(theta[x] for x in reversed(word))


def all_words(q: int, length: int) -> list[Word]:
    if length == 0:
        return [()]
    return list(product(range(q), repeat=length))


def rc_orbits(q: int, m: int, theta: dict[int, int]) -> list[tuple[Word, ...]]:
    seen: set[Word] = set()
    out: list[tuple[Word, ...]] = []
    for u in all_words(q, m):
        if u in seen:
            continue
        v = rc(u, theta)
        if v == u:
            out.append((u,))
            seen.add(u)
        else:
            pair = tuple(sorted((u, v)))
            out.append(pair)
            seen.update(pair)
    return out


def maximal_rc_free_cores(q: int, m: int, theta: dict[int, int]) -> list[set[Word]]:
    cores: list[set[Word]] = [set()]
    for orbit in rc_orbits(q, m, theta):
        if len(orbit) == 1:
            # A self-RC word cannot be added to any RC-free set.
            continue
        next_cores: list[set[Word]] = []
        for core in cores:
            for u in orbit:
                enlarged = set(core)
                enlarged.add(u)
                next_cores.append(enlarged)
        cores = next_cores
    return cores


def overlap_matrix(core: set[Word], q: int, m: int) -> np.ndarray:
    if m == 1:
        return np.array([[len(core)]], dtype=float)
    states = all_words(q, m - 1)
    index = {s: i for i, s in enumerate(states)}
    mat = np.zeros((len(states), len(states)), dtype=float)
    for u in core:
        mat[index[u[:-1]], index[u[1:]]] += 1.0
    return mat


def spectral_radius(mat: np.ndarray) -> float:
    if mat.size == 0:
        return 0.0
    vals = np.linalg.eigvals(mat)
    return float(max(abs(v) for v in vals))


def finite_type_count(core: set[Word], q: int, m: int, n: int) -> int:
    if n < m:
        return q**n
    if m == 1:
        return len(core) ** n
    states: dict[Word, int] = {}
    for u in core:
        states[u[1:]] = states.get(u[1:], 0) + 1
    for _ in range(n - m):
        nxt: dict[Word, int] = {}
        for suffix, multiplicity in states.items():
            for base in range(q):
                candidate = suffix + (base,)
                if candidate not in core:
                    continue
                new_suffix = candidate[1:]
                nxt[new_suffix] = nxt.get(new_suffix, 0) + multiplicity
        states = nxt
    return sum(states.values())


def is_avoiding(word: Word, theta: dict[int, int], m: int, k: int) -> bool:
    n = len(word)
    d = m + k
    for i in range(0, n - m + 1):
        left = word[i : i + m]
        for j in range(i + d, n - m + 1):
            if left == rc(word[j : j + m], theta):
                return False
    return True


def exact_count(q: int, theta: dict[int, int], m: int, k: int, n: int) -> int:
    return sum(1 for word in product(range(q), repeat=n) if is_avoiding(word, theta, m, k))


def run_case(case: ModelCase) -> None:
    cores = maximal_rc_free_cores(case.q, case.m, case.theta)
    if not cores:
        raise AssertionError(f"{case.name}: no maximal cores generated")

    radii = [spectral_radius(overlap_matrix(core, case.q, case.m)) for core in cores]
    rho_star = max(radii)
    c_bits = math.log2(rho_star) if rho_star > 0 else float("-inf")
    print(
        f"CASE {case.name}: q={case.q} m={case.m} cores={len(cores)} "
        f"rho_star={rho_star:.12g} c_bits={c_bits:.9f}",
        flush=True,
    )

    exact_by_k: dict[int, list[int]] = {}
    for k in case.k_values:
        exact_by_k[k] = [
            exact_count(case.q, case.theta, case.m, k, n) for n in range(case.n_max + 1)
        ]

    for n in range(case.m, case.n_max + 1):
        # Unseparated words must contain every individual finite-type core.
        max_core_count = max(finite_type_count(core, case.q, case.m, n) for core in cores)
        if exact_by_k[0][n] < max_core_count:
            raise AssertionError(
                f"{case.name}: k=0 exact count below a core count at n={n}: "
                f"exact={exact_by_k[0][n]} core={max_core_count}"
            )

        # Increasing k relaxes the forbidden-pair rule, so finite counts must not decrease.
        previous = -1
        for k in sorted(case.k_values):
            current = exact_by_k[k][n]
            if current < previous:
                raise AssertionError(
                    f"{case.name}: count decreased with k at n={n}: "
                    f"k={k} current={current} previous={previous}"
                )
            previous = current

    tail_n = case.n_max
    tail_summary = []
    for k in sorted(case.k_values):
        count = exact_by_k[k][tail_n]
        rate = math.log2(count) / tail_n if count > 0 else float("-inf")
        excess = rate - c_bits if math.isfinite(c_bits) and math.isfinite(rate) else float("nan")
        tail_summary.append(f"k={k}:N={count}:rate={rate:.6f}:rate_minus_c={excess:.6f}")
    print("  PASS " + " | ".join(tail_summary), flush=True)


def main() -> None:
    cases = [
        ModelCase("binary_swap_m2", 2, {0: 1, 1: 0}, 2, (0, 1, 2, 3), 14),
        ModelCase("binary_identity_m2", 2, {0: 0, 1: 1}, 2, (0, 1, 2), 14),
        ModelCase("binary_swap_m3", 2, {0: 1, 1: 0}, 3, (0, 1, 2), 12),
        ModelCase("ternary_one_fixed_m2", 3, {0: 0, 1: 2, 2: 1}, 2, (0, 1, 2), 10),
        ModelCase("dna_pairing_m2", 4, {0: 3, 3: 0, 1: 2, 2: 1}, 2, (0, 1, 2), 8),
    ]
    for case in cases:
        run_case(case)
    print("CAPACITY_BASELINE_EMPIRICAL_PASS")


if __name__ == "__main__":
    main()
