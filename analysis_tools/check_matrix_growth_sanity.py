#!/usr/bin/env python3
"""Finite sanity checks for Perron--Frobenius growth estimates.

The checks cover three cases that are easy to mishandle:

1. periodic strongly connected components should still give positive lower
   growth along every residue class when all start/end vertices are counted;
2. nilpotent zero-spectral-radius matrices should become eventually empty;
3. the empirical constants

       max_r 1^T A^r 1 / (rho(A)+epsilon)^r

   should grow at most polynomially in epsilon^{-1} on representative finite
   cases, as expected from the Jordan form.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MatrixCase:
    name: str
    matrix: np.ndarray
    comment: str


def spectral_radius(a: np.ndarray) -> float:
    vals = np.linalg.eigvals(a.astype(float))
    return float(max(abs(v) for v in vals))


def path_count(a: np.ndarray, r: int) -> int:
    if r == 0:
        p = np.eye(a.shape[0], dtype=object)
    else:
        p = np.linalg.matrix_power(a.astype(object), r)
    return int(np.sum(p))


def empirical_k(a: np.ndarray, rho: float, eps: float, max_r: int) -> float:
    return max(path_count(a, r) / ((rho + eps) ** r) for r in range(max_r + 1))


def positive_lower_ratio(a: np.ndarray, rho: float, burn_in: int, max_r: int) -> float:
    ratios = []
    for r in range(burn_in, max_r + 1):
        count = path_count(a, r)
        if count <= 0:
            raise AssertionError(f"zero path count after burn-in at r={r}")
        ratios.append(count / (rho**r))
    return min(ratios)


def nilpotent_cutoff(a: np.ndarray, max_r: int) -> int:
    last_nonzero = -1
    for r in range(max_r + 1):
        if path_count(a, r) != 0:
            last_nonzero = r
    return last_nonzero


def log_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.array([math.log(v) for v in xs], dtype=float)
    y = np.array([math.log(v) for v in ys], dtype=float)
    if len(x) < 2:
        return 0.0
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def main() -> None:
    cases = [
        MatrixCase(
            "primitive_fibonacci",
            np.array([[1, 1], [1, 0]], dtype=int),
            "primitive Perron component",
        ),
        MatrixCase(
            "period3_simple_cycle",
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int),
            "irreducible period-3 component with rho=1",
        ),
        MatrixCase(
            "period3_weighted_cycle",
            np.array([[0, 2, 0], [0, 0, 1], [1, 0, 0]], dtype=int),
            "irreducible period-3 component with rho>1",
        ),
        MatrixCase(
            "reducible_dominant_plus_nilpotent",
            np.array(
                [
                    [1, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                ],
                dtype=int,
            ),
            "dominant primitive block plus acyclic transient block",
        ),
        MatrixCase(
            "nilpotent_upper_shift",
            np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=int),
            "rho=0 acyclic graph",
        ),
    ]

    eps_grid = [2.0 ** (-j) for j in range(1, 13)]
    max_r = 80

    for case in cases:
        a = case.matrix
        rho = spectral_radius(a)
        d = a.shape[0]
        print(f"CASE {case.name}: d={d} rho={rho:.12g} -- {case.comment}")

        if rho == 0.0 or rho < 1e-12:
            cutoff = nilpotent_cutoff(a, max_r)
            if cutoff >= d:
                raise AssertionError(
                    f"{case.name}: nilpotent matrix has nonzero path count at r={cutoff} >= d={d}"
                )
            k_values = [empirical_k(a, rho, eps, max_r) for eps in eps_grid]
            slope = log_slope([1.0 / eps for eps in eps_grid], k_values)
            print(f"  PASS nilpotent_cutoff={cutoff}; empirical_K_log_slope={slope:.3f}")
            continue

        lower = positive_lower_ratio(a, rho, burn_in=d * 3, max_r=max_r)
        if lower <= 0:
            raise AssertionError(f"{case.name}: nonpositive lower ratio")

        k_values = [empirical_k(a, rho, eps, max_r) for eps in eps_grid]
        slope = log_slope([1.0 / eps for eps in eps_grid], k_values)
        # Only polynomial growth is needed for fixed dimension.
        # This loose gate catches exponential-looking blowups without trying to
        # estimate the sharp Jordan exponent.
        if slope > d + 1.0:
            raise AssertionError(
                f"{case.name}: empirical K slope {slope:.3f} exceeds loose polynomial gate d+1={d + 1}"
            )
        print(f"  PASS lower_ratio_min={lower:.6g}; empirical_K_log_slope={slope:.3f}")

    print("MATRIX_GROWTH_SANITY_PASS")


if __name__ == "__main__":
    main()
