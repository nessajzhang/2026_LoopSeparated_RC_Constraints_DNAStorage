"""Certified structured high-alpha construction for arbitrary DNA stem threshold."""

from __future__ import annotations

import math
from typing import Sequence

import mpmath as mp
import numpy as np


def linear_recurrence_nth(initial: Sequence[int], coefficients: Sequence[int], n: int) -> int:
    """Return an exact recurrence term by polynomial binary powering.

    ``coefficients=(c_1,...,c_d)`` means
    ``f_n=c_1 f_{n-1}+...+c_d f_{n-d}``.
    """

    init = tuple(int(value) for value in initial)
    coeffs = tuple(int(value) for value in coefficients)
    order = len(coeffs)
    if order < 1 or len(init) != order or n < 0:
        raise ValueError("invalid recurrence")
    if n < order:
        return init[n]

    def combine(left: list[int], right: list[int]) -> list[int]:
        product = [0] * (2 * order - 1)
        for i, left_value in enumerate(left):
            for j, right_value in enumerate(right):
                product[i + j] += left_value * right_value
        for degree in range(2 * order - 2, order - 1, -1):
            value = product[degree]
            if not value:
                continue
            for offset, coefficient in enumerate(coeffs, start=1):
                product[degree - offset] += value * coefficient
        return product[:order]

    result = [1] + [0] * (order - 1)
    if order == 1:
        base = [coeffs[0]]
    else:
        base = [0, 1] + [0] * (order - 2)
    exponent = n
    while exponent:
        if exponent & 1:
            result = combine(result, base)
        base = combine(base, base)
        exponent >>= 1
    return sum(weight * value for weight, value in zip(result, init))


def right_endpoint_count(length: int, m: int) -> int:
    """Count quaternary words in which every length-m window contains A."""

    if length < 0 or m < 1:
        raise ValueError("require length>=0 and m>=1")
    initial = tuple(4**index for index in range(m))
    coefficients = tuple(3**index for index in range(m))
    return linear_recurrence_nth(initial, coefficients, length)


def right_endpoint_radius(m: int) -> float:
    """Return the Perron root of the m-state trailing-non-A automaton."""

    if m < 1:
        raise ValueError("m must be positive")
    companion = np.zeros((m, m), dtype=float)
    companion[0, :] = [3**index for index in range(m)]
    if m > 1:
        companion[1:, :-1] = np.eye(m - 1)
    return float(max(abs(np.linalg.eigvals(companion))))


def right_endpoint_radius_mp(m: int, decimal_places: int = 60) -> mp.mpf:
    """High-precision Perron root, including the exponentially small gap to 4."""

    if m < 1 or decimal_places < 20:
        raise ValueError("require m>=1 and at least 20 decimal places")
    if m <= 3:
        with mp.workdps(decimal_places):
            coefficients = [mp.mpf(1)] + [-(mp.mpf(3) ** index) for index in range(m)]
            roots = mp.polyroots(coefficients, maxsteps=200, error=False)
            return max((root.real for root in roots if abs(root.imag) < mp.eps), default=mp.mpf(0))

    required = max(decimal_places, math.ceil(m * math.log10(4 / 3)) + 30)
    with mp.workdps(required):
        lo = mp.mpf(3) + mp.mpf("1e-8")
        hi = mp.mpf(4)

        def equation(value: mp.mpf) -> mp.mpf:
            return 4 - value - (mp.mpf(3) / value) ** m

        if equation(lo) <= 0 or equation(hi) >= 0:
            raise AssertionError("failed to bracket the Perron root")
        for _ in range(4 * required):
            mid = (lo + hi) / 2
            if equation(mid) > 0:
                lo = mid
            else:
                hi = mid
        return +(lo + hi) / 2


def structured_lambda_lower(m: int) -> float:
    """Certified lower bound 3*rho_m on Lambda_DNA,m."""

    return 3.0 * right_endpoint_radius(m)


def structured_capacity_lower(alpha: float, m: int) -> float:
    if not 0.5 <= alpha < 1:
        raise ValueError("requires 1/2<=alpha<1")
    return 2 * (2 * alpha - 1) + (1 - alpha) * math.log2(structured_lambda_lower(m))


def structured_codebook_size(n: int, k: int, m: int) -> int:
    """Exact size of the left-{A,C,G}, free-middle, right-SFT subcode."""

    if n < 0 or k < 0 or m < 1:
        raise ValueError("invalid parameters")
    distance = m + k
    endpoint_length = n - distance
    free_length = 2 * distance - n
    if endpoint_length < m - 1 or free_length < 0:
        raise ValueError("exact high-alpha endpoint decomposition is not active")
    return 3**endpoint_length * 4**free_length * right_endpoint_count(endpoint_length, m)
