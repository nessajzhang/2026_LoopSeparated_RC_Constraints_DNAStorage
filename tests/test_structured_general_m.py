from __future__ import annotations

from itertools import product

import mpmath as mp
import pytest

from loop_alpha_audit.structured_general_m import (
    linear_recurrence_nth,
    right_endpoint_count,
    right_endpoint_radius,
    right_endpoint_radius_mp,
    structured_codebook_size,
)


def brute_right_count(length: int, m: int) -> int:
    return sum(
        all(0 in word[start : start + m] for start in range(length - m + 1))
        for word in product(range(4), repeat=length)
    )


def transfer_right_count(length: int, m: int) -> int:
    states = [0] * m
    states[0] = 1
    for _ in range(length):
        next_states = [0] * m
        next_states[0] = sum(states)
        for run in range(m - 1):
            next_states[run + 1] = 3 * states[run]
        states = next_states
    return sum(states)


def test_linear_recurrence_matches_direct_iteration() -> None:
    initial = (1, 4, 16)
    coefficients = (1, 3, 9)
    values = list(initial)
    for _ in range(3, 50):
        values.append(sum(c * value for c, value in zip(coefficients, values[-1:-4:-1])))
    for n, expected in enumerate(values):
        assert linear_recurrence_nth(initial, coefficients, n) == expected


@pytest.mark.parametrize("m", range(1, 6))
def test_right_endpoint_count_against_bruteforce(m: int) -> None:
    for length in range(0, 8):
        assert right_endpoint_count(length, m) == brute_right_count(length, m)


def test_recurrence_against_independent_transfer_states() -> None:
    for m in range(1, 33):
        for length in (0, 1, 2, 3, 5, 8, 13, 21, 40, 75):
            assert right_endpoint_count(length, m) == transfer_right_count(length, m)


def test_m2_matches_previous_recurrence() -> None:
    values = [right_endpoint_count(length, 2) for length in range(1, 20)]
    assert values[:2] == [4, 7]
    assert all(
        current == previous + 3 * older
        for older, previous, current in zip(values, values[1:], values[2:])
    )


def test_radius_satisfies_characteristic_equation() -> None:
    for m in range(1, 17):
        radius = right_endpoint_radius(m)
        right = sum(3**index * radius ** (m - 1 - index) for index in range(m))
        assert abs(radius**m - right) <= 1e-8 * radius**m


def test_high_precision_radius_matches_float_and_resolves_large_m_gap() -> None:
    for m in range(1, 17):
        assert abs(float(right_endpoint_radius_mp(m)) - right_endpoint_radius(m)) < 1e-10
    radius_1024 = right_endpoint_radius_mp(1024, decimal_places=80)
    assert 3 < radius_1024 < 4
    assert mp.mpf(0) < 4 - radius_1024 < mp.mpf("1e-100")


def test_structured_codebook_small_identity() -> None:
    n, k, m = 12, 6, 3
    endpoint = n - m - k
    free = 2 * (m + k) - n
    assert structured_codebook_size(n, k, m) == (
        3**endpoint * 4**free * right_endpoint_count(endpoint, m)
    )
