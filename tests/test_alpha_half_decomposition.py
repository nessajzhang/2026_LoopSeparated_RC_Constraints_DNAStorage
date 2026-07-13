from __future__ import annotations

from itertools import product

import pytest

from loop_alpha_audit.paired import (
    brute_force_language_count,
    decomposed_language_count,
    decomposition_parameters,
    endpoint_pair_is_valid,
    paired_count_bruteforce,
    paired_count_dynamic,
)


CASES = (
    ((1, 0), 1, 10),
    ((1, 0), 2, 10),
    ((0, 1), 1, 10),
    ((0, 1), 2, 10),
    ((1, 0, 2), 1, 8),
    ((1, 0, 2), 2, 8),
    ((3, 2, 1, 0), 1, 7),
    ((3, 2, 1, 0), 2, 7),
)


@pytest.mark.parametrize(("theta", "m", "n_max"), CASES)
def test_exact_endpoint_decomposition_against_full_bruteforce(
    theta: tuple[int, ...], m: int, n_max: int
) -> None:
    q = len(theta)
    for n in range(max(0, 2 * m - 2), n_max + 1):
        for k in range(n + 1):
            _, _, r, free_length = decomposition_parameters(n, m, k)
            if r < 0 or free_length < 0:
                continue
            original = brute_force_language_count(q, n, m, k, theta, original_definition=True)
            reduced = brute_force_language_count(q, n, m, k, theta, original_definition=False)
            decomposed = decomposed_language_count(q, n, m, k, theta)
            assert original == reduced == decomposed


@pytest.mark.parametrize("theta", [(1, 0), (0, 1), (1, 0, 2), (3, 2, 1, 0)])
@pytest.mark.parametrize("m", [1, 2])
def test_paired_dynamic_program_against_pair_bruteforce(theta: tuple[int, ...], m: int) -> None:
    q = len(theta)
    max_r = 4 if q == 2 else 2
    for r in range(max_r + 1):
        assert paired_count_dynamic(q, r, m, theta) == paired_count_bruteforce(q, r, m, theta)


def test_r_zero_is_vacuous() -> None:
    theta = (3, 2, 1, 0)
    q = 4
    m = 2
    for n in range(2, 9):
        k = n - 2 * m + 1
        if k < 0:
            continue
        _, _, r, free_length = decomposition_parameters(n, m, k)
        assert r == 0
        assert free_length == n - 2 * (m - 1)
        assert decomposed_language_count(q, n, m, k, theta) == q**n


def test_endpoint_validity_matches_triangular_rule() -> None:
    theta = (1, 0)
    m = 1
    left = (0, 0, 0)
    assert endpoint_pair_is_valid(left, (0, 0, 0), m, theta)
    assert not endpoint_pair_is_valid(left, (1, 0, 0), m, theta)


def test_every_small_endpoint_pair_is_classified() -> None:
    theta = (1, 0)
    for m in [1, 2]:
        for r in range(4):
            endpoint_length = r + m - 1
            words = tuple(product(range(2), repeat=endpoint_length))
            classified = sum(
                endpoint_pair_is_valid(left, right, m, theta) for left in words for right in words
            )
            assert classified == paired_count_bruteforce(2, r, m, theta)
