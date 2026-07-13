from __future__ import annotations

from fractions import Fraction

import numpy as np

from loop_alpha_audit.spectral import (
    collatz_rational_upper,
    dna_m2_matrix,
    dna_m2_perp_matrix,
    matrix_to_mask,
    target_lambda_squared_comparison,
)


def numerical_radius(matrix: tuple[tuple[int, ...], ...]) -> float:
    return float(max(abs(np.linalg.eigvals(np.asarray(matrix, dtype=float)))))


def test_mask_matrix_round_trip() -> None:
    for mask in [0, 1, 1911, 4383, 65535]:
        assert matrix_to_mask(dna_m2_matrix(mask)) == mask


def test_perp_is_an_involution_on_subsets() -> None:
    for mask in [0, 1, 17, 1911, 4383, 32768, 65535]:
        matrix = dna_m2_matrix(mask)
        assert dna_m2_perp_matrix(dna_m2_perp_matrix(matrix)) == matrix


def test_collatz_bound_dominates_numerical_radius() -> None:
    for mask in [0, 1, 17, 1911, 4383, 8946, 32768, 65535]:
        for matrix in [dna_m2_matrix(mask), dna_m2_perp_matrix(dna_m2_matrix(mask))]:
            upper = collatz_rational_upper(matrix, steps=50)
            assert float(upper) + 1e-12 >= numerical_radius(matrix)


def test_exact_target_comparison() -> None:
    assert target_lambda_squared_comparison(Fraction(69, 10)) < 0
    assert target_lambda_squared_comparison(Fraction(7, 1)) > 0


def test_canonical_maximizer_matrices() -> None:
    matrix = dna_m2_matrix(1911)
    perpendicular = dna_m2_perp_matrix(matrix)
    assert [sum(row) for row in matrix] == [3, 3, 3, 0]
    assert [sum(row) for row in perpendicular] == [4, 1, 1, 1]
