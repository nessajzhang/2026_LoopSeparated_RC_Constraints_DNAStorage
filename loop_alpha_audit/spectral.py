"""Exact helpers for the DNA m=2 paired spectral maximization."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence

Matrix = tuple[tuple[int, ...], ...]


def dna_m2_matrix(mask: int) -> Matrix:
    if not 0 <= mask < 2**16:
        raise ValueError("a DNA m=2 subset mask must be in [0,2^16)")
    return tuple(tuple((mask >> (4 * row + column)) & 1 for column in range(4)) for row in range(4))


def dna_m2_perp_matrix(matrix: Sequence[Sequence[int]]) -> Matrix:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("DNA m=2 matrices must be 4 by 4")
    complement = (3, 2, 1, 0)
    result = [[1] * 4 for _ in range(4)]
    for left in range(4):
        for right in range(4):
            if matrix[left][right]:
                rc_left = complement[right]
                rc_right = complement[left]
                result[rc_left][rc_right] = 0
    return tuple(tuple(row) for row in result)


def matrix_to_mask(matrix: Sequence[Sequence[int]]) -> int:
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("matrix must be 4 by 4")
    mask = 0
    for row in range(4):
        for column in range(4):
            value = int(matrix[row][column])
            if value not in (0, 1):
                raise ValueError("matrix entries must be binary")
            mask |= value << (4 * row + column)
    return mask


def matrix_vector_product(
    matrix: Sequence[Sequence[int]], vector: Sequence[int]
) -> tuple[int, ...]:
    if len(matrix) != len(vector) or any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix and vector dimensions do not agree")
    return tuple(sum(int(value) * int(vector[j]) for j, value in enumerate(row)) for row in matrix)


def collatz_rational_upper(matrix: Sequence[Sequence[int]], steps: int = 50) -> Fraction:
    """Return an exact Collatz--Wielandt upper bound for the spectral radius.

    The positive integer vector is `(I+M)^steps 1`.  Since `Mx <= u x`
    coordinatewise for `u=max_i (Mx)_i/x_i`, Perron--Frobenius gives
    `rho(M)<=u` even when `M` is reducible or nilpotent.
    """

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("matrix must be nonempty and square")
    if any(int(value) < 0 for row in matrix for value in row):
        raise ValueError("matrix must be nonnegative")

    shifted = tuple(
        tuple(int(matrix[i][j]) + int(i == j) for j in range(size)) for i in range(size)
    )
    vector = (1,) * size
    for _ in range(steps):
        vector = matrix_vector_product(shifted, vector)
    image = matrix_vector_product(matrix, vector)
    return max(Fraction(image[i], vector[i]) for i in range(size))


def target_lambda_squared_comparison(bound: Fraction) -> int:
    """Compare a rational bound with 3(1+sqrt(13))/2 exactly.

    Returns `-1`, `0`, or `1`.  Equality cannot occur for a rational input,
    but the three-way interface prevents accidental use of floating point.
    """

    if bound < 0:
        return -1
    shifted = 2 * bound - 3
    if shifted <= 0:
        return -1
    square = shifted * shifted
    if square < 117:
        return -1
    if square > 117:
        return 1
    return 0


def target_lambda_decimal() -> str:
    return "3(1+sqrt(13))/2"
