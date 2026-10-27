from __future__ import annotations

import copy
import random
from itertools import product

import pytest
import sympy as sp

from loop_alpha_audit.regimes import (
    classify_graph,
    higher_block_core,
    higher_block_factors,
    involution_regime,
    maximal_rc_free_cores,
    overlap_matrix,
    verify_graph_certificate,
)


@pytest.mark.parametrize(
    "matrix,regime",
    [
        ([[0]], "zero"),
        ([[1]], "unit"),
        ([[2]], "positive"),
        ([[0, 1], [0, 0]], "zero"),
        ([[0, 1], [1, 0]], "unit"),
        ([[1, 1], [0, 1]], "unit"),
        ([[0, 2], [1, 0]], "positive"),
        ([[1, 1], [1, 0]], "positive"),
    ],
)
def test_exact_multigraph_regimes(matrix, regime):
    certificate = classify_graph(matrix)
    assert certificate["regime"] == regime
    assert verify_graph_certificate(certificate)


def test_m1_parallel_edges_and_invalid_matrices():
    for q in range(2, 7):
        for count in range(q + 1):
            matrix = overlap_matrix(q, 1, [(i,) for i in range(count)])
            assert matrix == ((count,),)
            assert classify_graph(matrix)["regime_index"] == min(count, 2)
    for bad in ([], [[-1]], [[0.5]], [[1, 0]], [["1"]]):
        with pytest.raises(ValueError):
            classify_graph(bad)


def test_certificate_corruption():
    certificate = classify_graph([[0, 2], [1, 0]])
    bad = copy.deepcopy(certificate)
    bad["branch"]["closed_walks"][1][0][2] = 2
    with pytest.raises(ValueError):
        verify_graph_certificate(bad)
    bad = copy.deepcopy(certificate)
    bad["components"] = [[0], [1]]
    with pytest.raises(ValueError):
        verify_graph_certificate(bad)
    bad = classify_graph([[0, 1], [0, 0]])
    bad["topological_order"].reverse()
    with pytest.raises(ValueError):
        verify_graph_certificate(bad)


def test_independent_random_multigraph_certificates():
    rng = random.Random(20261023)
    for n in range(1, 8):
        for _ in range(30):
            matrix = [
                [rng.randrange(3) if rng.random() < 0.23 else 0 for _ in range(n)] for _ in range(n)
            ]
            assert verify_graph_certificate(classify_graph(matrix))


def test_higher_block_factorization_charpoly_and_short_boundary():
    rng = random.Random(701)
    t = sp.Symbol("t")
    for q, m in ((2, 1), (3, 1), (2, 2), (2, 3), (3, 2)):
        all_words = list(product(range(q), repeat=m))
        for _ in range(8):
            core = {w for w in all_words if rng.randrange(2)}
            lift = higher_block_core(q, m, core)
            p, qq = map(sp.Matrix, higher_block_factors(q, m, core))
            old, new = (
                sp.Matrix(overlap_matrix(q, m, core)),
                sp.Matrix(overlap_matrix(q, m + 1, lift)),
            )
            assert p * qq == old
            assert qq * p == new
            assert (
                sp.expand(
                    new.charpoly(t).as_expr()
                    - t ** (q**m - q ** (m - 1)) * old.charpoly(t).as_expr()
                )
                == 0
            )
            # Length m is deliberately not asserted equal: lifted constraints
            # are vacuous there, whereas original length-m constraints are not.
            for n in (m + 1, m + 2):
                for word in product(range(q), repeat=n):
                    old_ok = all(word[i : i + m] in core for i in range(n - m + 1))
                    new_ok = all(word[i : i + m + 1] in lift for i in range(n - m))
                    assert old_ok == new_ok
    assert len(higher_block_core(2, 1, {(0,)})) == 1


def test_rc_free_lifting_and_general_classification():
    for theta in ((0, 1), (1, 0), (1, 0, 2), (0, 1, 2)):
        q = len(theta)
        for m in (1, 2):
            for _, core in maximal_rc_free_cores(q, m, theta):
                lift = higher_block_core(q, m, core)
                assert all(tuple(theta[a] for a in w[::-1]) not in lift for w in lift)
                assert (
                    classify_graph(overlap_matrix(q, m, core))["regime"]
                    == classify_graph(overlap_matrix(q, m + 1, lift))["regime"]
                )
    assert [involution_regime((0, 1), m)["regime"] for m in range(1, 8)] == ["zero"] * 4 + [
        "unit",
        "positive",
        "positive",
    ]
    assert involution_regime((1, 0, 3, 2), 1)["exact_radius"] == 2
    assert involution_regime((0, 1, 2), 2)["regime"] == "unit"
