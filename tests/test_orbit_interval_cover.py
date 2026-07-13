from __future__ import annotations

from itertools import product

import pytest

from loop_alpha_audit.model import (
    construct_orbit_interval_cover,
    covered_start_set,
    is_avoiding_full,
    is_avoiding_m_windows,
    is_maximal_rc_free,
    marked_letter_set,
    rc_orbits,
    reverse_complement,
    windows,
)


INVOLUTIONS = (
    (1, 0),  # binary swap
    (0, 1),  # binary identity: all m=1 symbols are self-RC
    (1, 0, 2),  # one fixed point and one complementary pair
    (3, 2, 1, 0),  # DNA encoding A,C,G,T -> T,G,C,A
)


@pytest.mark.parametrize("theta", INVOLUTIONS[:3])
@pytest.mark.parametrize("m", [1, 2])
def test_stem_reduction_exhaustive(theta: tuple[int, ...], m: int) -> None:
    q = len(theta)
    for n in range(0, 8):
        for k in range(0, max(1, n + 1)):
            for word in product(range(q), repeat=n):
                assert is_avoiding_full(word, m, k, theta) == is_avoiding_m_windows(
                    word, m, k, theta
                )


@pytest.mark.parametrize(
    ("theta", "m", "n_max"),
    [
        ((1, 0), 1, 9),
        ((1, 0), 2, 9),
        ((0, 1), 1, 9),
        ((0, 1), 2, 9),
        ((1, 0, 2), 1, 8),
        ((1, 0, 2), 2, 8),
        ((3, 2, 1, 0), 1, 7),
        ((3, 2, 1, 0), 2, 7),
    ],
)
def test_orbit_interval_cover_exhaustive(theta: tuple[int, ...], m: int, n_max: int) -> None:
    q = len(theta)
    orbit_count = len(rc_orbits(q, m, theta))
    for n in range(0, n_max + 1):
        blocks_count = max(0, n - m + 1)
        for k in range(0, max(1, n + 1)):
            distance = m + k
            for word in product(range(q), repeat=n):
                if not is_avoiding_m_windows(word, m, k, theta):
                    continue
                core, covers = construct_orbit_interval_cover(word, m, k, theta)
                assert is_maximal_rc_free(core, q, m, theta)
                assert len(covers) <= orbit_count
                assert all(cover.start_length <= 2 * distance - 1 for cover in covers)

                covered = covered_start_set(covers)
                for start, block in enumerate(windows(word, m)):
                    if block not in core:
                        assert start in covered

                marked = marked_letter_set(covers, m, n)
                assert len(marked) <= orbit_count * (2 * distance + m - 2)
                for start, block in enumerate(windows(word, m)):
                    coordinates = set(range(start, start + m))
                    if coordinates.isdisjoint(marked):
                        assert block in core

                if blocks_count == 0:
                    assert not covers


def test_self_rc_occurrences_are_clustered() -> None:
    theta = (0, 1)
    word = (0, 1, 0, 1)
    m = 1
    k = 2
    assert is_avoiding_m_windows(word, m, k, theta)
    core, covers = construct_orbit_interval_cover(word, m, k, theta)
    assert not core
    assert {cover.noncore for cover in covers} == {(0,), (1,)}


def test_nonavoiding_word_can_break_anchor_interval() -> None:
    theta = (1, 0)
    word = (0, 0, 0, 1)
    assert not is_avoiding_m_windows(word, m=1, k=0, theta=theta)
    with pytest.raises(ValueError):
        construct_orbit_interval_cover(word, m=1, k=0, theta=theta)


def test_reverse_complement_is_an_involution() -> None:
    for theta in INVOLUTIONS:
        q = len(theta)
        for m in range(1, 5):
            for word in product(range(q), repeat=m):
                assert reverse_complement(reverse_complement(word, theta), theta) == word
