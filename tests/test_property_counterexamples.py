from __future__ import annotations

from typing import Any

from hypothesis import HealthCheck, given, settings, strategies as st

from loop_alpha_audit.model import (
    construct_orbit_interval_cover,
    covered_start_set,
    is_avoiding_full,
    is_avoiding_m_windows,
    marked_letter_set,
    windows,
)
from loop_alpha_audit.paired import decomposition_parameters, endpoint_pair_is_valid


INVOLUTIONS = (
    (1, 0),
    (0, 1),
    (1, 0, 2),
    (0, 2, 1),
    (3, 2, 1, 0),
    (0, 1, 2, 3),
)


@settings(
    max_examples=1_000,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
@given(theta=st.sampled_from(INVOLUTIONS), data=st.data())
def test_randomized_reduction_and_orbit_cover(theta: tuple[int, ...], data: Any) -> None:
    q = len(theta)
    n = data.draw(st.integers(min_value=0, max_value=30), label="n")
    m = data.draw(st.integers(min_value=1, max_value=4), label="m")
    k = data.draw(st.integers(min_value=0, max_value=30), label="k")
    word = tuple(
        data.draw(
            st.lists(st.integers(min_value=0, max_value=q - 1), min_size=n, max_size=n),
            label="word",
        )
    )

    original = is_avoiding_full(word, m, k, theta)
    reduced = is_avoiding_m_windows(word, m, k, theta)
    assert original == reduced
    if not reduced:
        return

    core, covers = construct_orbit_interval_cover(word, m, k, theta)
    covered = covered_start_set(covers)
    for start, block in enumerate(windows(word, m)):
        if block not in core:
            assert start in covered

    marked = marked_letter_set(covers, m, n)
    for start, block in enumerate(windows(word, m)):
        if set(range(start, start + m)).isdisjoint(marked):
            assert block in core


@settings(max_examples=1_000, deadline=None)
@given(theta=st.sampled_from(INVOLUTIONS), data=st.data())
def test_randomized_endpoint_bijection_at_word_level(theta: tuple[int, ...], data: Any) -> None:
    q = len(theta)
    m = data.draw(st.integers(min_value=1, max_value=4), label="m")
    n = data.draw(st.integers(min_value=0, max_value=40), label="n")
    k = data.draw(st.integers(min_value=0, max_value=40), label="k")
    _, distance, r, free_length = decomposition_parameters(n, m, k)
    if r < 0 or free_length < 0 or distance > n:
        return

    word = tuple(
        data.draw(
            st.lists(st.integers(min_value=0, max_value=q - 1), min_size=n, max_size=n),
            label="word",
        )
    )
    endpoint_length = n - distance
    left = word[:endpoint_length]
    right = word[distance:]
    assert len(left) == len(right) == r + m - 1
    assert is_avoiding_full(word, m, k, theta) == endpoint_pair_is_valid(left, right, m, theta)
