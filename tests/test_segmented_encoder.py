from __future__ import annotations

import math
import random
from itertools import product

from loop_alpha_audit.encoder import (
    RIGHT_ADJACENCY,
    RIGHT_ENUMERATOR,
    SegmentedDNAEncoder,
    fixed_base_rank,
    fixed_base_unrank,
    right_sft_count_fast,
)
from loop_alpha_audit.model import is_avoiding_full, is_avoiding_m_windows


def brute_right_words(length: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        word
        for word in product(range(4), repeat=length)
        if all(RIGHT_ADJACENCY[left][right] for left, right in zip(word, word[1:]))
    )


def test_fixed_base_rank_unrank() -> None:
    for base in [2, 3, 4]:
        for length in range(6):
            for index in range(base**length):
                word = fixed_base_unrank(index, base, length)
                assert fixed_base_rank(word, base) == index


def test_right_sft_count_rank_unrank_exhaustive() -> None:
    for length in range(1, 8):
        words = brute_right_words(length)
        assert RIGHT_ENUMERATOR.count(length) == len(words)
        assert right_sft_count_fast(length) == len(words)
        for index, word in enumerate(words):
            assert RIGHT_ENUMERATOR.unrank(index, length) == word
            assert RIGHT_ENUMERATOR.rank(word) == index


def test_fast_right_count_matches_dynamic_count() -> None:
    for length in range(1, 201):
        assert right_sft_count_fast(length) == RIGHT_ENUMERATOR.count(length)


def test_segmented_encoder_exhaustive_small_codebooks() -> None:
    for n in range(4, 11):
        for k in range(n + 1):
            distance = 2 + k
            r = n - 3 - k
            free = 2 * distance - n
            if r < 0 or free < 0 or distance > n:
                continue
            encoder = SegmentedDNAEncoder(n=n, k=k)
            if encoder.codebook_size > 40_000:
                continue
            for index in range(encoder.codebook_size):
                word = encoder.encode_index(index)
                assert encoder.decode_index(word) == index
                assert is_avoiding_full(word, 2, k, (3, 2, 1, 0))


def test_segmented_encoder_random_larger_blocks() -> None:
    rng = random.Random(20260712)
    for alpha in [0.5, 0.6, 0.75, 0.9]:
        for n in [40, 80, 160]:
            k = math.floor(alpha * n)
            encoder = SegmentedDNAEncoder(n=n, k=k)
            for _ in range(12):
                index = rng.randrange(encoder.codebook_size)
                word = encoder.encode_index(index)
                assert encoder.decode_index(word) == index
                assert is_avoiding_m_windows(word, 2, k, (3, 2, 1, 0))


def test_codebook_size_matches_product_construction() -> None:
    encoder = SegmentedDNAEncoder(n=40, k=20)
    assert encoder.endpoint_length == 18
    assert encoder.free_length == 4
    assert encoder.codebook_size == 3**18 * 4**4 * RIGHT_ENUMERATOR.count(18)
