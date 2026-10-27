from __future__ import annotations

import random
from itertools import product

import pytest

from loop_alpha_audit.encoder import SegmentedDNAEncoder
from loop_alpha_audit.endpoint_general import EndpointMapper, PreparedWindowSFT
from loop_alpha_audit.model import is_avoiding_full, is_avoiding_m_windows
from loop_alpha_audit.structured_general_m import structured_codebook_size


def test_all_window_tables_exhaustive_small():
    for m in range(1, 4):
        windows = tuple(product(range(2), repeat=m))
        for mask in range(1 << len(windows)):
            allowed = {w for i, w in enumerate(windows) if mask >> i & 1}
            for n in range(5):
                prepared = PreparedWindowSFT(2, m, allowed, n)
                words = [
                    w
                    for w in product(range(2), repeat=n)
                    if all(w[i : i + m] in allowed for i in range(n - m + 1))
                ]
                assert prepared.codebook_size == len(words)
                for rank, word in enumerate(words):
                    assert prepared.unrank(rank) == word
                    assert prepared.rank(word) == rank


def test_m3_all_binary_window_sets_and_small_product_codebooks():
    windows = tuple(product(range(2), repeat=3))
    for mask in range(256):
        allowed = {w for i, w in enumerate(windows) if mask >> i & 1}
        for n, k in ((6, 0), (7, 1), (8, 1)):
            mapper = EndpointMapper(n, 3, k, allowed, q=2, theta=(1, 0))
            for rank in range(mapper.codebook_size):
                word = mapper.encode_index(rank)
                assert mapper.decode_index(word) == rank
                assert is_avoiding_full(word, 3, k, (1, 0))


def test_m2_matches_original_certified_encoder_when_all_window_active():
    allowed = set(product(range(3), repeat=2))
    for n in range(4, 11):
        for k in range(n):
            ell, free = n - 2 - k, 2 * (2 + k) - n
            if ell < 2 or free < 0:
                continue
            new, old = EndpointMapper(n, 2, k, allowed), SegmentedDNAEncoder(n, k)
            assert new.codebook_size == old.codebook_size
            if old.codebook_size > 5000:
                continue
            for rank in range(old.codebook_size):
                word = old.encode_index(rank)
                assert new.encode_index(rank) == word
                assert new.decode_index(word) == rank


def test_dna_m3_full_definition_small_and_exact_structured_counts():
    allowed = set(product(range(3), repeat=3))
    for n, k in ((6, 0), (7, 1), (8, 1)):
        mapper = EndpointMapper(n, 3, k, allowed)
        assert mapper.codebook_size == structured_codebook_size(n, k, 3)
        for rank in range(mapper.codebook_size):
            word = mapper.encode_index(rank)
            assert mapper.decode_index(word) == rank
            assert is_avoiding_full(word, 3, k, (3, 2, 1, 0))


def test_binary_boundary_unused_ranks_invalid_symbols_and_empty_codebook():
    mapper = EndpointMapper(6, 3, 0, product(range(3), repeat=3))
    assert mapper.codebook_size == 999
    assert mapper.K == 9
    for payload in (0, 1, (1 << mapper.K) - 1):
        word = mapper.encode_payload(payload)
        assert mapper.decode_payload(word) == payload
        assert mapper.encode_bits(mapper.decode_bits(word)) == word
    for payload in (-1, 1 << mapper.K):
        with pytest.raises(ValueError):
            mapper.encode_payload(payload)
    assert mapper.decode_index(mapper.encode_index(512)) == 512
    with pytest.raises(ValueError, match="unused rank"):
        mapper.decode_payload(mapper.encode_index(512))
    for bad in ((0,) * 5, (4,) * 6, (0.2,) * 6):
        with pytest.raises(ValueError):
            mapper.decode_index(bad)
    empty = EndpointMapper(6, 3, 0, [])
    assert empty.codebook_size == 0
    with pytest.raises(ValueError):
        _ = empty.payload_bits
    with pytest.raises(ValueError):
        empty.encode_index(0)
    with pytest.raises(ValueError):
        EndpointMapper(100, 3, 3, product(range(3), repeat=3))


def test_vacuity_all_lengths_and_g_zero():
    for n in range(8):
        mapper = EndpointMapper(n, 3, 8, [])
        assert mapper.regime == "vacuous"
        assert mapper.codebook_size == 4**n
        for rank in (0, 4**n - 1):
            assert mapper.decode_index(mapper.encode_index(rank)) == rank
    mapper = EndpointMapper(8, 3, 1, product(range(3), repeat=3))
    assert mapper.free_length == 0
    assert mapper.endpoint_length == 4


def test_seeded_large_roundtrips_original_definition_and_exact_k():
    allowed = set(product(range(3), repeat=3))
    for seed in (20261023, 71, 193):
        rng = random.Random(seed)
        for n in (100, 150, 200, 300):
            for alpha in (0.5, 0.6, 0.75, 0.9, 1.0):
                k = int(alpha * n)
                mapper = EndpointMapper(n, 3, k, allowed)
                if mapper.regime != "vacuous":
                    assert mapper.codebook_size == structured_codebook_size(n, k, 3)
                assert 1 << mapper.K <= mapper.codebook_size < 1 << (mapper.K + 1)
                for payload in (
                    0,
                    (1 << mapper.K) - 1,
                    *(rng.randrange(1 << mapper.K) for _ in range(3)),
                ):
                    word = mapper.encode_payload(payload)
                    assert mapper.decode_payload(word) == payload
                    assert is_avoiding_m_windows(word, 3, k, (3, 2, 1, 0))
                    # The original definition independently checks all stem
                    # lengths for boundary payloads, not just a reduced test.
                    if payload == 0:
                        assert is_avoiding_full(word, 3, k, (3, 2, 1, 0))
