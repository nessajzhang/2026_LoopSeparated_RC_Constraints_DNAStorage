from __future__ import annotations

from itertools import product
import random

import pytest

from loop_alpha_audit.encoder import cache_info, clear_caches
from loop_alpha_audit.model import is_avoiding_full
from loop_alpha_audit.prepared_codec import DNA, prepare
from loop_alpha_audit.short_loop_codec import (
    GCEnumerator,
    MembershipEnumerator,
    PAIR_TO_DNA,
    avoids_rc_windows,
    full_oligo_failure,
)


def majority(bits, w):
    return all(sum(bits[i : i + w]) > w // 2 for i in range(len(bits) - w + 1))


def runs(bits, limit=4):
    return all(
        not all(bit == bits[i] for bit in bits[i : i + limit + 1]) for i in range(len(bits) - limit)
    )


def test_binary_components_against_full_lexical_enumeration():
    for prefix in ((), (1, 0, 1), (1, 1, 1, 1)):
        for length in range(9):
            words = list(product((0, 1), repeat=length))
            for w in (5, 7):
                enumerator = MembershipEnumerator(length, prefix, w)
                expected = [word for word in words if majority(prefix + word, w)]
                assert enumerator.M == len(expected)
                for index, word in enumerate(expected):
                    assert enumerator.unrank(index) == word
                    assert enumerator.rank(word) == index
            target = (len(prefix) + length) // 2
            enumerator = GCEnumerator(length, prefix, target)
            expected = [
                word for word in words if sum(prefix + word) == target and runs(prefix + word)
            ]
            assert enumerator.M == len(expected)
            for index, word in enumerate(expected):
                assert enumerator.unrank(index) == word
                assert enumerator.rank(word) == index


@pytest.mark.parametrize("m", [6, 7])
def test_small_product_codebooks_exhaustive_and_independent_order(m):
    w = 5 if m == 6 else 7
    for n in range(8):
        codec = prepare(dict(kind="short_core", n=n, m=m, k=0))
        members = [word for word in product((0, 1), repeat=n) if majority(word, w)]
        gc_words = [
            word for word in product((0, 1), repeat=n) if sum(word) == n // 2 and runs(word)
        ]
        assert codec.M == len(members) * len(gc_words)
        expected = (
            "".join(PAIR_TO_DNA[pair] for pair in zip(member, gc, strict=True))
            for member in members
            for gc in gc_words
        )
        for index, word in enumerate(expected):
            if index >= 1 << codec.K:
                with pytest.raises(ValueError, match="unused_binary_rank"):
                    codec.decode_index(word)
                continue
            assert codec.encode_index(index) == word
            assert codec.decode_index(word) == index
            assert is_avoiding_full(tuple(DNA.index(base) for base in word), m, 0, (3, 2, 1, 0))


def test_rolling_checker_against_original_all_stem_definition():
    for n in range(7):
        for word in product(range(4), repeat=n):
            sequence = "".join(DNA[symbol] for symbol in word)
            for m, k in ((1, 0), (1, 2), (2, 0), (2, 1), (3, 0)):
                assert avoids_rc_windows(sequence, m, k) == is_avoiding_full(
                    word, m, k, (3, 2, 1, 0)
                )


@pytest.mark.parametrize("seed", [20261023, 20261024, 20261025])
def test_real_short_core_messages_full_prefix_constraints(seed):
    rng = random.Random(seed)
    for m in (6, 7):
        for k in (0, 3, 4, 5):
            codec = prepare(dict(kind="short_core", n=150, m=m, k=k, prefix="TCTCTTCC"))
            assert codec.metadata["core_window"] == (5 if m == 6 else 7)
            for index in (0, (1 << codec.K) - 1, *[rng.randrange(1 << codec.K) for _ in range(12)]):
                sequence = codec.encode_index(index)
                assert sequence.startswith(codec.prefix) and len(sequence) == 150
                assert sum(base in "GC" for base in sequence) == 75
                assert full_oligo_failure(sequence, m, k) is None
                assert codec.decode_index(sequence) == index


def test_address_prefixes_share_large_tables_and_capacity():
    # Codec-state check only; public address edit distance belongs to framing.
    rng = random.Random(20261023)
    addresses = set()
    while len(addresses) < 63:
        bases = ["C"] * 12 + ["T"] * 10
        rng.shuffle(bases)
        address = "".join(bases) + "TT"
        if runs(tuple(int(base == "C") for base in address)):
            addresses.add(address)
    prefixes = [address + "CC" for address in sorted(addresses)]
    clear_caches()
    codecs = [
        prepare(dict(kind="short_core", n=150, m=7, k=3, prefix=prefix)) for prefix in prefixes
    ]
    assert len(codecs) == 63 and codecs[0].K == 210
    assert len({codec.K for codec in codecs}) == 1
    assert len({id(codec._membership.table) for codec in codecs}) == 1
    assert len({id(codec._gc.table) for codec in codecs}) == 1
    assert cache_info()["misses"] == 2
    for index, codec in enumerate(codecs):
        sequence = codec.encode_index(index)
        assert sequence.startswith(codec.prefix)
        assert codec.decode_index(sequence) == index


@pytest.mark.parametrize("kind", ["nonce", "gc_h_nonce"])
@pytest.mark.parametrize("budget", [64, 256])
def test_nonce_actual_random_bits_and_explicit_candidate_counts(kind, budget):
    for seed in (20261023, 20261024, 20261025):
        rng = random.Random(seed)
        config = dict(
            kind=kind, n=150, m=7, k=3, prefix="TCTCTTAT", nonce_nt=4, nonce_budget=budget
        )
        codec = prepare(config)
        assert config["nonce_budget"] == budget and codec.M is None
        assert codec.K == 2 * (150 - len(codec.prefix) - 4)
        for index in (0, (1 << codec.K) - 1, *[rng.randrange(1 << codec.K) for _ in range(20)]):
            bits = format(index, f"0{codec.K}b")
            result = codec.encode_bits(bits)
            assert 1 <= result.attempts <= budget
            if result.status == "ENCODING_FAILED":
                assert result.sequence is None and result.attempts == budget
                continue
            assert result.status == "OK"
            assert full_oligo_failure(result.sequence, 7, 3, check_rc=kind == "nonce") is None
            assert codec.decode_bits(result.sequence).bits == bits


def test_candidate_exhaustion_is_not_hidden_payload_redraw():
    codec = prepare(dict(kind="nonce", n=12, m=2, k=0, prefix="", nonce_nt=1, nonce_budget=1))
    failures = []
    successes = []
    for index in range(256):
        bits = format(index, f"0{codec.K}b")
        result = codec.encode_bits(bits)
        if result.status == "ENCODING_FAILED":
            failures.append(result)
            assert result.attempts == 1 and result.failure_reason == "candidate_budget_exhausted"
        else:
            successes.append(result)
            assert codec.decode_bits(result.sequence).bits == bits
    assert failures


def test_scrambling_changes_entire_variable_body_not_only_a_trailer():
    codec = prepare(dict(kind="nonce", n=60, m=7, k=3, nonce_nt=4, nonce_budget=256))
    masks = [codec._mask(nonce) for nonce in range(16)]
    assert len(set(masks)) == 16
    assert len({mask >> (codec.K - 16) for mask in masks}) > 1
    assert len({mask & 65535 for mask in masks}) > 1
