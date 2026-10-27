from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import product
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from loop_alpha_audit import encoder
from loop_alpha_audit.encoder import EXACT_TABLE_CACHE, SFTEnumerator, cache_info, clear_caches
from loop_alpha_audit.prepared_codec import DNA, prepare
from loop_alpha_audit.short_loop_codec import avoids_rc_windows


def reference_encoder():
    source = (
        Path(__file__).resolve().parents[2]
        / "original/code_archive_extract/2026_LoopSeparated_RC_Constraints_DNAStorage-318a65699470a29e37332ca5fa0bdb03578d7c7d/loop_alpha_audit/encoder.py"
    )
    if not source.is_file():
        # Standalone public code packages need no private workspace. This
        # uncached reference retains the pinned baseline's exact recurrence
        # and mixed-radix order; the author audit additionally loads the actual
        # immutable baseline module whenever available.
        return SimpleNamespace(SFTEnumerator=ReferenceSFT, SegmentedDNAEncoder=ReferenceSegmented)
    raw = source.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == "3e8a5b2b8e4bd6605345751f73a6cbcc94fe5ba3"
    )
    name = "loop_alpha_audit._frozen_reference_encoder"
    module = ModuleType(name)
    module.__file__ = str(source)
    module.__package__ = "loop_alpha_audit"
    sys.modules[name] = module
    # Reading and compiling explicitly avoids SourceFileLoader's automatic
    # .pyc write into the immutable archive extraction.
    exec(compile(raw, str(source), "exec"), module.__dict__)
    return module


class ReferenceSFT:
    def __init__(self, graph):
        self.graph = graph

    def suffix_counts(self, length):
        rows = [(1,) * len(self.graph)]
        for _ in range(length - 1):
            rows.append(
                tuple(
                    sum(edge * previous for edge, previous in zip(row, rows[-1], strict=True))
                    for row in self.graph
                )
            )
        return tuple(rows)

    def count(self, length):
        return sum(self.suffix_counts(length)[-1])

    def unrank(self, index, length):
        counts = self.suffix_counts(length)
        result = []
        for remaining in range(length - 1, -1, -1):
            for symbol in range(len(self.graph)):
                if result and not self.graph[result[-1]][symbol]:
                    continue
                if index < counts[remaining][symbol]:
                    result.append(symbol)
                    break
                index -= counts[remaining][symbol]
        return tuple(result)

    def rank(self, word):
        counts = self.suffix_counts(len(word))
        return sum(
            counts[len(word) - position - 1][candidate]
            for position, actual in enumerate(word)
            for candidate in range(actual)
            if position == 0 or self.graph[word[position - 1]][candidate]
        )


class ReferenceSegmented:
    def __init__(self, n, k):
        self.n, self.ell, self.free = n, n - (2 + k), 2 * (2 + k) - n
        self.right = ReferenceSFT(encoder.RIGHT_ADJACENCY)
        self.right_count = self.right.count(self.ell)
        self.codebook_size = 3**self.ell * 4**self.free * self.right_count

    def encode_index(self, index):
        left_rank, remainder = divmod(index, 4**self.free * self.right_count)
        middle_rank, right_rank = divmod(remainder, self.right_count)

        def digits(value, base, length):
            result = [0] * length
            for position in range(length - 1, -1, -1):
                value, result[position] = divmod(value, base)
            return tuple(result)

        return (
            digits(left_rank, 3, self.ell)
            + digits(middle_rank, 4, self.free)
            + self.right.unrank(right_rank, self.ell)
        )


def test_portable_reference_fallback_matches_actual_archived_module():
    actual = reference_encoder()
    for graph in (((1, 1), (1, 0)), ((0, 1), (1, 0))):
        portable, archived = ReferenceSFT(graph), actual.SFTEnumerator(graph)
        for length in range(1, 8):
            assert portable.count(length) == archived.count(length)
            for index in range(portable.count(length)):
                assert portable.unrank(index, length) == archived.unrank(index, length)
    for n, k in ((40, 20), (100, 60), (150, 135)):
        portable, archived = ReferenceSegmented(n, k), actual.SegmentedDNAEncoder(n, k)
        assert portable.codebook_size == archived.codebook_size
        for rank in (0, 1, portable.codebook_size - 1):
            assert portable.encode_index(rank) == archived.encode_index(rank)


def test_sft_cache_matches_independent_frozen_code_and_lexical_enumeration():
    reference = reference_encoder()
    for graph in (((1, 1), (1, 0)), ((0, 1), (1, 0)), ((1, 1), (1, 1))):
        new, old = SFTEnumerator(graph), reference.SFTEnumerator(graph)
        for length in range(1, 7):
            words = [
                word
                for word in product(range(2), repeat=length)
                if all(graph[a][b] for a, b in zip(word, word[1:]))
            ]
            assert new.count(length) == old.count(length) == len(words)
            assert new.suffix_counts(length) is new.suffix_counts(length)
            for index, word in enumerate(words):
                assert new.unrank(index, length) == old.unrank(index, length) == word
                assert new.rank(word) == old.rank(word) == index


def test_cache_keys_limits_eviction_and_large_table_refusal():
    try:
        EXACT_TABLE_CACHE.configure(max_bytes=1024 * 1024, max_entries=2)
        first = SFTEnumerator(((1, 1), (1, 0)))
        other = SFTEnumerator(((1, 1), (0, 1)))
        a = first.suffix_counts(5)
        assert first.suffix_counts(5) is a
        assert other.suffix_counts(5) != a
        first.suffix_counts(6)
        info = cache_info()
        assert info["evictions"] >= 1 and info["entries"] <= 2
        assert info["retained_bytes"] <= info["max_bytes"]
        with pytest.raises(ValueError, match="budget_exceeded"):
            first.suffix_counts(1_000_000)
    finally:
        EXACT_TABLE_CACHE.configure(max_bytes=64 * 1024**2, max_entries=64)


def test_constants_computed_once_and_prepare_is_explicit(monkeypatch):
    calls = []
    original = encoder.right_sft_count_fast

    def counted(length):
        calls.append(length)
        return original(length)

    monkeypatch.setattr(encoder, "right_sft_count_fast", counted)
    old_api = encoder.SegmentedDNAEncoder(100, 50)
    assert calls == [48]
    old_api.prepare()
    for index in range(10):
        assert old_api.decode_index(old_api.encode_index(index)) == index
        assert old_api.codebook_size > 0 and old_api.right_count > 0
    assert calls == [48]


def test_concurrent_equal_configuration_builds_one_suffix_table():
    clear_caches()
    config = dict(kind="endpoint_m2", n=150, m=2, k=75)
    with ThreadPoolExecutor(max_workers=8) as pool:
        codecs = list(pool.map(lambda _: prepare(config), range(16)))
    assert cache_info()["misses"] == 1
    assert len({id(codec._table) for codec in codecs}) == 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda codec: codec.decode_index(codec.encode_index(111)), codecs))
    assert results == [111] * 16


@pytest.mark.parametrize("seed", [20261023, 20261024, 20261025])
def test_endpoint_against_frozen_reference_outputs(seed):
    reference = reference_encoder()
    rng = random.Random(seed)
    for n in (40, 100, 150, 200, 300):
        for numerator, denominator in ((1, 2), (3, 5), (3, 4), (9, 10)):
            k = n * numerator // denominator
            old = reference.SegmentedDNAEncoder(n, k)
            new = prepare(dict(kind="endpoint_m2", n=n, m=2, k=k))
            assert new.M == old.codebook_size
            assert new.K == new.M.bit_length() - 1
            for index in (0, (1 << new.K) - 1, rng.randrange(1 << new.K)):
                expected = "".join(DNA[symbol] for symbol in old.encode_index(index))
                actual = new.encode_index(index)
                assert actual == expected
                assert new.decode_index(actual) == index
                assert avoids_rc_windows(actual, 2, k)


def test_nonempty_prefix_conditioning_matches_filtered_legacy_order():
    reference = reference_encoder()
    for n, k, prefix in ((6, 2, "A"), (6, 1, "AC"), (6, 1, "ACGA"), (6, 1, "ACGAA")):
        old = reference.SegmentedDNAEncoder(n, k)
        expected = [
            "".join(DNA[symbol] for symbol in old.encode_index(i)) for i in range(old.codebook_size)
        ]
        expected = [word for word in expected if word.startswith(prefix)]
        codec = prepare(dict(kind="endpoint_m2", n=n, m=2, k=k, prefix=prefix))
        assert codec.M == len(expected)
        for index, word in enumerate(expected[: 1 << codec.K]):
            assert codec.encode_index(index) == word
            assert codec.decode_index(word) == index


def test_count_only_legacy_does_not_allocate_suffix_table():
    clear_caches()
    old = encoder.SegmentedDNAEncoder(n=10_000, k=5_000)
    assert old.codebook_size > 0
    assert cache_info()["entries"] == 0
