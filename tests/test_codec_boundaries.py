from __future__ import annotations

from itertools import product
import inspect

import pytest

from loop_alpha_audit.encoder import SegmentedDNAEncoder, SFTEnumerator
from loop_alpha_audit.prepared_codec import DNA, prepare
from loop_alpha_audit.short_loop_codec import full_oligo_failure


@pytest.mark.parametrize(
    "n,k,prefix", [(0, 0, ""), (1, 0, ""), (2, 0, "T"), (3, 0, "T"), (5, 2, "TC"), (8, 5, "TCTCT")]
)
def test_vacuous_endpoint_language_and_zero_payload(n, k, prefix):
    codec = prepare(dict(kind="endpoint_m2", n=n, m=2, k=k, prefix=prefix))
    assert codec.metadata["vacuous"] and codec.M == 4 ** (n - len(prefix))
    for rank in range(1 << codec.K):
        word = codec.encode_index(rank)
        assert word.startswith(prefix)
        assert codec.decode_index(word) == rank
    assert codec.decode_bits(codec.encode_bits("0" * codec.K).sequence).bits == "0" * codec.K


def test_high_alpha_geometry_g_zero_and_invalid_prefix():
    codec = prepare(dict(kind="endpoint_m2", n=8, m=2, k=2))
    assert codec.metadata["free_length"] == 0
    assert codec.decode_index(codec.encode_index((1 << codec.K) - 1)) == (1 << codec.K) - 1
    for config in (
        dict(kind="endpoint_m2", n=150, m=2, k=3),
        dict(kind="endpoint_m2", n=8, m=3, k=4),
        dict(kind="endpoint_m2", n=8, m=2, k=2, prefix="T"),
        dict(kind="endpoint_m2", n=8, m=2, k=2, prefix="ACGGCC"),
    ):
        with pytest.raises(ValueError):
            prepare(config)


def test_unused_binary_ranks_rejected_not_modulo_decoded():
    config = dict(kind="endpoint_m2", n=8, m=2, k=4)
    codec = prepare(config)
    assert 1 << codec.K < codec.M
    legacy = SegmentedDNAEncoder(n=8, k=4)
    unused = "".join(DNA[base] for base in legacy.encode_index(1 << codec.K))
    with pytest.raises(ValueError, match="unused_binary_rank"):
        codec.decode_index(unused)
    assert codec.decode_bits(unused).status == "INVALID_SEQUENCE"
    for kind in ("endpoint_m2", "short_core", "nonce", "gc_h_nonce", "two_bit"):
        config = dict(
            kind=kind,
            n=30,
            m=2 if kind == "endpoint_m2" else 7,
            k=15 if kind == "endpoint_m2" else 3,
        )
        current = prepare(config)
        for index in (-1, 1 << current.K, True, 1.5, "0"):
            with pytest.raises(ValueError):
                current.encode_index(index)


@pytest.mark.parametrize("kind", ["endpoint_m2", "short_core", "nonce", "gc_h_nonce", "two_bit"])
def test_result_schema_strict_inputs_and_no_truth_receiver_parameters(kind):
    config = dict(
        kind=kind, n=40, m=2 if kind == "endpoint_m2" else 7, k=20 if kind == "endpoint_m2" else 3
    )
    codec = prepare(config)
    for bits in (None, [], "0" * (codec.K + 1), "0" * (codec.K - 1), "2" * codec.K):
        result = codec.encode_bits(bits)
        assert result.status == "INVALID_INPUT" and result.sequence is None
        assert result.failure_reason and result.elapsed_ns >= 0 and result.attempts == 0
    bits = "0" * codec.K
    encoded = codec.encode_bits(bits)
    if encoded.status == "ENCODING_FAILED":
        assert encoded.failure_reason
        return
    assert encoded.status == "OK" and encoded.attempts >= 1
    assert codec.decode_bits(encoded.sequence).bits == bits
    for word in (
        None,
        [],
        encoded.sequence + "A",
        encoded.sequence[:-1],
        "N" + encoded.sequence[1:],
    ):
        result = codec.decode_bits(word)
        assert result.status == "INVALID_SEQUENCE" and result.bits is None
        assert result.failure_reason and result.elapsed_ns >= 0
    assert list(inspect.signature(codec.decode_bits).parameters) == ["sequence"]
    assert list(inspect.signature(codec.decode_index).parameters) == ["sequence"]
    assert config == dict(
        kind=kind, n=40, m=2 if kind == "endpoint_m2" else 7, k=20 if kind == "endpoint_m2" else 3
    )


def test_complete_fixed_prefix_and_majority_gc_rll_prefix_validation():
    for n, m, k, word in ((6, 2, 2, "ACAAAA"), (4, 7, 0, "TCTC")):
        kind = "endpoint_m2" if m == 2 else "short_core"
        codec = prepare(dict(kind=kind, n=n, m=m, k=k, prefix=word))
        assert codec.K == 0 and codec.M == 1
        assert codec.encode_bits("").sequence == word
        assert codec.decode_bits(word).bits == ""
    for prefix, match in (("AAAAAAA", "TC_majority"), ("ATATA", "GC_bit_run"), ("CCCC", "empty")):
        with pytest.raises(ValueError, match=match):
            prepare(
                dict(kind="short_core", n=7 if prefix != "CCCC" else 5, m=7, k=3, prefix=prefix)
            )


def test_plain_mapper_does_not_claim_constraint_or_error_correction():
    codec = prepare(dict(kind="two_bit", n=150, m=7, k=3, prefix="TCTCTTGG"))
    encoded = codec.encode_bits("0" * codec.K)
    assert codec.metadata["constraint_scope"] == "none"
    assert full_oligo_failure(encoded.sequence, 7, 3) is not None
    mutated = encoded.sequence[:-1] + "C"
    decoded = codec.decode_bits(mutated)
    assert decoded.status == "OK" and decoded.bits != "0" * codec.K


def test_invalid_config_and_prefix_fields():
    for config in (
        {},
        {"kind": "unknown", "n": 1},
        {"kind": "two_bit", "n": True},
        {"kind": "two_bit", "n": 4, "prefix": "N"},
        {"kind": "two_bit", "n": 4, "prefix": "AAAAA"},
        {"kind": "two_bit", "n": 4, "secret_truth": "not allowed"},
        {"kind": "nonce", "n": 3, "nonce_nt": 4},
        {"kind": "nonce", "n": 20, "nonce_budget": 257},
        {"kind": "nonce", "n": 20, "prefix": "AAAAA"},
        {"kind": "nonce", "n": 20, "gc_low": 0.8, "gc_high": 0.2},
    ):
        with pytest.raises(ValueError):
            prepare(config)


def test_single_symbol_invalid_mutations_and_full_prefix_rejection():
    codec = prepare(dict(kind="short_core", n=30, m=7, k=3, prefix="TCTTCC"))
    sequence = codec.encode_index(0)
    for position, invalid in product((0, 5, 29), ("N", "a", "-")):
        assert (
            codec.decode_bits(sequence[:position] + invalid + sequence[position + 1 :]).status
            == "INVALID_SEQUENCE"
        )
    assert codec.decode_bits("A" + sequence[1:]).failure_reason == "fixed_prefix_mismatch"


def test_no_float_or_boolean_branch_counts_in_legacy_or_new_interfaces():
    with pytest.raises(ValueError):
        SFTEnumerator(((1.0, 1), (0, 1)))
    for value in (True, 1.0, "1"):
        with pytest.raises(ValueError):
            SegmentedDNAEncoder(value, 2)
        with pytest.raises(ValueError):
            SFTEnumerator(((1, 1), (1, 0))).suffix_counts(value)
    codec = prepare(dict(kind="nonce", n=4, m=7, k=0, nonce_nt=4, nonce_budget=256))
    assert codec.K == 0
    encoded = codec.encode_bits("")
    assert encoded.status == "OK" and codec.decode_bits(encoded.sequence).bits == ""
    with pytest.raises(ValueError):
        prepare(dict(kind=[], n=1))
