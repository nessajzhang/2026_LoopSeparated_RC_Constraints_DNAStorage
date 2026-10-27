from __future__ import annotations

import dataclasses
import random

import pytest

from loop_alpha_audit.framing import (
    PublicProtocol,
    decode_frame,
    encode_file,
    encode_frame,
    outer_decode,
    outer_encode,
    prepared_for,
    rate_accounting,
)


def protocol(mapper="short_core", **kwargs):
    return PublicProtocol(rs_n=7, rs_k=5, mapper=mapper, **kwargs)


def test_no_truth_or_channel_seed_in_public_protocol() -> None:
    names = {field.name for field in dataclasses.fields(PublicProtocol)}
    assert not names & {"payload", "truth", "sequence", "seed", "origin", "events"}


def test_shortened_rs_column_interleaving_and_error_erasure_bound() -> None:
    public = protocol("two_bit")
    user = random.Random(20261023).randbytes(public.maximum_user_bytes)
    payloads = outer_encode(user, public, padding_seed=3)
    assert len(payloads) == 7 and len(payloads[0]) == public.payload_bytes
    assert outer_decode(dict(enumerate(payloads)), public).user == user
    for erased in ((0,), (1, 5), (0, 6)):
        result = outer_decode({i: p for i, p in enumerate(payloads) if i not in erased}, public)
        assert result.status == "OK" and result.user == user
        assert result.erasures == erased
    corrupted = dict(enumerate(payloads))
    corrupted[3] = bytes(byte ^ 0x55 for byte in corrupted[3])
    assert outer_decode(corrupted, public).user == user
    assert outer_decode(dict(enumerate(payloads[:4])), public).status == "RS_FAILED"
    assert outer_decode({99: payloads[0]}, public).status == "INVALID_INPUT"


def test_zero_parity_and_file_length_and_checksum() -> None:
    public = PublicProtocol(rs_n=7, rs_k=7, mapper="two_bit")
    for user in (b"", b"abc", bytes(public.maximum_user_bytes)):
        payloads = outer_encode(user, public, padding_seed=7)
        assert outer_decode(dict(enumerate(payloads)), public).user == user
        corrupted = dict(enumerate(payloads))
        corrupted[0] = bytes([corrupted[0][0] ^ 128]) + corrupted[0][1:]
        assert outer_decode(corrupted, public).status == "FILE_CHECK_FAILED"
        assert (
            outer_decode({i: p for i, p in enumerate(payloads) if i != 3}, public).status
            == "RS_FAILED"
        )


@pytest.mark.parametrize("mapper", ["two_bit", "gc_h_nonce", "short_core", "nonce"])
def test_full_file_noiseless_round_trip_and_complete_accounting(mapper) -> None:
    public = protocol(mapper)
    user = random.Random(20261024).randbytes(public.maximum_user_bytes)
    frames = encode_file(user, public, seed=8)
    assert all(frame.status == "OK" for frame in frames)
    decoded = {}
    for frame in frames:
        result = decode_frame(frame.sequence, frame.strand_index, public)
        assert result.status == "OK" and result.payload == frame.payload
        decoded[frame.strand_index] = result.payload
        if mapper in ("short_core", "nonce"):
            assert frame.constraint_ok and frame.gc_h_filter_ok
        if mapper != "two_bit":
            assert frame.gc_h_filter_ok
    assert outer_decode(decoded, public).user == user
    accounting = rate_accounting(public, len(user), frames)
    assert accounting["total_synthesized_nt"] == 7 * 150
    assert accounting["R_net"] == len(user) * 8 / (7 * 150)
    assert accounting["header_nt_per_strand"] == 26
    assert accounting["frame_CRC_bits_per_strand"] == 32
    assert accounting["file_metadata_bits"] == 64
    assert accounting["parity_strands"] == 2
    assert accounting["mapper_input_bits"] == sum(frame.mapper_input_bits for frame in frames)


def test_receiver_inverse_rejects_wrong_address_mode_length_and_crc() -> None:
    public = protocol("two_bit")
    frame = encode_frame(b"x" * public.payload_bytes, 1, public, seed=10)
    assert decode_frame(frame.sequence, 0, public).status == "INVALID_SEQUENCE"
    assert decode_frame(frame.sequence[:-1], 1, public).status == "INVALID_SEQUENCE"
    address_nt = public.address_book.length
    invalid_tag = frame.sequence[:address_nt] + "AA" + frame.sequence[address_nt + 2 :]
    assert decode_frame(invalid_tag, 1, public).status == "INVALID_SEQUENCE"
    position = public.header_nt + 5
    base = frame.sequence[position]
    edited = (
        frame.sequence[:position] + ("A" if base != "A" else "C") + frame.sequence[position + 1 :]
    )
    assert decode_frame(edited, 1, public).status == "CRC_FAILED"
    codec = prepared_for(public, 0, "short_core")
    assert all(prepared_for(public, i, "short_core").K == codec.K for i in range(public.rs_n))


def test_protocol_does_not_hide_primers_or_oversized_payload() -> None:
    with pytest.raises(ValueError):
        protocol(primer_nt=20)
    with pytest.raises(ValueError):
        protocol(payload_bytes=1000)
    with pytest.raises(ValueError):
        protocol(repair_edit_radius=2)
    public = protocol()
    with pytest.raises(ValueError):
        outer_encode(bytes(public.maximum_user_bytes + 1), public, padding_seed=2)


def test_primary_failure_is_counted_without_fallback_and_with_fallback() -> None:
    terminal = protocol("gc_h_nonce", nonce_budget=1)
    frames = encode_file(bytes(terminal.maximum_user_bytes), terminal, seed=0)
    failures = sum(frame.status == "ENCODING_FAILED" for frame in frames)
    assert failures > 0
    accounting = rate_accounting(terminal, terminal.maximum_user_bytes, frames)
    assert accounting["primary_mapper_failures"] == failures
    assert accounting["terminal_mapper_failures"] == failures
    assert accounting["fallback_attempts"] == accounting["fallback_used"] == 0
    fallback = protocol("nonce", nonce_budget=1)
    frames = encode_file(bytes(fallback.maximum_user_bytes), fallback, seed=0)
    accounting = rate_accounting(fallback, fallback.maximum_user_bytes, frames)
    assert accounting["primary_mapper_failures"] > 0
    assert accounting["primary_mapper_failures"] == accounting["fallback_attempts"]
    assert accounting["fallback_attempts"] == accounting["fallback_used"]
    assert accounting["terminal_mapper_failures"] == 0
    assert all(frame.status == "OK" for frame in frames)
