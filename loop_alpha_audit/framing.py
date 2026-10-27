"""Public framing, CRC and shortened RS code around exact constrained mapping.

This module is not a new error-correcting-code construction. Rank inversion is
only an exact noiseless inverse. The separately implemented receiver produces
CRC-checked frame payloads; this module then invokes an established byte RS
decoder across strands. All addressing, nonce, CRC, parity and padding costs
are included in the public accounting interface.
"""

from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter_ns
from typing import Mapping

from reedsolo import RSCodec, ReedSolomonError

from .addresses import AddressBook, make_address_book
from .channel import derived_seed
from .prepared_codec import PreparedCodec, prepare
from .short_loop_codec import avoids_rc_windows, full_oligo_failure

MODE_TAGS = {"short_core": "CC", "nonce": "AT", "gc_h_nonce": "AG", "two_bit": "GG"}
CRC_BITS = 32
FILE_METADATA_BYTES = 8  # Four-byte length and four-byte CRC, both RS protected.


def bytes_to_bits(value: bytes) -> str:
    return "".join(f"{byte:08b}" for byte in value)


def bits_to_bytes(value: str) -> bytes:
    if len(value) % 8 or set(value) - set("01"):
        raise ValueError("a byte string requires a multiple of eight binary digits")
    return bytes(int(value[start : start + 8], 2) for start in range(0, len(value), 8))


def frame_crc(strand_index: int, payload: bytes, mode: str) -> bytes:
    if not 0 <= strand_index <= 255:
        raise ValueError("strand index outside byte RS frame range")
    if mode not in MODE_TAGS:
        raise ValueError("unknown frame mode")
    return struct.pack(
        ">I",
        zlib.crc32(struct.pack(">H", strand_index) + MODE_TAGS[mode].encode("ascii") + payload)
        & 0xFFFFFFFF,
    )


@dataclass(frozen=True)
class PublicProtocol:
    """Receiver-visible configuration. No payload, truth, seed or edit script."""

    n: int = 150
    m: int = 7
    k: int = 3
    mapper: str = "short_core"
    rs_n: int = 63
    rs_k: int = 55
    address_book: AddressBook | None = None
    payload_bytes: int | None = None
    nonce_nt: int = 4
    nonce_budget: int = 64
    fallback_to_core: bool = True
    address_radius: int = 2
    boundary_slack: int = 3
    repair_edit_radius: int = 1
    repair_candidate_budget: int = 2500
    cluster_read_cap: int = 20
    primer_nt: int = 0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("n", 1),
            ("m", 1),
            ("k", 0),
            ("rs_n", 1),
            ("rs_k", 1),
            ("nonce_nt", 1),
            ("nonce_budget", 1),
            ("address_radius", 0),
            ("boundary_slack", 0),
            ("repair_candidate_budget", 1),
            ("cluster_read_cap", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not self.rs_k <= self.rs_n <= 255:
            raise ValueError("require 1 <= rs_k <= rs_n <= 255")
        if self.mapper not in MODE_TAGS:
            raise ValueError("unknown practical mapper")
        if self.repair_edit_radius not in (0, 1):
            raise ValueError("receiver implements only zero- or one-edit repair")
        if self.primer_nt != 0:
            raise ValueError("this diagnostic protocol has no primers; do not hide added primers")
        book = self.address_book or make_address_book(self.rs_n)
        if len(book.sequences) != self.rs_n:
            raise ValueError("address count must equal rs_n")
        # The shared-rate design relies on identical continuation states, not
        # on an unverified assumption about arbitrary address sequences.
        for address in book.sequences:
            if (
                set(address) - set("TC")
                or not address.endswith("TT")
                or address.count("C") * 2 != len(address)
            ):
                raise ValueError("diagnostic public addresses must be balanced TC and end TT")
        object.__setattr__(self, "address_book", book)
        if self.n < book.length + 2 + self.nonce_nt:
            raise ValueError("oligo too short for public address, mode and nonce")
        reference = prepared_for(self, 0, "short_core")
        maximum_common = (reference.K - CRC_BITS) // 8
        chosen = maximum_common if self.payload_bytes is None else self.payload_bytes
        if type(chosen) is not int or chosen < 1 or chosen > maximum_common:
            raise ValueError("useful frame payload exceeds the common guaranteed-core capacity")
        object.__setattr__(self, "payload_bytes", chosen)
        if self.rs_k * chosen <= FILE_METADATA_BYTES:
            raise ValueError("file is too small after protected file metadata")
        if prepared_for(self, 0, self.mapper).K < 8 * chosen + CRC_BITS:
            raise ValueError("primary mapper cannot carry the declared useful frame payload")

    @property
    def maximum_user_bytes(self) -> int:
        assert self.payload_bytes is not None
        return self.rs_k * self.payload_bytes - FILE_METADATA_BYTES

    @property
    def header_nt(self) -> int:
        assert self.address_book is not None
        return self.address_book.length + 2

    @property
    def allowed_modes(self) -> tuple[str, ...]:
        if self.mapper == "nonce" and self.fallback_to_core:
            return ("nonce", "short_core")
        return (self.mapper,)


def prepared_for(protocol: PublicProtocol, strand_index: int, kind: str) -> PreparedCodec:
    if not 0 <= strand_index < protocol.rs_n or kind not in MODE_TAGS:
        raise ValueError("invalid public frame codec selection")
    assert protocol.address_book is not None
    prefix = protocol.address_book.sequences[strand_index] + MODE_TAGS[kind]
    return _prepared_from_fields(
        kind, protocol.n, protocol.m, protocol.k, prefix, protocol.nonce_nt, protocol.nonce_budget
    )


@lru_cache(maxsize=256)
def _prepared_from_fields(
    kind: str, n: int, m: int, k: int, prefix: str, nonce_nt: int, nonce_budget: int
) -> PreparedCodec:
    # Immutable primitive cache keys avoid hashing a dataclass during its own
    # __post_init__, before its derived payload_bytes field has been finalized.
    config = {"kind": kind, "n": n, "m": m, "k": k, "prefix": prefix}
    if kind in ("nonce", "gc_h_nonce"):
        config.update(nonce_nt=nonce_nt, nonce_budget=nonce_budget)
    return prepare(config)


def _rs(protocol: PublicProtocol) -> RSCodec | None:
    parity = protocol.rs_n - protocol.rs_k
    return RSCodec(parity, nsize=255, c_exp=8, prim=0x11D, generator=2, fcr=0) if parity else None


def outer_encode(user: bytes, protocol: PublicProtocol, *, padding_seed: int) -> tuple[bytes, ...]:
    """Transpose bytewise shortened RS codewords into strand payloads."""
    if not isinstance(user, bytes) or len(user) > protocol.maximum_user_bytes:
        raise ValueError("file payload is not bytes or exceeds protocol capacity")
    assert protocol.payload_bytes is not None
    protected = struct.pack(">II", len(user), zlib.crc32(user) & 0xFFFFFFFF) + user
    padding_size = protocol.rs_k * protocol.payload_bytes - len(protected)
    rng = random.Random(derived_seed(padding_seed, "outer-padding"))
    protected += rng.randbytes(padding_size)
    rows = [
        protected[i * protocol.payload_bytes : (i + 1) * protocol.payload_bytes]
        for i in range(protocol.rs_k)
    ]
    rs = _rs(protocol)
    output = [bytearray() for _ in range(protocol.rs_n)]
    for column in range(protocol.payload_bytes):
        message = bytes(row[column] for row in rows)
        coded = bytes(rs.encode(message)) if rs is not None else message
        if len(coded) != protocol.rs_n:
            raise AssertionError("RS library returned an unexpected shortened codeword length")
        for index, symbol in enumerate(coded):
            output[index].append(symbol)
    return tuple(bytes(row) for row in output)


@dataclass(frozen=True)
class OuterDecodeResult:
    status: str
    user: bytes | None
    erasures: tuple[int, ...]
    corrected_positions: tuple[int, ...]
    elapsed_ns: int
    failure_reason: str | None


def outer_decode(payloads: Mapping[int, bytes], protocol: PublicProtocol) -> OuterDecodeResult:
    """Recover from receiver output only; no expected data are accepted."""
    start = perf_counter_ns()
    assert protocol.payload_bytes is not None
    if any(
        type(index) is not int
        or not 0 <= index < protocol.rs_n
        or not isinstance(value, bytes)
        or len(value) != protocol.payload_bytes
        for index, value in payloads.items()
    ):
        return OuterDecodeResult(
            "INVALID_INPUT",
            None,
            (),
            (),
            perf_counter_ns() - start,
            "invalid receiver payload table",
        )
    erased = tuple(index for index in range(protocol.rs_n) if index not in payloads)
    parity = protocol.rs_n - protocol.rs_k
    if len(erased) > parity:
        return OuterDecodeResult(
            "RS_FAILED",
            None,
            erased,
            (),
            perf_counter_ns() - start,
            "erasures exceed parity symbols",
        )
    recovered = [bytearray() for _ in range(protocol.rs_k)]
    corrected: set[int] = set()
    rs = _rs(protocol)
    try:
        for column in range(protocol.payload_bytes):
            coded = bytes(
                payloads[index][column] if index in payloads else 0
                for index in range(protocol.rs_n)
            )
            if rs is None:
                message = coded
            else:
                message, _, errata = rs.decode(coded, erase_pos=list(erased))
                corrected.update(errata)
            if len(message) != protocol.rs_k:
                raise ReedSolomonError("decoded shortened RS message has wrong size")
            for index, symbol in enumerate(message):
                recovered[index].append(symbol)
    except (ReedSolomonError, ValueError, IndexError) as exc:
        return OuterDecodeResult(
            "RS_FAILED", None, erased, tuple(sorted(corrected)), perf_counter_ns() - start, str(exc)
        )
    file_bytes = b"".join(bytes(row) for row in recovered)
    size, checksum = struct.unpack(">II", file_bytes[:FILE_METADATA_BYTES])
    user = file_bytes[FILE_METADATA_BYTES : FILE_METADATA_BYTES + size]
    if size > protocol.maximum_user_bytes or len(user) != size:
        return OuterDecodeResult(
            "FILE_CHECK_FAILED",
            None,
            erased,
            tuple(sorted(corrected)),
            perf_counter_ns() - start,
            "protected length invalid",
        )
    if zlib.crc32(user) & 0xFFFFFFFF != checksum:
        return OuterDecodeResult(
            "FILE_CHECK_FAILED",
            None,
            erased,
            tuple(sorted(corrected)),
            perf_counter_ns() - start,
            "protected file CRC mismatch",
        )
    return OuterDecodeResult(
        "OK", user, erased, tuple(sorted(corrected)), perf_counter_ns() - start, None
    )


@dataclass(frozen=True)
class EncodedFrame:
    """Sender/evaluator record. Never admitted as a receiver argument."""

    strand_index: int
    status: str
    sequence: str | None
    payload: bytes
    mode: str | None
    attempts: int
    primary_failed: bool
    fallback_attempted: bool
    mapper_input_bits: int
    filler_bits: int
    elapsed_ns: int
    constraint_ok: bool | None
    gc_h_filter_ok: bool | None
    failure_reason: str | None


def encode_frame(
    payload: bytes, strand_index: int, protocol: PublicProtocol, *, seed: int
) -> EncodedFrame:
    start = perf_counter_ns()
    assert protocol.payload_bytes is not None
    if not isinstance(payload, bytes) or len(payload) != protocol.payload_bytes:
        raise ValueError("frame useful payload has incorrect length")
    if type(strand_index) is not int or not 0 <= strand_index < protocol.rs_n:
        raise ValueError("invalid public strand index")
    attempts = 0
    primary_failed = False
    for mode_number, mode in enumerate(protocol.allowed_modes):
        fallback_attempted = mode_number > 0
        protected_bits = bytes_to_bits(payload + frame_crc(strand_index, payload, mode))
        codec = prepared_for(protocol, strand_index, mode)
        filler = codec.K - len(protected_bits)
        if filler < 0:
            raise AssertionError("protocol did not reserve enough mapper capacity")
        rng = random.Random(derived_seed(seed, "mapper-filler", strand_index, mode))
        filler_bits = format(rng.getrandbits(filler), f"0{filler}b") if filler else ""
        result = codec.encode_bits(protected_bits + filler_bits)
        attempts += result.attempts
        if result.status != "OK":
            if mode_number == 0:
                primary_failed = True
            if result.status == "ENCODING_FAILED" and mode != protocol.allowed_modes[-1]:
                continue
            return EncodedFrame(
                strand_index,
                result.status,
                None,
                payload,
                None,
                attempts,
                primary_failed,
                fallback_attempted,
                codec.K,
                filler,
                perf_counter_ns() - start,
                None,
                None,
                result.failure_reason,
            )
        assert result.sequence is not None
        sequence = result.sequence
        constraint_ok = avoids_rc_windows(sequence, protocol.m, protocol.k)
        gc_h_ok = full_oligo_failure(sequence, protocol.m, protocol.k, check_rc=False) is None
        if mode in ("short_core", "nonce") and not (constraint_ok and gc_h_ok):
            raise AssertionError("constrained mapper emitted a violating complete oligo")
        if mode == "gc_h_nonce" and not gc_h_ok:
            raise AssertionError("filter-only mapper emitted a GC/homopolymer violation")
        return EncodedFrame(
            strand_index,
            "OK",
            sequence,
            payload,
            mode,
            attempts,
            primary_failed,
            fallback_attempted,
            codec.K,
            filler,
            perf_counter_ns() - start,
            constraint_ok,
            gc_h_ok,
            None,
        )
    raise AssertionError("protocol has no encoding modes")


@dataclass(frozen=True)
class FrameDecodeResult:
    status: str
    payload: bytes | None
    mode: str | None
    elapsed_ns: int
    failure_reason: str | None


def decode_frame(sequence: str, strand_index: int, protocol: PublicProtocol) -> FrameDecodeResult:
    """Strict inverse mapping plus CRC; not a noisy-read reconstruction routine."""
    start = perf_counter_ns()
    assert protocol.payload_bytes is not None and protocol.address_book is not None
    if (
        type(strand_index) is not int
        or not 0 <= strand_index < protocol.rs_n
        or not isinstance(sequence, str)
        or len(sequence) != protocol.n
    ):
        return FrameDecodeResult(
            "INVALID_SEQUENCE",
            None,
            None,
            perf_counter_ns() - start,
            "wrong frame length or public address index",
        )
    address = protocol.address_book.sequences[strand_index]
    if not sequence.startswith(address):
        return FrameDecodeResult(
            "INVALID_SEQUENCE",
            None,
            None,
            perf_counter_ns() - start,
            "observed address disagrees with exact frame prefix",
        )
    tag = sequence[len(address) : len(address) + 2]
    mode = next((name for name in protocol.allowed_modes if MODE_TAGS[name] == tag), None)
    if mode is None:
        return FrameDecodeResult(
            "INVALID_SEQUENCE",
            None,
            None,
            perf_counter_ns() - start,
            "unknown or disallowed transmitted mode tag",
        )
    decoded = prepared_for(protocol, strand_index, mode).decode_bits(sequence)
    if decoded.status != "OK" or decoded.bits is None:
        return FrameDecodeResult(
            "INVALID_SEQUENCE", None, mode, perf_counter_ns() - start, decoded.failure_reason
        )
    size = protocol.payload_bytes
    protected = bits_to_bytes(decoded.bits[: 8 * size + CRC_BITS])
    payload, checksum = protected[:size], protected[size:]
    if checksum != frame_crc(strand_index, payload, mode):
        return FrameDecodeResult(
            "CRC_FAILED", None, mode, perf_counter_ns() - start, "address-bound frame CRC mismatch"
        )
    return FrameDecodeResult("OK", payload, mode, perf_counter_ns() - start, None)


def encode_file(user: bytes, protocol: PublicProtocol, *, seed: int) -> tuple[EncodedFrame, ...]:
    payloads = outer_encode(user, protocol, padding_seed=seed)
    return tuple(
        encode_frame(payload, index, protocol, seed=seed) for index, payload in enumerate(payloads)
    )


def rate_accounting(
    protocol: PublicProtocol, user_bytes: int, frames: tuple[EncodedFrame, ...]
) -> dict:
    """Actual synthesis denominator and explicit unused mapper capacity."""
    if len(frames) != protocol.rs_n or not 0 <= user_bytes <= protocol.maximum_user_bytes:
        raise ValueError("accounting input does not match public protocol")
    sent = [frame for frame in frames if frame.sequence is not None and frame.status == "OK"]
    nt = sum(len(frame.sequence) for frame in sent)
    return {
        "user_bits": 8 * user_bytes,
        "total_synthesized_nt": nt,
        "planned_synthesized_nt": protocol.n * protocol.rs_n,
        "R_net": 8 * user_bytes / nt if nt else None,
        "R_net_per_planned_nt": 8 * user_bytes / (protocol.n * protocol.rs_n),
        "all_strands_encoded": len(sent) == protocol.rs_n,
        "outer_code": {"n": protocol.rs_n, "k": protocol.rs_k, "symbol_bits": 8},
        "header_nt_per_strand": protocol.header_nt,
        "primer_nt_per_strand": protocol.primer_nt,
        "frame_CRC_bits_per_strand": CRC_BITS,
        "file_metadata_bits": FILE_METADATA_BYTES * 8,
        "outer_file_padding_bits": 8
        * (protocol.rs_k * protocol.payload_bytes - FILE_METADATA_BYTES - user_bytes),
        "parity_strands": protocol.rs_n - protocol.rs_k,
        "mapper_input_bits": sum(frame.mapper_input_bits for frame in sent),
        "filler_bits": sum(frame.filler_bits for frame in sent),
        "R_map_protected_input_excluding_filler": (
            sum(frame.mapper_input_bits - frame.filler_bits for frame in sent)
            / sum(protocol.n - protocol.header_nt for _ in sent)
            if sent
            else None
        ),
        "R_map_input_including_frame_CRC_parity_and_filler": (
            sum(frame.mapper_input_bits for frame in sent)
            / sum(protocol.n - protocol.header_nt for _ in sent)
            if sent
            else None
        ),
        "nonce_nt": sum(
            protocol.nonce_nt for frame in sent if frame.mode in ("nonce", "gc_h_nonce")
        ),
        "primary_mapper_failures": sum(frame.primary_failed for frame in frames),
        "fallback_attempts": sum(frame.fallback_attempted for frame in frames),
        "fallback_used": sum(
            frame.status == "OK" and frame.mode != protocol.mapper for frame in frames
        ),
        "terminal_mapper_failures": len(frames) - len(sent),
        "attempts_including_failures": sum(frame.attempts for frame in frames),
        "claim": "no primer or biochemical error-reduction guarantee; fixed-resource diagnostic protocol",
    }
