"""Prepared exact message codecs; binary rank interfaces are not error correction.

All lengths include the transmitted fixed prefix. No decoder accepts sender
truth. The legacy encoder keeps its full-codebook API; this facade deliberately
uses only ranks below 2**K, where K=M.bit_length()-1 for exact codebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from types import MappingProxyType
from typing import Iterable, Mapping

from .encoder import RIGHT_ADJACENCY, RIGHT_ENUMERATOR, fixed_base_rank, fixed_base_unrank

DNA = "ACGT"


@dataclass(frozen=True)
class EncodeResult:
    status: str
    sequence: str | None
    elapsed_ns: int
    failure_reason: str | None
    attempts: int


@dataclass(frozen=True)
class DecodeResult:
    status: str
    bits: str | None
    elapsed_ns: int
    failure_reason: str | None


class EncodingFailure(ValueError):
    def __init__(self, reason: str, attempts: int):
        super().__init__(reason)
        self.attempts = attempts


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def common_config(config: dict, allowed_extra: tuple[str, ...] = ()) -> tuple[int, int, int, str]:
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    allowed = {"kind", "n", "m", "k", "prefix", *allowed_extra}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown codec config keys: {sorted(unknown)}")
    n = integer(config.get("n"), "n")
    m = integer(config.get("m", 2 if config.get("kind") == "endpoint_m2" else 7), "m", 1)
    k = integer(config.get("k", 3), "k")
    prefix = config.get("prefix", "")
    if not isinstance(prefix, str) or any(base not in DNA for base in prefix):
        raise ValueError("prefix must contain only uppercase A/C/G/T")
    if len(prefix) > n:
        raise ValueError("prefix is longer than total n")
    return n, m, k, prefix


@dataclass(frozen=True)
class PreparedCodec:
    kind: str
    n: int
    m: int
    k: int
    prefix: str
    M: int | None
    K: int
    metadata: Mapping[str, object]

    @property
    def maximum_payload_bits(self) -> int:
        return self.K

    def _check_index(self, index: int) -> None:
        integer(index, "index")
        if index >= 1 << self.K:
            raise ValueError("unused_binary_rank: require index < 2**K")

    def _check_sequence(self, sequence: str) -> None:
        if not isinstance(sequence, str) or len(sequence) != self.n:
            raise ValueError("sequence_length_mismatch")
        if any(base not in DNA for base in sequence):
            raise ValueError("invalid_DNA_symbol")
        if not sequence.startswith(self.prefix):
            raise ValueError("fixed_prefix_mismatch")

    def _encode_with_attempts(self, index: int) -> tuple[str, int]:
        raise NotImplementedError

    def _decode_raw(self, sequence: str) -> int:
        raise NotImplementedError

    def encode_index(self, index: int) -> str:
        self._check_index(index)
        return self._encode_with_attempts(index)[0]

    def decode_index(self, sequence: str) -> int:
        self._check_sequence(sequence)
        index = self._decode_raw(sequence)
        self._check_index(index)
        return index

    def encode_bits(self, bits: str) -> EncodeResult:
        start = perf_counter_ns()
        try:
            if (
                not isinstance(bits, str)
                or len(bits) != self.K
                or any(bit not in "01" for bit in bits)
            ):
                raise ValueError("payload must be a binary string of exactly K bits")
            index = int(bits, 2) if bits else 0
            self._check_index(index)
            sequence, attempts = self._encode_with_attempts(index)
            return EncodeResult("OK", sequence, perf_counter_ns() - start, None, attempts)
        except EncodingFailure as exc:
            return EncodeResult(
                "ENCODING_FAILED", None, perf_counter_ns() - start, str(exc), exc.attempts
            )
        except ValueError as exc:
            return EncodeResult("INVALID_INPUT", None, perf_counter_ns() - start, str(exc), 0)

    def decode_bits(self, sequence: str) -> DecodeResult:
        start = perf_counter_ns()
        try:
            index = self.decode_index(sequence)
            bits = format(index, f"0{self.K}b") if self.K else ""
            return DecodeResult("OK", bits, perf_counter_ns() - start, None)
        except ValueError as exc:
            return DecodeResult("INVALID_SEQUENCE", None, perf_counter_ns() - start, str(exc))

    def encode_many(self, messages: Iterable[str]) -> list[EncodeResult]:
        return [self.encode_bits(bits) for bits in messages]

    def decode_many(self, sequences: Iterable[str]) -> list[DecodeResult]:
        return [self.decode_bits(sequence) for sequence in sequences]


def _right_count(table: tuple, length: int, fixed: tuple[int, ...]) -> int:
    if not fixed:
        return sum(table[length - 1])
    return table[length - len(fixed)][fixed[-1]]


def _right_unrank(index: int, length: int, fixed: tuple[int, ...], table: tuple) -> tuple[int, ...]:
    result = list(fixed)
    while len(result) < length:
        remaining = length - len(result) - 1
        for candidate in range(4):
            if result and not RIGHT_ADJACENCY[result[-1]][candidate]:
                continue
            block = table[remaining][candidate]
            if index < block:
                result.append(candidate)
                break
            index -= block
        else:
            raise AssertionError("right endpoint unrank exhausted exact branches")
    return tuple(result)


def _right_rank(word: tuple[int, ...], fixed: tuple[int, ...], table: tuple) -> int:
    if word[: len(fixed)] != fixed:
        raise ValueError("right_prefix_mismatch")
    index = 0
    for position, actual in enumerate(word):
        if position and not RIGHT_ADJACENCY[word[position - 1]][actual]:
            raise ValueError("right_endpoint_adjacency_violation")
        if position < len(fixed):
            continue
        remaining = len(word) - position - 1
        index += sum(
            table[remaining][candidate]
            for candidate in range(actual)
            if position == 0 or RIGHT_ADJACENCY[word[position - 1]][candidate]
        )
    return index


class EndpointCodec(PreparedCodec):
    def __init__(self, config: dict):
        n, m, k, prefix = common_config(config)
        if m != 2:
            raise ValueError("endpoint_m2 requires m=2")
        vacuous = n < 2 * m + k
        if not vacuous and 2 * (m + k) < n:
            raise ValueError("endpoint geometry requires 2*(m+k)>=n; short-k is not supported")
        if vacuous:
            ell = free = 0
            left_fixed = middle_fixed = right_fixed = ()
            table = ()
            left_count, middle_count, right_count = 1, 4 ** (n - len(prefix)), 1
        else:
            ell, free = n - (m + k), 2 * (m + k) - n
            encoded_prefix = tuple(DNA.index(base) for base in prefix)
            left_fixed = encoded_prefix[:ell]
            middle_fixed = encoded_prefix[ell : ell + free]
            right_fixed = encoded_prefix[ell + free :]
            if any(symbol == 3 for symbol in left_fixed):
                raise ValueError("fixed prefix is outside ACG left-endpoint subcode")
            if any(not RIGHT_ADJACENCY[a][b] for a, b in zip(right_fixed, right_fixed[1:])):
                raise ValueError("fixed prefix violates right-endpoint adjacency")
            table = RIGHT_ENUMERATOR.suffix_counts(ell)
            left_count = 3 ** (ell - len(left_fixed))
            middle_count = 4 ** (free - len(middle_fixed))
            right_count = _right_count(table, ell, right_fixed)
        total = left_count * middle_count * right_count
        if total < 1:
            raise ValueError("empty_prefix_conditioned_codebook")
        super().__init__(
            "endpoint_m2",
            n,
            m,
            k,
            prefix,
            total,
            total.bit_length() - 1,
            MappingProxyType(
                {
                    "vacuous": vacuous,
                    "endpoint_length": ell,
                    "free_length": free,
                    "prefix_conditioned": bool(prefix),
                    "constraint_scope": "full_oligo",
                    "error_correction": False,
                }
            ),
        )
        for name, value in {
            "_vacuous": vacuous,
            "_ell": ell,
            "_free": free,
            "_left_fixed": left_fixed,
            "_middle_fixed": middle_fixed,
            "_right_fixed": right_fixed,
            "_table": table,
            "_middle_count": middle_count,
            "_right_count": right_count,
        }.items():
            object.__setattr__(self, name, value)

    def _encode_with_attempts(self, index: int) -> tuple[str, int]:
        if self._vacuous:
            suffix = fixed_base_unrank(index, 4, self.n - len(self.prefix))
            return self.prefix + "".join(DNA[symbol] for symbol in suffix), 1
        left_index, remainder = divmod(index, self._middle_count * self._right_count)
        middle_index, right_index = divmod(remainder, self._right_count)
        left = self._left_fixed + fixed_base_unrank(
            left_index, 3, self._ell - len(self._left_fixed)
        )
        middle = self._middle_fixed + fixed_base_unrank(
            middle_index, 4, self._free - len(self._middle_fixed)
        )
        right = _right_unrank(right_index, self._ell, self._right_fixed, self._table)
        return "".join(DNA[symbol] for symbol in left + middle + right), 1

    def _decode_raw(self, sequence: str) -> int:
        if self._vacuous:
            return fixed_base_rank(
                tuple(DNA.index(base) for base in sequence[len(self.prefix) :]), 4
            )
        digits = tuple(DNA.index(base) for base in sequence)
        left, middle, right = (
            digits[: self._ell],
            digits[self._ell : self._ell + self._free],
            digits[self._ell + self._free :],
        )
        if any(symbol == 3 for symbol in left):
            raise ValueError("left_endpoint_alphabet_violation")
        left_rank = fixed_base_rank(left[len(self._left_fixed) :], 3)
        middle_rank = fixed_base_rank(middle[len(self._middle_fixed) :], 4)
        right_rank = _right_rank(right, self._right_fixed, self._table)
        return (left_rank * self._middle_count + middle_rank) * self._right_count + right_rank


def prepare(config: dict) -> PreparedCodec:
    """Dispatch an immutable prepared codec; invalid configuration raises ValueError."""
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    kind = config.get("kind")
    if not isinstance(kind, str):
        raise ValueError("kind must be a known codec string")
    if kind == "endpoint_m2":
        return EndpointCodec(config)
    if kind in {"short_core", "nonce", "two_bit", "gc_h_nonce"}:
        from .short_loop_codec import NonceCodec, ShortCoreCodec, TwoBitCodec

        if kind == "short_core":
            return ShortCoreCodec(config)
        return TwoBitCodec(config) if kind == "two_bit" else NonceCodec(config)
    raise ValueError(f"unknown codec kind: {kind!r}")
