"""Prepared, exact all-window SFT and generic endpoint rank/unrank.

This is a noiseless inverse mapping, not an error-correcting decoder.  The
window-set product is a subcode of the complete triangular endpoint language.
For lengths shorter than m, all-window SFT constraints are vacuous; this finite
boundary differs from a construction that restricts every letter to a subalphabet.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import Iterable, Sequence

Word = tuple[int, ...]


def _word(sequence: Sequence[int], q: int) -> Word:
    result = tuple(sequence)
    if any(not isinstance(a, Integral) or not 0 <= a < q for a in result):
        raise ValueError("sequence contains a noninteger or out-of-alphabet symbol")
    return tuple(map(int, result))


def _base_unrank(index: int, q: int, length: int) -> Word:
    if not isinstance(index, int) or not 0 <= index < q**length:
        raise ValueError("index is outside the free-word codebook")
    result = [0] * length
    for i in range(length - 1, -1, -1):
        index, result[i] = divmod(index, q)
    return tuple(result)


def _base_rank(word: Sequence[int], q: int) -> int:
    value = 0
    for a in _word(word, q):
        value = q * value + a
    return value


@dataclass(frozen=True, init=False)
class PreparedWindowSFT:
    """Suffix-count DP prepared once for a fixed length and allowed m-windows."""

    q: int
    m: int
    length: int
    allowed: frozenset[Word]
    states: tuple[Word, ...]
    transitions: tuple[tuple[int, ...], ...]
    suffix_counts: tuple[tuple[int, ...], ...]

    def __init__(self, q: int, m: int, allowed: Iterable[Sequence[int]], length: int):
        if (
            not isinstance(q, int)
            or not isinstance(m, int)
            or not isinstance(length, int)
            or q < 2
            or m < 1
            or length < 0
        ):
            raise ValueError("require integer q>=2, m>=1, length>=0")
        chosen = frozenset(_word(w, q) for w in allowed)
        if any(len(w) != m for w in chosen):
            raise ValueError("allowed set contains a word of the wrong length")
        states = tuple(w for ell in range(m) for w in product(range(q), repeat=ell))
        index = {state: i for i, state in enumerate(states)}
        transitions = []
        for state in states:
            row = []
            for symbol in range(q):
                window = state + (symbol,)
                if len(window) == m and window not in chosen:
                    row.append(-1)
                else:
                    suffix = window[-(m - 1) :] if m > 1 else ()
                    row.append(index[suffix])
            transitions.append(tuple(row))
        rows = [(1,) * len(states)]
        for _ in range(length):
            previous = rows[-1]
            rows.append(
                tuple(sum(previous[target] for target in row if target >= 0) for row in transitions)
            )
        for name, value in (
            ("q", q),
            ("m", m),
            ("length", length),
            ("allowed", chosen),
            ("states", states),
            ("transitions", tuple(transitions)),
            ("suffix_counts", tuple(rows)),
        ):
            object.__setattr__(self, name, value)

    @property
    def codebook_size(self) -> int:
        return self.suffix_counts[self.length][0]

    def unrank(self, index: int) -> Word:
        if not isinstance(index, int) or not 0 <= index < self.codebook_size:
            raise ValueError("index is outside the prepared SFT codebook")
        state, result = 0, []
        for remaining in range(self.length - 1, -1, -1):
            for symbol, target in enumerate(self.transitions[state]):
                if target < 0:
                    continue
                block = self.suffix_counts[remaining][target]
                if index < block:
                    state = target
                    result.append(symbol)
                    break
                index -= block
            else:
                raise AssertionError("exact suffix counts failed to choose a transition")
        return tuple(result)

    def rank(self, word: Sequence[int]) -> int:
        sequence = _word(word, self.q)
        if len(sequence) != self.length:
            raise ValueError("word length does not match the prepared SFT")
        state, rank = 0, 0
        for actual, remaining in zip(sequence, range(self.length - 1, -1, -1)):
            row = self.transitions[state]
            rank += sum(
                self.suffix_counts[remaining][target] for target in row[:actual] if target >= 0
            )
            state = row[actual]
            if state < 0:
                raise ValueError("word contains a forbidden SFT transition")
        return rank


@dataclass(frozen=True, init=False)
class EndpointMapper:
    n: int
    m: int
    k: int
    q: int
    theta: tuple[int, ...]
    allowed: frozenset[Word]
    perpendicular: frozenset[Word]
    regime: str
    endpoint_length: int
    free_length: int
    left: PreparedWindowSFT | None
    right: PreparedWindowSFT | None
    left_count: int
    middle_count: int
    right_count: int
    codebook_size: int

    def __init__(
        self,
        n: int,
        m: int,
        k: int,
        allowed: Iterable[Sequence[int]],
        *,
        q: int = 4,
        theta: Sequence[int] = (3, 2, 1, 0),
    ):
        if any(not isinstance(x, int) for x in (n, m, k, q)) or n < 0 or m < 1 or k < 0 or q < 2:
            raise ValueError("invalid integer endpoint parameters")
        theta = tuple(theta)
        if (
            len(theta) != q
            or sorted(theta) != list(range(q))
            or any(theta[theta[a]] != a for a in range(q))
        ):
            raise ValueError("theta must be an involution of range(q)")
        chosen = frozenset(_word(w, q) for w in allowed)
        if any(len(w) != m for w in chosen):
            raise ValueError("allowed set contains a word of the wrong length")
        rc = {tuple(theta[a] for a in w[::-1]) for w in chosen}
        perpendicular = frozenset(w for w in product(range(q), repeat=m) if w not in rc)
        if n < 2 * m + k:
            # No pair of length-m windows has the required nonoverlap+gap.
            regime, ell, free = "vacuous", 0, n
            left = right = None
            lc, mc, rr = 1, q**n, 1
        else:
            distance = m + k
            ell, free = n - distance, 2 * distance - n
            if free < 0:
                raise ValueError(
                    "unsupported coupled regime: endpoint geometry requires 2*(m+k)>=n"
                )
            regime = "endpoint_product_subcode"
            left = PreparedWindowSFT(q, m, chosen, ell)
            right = PreparedWindowSFT(q, m, perpendicular, ell)
            lc, mc, rr = left.codebook_size, q**free, right.codebook_size
        attributes = {
            "n": n,
            "m": m,
            "k": k,
            "q": q,
            "theta": theta,
            "allowed": chosen,
            "perpendicular": perpendicular,
            "regime": regime,
            "endpoint_length": ell,
            "free_length": free,
            "left": left,
            "right": right,
            "left_count": lc,
            "middle_count": mc,
            "right_count": rr,
            "codebook_size": lc * mc * rr,
        }
        for name, value in attributes.items():
            object.__setattr__(self, name, value)

    @property
    def payload_bits(self) -> int:
        if self.codebook_size == 0:
            raise ValueError("empty endpoint codebook has no binary payload length")
        return self.codebook_size.bit_length() - 1

    @property
    def K(self) -> int:
        return self.payload_bits

    def encode_index(self, index: int) -> Word:
        if not isinstance(index, int) or not 0 <= index < self.codebook_size:
            raise ValueError("index is outside the endpoint product codebook")
        if self.regime == "vacuous":
            return _base_unrank(index, self.q, self.n)
        left_index, remainder = divmod(index, self.middle_count * self.right_count)
        middle_index, right_index = divmod(remainder, self.right_count)
        return (
            self.left.unrank(left_index)
            + _base_unrank(middle_index, self.q, self.free_length)
            + self.right.unrank(right_index)
        )

    def decode_index(self, word: Sequence[int]) -> int:
        sequence = _word(word, self.q)
        if len(sequence) != self.n:
            raise ValueError("word length does not match the endpoint mapper")
        if self.regime == "vacuous":
            return _base_rank(sequence, self.q)
        ell = self.endpoint_length
        li = self.left.rank(sequence[:ell])
        mi = _base_rank(sequence[ell : ell + self.free_length], self.q)
        ri = self.right.rank(sequence[ell + self.free_length :])
        return (li * self.middle_count + mi) * self.right_count + ri

    def encode_payload(self, payload: int) -> Word:
        if not isinstance(payload, int) or not 0 <= payload < 1 << self.payload_bits:
            raise ValueError("binary payload is outside [0,2^K)")
        return self.encode_index(payload)

    def decode_payload(self, word: Sequence[int]) -> int:
        index = self.decode_index(word)
        if index >= 1 << self.payload_bits:
            raise ValueError("valid codeword has an unused rank outside the binary subcode")
        return index

    def encode_bits(self, bits: Sequence[int]) -> Word:
        values = _word(bits, 2)
        if len(values) != self.payload_bits:
            raise ValueError("binary message length must equal K exactly")
        return self.encode_payload(_base_rank(values, 2))

    def decode_bits(self, word: Sequence[int]) -> Word:
        return _base_unrank(self.decode_payload(word), 2, self.payload_bits)


GenericEndpointMapper = EndpointMapper
