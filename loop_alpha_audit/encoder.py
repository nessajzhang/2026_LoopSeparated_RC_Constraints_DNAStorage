"""Exact enumerative encoder for the certified DNA m=2 high-alpha subcode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .model import Word
from .paired import decomposition_parameters


DNA_THETA = (3, 2, 1, 0)
LEFT_ALPHABET = (0, 1, 2)  # A,C,G
RIGHT_ADJACENCY = (
    (1, 1, 1, 1),
    (1, 0, 0, 0),
    (1, 0, 0, 0),
    (1, 0, 0, 0),
)


def right_sft_count_fast(length: int) -> int:
    """Count right-endpoint words by exact binary matrix powering.

    The total counts satisfy ``f_L=f_{L-1}+3f_{L-2}``, with
    ``f_1=4`` and ``f_2=7``. Matrix powering avoids storing all suffix-count
    rows when only a very large finite-block rate is needed.
    """

    if length < 1:
        raise ValueError("SFT words must have positive length")
    if length == 1:
        return 4

    def multiply(
        left: tuple[tuple[int, int], tuple[int, int]],
        right: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (
                left[0][0] * right[0][0] + left[0][1] * right[1][0],
                left[0][0] * right[0][1] + left[0][1] * right[1][1],
            ),
            (
                left[1][0] * right[0][0] + left[1][1] * right[1][0],
                left[1][0] * right[0][1] + left[1][1] * right[1][1],
            ),
        )

    exponent = length - 2
    result = ((1, 0), (0, 1))
    base = ((1, 3), (1, 0))
    while exponent:
        if exponent & 1:
            result = multiply(result, base)
        base = multiply(base, base)
        exponent >>= 1
    return result[0][0] * 7 + result[0][1] * 4


def fixed_base_unrank(index: int, base: int, length: int) -> Word:
    if base < 2 or length < 0 or not 0 <= index < base**length:
        raise ValueError("index is outside the fixed-base codebook")
    digits = [0] * length
    for position in range(length - 1, -1, -1):
        index, digits[position] = divmod(index, base)
    return tuple(digits)


def fixed_base_rank(word: Sequence[int], base: int) -> int:
    result = 0
    for symbol in word:
        if not 0 <= int(symbol) < base:
            raise ValueError("word contains a symbol outside the requested base")
        result = result * base + int(symbol)
    return result


@dataclass(frozen=True)
class SFTEnumerator:
    adjacency: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        size = len(self.adjacency)
        if size < 1 or any(len(row) != size for row in self.adjacency):
            raise ValueError("adjacency must be nonempty and square")
        if any(value not in (0, 1) for row in self.adjacency for value in row):
            raise ValueError("adjacency must be binary")

    @property
    def alphabet_size(self) -> int:
        return len(self.adjacency)

    def suffix_counts(self, length: int) -> tuple[tuple[int, ...], ...]:
        """Return counts indexed by remaining edge count and current state."""

        if length < 1:
            raise ValueError("SFT words must have positive length")
        rows: list[tuple[int, ...]] = [(1,) * self.alphabet_size]
        for _ in range(length - 1):
            previous = rows[-1]
            current = tuple(
                sum(self.adjacency[state][nxt] * previous[nxt] for nxt in range(self.alphabet_size))
                for state in range(self.alphabet_size)
            )
            rows.append(current)
        return tuple(rows)

    def count(self, length: int) -> int:
        counts = self.suffix_counts(length)
        return sum(counts[length - 1])

    def unrank(self, index: int, length: int) -> Word:
        counts = self.suffix_counts(length)
        total = sum(counts[length - 1])
        if not 0 <= index < total:
            raise ValueError("index is outside the SFT codebook")

        result: list[int] = []
        remaining_edges = length - 1
        current = -1
        for symbol in range(self.alphabet_size):
            block = counts[remaining_edges][symbol]
            if index < block:
                current = symbol
                result.append(symbol)
                break
            index -= block
        if current < 0:
            raise AssertionError("failed to select an initial SFT state")

        while remaining_edges:
            next_remaining = remaining_edges - 1
            for nxt in range(self.alphabet_size):
                if not self.adjacency[current][nxt]:
                    continue
                block = counts[next_remaining][nxt]
                if index < block:
                    result.append(nxt)
                    current = nxt
                    break
                index -= block
            else:
                raise AssertionError("failed to select an SFT transition")
            remaining_edges = next_remaining
        return tuple(result)

    def rank(self, word: Sequence[int]) -> int:
        sequence = tuple(int(symbol) for symbol in word)
        if not sequence:
            raise ValueError("SFT words must have positive length")
        if any(not 0 <= symbol < self.alphabet_size for symbol in sequence):
            raise ValueError("word contains a symbol outside the SFT alphabet")
        if any(not self.adjacency[left][right] for left, right in zip(sequence, sequence[1:])):
            raise ValueError("word violates the SFT adjacency")

        counts = self.suffix_counts(len(sequence))
        remaining_edges = len(sequence) - 1
        index = sum(counts[remaining_edges][symbol] for symbol in range(sequence[0]))
        current = sequence[0]
        for actual in sequence[1:]:
            next_remaining = remaining_edges - 1
            index += sum(
                counts[next_remaining][candidate]
                for candidate in range(actual)
                if self.adjacency[current][candidate]
            )
            current = actual
            remaining_edges = next_remaining
        return index


RIGHT_ENUMERATOR = SFTEnumerator(RIGHT_ADJACENCY)


@dataclass(frozen=True)
class SegmentedDNAEncoder:
    """Exact rank/unrank map for the certified high-alpha DNA subcode."""

    n: int
    k: int
    m: int = 2

    def __post_init__(self) -> None:
        if self.m != 2:
            raise ValueError("the certified segmented encoder currently supports DNA m=2")
        _, distance, r, free = decomposition_parameters(self.n, self.m, self.k)
        if r < 0 or free < 0:
            raise ValueError("requires r>=0 and 2(m+k)>=n")
        if distance > self.n:
            raise ValueError("use the vacuous full-alphabet encoder when m+k>n")

    @property
    def distance(self) -> int:
        return self.m + self.k

    @property
    def endpoint_length(self) -> int:
        return self.n - self.distance

    @property
    def free_length(self) -> int:
        return 2 * self.distance - self.n

    @property
    def left_count(self) -> int:
        return len(LEFT_ALPHABET) ** self.endpoint_length

    @property
    def middle_count(self) -> int:
        return 4**self.free_length

    @property
    def right_count(self) -> int:
        return right_sft_count_fast(self.endpoint_length)

    @property
    def codebook_size(self) -> int:
        return self.left_count * self.middle_count * self.right_count

    def encode_index(self, index: int) -> Word:
        if not 0 <= index < self.codebook_size:
            raise ValueError("index is outside the segmented codebook")
        left_index, remainder = divmod(index, self.middle_count * self.right_count)
        middle_index, right_index = divmod(remainder, self.right_count)
        left_digits = fixed_base_unrank(left_index, 3, self.endpoint_length)
        left = tuple(LEFT_ALPHABET[digit] for digit in left_digits)
        middle = fixed_base_unrank(middle_index, 4, self.free_length)
        right = RIGHT_ENUMERATOR.unrank(right_index, self.endpoint_length)
        result = left + middle + right
        if len(result) != self.n:
            raise AssertionError("segmented encoder produced an incorrect length")
        return result

    def decode_index(self, word: Sequence[int]) -> int:
        sequence = tuple(int(symbol) for symbol in word)
        if len(sequence) != self.n:
            raise ValueError("word length does not match the encoder")
        ell = self.endpoint_length
        left = sequence[:ell]
        middle = sequence[ell : ell + self.free_length]
        right = sequence[ell + self.free_length :]
        if any(symbol not in LEFT_ALPHABET for symbol in left):
            raise ValueError("left endpoint violates the certified alphabet")
        left_index = fixed_base_rank(left, 3)
        middle_index = fixed_base_rank(middle, 4)
        right_index = RIGHT_ENUMERATOR.rank(right)
        return (left_index * self.middle_count + middle_index) * self.right_count + right_index
