"""Message-preserving short-loop subcodes and explicit-failure nonce mapping.

The TC-majority component is an inherited dominant-core construction, not a
new capacity theorem. Its product with an exact-GC binary RLL component is a
conservative reversible subcode. Nonce screening is a different partial map;
it must not be described as a guaranteed fixed-length constrained code.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from types import MappingProxyType

from .encoder import EXACT_TABLE_CACHE, fixed_base_rank, fixed_base_unrank, table_size_upper_bound
from .prepared_codec import DNA, EncodingFailure, PreparedCodec, common_config, integer

PAIR_TO_DNA = {(1, 0): "T", (1, 1): "C", (0, 0): "A", (0, 1): "G"}


def avoids_rc_windows(sequence: str, m: int, k: int) -> bool:
    """Fixed-m O(n) rolling-code check of all separated m-window pairs.

    Stores the earliest start for each window, not sender truth or a sampled
    subset. Tests compare it with the independent all-stem definition.
    """
    integer(m, "m", 1)
    integer(k, "k")
    if not isinstance(sequence, str) or any(base not in DNA for base in sequence):
        raise ValueError("invalid_DNA_symbol")
    if len(sequence) < 2 * m + k:
        return True
    mask = (1 << (2 * m)) - 1
    shift = 2 * (m - 1)
    word = reverse = 0
    earliest: dict[int, int] = {}
    for position, base in enumerate(sequence):
        digit = DNA.index(base)
        word = ((word << 2) | digit) & mask
        reverse = (reverse >> 2) | ((3 - digit) << shift)
        if position + 1 < m:
            continue
        start = position - m + 1
        if reverse in earliest and start - earliest[reverse] >= m + k:
            return False
        earliest.setdefault(word, start)
    return True


def full_oligo_failure(
    sequence: str,
    m: int,
    k: int,
    *,
    gc_low: Fraction = Fraction(2, 5),
    gc_high: Fraction = Fraction(3, 5),
    max_homopolymer: int = 4,
    check_rc: bool = True,
) -> str | None:
    if not isinstance(sequence, str) or any(base not in DNA for base in sequence):
        return "invalid_DNA_symbol"
    count = sum(base in "GC" for base in sequence)
    if count * gc_low.denominator < len(sequence) * gc_low.numerator:
        return "GC_below_bound"
    if count * gc_high.denominator > len(sequence) * gc_high.numerator:
        return "GC_above_bound"
    previous, run = "", 0
    for base in sequence:
        run = run + 1 if base == previous else 1
        previous = base
        if run > max_homopolymer:
            return "homopolymer_bound_exceeded"
    if check_rc and not avoids_rc_windows(sequence, m, k):
        return "RC_window_violation"
    return None


def _binary_word(bits: tuple[int, ...], length: int) -> None:
    if len(bits) != length or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("invalid_binary_component")


class MembershipEnumerator:
    """Prefix-conditioned strict-majority odd-window binary language."""

    def __init__(self, length: int, prefix: tuple[int, ...], w: int):
        self.length, self.prefix, self.w = length, prefix, w
        states = (1 << w) - 1
        transitions: list[tuple[int, int]] = []
        for history_length in range(w):
            for mask in range(1 << history_length):
                following = []
                for bit in (0, 1):
                    window = (mask << 1) | bit
                    if history_length == w - 1:
                        next_mask = window & ((1 << (w - 1)) - 1)
                        nxt = (1 << (w - 1)) - 1 + next_mask if window.bit_count() > w // 2 else -1
                    else:
                        nxt = (1 << (history_length + 1)) - 1 + window
                    following.append(nxt)
                transitions.append(tuple(following))
        self.transitions = tuple(transitions)
        state = 0
        for bit in prefix:
            state = self.transitions[state][bit]
            if state < 0:
                raise ValueError("prefix_violates_TC_majority_core")
        self.state = state

        def build() -> tuple:
            rows = [(1,) * states]
            for _ in range(length):
                previous = rows[-1]
                rows.append(
                    tuple(sum(previous[nxt] for nxt in row if nxt >= 0) for row in self.transitions)
                )
            return tuple(rows)

        self.table = EXACT_TABLE_CACHE.get(
            ("majority", w, length), build, table_size_upper_bound(length + 1, states, 1)
        )
        self.M = self.table[length][state]

    def unrank(self, index: int) -> tuple[int, ...]:
        if type(index) is not int or not 0 <= index < self.M:
            raise ValueError("membership_rank_outside_codebook")
        state = self.state
        result = []
        for remaining in range(self.length - 1, -1, -1):
            for bit, nxt in enumerate(self.transitions[state]):
                block = self.table[remaining][nxt] if nxt >= 0 else 0
                if index < block:
                    result.append(bit)
                    state = nxt
                    break
                index -= block
            else:
                raise AssertionError("majority unrank exhausted exact branches")
        return tuple(result)

    def rank(self, word: tuple[int, ...]) -> int:
        _binary_word(word, self.length)
        index, state = 0, self.state
        for position, bit in enumerate(word):
            nxt0, nxt1 = self.transitions[state]
            if bit and nxt0 >= 0:
                index += self.table[self.length - position - 1][nxt0]
            state = nxt1 if bit else nxt0
            if state < 0:
                raise ValueError("TC_majority_core_violation")
        return index


class GCEnumerator:
    """Exact-one-count binary RLL language, conditioned on the entire prefix."""

    def __init__(self, length: int, prefix: tuple[int, ...], total_gc: int, max_run: int = 4):
        self.length, self.prefix, self.max_run = length, prefix, max_run
        self.needed = total_gc - sum(prefix)
        states = 1 + 2 * max_run
        transitions = [(1, 1 + max_run)]
        for last in (0, 1):
            for run in range(1, max_run + 1):
                transitions.append(
                    tuple(
                        1 + bit * max_run + (run if bit == last else 0)
                        if bit != last or run < max_run
                        else -1
                        for bit in (0, 1)
                    )
                )
        self.transitions = tuple(transitions)
        state = 0
        for bit in prefix:
            state = self.transitions[state][bit]
            if state < 0:
                raise ValueError("prefix_violates_GC_bit_run_bound")
        self.state = state

        def build() -> tuple:
            rows = [((1,) * states,)]
            for remaining in range(1, length + 1):
                previous = rows[-1]
                layer = []
                for ones in range(remaining + 1):
                    layer.append(
                        tuple(
                            sum(
                                previous[ones - bit][nxt]
                                for bit, nxt in enumerate(pair)
                                if nxt >= 0 and 0 <= ones - bit < remaining
                            )
                            for pair in self.transitions
                        )
                    )
                rows.append(tuple(layer))
            return tuple(rows)

        self.table = EXACT_TABLE_CACHE.get(
            ("exact-gc-rll", length, max_run),
            build,
            table_size_upper_bound(length + 1, (length + 1) * states, 1),
        )
        self.M = self.table[length][self.needed][state] if 0 <= self.needed <= length else 0

    def _block(self, remaining: int, ones: int, state: int) -> int:
        return self.table[remaining][ones][state] if state >= 0 and 0 <= ones <= remaining else 0

    def unrank(self, index: int) -> tuple[int, ...]:
        if type(index) is not int or not 0 <= index < self.M:
            raise ValueError("GC_rank_outside_codebook")
        ones, state, result = self.needed, self.state, []
        for remaining in range(self.length - 1, -1, -1):
            for bit, nxt in enumerate(self.transitions[state]):
                block = self._block(remaining, ones - bit, nxt)
                if index < block:
                    result.append(bit)
                    state, ones = nxt, ones - bit
                    break
                index -= block
            else:
                raise AssertionError("GC unrank exhausted exact branches")
        return tuple(result)

    def rank(self, word: tuple[int, ...]) -> int:
        _binary_word(word, self.length)
        ones, state, index = self.needed, self.state, 0
        for position, bit in enumerate(word):
            nxt0, nxt1 = self.transitions[state]
            if bit:
                index += self._block(self.length - position - 1, ones, nxt0)
            state = nxt1 if bit else nxt0
            ones -= bit
            if state < 0 or ones < 0:
                raise ValueError("GC_count_or_run_violation")
        if ones:
            raise ValueError("GC_count_violation")
        return index


class ShortCoreCodec(PreparedCodec):
    def __init__(self, config: dict):
        n, m, k, prefix = common_config(config)
        if m not in (6, 7):
            raise ValueError("short_core supports target m=6 or m=7")
        w = 5 if m == 6 else 7
        tc_prefix = tuple(int(base in "TC") for base in prefix)
        gc_prefix = tuple(int(base in "GC") for base in prefix)
        membership = MembershipEnumerator(n - len(prefix), tc_prefix, w)
        gc = GCEnumerator(n - len(prefix), gc_prefix, n // 2)
        total = membership.M * gc.M
        if total == 0:
            raise ValueError("empty_prefix_conditioned_short_core_codebook")
        super().__init__(
            "short_core",
            n,
            m,
            k,
            prefix,
            total,
            total.bit_length() - 1,
            MappingProxyType(
                {
                    "core_window": w,
                    "target_m": m,
                    "total_gc": n // 2,
                    "GC_bit_max_run": 4,
                    "membership_count": membership.M,
                    "GC_component_count": gc.M,
                    "constraint_scope": "full_oligo",
                    "zero_failure_guarantee": True,
                    "construction": "inherited_TC_majority_core_plus_stricter_GC_bit_RLL",
                }
            ),
        )
        object.__setattr__(self, "_membership", membership)
        object.__setattr__(self, "_gc", gc)

    def _encode_with_attempts(self, index: int) -> tuple[str, int]:
        tc_rank, gc_rank = divmod(index, self._gc.M)
        tc = self._membership.prefix + self._membership.unrank(tc_rank)
        gc = self._gc.prefix + self._gc.unrank(gc_rank)
        return "".join(PAIR_TO_DNA[pair] for pair in zip(tc, gc, strict=True)), 1

    def _decode_raw(self, sequence: str) -> int:
        body = sequence[len(self.prefix) :]
        tc_rank = self._membership.rank(tuple(int(base in "TC") for base in body))
        gc_rank = self._gc.rank(tuple(int(base in "GC") for base in body))
        return tc_rank * self._gc.M + gc_rank


class TwoBitCodec(PreparedCodec):
    def __init__(self, config: dict):
        n, m, k, prefix = common_config(config)
        total = 4 ** (n - len(prefix))
        super().__init__(
            "two_bit",
            n,
            m,
            k,
            prefix,
            total,
            total.bit_length() - 1,
            MappingProxyType(
                {
                    "constraint_scope": "none",
                    "zero_failure_guarantee": True,
                    "error_correction": False,
                }
            ),
        )

    def _encode_with_attempts(self, index: int) -> tuple[str, int]:
        word = fixed_base_unrank(index, 4, self.n - len(self.prefix))
        return self.prefix + "".join(DNA[symbol] for symbol in word), 1

    def _decode_raw(self, sequence: str) -> int:
        return fixed_base_rank(tuple(DNA.index(base) for base in sequence[len(self.prefix) :]), 4)


class NonceCodec(PreparedCodec):
    def __init__(self, config: dict):
        n, m, k, prefix = common_config(
            config, ("nonce_nt", "nonce_budget", "gc_low", "gc_high", "max_homopolymer")
        )
        nonce_nt = integer(config.get("nonce_nt", 4), "nonce_nt", 1)
        budget = integer(config.get("nonce_budget", 64), "nonce_budget", 1)
        if budget > 4**nonce_nt:
            raise ValueError("nonce_budget exceeds transmitted nonce alphabet")
        body_length = n - len(prefix) - nonce_nt
        if body_length < 0:
            raise ValueError("fixed_prefix_and_nonce_exceed_total_n")
        try:
            low, high = (
                Fraction(str(config.get("gc_low", "0.4"))),
                Fraction(str(config.get("gc_high", "0.6"))),
            )
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("invalid_GC_bounds") from exc
        if not 0 <= low <= high <= 1:
            raise ValueError("require 0<=gc_low<=gc_high<=1")
        max_h = integer(config.get("max_homopolymer", 4), "max_homopolymer", 1)
        check_rc = config["kind"] == "nonce"
        prefix_failure = full_oligo_failure(
            prefix,
            m,
            k,
            gc_low=Fraction(0),
            gc_high=Fraction(1),
            max_homopolymer=max_h,
            check_rc=check_rc,
        )
        if prefix_failure:
            raise ValueError("fixed_prefix_" + prefix_failure)
        prefix_gc = sum(base in "GC" for base in prefix)
        if prefix_gc > high * n or prefix_gc + n - len(prefix) < low * n:
            raise ValueError("fixed_prefix_has_infeasible_total_GC")
        bits = 2 * body_length
        super().__init__(
            config["kind"],
            n,
            m,
            k,
            prefix,
            None,
            bits,
            MappingProxyType(
                {
                    "nonce_nt": nonce_nt,
                    "nonce_budget": budget,
                    "gc_low": str(low),
                    "gc_high": str(high),
                    "max_homopolymer": max_h,
                    "constraint_scope": "full_oligo",
                    "checks_RC": check_rc,
                    "zero_failure_guarantee": False,
                    "M_is_undefined_for_partial_mapping": True,
                    "decode_accepts_legal_nonce_aliases": True,
                }
            ),
        )
        protocol = {
            "domain": "loop-alpha-whole-body-shake256-v1",
            "kind": self.kind,
            "n": n,
            "m": m,
            "k": k,
            "prefix": prefix,
            "nonce_nt": nonce_nt,
        }
        domain = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode() + b"\0"
        for name, value in {
            "_nonce_nt": nonce_nt,
            "_budget": budget,
            "_body_length": body_length,
            "_gc_low": low,
            "_gc_high": high,
            "_max_h": max_h,
            "_check_rc": check_rc,
            "_domain": domain,
        }.items():
            object.__setattr__(self, name, value)

    def _mask(self, nonce: int) -> int:
        encoded = nonce.to_bytes((2 * self._nonce_nt + 7) // 8, "big")
        raw = hashlib.shake_256(self._domain + encoded).digest((self.K + 7) // 8)
        return int.from_bytes(raw, "big") & ((1 << self.K) - 1)

    def _failure(self, sequence: str) -> str | None:
        return full_oligo_failure(
            sequence,
            self.m,
            self.k,
            gc_low=self._gc_low,
            gc_high=self._gc_high,
            max_homopolymer=self._max_h,
            check_rc=self._check_rc,
        )

    def _encode_with_attempts(self, index: int) -> tuple[str, int]:
        for nonce in range(self._budget):
            header = "".join(DNA[symbol] for symbol in fixed_base_unrank(nonce, 4, self._nonce_nt))
            scrambled = index ^ self._mask(nonce)
            body = "".join(
                DNA[symbol] for symbol in fixed_base_unrank(scrambled, 4, self._body_length)
            )
            sequence = self.prefix + header + body
            if self._failure(sequence) is None:
                return sequence, nonce + 1
        raise EncodingFailure("candidate_budget_exhausted", self._budget)

    def _decode_raw(self, sequence: str) -> int:
        failure = self._failure(sequence)
        if failure:
            raise ValueError(failure)
        header_end = len(self.prefix) + self._nonce_nt
        nonce = fixed_base_rank(
            tuple(DNA.index(base) for base in sequence[len(self.prefix) : header_end]), 4
        )
        if nonce >= self._budget:
            raise ValueError("nonce_outside_public_budget")
        scrambled = fixed_base_rank(tuple(DNA.index(base) for base in sequence[header_end:]), 4)
        return scrambled ^ self._mask(nonce)
