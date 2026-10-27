"""Public finite address code and non-oracle, bounded-distance assignment.

The distance certificate concerns complete address strings. In a noisy oligo,
the address/body boundary is not known; assignment tests a declared bounded
prefix-length range and rejects nearest-address ties. CRC and the outer code,
not this distance statement alone, assess the complete receiving pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, groupby

from rapidfuzz.distance import Levenshtein


def longest_run(sequence: str) -> int:
    return max((sum(1 for _ in group) for _, group in groupby(sequence)), default=0)


def reference_edit_distance(left: str, right: str) -> int:
    """Independent scalar dynamic program used to audit the fast distance code."""
    previous = list(range(len(right) + 1))
    for i, left_base in enumerate(left, 1):
        current = [i]
        for j, right_base in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_base != right_base),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class AddressBook:
    sequences: tuple[str, ...]
    required_distance: int = 5

    def __post_init__(self) -> None:
        if not self.sequences or len(set(self.sequences)) != len(self.sequences):
            raise ValueError("address book must be nonempty and contain unique addresses")
        lengths = {len(sequence) for sequence in self.sequences}
        if len(lengths) != 1 or next(iter(lengths)) < 1:
            raise ValueError("addresses must have one positive common length")
        if any(set(sequence) - set("ACGT") for sequence in self.sequences):
            raise ValueError("addresses must contain uppercase DNA only")
        if self.required_distance < 1:
            raise ValueError("required distance must be positive")
        if any(
            Levenshtein.distance(left, right) < self.required_distance
            for left, right in combinations(self.sequences, 2)
        ):
            raise ValueError("address code does not meet its declared distance")

    @property
    def length(self) -> int:
        return len(self.sequences[0])

    def certificate(self, *, independent: bool = True) -> dict[str, object]:
        pairs = list(combinations(self.sequences, 2))
        minimum = min((Levenshtein.distance(a, b) for a, b in pairs), default=None)
        reference_minimum = (
            min((reference_edit_distance(a, b) for a, b in pairs), default=None)
            if independent
            else None
        )
        if independent and reference_minimum != minimum:
            raise AssertionError("independent address-distance check disagrees")
        return {
            "address_count": len(self.sequences),
            "address_nt": self.length,
            "required_levenshtein_distance": self.required_distance,
            "minimum_levenshtein_distance": minimum,
            "reference_minimum_levenshtein_distance": reference_minimum,
            "pairs_checked": len(pairs),
            "sequences": list(self.sequences),
            "alphabet": sorted(set("".join(self.sequences))),
            "gc_counts": [sequence.count("C") + sequence.count("G") for sequence in self.sequences],
            "longest_runs": [longest_run(sequence) for sequence in self.sequences],
            "scope": "complete address strings, not a guarantee for unknown noisy boundaries",
        }


@lru_cache(maxsize=8)
def make_address_book(
    count: int = 63,
    *,
    length: int = 24,
    distance: int = 5,
    seed: int = 2026102301,
    max_trials: int = 500_000,
) -> AddressBook:
    """Deterministic greedy public code: balanced T/C, ending TT, runs <=4.

    This is an explicit finite construction, not an optimal address-code claim.
    Trial exhaustion is a reported error, never an unverified smaller distance.
    """
    if not 1 <= count <= 255 or length < 8 or length % 2 or distance < 1:
        raise ValueError("invalid address-code parameters")
    if max_trials < count:
        raise ValueError("trial budget is smaller than requested code size")
    rng = random.Random(seed)
    selected: list[str] = []
    seen: set[str] = set()
    template = ["C"] * (length // 2) + ["T"] * (length // 2 - 2)
    for _ in range(max_trials):
        candidate_letters = template.copy()
        rng.shuffle(candidate_letters)
        candidate = "".join(candidate_letters) + "TT"
        if candidate in seen:
            continue
        seen.add(candidate)
        if longest_run(candidate) > 4:
            continue
        # Hamming distance is an inexpensive necessary, not sufficient, check.
        if any(sum(a != b for a, b in zip(candidate, prior)) < distance for prior in selected):
            continue
        if any(Levenshtein.distance(candidate, prior) < distance for prior in selected):
            continue
        selected.append(candidate)
        if len(selected) == count:
            return AddressBook(tuple(selected), distance)
    raise RuntimeError(
        f"address construction exhausted {max_trials} trials at size {len(selected)}"
    )


@dataclass(frozen=True)
class AddressAssignment:
    status: str
    strand_index: int | None
    distance: int | None
    boundary_lengths: tuple[int, ...]


def assign_address(
    sequence: str,
    book: AddressBook,
    *,
    correction_radius: int = 2,
    boundary_slack: int = 3,
) -> AddressAssignment:
    """Use only observed bases and the public code, with explicit tie rejection."""
    if correction_radius < 0 or boundary_slack < 0:
        raise ValueError("assignment radii must be nonnegative")
    if not isinstance(sequence, str) or set(sequence) - set("ACGT"):
        return AddressAssignment("INVALID_READ", None, None, ())
    low = max(1, book.length - boundary_slack)
    high = min(len(sequence), book.length + boundary_slack)
    if high < low:
        return AddressAssignment("TOO_SHORT", None, None, ())
    scores: list[tuple[int, tuple[int, ...]]] = []
    for address in book.sequences:
        distances = [
            (Levenshtein.distance(address, sequence[:end]), end) for end in range(low, high + 1)
        ]
        best = min(distance for distance, _ in distances)
        scores.append((best, tuple(end for distance, end in distances if distance == best)))
    best = min(distance for distance, _ in scores)
    if best > correction_radius:
        return AddressAssignment("OUT_OF_RADIUS", None, best, ())
    winners = [index for index, (distance, _) in enumerate(scores) if distance == best]
    if len(winners) != 1:
        return AddressAssignment("AMBIGUOUS", None, best, ())
    index = winners[0]
    return AddressAssignment("OK", index, best, scores[index][1])
