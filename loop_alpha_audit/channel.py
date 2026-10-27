"""Explicit synthetic IDS channel; receiver observations never carry sender truth.

An independent Bernoulli insertion opportunity precedes each original base and
follows the last base. At most one uniformly selected base is inserted per gap.
Each original base is then independently deleted; surviving original bases are
substituted. Inserted bases are not subsequently deleted or substituted. These
are per-opportunity parameters, not an asserted experimental error profile.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Mapping

DNA = "ACGT"


def derived_seed(master: int, *labels: object) -> int:
    """Stable independent streams, unaffected by process scheduling or Python hash."""
    material = "\x1f".join((str(master), *(str(item) for item in labels))).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")


def _validate_sequence(sequence: str) -> None:
    if not isinstance(sequence, str) or any(base not in DNA for base in sequence):
        raise ValueError("sequence must be an uppercase ACGT string")


@dataclass(frozen=True)
class IDSConfig:
    p_insert: float = 0.0
    p_delete: float = 0.0
    p_substitute: float = 0.0
    substitution_matrix: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        for name in ("p_insert", "p_delete", "p_substitute"):
            _probability(getattr(self, name), name)
        if self.substitution_matrix is not None:
            matrix = self.substitution_matrix
            if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
                raise ValueError("substitution matrix must be four by four")
            for index, row in enumerate(matrix):
                if any(not math.isfinite(p) or p < 0 for p in row):
                    raise ValueError("invalid substitution probability")
                if row[index] != 0 or not math.isclose(sum(row), 1.0, abs_tol=1e-12):
                    raise ValueError("substitution rows must sum to one with zero diagonal")

    @classmethod
    def diagnostic(cls, error_scale: float, mode: str = "mixed") -> IDSConfig:
        """Declared diagnostic grid, not fitted or labelled 'realistic'."""
        _probability(error_scale, "error_scale")
        if mode == "mixed":
            return cls(error_scale / 4, error_scale / 4, error_scale / 2)
        if mode == "substitution":
            return cls(p_substitute=error_scale)
        if mode == "indel":
            return cls(p_insert=error_scale / 2, p_delete=error_scale / 2)
        raise ValueError("unknown diagnostic channel")


@dataclass(frozen=True)
class ReadObservation:
    """The ONLY per-read data type admitted by the receiver interface."""

    read_id: str
    sequence: str


@dataclass(frozen=True)
class EditEvent:
    operation: str
    original_coordinate: int
    before: str
    after: str


@dataclass(frozen=True)
class ReadTruth:
    """Private evaluator/channel data; never pass this to reconstruction."""

    read_id: str
    strand_id: int
    original_sequence: str
    events: tuple[EditEvent, ...]
    insertion_opportunities: int
    deletion_opportunities: int
    substitution_opportunities: int


@dataclass(frozen=True)
class ChannelOutput:
    observations: tuple[ReadObservation, ...]
    truth: tuple[ReadTruth, ...]
    dropped_strands: tuple[int, ...]
    coverage_mode: str


def simulate_read(
    sequence: str, config: IDSConfig, rng: random.Random
) -> tuple[str, tuple[EditEvent, ...], int]:
    """Return a read, a separately held edit script, and surviving-base count.

    The script records channel operations, which need not equal the minimum
    Levenshtein distance: different edit operations can cancel each other.
    """
    _validate_sequence(sequence)
    output: list[str] = []
    events: list[EditEvent] = []
    survivors = 0
    for position in range(len(sequence) + 1):
        if rng.random() < config.p_insert:
            inserted = DNA[rng.randrange(4)]
            output.append(inserted)
            events.append(EditEvent("insertion", position, "", inserted))
        if position == len(sequence):
            break
        base = sequence[position]
        if rng.random() < config.p_delete:
            events.append(EditEvent("deletion", position, base, ""))
            continue
        survivors += 1
        if rng.random() < config.p_substitute:
            source = DNA.index(base)
            if config.substitution_matrix is None:
                replaced = DNA[(source + 1 + rng.randrange(3)) % 4]
            else:
                target = rng.random()
                cumulative = 0.0
                replacement_index = -1
                for candidate, probability in enumerate(config.substitution_matrix[source]):
                    cumulative += probability
                    if probability and target < cumulative:
                        replacement_index = candidate
                        break
                if replacement_index < 0:
                    replacement_index = max(
                        i for i, p in enumerate(config.substitution_matrix[source]) if p > 0
                    )
                replaced = DNA[replacement_index]
            if replaced == base:
                raise AssertionError("a substitution must change the retained base")
            output.append(replaced)
            events.append(EditEvent("substitution", position, base, replaced))
        else:
            output.append(base)
    return "".join(output), tuple(events), survivors


def _poisson(mean: float, rng: random.Random) -> int:
    if mean == 0:
        return 0
    # Coverage in the prespecified oligo grid is <=10; refuse a numerically
    # inappropriate very-large-mean use rather than silently underflowing.
    if not 0 < mean <= 100:
        raise ValueError("Poisson coverage mean must lie in [0,100]")
    limit = math.exp(-mean)
    product = 1.0
    count = 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def simulate_pool(
    strands: Mapping[int, str],
    config: IDSConfig,
    *,
    coverage: float,
    seed: int,
    coverage_mode: str = "fixed",
    dropout: float = 0.0,
) -> ChannelOutput:
    """Sample and shuffle a read pool, with source identities isolated in truth.

    Receiver code must be called with ``result.observations``, never with this
    container or its truth member. Fixed and Poisson coverage are distinct modes.
    """
    _probability(dropout, "dropout")
    if not math.isfinite(coverage) or not 0 <= coverage <= 100:
        raise ValueError("coverage must lie in [0,100]")
    if coverage_mode not in ("fixed", "poisson"):
        raise ValueError("coverage_mode must be fixed or poisson")
    if coverage_mode == "fixed" and int(coverage) != coverage:
        raise ValueError("fixed coverage must be an integer")
    # These records are private simulator state. Opaque IDs are assigned only
    # after shuffling, and are never derived from strand IDs or replicate IDs.
    pending: list[tuple[str, int, str, tuple[EditEvent, ...], int]] = []
    dropped: list[int] = []
    for strand_id, sequence in sorted(strands.items()):
        if not isinstance(strand_id, int):
            raise ValueError("strand IDs must be integers")
        _validate_sequence(sequence)
        strand_rng = random.Random(derived_seed(seed, "strand-sampling", strand_id))
        if strand_rng.random() < dropout:
            dropped.append(strand_id)
            continue
        count = int(coverage) if coverage_mode == "fixed" else _poisson(coverage, strand_rng)
        for replicate in range(count):
            rng = random.Random(derived_seed(seed, "read-errors", strand_id, replicate))
            observed, script, survivors = simulate_read(sequence, config, rng)
            pending.append((observed, strand_id, sequence, script, survivors))
    random.Random(derived_seed(seed, "pool-shuffle")).shuffle(pending)
    id_rng = random.Random(derived_seed(seed, "opaque-read-identifiers"))
    used_ids: set[str] = set()
    observations: list[ReadObservation] = []
    truth: list[ReadTruth] = []
    for observed, strand_id, sequence, script, survivors in pending:
        read_id = f"r{id_rng.getrandbits(128):032x}"
        if read_id in used_ids:
            raise AssertionError("duplicate opaque read identifier")
        used_ids.add(read_id)
        observations.append(ReadObservation(read_id, observed))
        truth.append(
            ReadTruth(
                read_id,
                strand_id,
                sequence,
                script,
                len(sequence) + 1,
                len(sequence),
                survivors,
            )
        )
    return ChannelOutput(tuple(observations), tuple(truth), tuple(dropped), coverage_mode)
