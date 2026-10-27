"""Bounded, non-oracle reconstruction of an unordered noisy oligo read pool.

Inputs are only observed read IDs/bases and the public protocol. Public-address
assignment precedes medoid alignment and voting. Exact mapping inversion/CRC
checks and bounded single-edit candidate search precede the across-strand RS
decoder. This is an explicit benchmark receiver, not maximum-likelihood IDS
decoding and not an error-correcting property of rank/unrank.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Iterable, Sequence

from rapidfuzz.distance import Levenshtein

from .addresses import assign_address
from .channel import ReadObservation
from .framing import PublicProtocol, decode_frame, outer_decode


def _aligned_votes(reference: str, read: str) -> tuple[list[str], list[str]]:
    """Align an observation to an observed medoid; no transmitted template."""
    bases = [""] * len(reference)
    gaps = [""] * (len(reference) + 1)
    source = target = 0
    for edit in Levenshtein.editops(reference, read):
        while source < edit.src_pos and target < edit.dest_pos:
            bases[source] = read[target]
            source += 1
            target += 1
        if edit.tag == "replace":
            bases[source] = read[target]
            source += 1
            target += 1
        elif edit.tag == "delete":
            source += 1
        elif edit.tag == "insert":
            gaps[source] += read[target]
            target += 1
        else:
            raise AssertionError("unknown Levenshtein operation")
    while source < len(reference) and target < len(read):
        bases[source] = read[target]
        source += 1
        target += 1
    while target < len(read):
        gaps[source] += read[target]
        target += 1
    if source < len(reference):
        # Only source deletions can remain; the empty initialized votes are
        # precisely their contribution. No bases are fabricated from a sender.
        source = len(reference)
    if source != len(reference) or target != len(read):
        raise AssertionError("alignment did not consume both observed strings")
    return bases, gaps


def medoid_consensus(reads: Sequence[str]) -> tuple[str, str]:
    """Deterministic edit-distance medoid and majority alignment consensus.

    Ties retain the observed medoid base; insertion ties prefer no insertion.
    Multiple observations of the same sequence are retained as repeated reads.
    """
    if not reads:
        raise ValueError("cannot form consensus without observations")
    medoid = min(
        set(reads),
        key=lambda candidate: (
            sum(Levenshtein.distance(candidate, read) for read in reads),
            candidate,
        ),
    )
    base_votes = [Counter() for _ in medoid]
    gap_votes = [Counter() for _ in range(len(medoid) + 1)]
    for read in reads:
        bases, gaps = _aligned_votes(medoid, read)
        for index, base in enumerate(bases):
            base_votes[index][base] += 1
        for index, inserted in enumerate(gaps):
            gap_votes[index][inserted] += 1
    output: list[str] = []
    for index in range(len(medoid) + 1):
        gaps = gap_votes[index]
        insertion = min(gaps, key=lambda value: (-gaps[value], value != "", value))
        output.append(insertion)
        if index < len(medoid):
            bases = base_votes[index]
            base = min(bases, key=lambda value: (-bases[value], value != medoid[index], value))
            output.append(base)
    return "".join(output), medoid


def one_edit_candidates(sequence: str, length: int) -> Iterable[str]:
    """All strings of the declared length within one edit, in fixed order."""
    if len(sequence) == length:
        yield sequence
        for position, before in enumerate(sequence):
            for after in "ACGT":
                if before != after:
                    yield sequence[:position] + after + sequence[position + 1 :]
    elif len(sequence) + 1 == length:
        for position in range(len(sequence) + 1):
            for base in "ACGT":
                yield sequence[:position] + base + sequence[position:]
    elif len(sequence) == length + 1:
        for position in range(len(sequence)):
            yield sequence[:position] + sequence[position + 1 :]


@dataclass(frozen=True)
class FrameReconstruction:
    strand_index: int  # Inferred public address, not a simulator source label.
    status: str
    read_count: int
    reads_used: int
    sequence: str | None
    payload: bytes | None
    mode: str | None
    candidates_checked: int
    crc_failures: int
    search_truncated: bool
    elapsed_ns: int
    failure_reason: str | None


def reconstruct_cluster(
    reads: Sequence[ReadObservation],
    strand_index: int,
    protocol: PublicProtocol,
) -> FrameReconstruction:
    """Receive an address-assigned cluster, with neither expected word nor data."""
    start = perf_counter_ns()
    assert protocol.address_book is not None
    if type(strand_index) is not int or not 0 <= strand_index < protocol.rs_n:
        raise ValueError("invalid public address index")
    if not reads:
        return FrameReconstruction(
            strand_index,
            "NO_READS",
            0,
            0,
            None,
            None,
            None,
            0,
            0,
            False,
            perf_counter_ns() - start,
            "empty inferred cluster",
        )
    normalized: list[str] = []
    selected = sorted(reads, key=lambda read: read.read_id)[: protocol.cluster_read_cap]
    for read in selected:
        if type(read) is not ReadObservation:
            raise TypeError("receiver admits ReadObservation objects only")
        assigned = assign_address(
            read.sequence,
            protocol.address_book,
            correction_radius=protocol.address_radius,
            boundary_slack=protocol.boundary_slack,
        )
        if assigned.status != "OK" or assigned.strand_index != strand_index:
            raise ValueError("cluster contains a read without the declared inferred address")
        # This is ordinary correction to an inferred public address, not use
        # of a hidden true index. Body and mode remain observed and uncertain.
        boundary = min(
            assigned.boundary_lengths,
            key=lambda value: (abs(value - protocol.address_book.length), value),
        )
        normalized.append(protocol.address_book.sequences[strand_index] + read.sequence[boundary:])
    consensus, medoid = medoid_consensus(normalized)
    seeds = [consensus, medoid] + sorted(
        set(normalized),
        key=lambda word: (Levenshtein.distance(word, medoid), abs(len(word) - protocol.n), word),
    )
    seeds = list(dict.fromkeys(seeds))
    tried: set[str] = set()
    accepted: dict[bytes, tuple[str, str | None]] = {}
    checked = crc_failures = 0
    truncated = False

    def inspect(candidate: str) -> None:
        nonlocal checked, crc_failures
        if candidate in tried:
            return
        tried.add(candidate)
        checked += 1
        decoded = decode_frame(candidate, strand_index, protocol)
        if decoded.status == "CRC_FAILED":
            crc_failures += 1
        if decoded.status == "OK":
            assert decoded.payload is not None
            accepted.setdefault(decoded.payload, (candidate, decoded.mode))

    for candidate in seeds:
        if checked >= protocol.repair_candidate_budget:
            truncated = True
            break
        if len(candidate) == protocol.n:
            inspect(candidate)
    if not accepted and protocol.repair_edit_radius:
        for seed in seeds:
            for candidate in one_edit_candidates(seed, protocol.n):
                if candidate in tried:
                    continue
                if checked >= protocol.repair_candidate_budget:
                    truncated = True
                    break
                inspect(candidate)
            if truncated or len(accepted) > 1:
                break
    if len(accepted) == 1:
        payload, (sequence, mode) = next(iter(accepted.items()))
        return FrameReconstruction(
            strand_index,
            "OK",
            len(reads),
            len(selected),
            sequence,
            payload,
            mode,
            checked,
            crc_failures,
            truncated,
            perf_counter_ns() - start,
            None,
        )
    status = (
        "AMBIGUOUS_CRC_VALID_PAYLOADS"
        if len(accepted) > 1
        else "CANDIDATE_BUDGET_EXHAUSTED"
        if truncated
        else "CRC_ERASURE"
    )
    return FrameReconstruction(
        strand_index,
        status,
        len(reads),
        len(selected),
        None,
        None,
        None,
        checked,
        crc_failures,
        truncated,
        perf_counter_ns() - start,
        "bounded public-information search did not yield one payload",
    )


@dataclass(frozen=True)
class PoolReconstruction:
    status: str
    user: bytes | None
    frames: tuple[FrameReconstruction, ...]
    observed_read_count: int
    assigned_read_count: int
    unassigned_read_count: int
    assignment_counts: dict[str, int]
    outer_erasures: tuple[int, ...]
    outer_corrected_positions: tuple[int, ...]
    elapsed_ns: int
    failure_reason: str | None


def reconstruct_pool(
    reads: Sequence[ReadObservation], protocol: PublicProtocol
) -> PoolReconstruction:
    """Complete non-oracle pool receiver. Sender/evaluation records are excluded."""
    start = perf_counter_ns()
    if not isinstance(protocol, PublicProtocol):
        raise TypeError("a public protocol is required")
    assert protocol.address_book is not None
    if any(type(read) is not ReadObservation for read in reads):
        raise TypeError("only receiver observations may enter reconstruction")
    identifiers = [read.read_id for read in reads]
    if any(not isinstance(identifier, str) for identifier in identifiers) or len(
        set(identifiers)
    ) != len(reads):
        raise ValueError("read identifiers must be unique opaque strings")
    clusters: list[list[ReadObservation]] = [[] for _ in range(protocol.rs_n)]
    counts: Counter[str] = Counter()
    for read in reads:
        assigned = assign_address(
            read.sequence,
            protocol.address_book,
            correction_radius=protocol.address_radius,
            boundary_slack=protocol.boundary_slack,
        )
        counts[assigned.status] += 1
        if assigned.status == "OK":
            assert assigned.strand_index is not None
            clusters[assigned.strand_index].append(read)
    frames = tuple(
        reconstruct_cluster(cluster, index, protocol) for index, cluster in enumerate(clusters)
    )
    payloads = {
        frame.strand_index: frame.payload
        for frame in frames
        if frame.status == "OK" and frame.payload is not None
    }
    outer = outer_decode(payloads, protocol)
    return PoolReconstruction(
        outer.status,
        outer.user,
        frames,
        len(reads),
        counts["OK"],
        len(reads) - counts["OK"],
        dict(counts),
        outer.erasures,
        outer.corrected_positions,
        perf_counter_ns() - start,
        outer.failure_reason,
    )
