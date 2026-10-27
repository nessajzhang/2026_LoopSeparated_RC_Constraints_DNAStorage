"""Post-reconstruction truth scoring. Never imported by the receiver modules.

The sender payload/sequence, simulated source and edit scripts are admitted
here only after receiving is complete. Returned-frame BER includes wrong
CRC-valid returns. Erasures are charged separately as lost bits, not silently
removed or relabeled as observed bit errors.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from rapidfuzz.distance import Levenshtein

from .addresses import assign_address
from .channel import ChannelOutput
from .framing import EncodedFrame, PublicProtocol, rate_accounting
from .receiver import PoolReconstruction


def bit_errors(left: bytes, right: bytes) -> int:
    if len(left) != len(right):
        raise ValueError("bit errors require equal-length byte payloads")
    return sum((a ^ b).bit_count() for a, b in zip(left, right))


def score_file(
    expected_user: bytes,
    encoded: tuple[EncodedFrame, ...],
    channel: ChannelOutput,
    received: PoolReconstruction,
    protocol: PublicProtocol,
    *,
    seed: int,
    file_id: int,
    receiver_failure: str | None = None,
) -> tuple[dict, list[dict]]:
    """Score one independent file without altering the receiver's frozen output."""
    if len(encoded) != protocol.rs_n or len(received.frames) != protocol.rs_n:
        raise ValueError("complete attempted sender/receiver frame records required")
    assert protocol.payload_bytes is not None and protocol.address_book is not None
    accounting = rate_accounting(protocol, len(expected_user), encoded)
    read_truth = {row.read_id: row for row in channel.truth}
    source_counts = Counter(row.strand_id for row in channel.truth)
    if set(read_truth) != {read.read_id for read in channel.observations}:
        raise ValueError("observation/truth IDs do not reconcile in evaluator")
    correct_assignments = wrong_assignments = rejected_assignments = 0
    for observation in channel.observations:
        assigned = assign_address(
            observation.sequence,
            protocol.address_book,
            correction_radius=protocol.address_radius,
            boundary_slack=protocol.boundary_slack,
        )
        if assigned.status != "OK":
            rejected_assignments += 1
        elif assigned.strand_index == read_truth[observation.read_id].strand_id:
            correct_assignments += 1
        else:
            wrong_assignments += 1
    frame_rows = []
    for index, (sent, result) in enumerate(zip(encoded, received.frames)):
        if sent.strand_index != index or result.strand_index != index:
            raise ValueError("public frame ordering changed before post hoc scoring")
        returned = result.status == "OK" and result.payload is not None
        correct = returned and result.payload == sent.payload
        erasure = not returned
        undetected = returned and not correct
        errors = bit_errors(result.payload, sent.payload) if returned else 0
        distance = (
            Levenshtein.distance(result.sequence, sent.sequence)
            if result.sequence is not None and sent.sequence is not None
            else None
        )
        frame_rows.append(
            {
                "seed": seed,
                "file_id": file_id,
                "strand_index": index,
                "n_total": protocol.n,
                "m": protocol.m,
                "k": protocol.k,
                "mapper": protocol.mapper,
                "transmitted_mode": sent.mode,
                "payload_bytes_including_outer_parity": protocol.payload_bytes,
                "mapper_input_bits": sent.mapper_input_bits,
                "filler_bits": sent.filler_bits,
                "header_nt": protocol.header_nt,
                "primer_nt": protocol.primer_nt,
                "nonce_nt": protocol.nonce_nt if sent.mode in ("nonce", "gc_h_nonce") else 0,
                "frame_crc_bits": 32,
                "encode_status": sent.status,
                "encode_failure_reason": sent.failure_reason,
                "attempts": sent.attempts,
                "primary_mapper_failed": sent.primary_failed,
                "fallback_attempted": sent.fallback_attempted,
                "fallback_used": sent.status == "OK" and sent.mode != protocol.mapper,
                "whole_oligo_rc_ok": sent.constraint_ok,
                "whole_oligo_gc_h_ok": sent.gc_h_filter_ok,
                "true_read_count_evaluator_only": source_counts[index],
                "inferred_cluster_read_count": result.read_count,
                "reads_used": result.reads_used,
                "receiver_status": result.status,
                "receiver_failure_reason": result.failure_reason,
                "returned_payload": returned,
                "decoded_correct": correct,
                "erasure": erasure,
                "undetected_error": undetected,
                "returned_payload_bit_errors": errors,
                "returned_payload_bits": 8 * protocol.payload_bytes if returned else 0,
                "erasure_aware_lost_bits": 8 * protocol.payload_bytes if erasure else errors,
                "residual_edit_distance_evaluator_only": distance,
                "candidates_checked": result.candidates_checked,
                "crc_failures": result.crc_failures,
                "candidate_search_truncated": result.search_truncated,
                "encode_ns": sent.elapsed_ns,
                "receive_ns": result.elapsed_ns,
                "truth_isolated_from_receiver": True,
            }
        )
    returned_bits = sum(row["returned_payload_bits"] for row in frame_rows)
    wrong_bits = sum(row["returned_payload_bit_errors"] for row in frame_rows)
    file_correct = received.status == "OK" and received.user == expected_user
    operation_counts = Counter(event.operation for truth in channel.truth for event in truth.events)
    file_row = {
        "seed": seed,
        "file_id": file_id,
        "mapper": protocol.mapper,
        "file_receiver_status": received.status,
        "file_correct": file_correct,
        "file_undetected_error": received.status == "OK" and not file_correct,
        "receiver_exception_or_timeout": receiver_failure,
        "frames_attempted": protocol.rs_n,
        "frame_correct": sum(row["decoded_correct"] for row in frame_rows),
        "frame_failures": sum(not row["decoded_correct"] for row in frame_rows),
        "frame_erasures": sum(row["erasure"] for row in frame_rows),
        "frame_undetected_errors": sum(row["undetected_error"] for row in frame_rows),
        "returned_payload_bit_errors": wrong_bits,
        "returned_payload_bits": returned_bits,
        "conditional_returned_frame_BER": wrong_bits / returned_bits if returned_bits else None,
        "erasure_aware_bit_loss_score": sum(row["erasure_aware_lost_bits"] for row in frame_rows)
        / (protocol.rs_n * protocol.payload_bytes * 8),
        "read_count": len(channel.observations),
        "total_sequenced_nt": sum(len(read.sequence) for read in channel.observations),
        "assignment_correct_evaluator_only": correct_assignments,
        "assignment_wrong_evaluator_only": wrong_assignments,
        "assignment_rejected": rejected_assignments,
        "coverage_mode": channel.coverage_mode,
        "dropped_strands": len(channel.dropped_strands),
        "realized_insertions": operation_counts["insertion"],
        "realized_deletions": operation_counts["deletion"],
        "realized_substitutions": operation_counts["substitution"],
        "encode_kernel_ns": sum(row["encode_ns"] for row in frame_rows),
        "receive_total_ns": received.elapsed_ns,
        "truth_isolated_from_receiver": True,
        **accounting,
    }
    return file_row, frame_rows


def private_truth_record(
    expected_user: bytes,
    encoded: tuple[EncodedFrame, ...],
    channel: ChannelOutput,
    *,
    seed: int,
    file_id: int,
) -> dict:
    """Separate provenance record; never send this object to a receiver actor."""
    return {
        "seed": seed,
        "file_id": file_id,
        "expected_user_hex": expected_user.hex(),
        "sender": [{**asdict(frame), "payload": frame.payload.hex()} for frame in encoded],
        "read_truth": [asdict(truth) for truth in channel.truth],
        "dropped_strands": list(channel.dropped_strands),
    }
