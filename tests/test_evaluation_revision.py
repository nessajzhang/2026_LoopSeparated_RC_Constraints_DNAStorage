from __future__ import annotations

from dataclasses import replace

from loop_alpha_audit.channel import IDSConfig, simulate_pool
from loop_alpha_audit.evaluation_revision import score_file
from loop_alpha_audit.framing import PublicProtocol, encode_file
from loop_alpha_audit.receiver import reconstruct_pool
from loop_alpha_audit.statistics_revision import summarize_independent_files


def test_truth_scoring_is_post_hoc_and_keeps_full_failure_denominators() -> None:
    public = PublicProtocol(rs_n=7, rs_k=5, mapper="two_bit")
    user = b"evaluation only"
    frames = encode_file(user, public, seed=42)
    channel = simulate_pool(
        {frame.strand_index: frame.sequence for frame in frames}, IDSConfig(), coverage=3, seed=43
    )
    received = reconstruct_pool(channel.observations, public)
    row, scored = score_file(user, frames, channel, received, public, seed=1, file_id=0)
    assert row["file_correct"] and row["frame_correct"] == 7
    assert row["assignment_correct_evaluator_only"] == 21
    assert summarize_independent_files([row], expected_keys=[(1, 0)])["file_FER"] == 0
    # Inject a hypothetical CRC-valid wrong receiver return and an erasure into
    # immutable output AFTER decoding; the scorer must retain both as failures.
    wrong_payload = bytes([received.frames[0].payload[0] ^ 1]) + received.frames[0].payload[1:]
    new_frames = (
        replace(received.frames[0], payload=wrong_payload),
        replace(received.frames[1], status="CRC_ERASURE", payload=None),
    ) + received.frames[2:]
    changed = replace(received, frames=new_frames, user=b"wrong")
    row, scored = score_file(user, frames, channel, changed, public, seed=1, file_id=0)
    assert row["file_undetected_error"] and not row["file_correct"]
    assert row["frame_failures"] == 2
    assert row["frame_erasures"] == row["frame_undetected_errors"] == 1
    assert row["returned_payload_bit_errors"] == 1
    assert scored[0]["undetected_error"] and scored[1]["erasure"]
    assert summarize_independent_files([row], expected_keys=[(1, 0)])["file_FER"] == 1
    assert received.user == user  # Scoring did not alter the original decoder output.
