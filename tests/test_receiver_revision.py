from __future__ import annotations

import ast
import inspect
import random

import pytest

from loop_alpha_audit.channel import IDSConfig, ReadObservation, simulate_pool
from loop_alpha_audit.framing import PublicProtocol, encode_file
from loop_alpha_audit import receiver
from loop_alpha_audit.receiver import (
    _aligned_votes,
    medoid_consensus,
    one_edit_candidates,
    reconstruct_pool,
)


def test_alignment_votes_rebuild_observation() -> None:
    rng = random.Random(20261023)
    for _ in range(300):
        reference = "".join(rng.choices("ACGT", k=rng.randrange(20)))
        observed = "".join(rng.choices("ACGT", k=rng.randrange(20)))
        bases, gaps = _aligned_votes(reference, observed)
        rebuilt = "".join(gaps[i] + bases[i] for i in range(len(reference))) + gaps[-1]
        assert rebuilt == observed
    assert medoid_consensus(["ACGT", "ACGT", "ACGTT"])[0] == "ACGT"
    assert medoid_consensus(["ACGT"])[0] == "ACGT"


def test_single_edit_ball_is_complete_on_tiny_alphabet() -> None:
    assert {"ACGT", "TCGT", "AAGT", "ACTT"} <= set(one_edit_candidates("ACGT", 4))
    assert "ACGT" in set(one_edit_candidates("AGT", 4))
    assert "ACGT" in set(one_edit_candidates("AACGT", 4))
    assert list(one_edit_candidates("A", 4)) == []


@pytest.mark.parametrize("mapper", ["two_bit", "gc_h_nonce", "short_core", "nonce"])
def test_noiseless_pool_receiver_is_order_invariant(mapper) -> None:
    public = PublicProtocol(rs_n=7, rs_k=5, mapper=mapper)
    user = random.Random(17).randbytes(public.maximum_user_bytes)
    encoded = encode_file(user, public, seed=10)
    channel = simulate_pool(
        {frame.strand_index: frame.sequence for frame in encoded}, IDSConfig(), coverage=3, seed=22
    )
    reconstructed = reconstruct_pool(channel.observations, public)
    assert reconstructed.status == "OK" and reconstructed.user == user
    assert reconstructed.observed_read_count == reconstructed.assigned_read_count == 21
    assert all(frame.status == "OK" for frame in reconstructed.frames)
    assert reconstruct_pool(tuple(reversed(channel.observations)), public).user == user


def test_single_physical_edits_are_repaired_without_sender_template() -> None:
    public = PublicProtocol(rs_n=7, rs_k=5, mapper="short_core")
    user = random.Random(18).randbytes(public.maximum_user_bytes)
    encoded = encode_file(user, public, seed=11)
    for operation in ("insertion", "deletion", "substitution"):
        observations = []
        for index, frame in enumerate(encoded):
            sequence = frame.sequence
            position = (0, 12, 24, 26, 80, 149, 100)[index]
            if operation == "insertion":
                edited = sequence[:position] + "A" + sequence[position:]
            elif operation == "deletion":
                edited = sequence[:position] + sequence[position + 1 :]
            else:
                edited = (
                    sequence[:position]
                    + ("A" if sequence[position] != "A" else "C")
                    + sequence[position + 1 :]
                )
            observations.append(ReadObservation(f"opaque-{index * 17}", edited))
        result = reconstruct_pool(observations, public)
        assert result.status == "OK" and result.user == user, (operation, result)


def test_erasures_budget_and_invalid_pools_are_not_success_only() -> None:
    public = PublicProtocol(rs_n=7, rs_k=5, mapper="two_bit", repair_edit_radius=0)
    user = b"not a sender template"
    encoded = encode_file(user, public, seed=20)
    reads = [ReadObservation(str(index), frame.sequence) for index, frame in enumerate(encoded)]
    assert reconstruct_pool(reads[:5], public).user == user
    assert reconstruct_pool(reads[:4], public).status == "RS_FAILED"
    assert reconstruct_pool([], public).status == "RS_FAILED"
    with pytest.raises(TypeError):
        reconstruct_pool(encoded, public)
    with pytest.raises(ValueError):
        reconstruct_pool([reads[0], reads[0]], public)
    limited = PublicProtocol(rs_n=7, rs_k=5, mapper="two_bit", repair_candidate_budget=1)
    damaged = []
    for read in reads:
        sequence = read.sequence
        position = 60
        damaged.append(
            ReadObservation(
                read.read_id,
                sequence[:position]
                + ("A" if sequence[position] != "A" else "C")
                + sequence[position + 1 :],
            )
        )
    result = reconstruct_pool(damaged, limited)
    assert result.status == "RS_FAILED"
    assert all(frame.status == "CANDIDATE_BUDGET_EXHAUSTED" for frame in result.frames)


def test_receiver_interface_and_ast_exclude_truth_access() -> None:
    assert list(inspect.signature(reconstruct_pool).parameters) == ["reads", "protocol"]
    tree = ast.parse(inspect.getsource(receiver))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not attributes & {
        "truth",
        "events",
        "original_sequence",
        "strand_id",
        "expected_payload",
    }
    assert all(
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id not in {"eval", "exec", "open"}
        for node in ast.walk(tree)
    )
