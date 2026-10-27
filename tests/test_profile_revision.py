"""P3 schema/isolation/accounting tests; no dt4dds simulation in unit tests."""

from __future__ import annotations

import copy
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

import pytest

from loop_alpha_audit.framing import EncodedFrame
from loop_alpha_audit.receiver import PoolReconstruction

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PRIVATE = WORKSPACE / "literature/profiles/p3_implementation"
if not (PRIVATE / "schema.py").is_file():
    PRIVATE = ROOT / "reproducibility/profiles/p3_implementation"
CONFIG = ROOT / "configs/revision/profile_p3_v1.json"

if not (PRIVATE / "schema.py").is_file():
    pytest.skip(
        "optional P3 external-tool fixtures are not installed in the declared workspace layout",
        allow_module_level=True,
    )


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p3 = load(ROOT / "code/run_profile_revision.py", "profile_revision_test")
schema = load(PRIVATE / "schema.py", "profile_schema_test")


@pytest.fixture
def config():
    return json.loads(CONFIG.read_text())


def test_frozen_eight_smoke_and_future_not_authorized(config):
    assert p3.validate_config(config) == config
    assert (
        config["future_formal_not_authorized"]["planned_file_method_budget_cells"] == 3 * 32 * 4 * 2
    )
    for field, changed in (
        ("smoke", {**config["smoke"], "files": 32}),
        ("limits", {**config["limits"], "external_profile_jobs": 2}),
    ):
        corrupted = copy.deepcopy(config)
        corrupted[field] = changed
        with pytest.raises(ValueError):
            p3.validate_config(corrupted)


@pytest.mark.parametrize(
    "field",
    [
        "original_sequence",
        "strand_id",
        "frame_payload",
        "edit_script",
        "true_origin",
        "seed",
        "group",
        "raw_index",
    ],
)
def test_poison_truth_fields_rejected_by_both_schemas(tmp_path, field):
    record = {"read_id": "r" + "a" * 32, "sequence": "ACG" * 50, field: "poison_truth"}
    path = tmp_path / "reads.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="forbidden"):
        p3.read_public(path)
    with pytest.raises(ValueError, match="forbidden"):
        schema.validate_observation(record)


def test_public_only_empty_and_duplicate_ids(tmp_path):
    path = tmp_path / "reads.jsonl"
    path.write_text("")
    assert p3.read_public(path) == ()
    record = {"read_id": "r" + "0" * 32, "sequence": "ACG" * 50}
    path.write_text(json.dumps(record) + "\n")
    observations = p3.read_public(path)
    assert len(observations) == 1 and observations[0].sequence == record["sequence"]
    path.write_text((json.dumps(record) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        p3.read_public(path)
    with pytest.raises(ValueError, match="duplicate"):
        schema.validate_observations([record, record])


def job_from(config):
    physical = config["physical"]
    template = (
        physical["forward_primer"]
        + "ACG" * 50
        + schema.reverse_complement(physical["reverse_primer"])
    )
    return {
        "schema_version": 1,
        "mode": "noisy",
        "body_nt": 150,
        "copies_per_template": 8,
        "read_pairs": 315,
        "seed": 20261023899,
        "shuffle_seed": 17,
        "id_seed": 19,
        "templates": [template] * 63,
        "forward_primer": physical["forward_primer"],
        "reverse_primer": physical["reverse_primer"],
    }


def test_guard_before_pool_add_and_independent_budget_calls(config):
    job = job_from(config)
    assert schema.validate_job(job) is job
    job["read_pairs"] = 192
    assert schema.validate_job(job) is job
    job["templates"].append(job["templates"][0])
    with pytest.raises(ValueError, match="count cap"):
        schema.validate_job(job)
    job = job_from(config)
    job["read_pairs"] = 1000000
    with pytest.raises(ValueError, match="read-pair"):
        schema.validate_job(job)
    assert p3.seed_for(20261023801, "profile", "nonce", "noisy", 192, "source") != p3.seed_for(
        20261023801, "profile", "nonce", "noisy", 315, "source"
    )


def test_string_rc_checker_against_bruteforce_and_cross_boundary():
    rng = random.Random(940132)
    for _ in range(80):
        sequence = "".join(rng.choice("ACGT") for _ in range(24))
        m, k = rng.choice((2, 3, 4)), rng.choice((0, 1, 3))
        expected = {
            (i, j)
            for i in range(25 - m)
            for j in range(i + m + k, 25 - m)
            if sequence[i : i + m] == schema.reverse_complement(sequence[j : j + m])
        }
        record = p3.constraint_record(sequence, m, k)
        assert {(p["left_start"], p["right_start"]) for p in record["rc_violations"]} == expected
    record = p3.constraint_record("C" * 5 + "A" * 10 + "T" * 5, 2, 0, (10, 12))
    assert record["longest_homopolymer"] == 10
    assert record["cross_boundary_homopolymers"]
    assert record["cross_region_or_boundary_rc_count"] > 0


def test_crc_pass_is_not_payload_or_user_correctness():
    sent = [SimpleNamespace(strand_index=i, payload=b"a") for i in range(63)]
    frames = [
        SimpleNamespace(
            strand_index=i, payload=b"b" if i == 7 else b"a", status="OK", failure_reason=None
        )
        for i in range(63)
    ]
    result = SimpleNamespace(frames=frames, status="OK", user=b"wrong")
    metrics, rows = p3.score(b"right", sent, result)
    assert metrics["returned_frames"] == 63 and metrics["undetected_frame_payload_errors"] == 1
    assert metrics["file_failed"] and metrics["undetected_file_error"]
    assert rows[7]["returned_crc_ok"] and not rows[7]["payload_correct_posthoc"]
    assert all(row["assignment_accuracy"] is None for row in rows)
    failed, failed_rows = p3.score(b"right", sent, None)
    assert failed["frame_erasures"] == 63 and failed["file_failed"]
    assert all(row["erasure"] for row in failed_rows)


@pytest.mark.parametrize("method", ["two_bit", "gc_h_nonce", "short_core", "nonce"])
def test_planned_actual_primer_molecule_and_mate_costs(config, method):
    frames = tuple(
        EncodedFrame(
            i, "OK", "ACG" * 50, b"x" * 22, method, 1, False, False, 240, 32, 1, True, True, None
        )
        for i in range(63)
    )
    costs = p3.costs(
        config,
        b"x" * 1202,
        frames,
        {"actual_R1": 315, "actual_R2": 315, "initial_molecules": 504},
        True,
        method,
    )
    assert costs["fallback_used"] == 0
    assert costs["planned_full_design_nt"] == costs["actual_full_design_nt"] == 63 * 272
    assert costs["actual_molecule_weighted_nt"] == 504 * 272
    assert costs["actual_sequenced_nt_both_mates"] == 630 * 150
    assert costs["full_template_net_rate_actual"] == 9616 / 17136
    assert costs["body_net_rate_planned"] == 9616 / 9450
    assert costs["assignment_accuracy"] is None and costs["R2_received"] is False


def test_global_paired_tuple_shuffle_and_opaque_public_only(tmp_path, config, monkeypatch):
    monkeypatch.syspath_prepend(str(PRIVATE))
    worker = load(PRIVATE / "generate_worker.py", "profile_worker_test")
    for folder in ("raw", "public", "evaluator_only"):
        (tmp_path / folder).mkdir()
    r1 = [(f"@Read{i}", "ACG" * 49 + "ACGT"[i % 4] * 3, "+", "F" * 150) for i in range(12)]
    r2 = [(r[0], schema.reverse_complement(r[1]), "+", r[3]) for r in r1]
    for mate, rows in ((1, r1), (2, r2)):
        worker.write_fastq(tmp_path / "raw" / f"R{mate}.fq.gz", rows)
    assert worker.publish(tmp_path, job_from(config)) == 12
    reads = p3.read_public(tmp_path / "public/R1.jsonl")
    assert len(reads) == 12 and all(OPAQUE_ID(read.read_id) for read in reads)
    mapping = json.loads((tmp_path / "evaluator_only/shuffle_map.json").read_text())
    assert mapping["paired_tuple_multiset_preserved"]
    assert mapping["whole_file_uniform_permutation"] != list(range(12))
    assert sorted(mapping["whole_file_uniform_permutation"]) == list(range(12))


def OPAQUE_ID(value):
    return p3.OPAQUE.fullmatch(value) is not None


def test_watchdog_timeout_preserved_without_retry(tmp_path):
    resource = p3.supervise(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path,
        os.environ.copy(),
        seconds=0.02,
    )
    assert resource["watchdog_reason"] == "PROFILE_WALL_TIMEOUT"
    assert resource["exit_code"] != 0 and resource["process_group_confirmed_empty"]
    assert not resource["automatic_retry"]
    assert (tmp_path / "resource.json").is_file()


def test_existing_attempt_refused_before_encode(tmp_path):
    args = SimpleNamespace(
        config=CONFIG,
        out=tmp_path,
        profile_python=WORKSPACE / "profile_runtime/.venv/bin/python",
        worker=PRIVATE / "generate_worker.py",
        source=WORKSPACE / "literature/profiles/dt4dds_v1.0.0",
    )
    with pytest.raises(ValueError, match="existing output refused"):
        p3.run(args)


def failed_resource():
    return {"exit_code": -9, "watchdog_reason": "PROFILE_WALL_TIMEOUT"}


def write_pool(path, counts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(counts)


def test_real_source_headerless_pool_fixture_and_overdraw_not_hidden(tmp_path):
    initial = tmp_path / "evaluator_only/initial_pool.csv"
    selected = tmp_path / "evaluator_only/selected_pool.csv"
    sequence = "ACGT" * 68
    write_pool(initial, [(sequence, 8)])
    write_pool(selected, [(sequence, 9)])
    result = p3.failed_generation(tmp_path, failed_resource())
    assert result["initial_molecules"] == 8 and result["selected_molecules"] == 9
    assert list(result["selected_template_overdraw"].values()) == [1]
    assert result["actual_R1"] == result["actual_R2"] == 0
    assert result["status"] == "FAILED" and result["no_automatic_retry"]
    # Exact source-emitted smoke CSV bytes, including its genuine stock
    # overdraw. No header is added and the first molecule row must survive.
    real = tmp_path / "source_emitted/evaluator_only"
    real.mkdir(parents=True)
    for source, destination in (
        ("initial_pool_headerless.csv", "initial_pool.csv"),
        ("selected_pool_headerless_overdraw.csv", "selected_pool.csv"),
    ):
        shutil.copyfile(PRIVATE / "fixtures" / source, real / destination)
    observed = p3.failed_generation(real.parent, failed_resource())
    assert observed["initial_molecules"] == 504 and observed["selected_molecules"] == 315
    assert sorted(observed["selected_template_overdraw"].values()) == [1, 1]


def test_missing_empty_malformed_and_excess_pool_evidence(tmp_path):
    result = p3.failed_generation(tmp_path, failed_resource())
    assert result["initial_molecules"] is None and result["selected_molecules"] is None
    initial = tmp_path / "evaluator_only/initial_pool.csv"
    write_pool(initial, [])
    assert p3.failed_generation(tmp_path, failed_resource())["initial_molecules"] == 0
    write_pool(initial, [("ACGT" * 68, 505)])
    result = p3.failed_generation(tmp_path, failed_resource())
    assert result["initial_molecules"] == 505 and result["initial_count_exceeds_cap"]
    write_pool(initial, [("sequence", "value")])
    result = p3.failed_generation(tmp_path, failed_resource())
    assert result["initial_molecules"] is None and "initial_pool" in result["partial_read_errors"]


def test_complete_partial_corrupt_fastq_retained_not_zero(tmp_path):
    (tmp_path / "raw").mkdir()
    valid = "@Read0\n" + "ACG" * 50 + "\n+\n" + "F" * 150 + "\n"
    path = tmp_path / "raw/R1.fq.gz"
    with gzip.open(path, "wt") as stream:
        stream.write(valid)
    assert p3.failed_generation(tmp_path, failed_resource())["actual_R1"] == 1
    with gzip.open(path, "wt") as stream:
        stream.write(valid + "@Read1\nAAA\n")
    result = p3.failed_generation(tmp_path, failed_resource())
    assert result["actual_R1"] is None
    assert result["persisted_complete_record_prefix_count"]["R1"] == 1
    assert "R1" in result["partial_read_errors"]
    path.write_bytes(b"not gzip")
    assert p3.failed_generation(tmp_path, failed_resource())["actual_R1"] is None
    with gzip.open(path, "wt") as stream:
        stream.write(valid * 316)
    result = p3.failed_generation(tmp_path, failed_resource())
    assert result["actual_R1"] == 316 and result["actual_read_count_exceeds_requested"]["R1"]
    assert result["requested_minus_actual_by_mate"]["R1"] == -1


def test_watchdog_rss_and_no_hidden_retry(tmp_path, monkeypatch):
    calls = 0

    def snapshots(pid):
        nonlocal calls
        calls += 1
        return [{"pid": pid, "rss_bytes": 900 * 1024**2}] if calls == 1 else []

    monkeypatch.setattr(p3, "group_snapshot", snapshots)
    result = p3.supervise(
        [sys.executable, "-c", "import time; time.sleep(5)"], tmp_path, os.environ.copy()
    )
    assert result["watchdog_reason"] == "PROFILE_RSS_WATCHDOG"
    assert result["process_group_confirmed_empty"] and result["exit_code"] != 0


def test_watchdog_infrastructure_failure_seals_resource_and_blocks(tmp_path, monkeypatch):
    calls = 0

    def snapshots(pid):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("ps unavailable")
        return []

    monkeypatch.setattr(p3, "group_snapshot", snapshots)
    with pytest.raises(p3.ReceiverTerminationUnconfirmedError):
        p3.supervise(
            [sys.executable, "-c", "import time; time.sleep(5)"], tmp_path, os.environ.copy()
        )
    result = json.loads((tmp_path / "resource.json").read_text())
    assert result["cleanup_errors"] and result["process_group_confirmed_empty"]


def test_process_start_failure_preserves_unconfirmed_resource(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("start failed")

    monkeypatch.setattr(p3.subprocess, "Popen", fail)
    with pytest.raises(p3.ReceiverTerminationUnconfirmedError):
        p3.supervise(["unavailable"], tmp_path, os.environ.copy())
    result = json.loads((tmp_path / "resource.json").read_text())
    assert result["pid"] is None and not result["process_group_confirmed_empty"]
    assert result["cleanup_errors"]


def test_unconfirmed_descendant_cannot_authorize_receiver(tmp_path, monkeypatch):
    # Mock an unconfirmed group after an already-exited child; never signal
    # another real process group in this negative test.
    monkeypatch.setattr(p3, "group_snapshot", lambda pid: [{"pid": pid, "rss_bytes": 1}])
    monkeypatch.setattr(p3.os, "killpg", lambda *args: None)
    with pytest.raises(p3.ReceiverTerminationUnconfirmedError):
        p3.supervise([sys.executable, "-c", "pass"], tmp_path, os.environ.copy())
    result = json.loads((tmp_path / "resource.json").read_text())
    assert not result["process_group_confirmed_empty"] and result["remaining_at_worker_exit"]


def test_wait_and_group_query_failures_preserve_fatal_resource(tmp_path, monkeypatch):
    class Process:
        pid = 999999
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout):
            raise OSError("join unavailable")

    monkeypatch.setattr(p3.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(p3, "group_snapshot", lambda pid: [])
    with pytest.raises(p3.ReceiverTerminationUnconfirmedError):
        p3.supervise(["mock-process-no-real-child"], tmp_path, os.environ.copy())
    result = json.loads((tmp_path / "resource.json").read_text())
    assert any("WAIT_OR_KILL_FAILED" in error for error in result["cleanup_errors"])
    assert not result["automatic_retry"]


def synthetic_frames(method="two_bit"):
    return tuple(
        EncodedFrame(
            i, "OK", "ACG" * 50, b"x" * 22, method, 1, False, False, 240, 32, 1, True, True, None
        )
        for i in range(63)
    )


def mock_profile(monkeypatch, complete):
    def supervisor(command, directory, env):
        channel = directory / "channel"
        (channel / "public").mkdir(parents=True)
        if complete:
            (channel / "summary.json").write_text(
                json.dumps(
                    {"status": "COMPLETE", "actual_R1": 0, "actual_R2": 0, "initial_molecules": 504}
                )
            )
            (channel / "public/R1.jsonl").write_text("")
        resource = {
            "exit_code": 0 if complete else 1,
            "process_group_confirmed_empty": True,
            "watchdog_reason": None,
        }
        p3.write_json(directory / "resource.json", resource)
        return resource

    monkeypatch.setattr(p3, "supervise", supervisor)


def cell_args():
    return SimpleNamespace(
        profile_python=Path(sys.executable),
        worker=PRIVATE / "generate_worker.py",
        source=WORKSPACE / "literature/profiles/dt4dds_v1.0.0",
    )


def test_profile_failure_never_spawns_receiver_and_is_sealed(tmp_path, config, monkeypatch):
    mock_profile(monkeypatch, complete=False)

    def forbidden(*args, **kwargs):
        pytest.fail("failed profile cannot spawn a receiver")

    monkeypatch.setattr(p3, "ReceiverActor", forbidden)
    row = p3.run_cell(
        config, "two_bit", "noisy", b"x" * 1202, synthetic_frames(), {}, cell_args(), tmp_path, {}
    )
    assert row["file_failed"] and row["frame_erasures"] == 63
    assert row["generator_status"] == "FAILED" and row["receiver_error"] == "NOT_RUN_PROFILE_FAILED"
    assert (tmp_path / "cell_01/complete.json").is_file()


def test_receiver_timeout_closes_actor_and_preserves_failure(tmp_path, config, monkeypatch):
    mock_profile(monkeypatch, complete=True)
    actors = []

    class Actor:
        def __init__(self, protocol, timeout_seconds):
            assert len(sys.argv) == 1  # No private generator CLI paths in spawn preparation.
            self.server_metadata = {"kind": "ready", "pid": 1234, "start_method": "spawn"}
            self._process = SimpleNamespace(is_alive=lambda: False)
            self.closed = False
            actors.append(self)

        def receive(self, observations):
            assert observations == ()
            raise TimeoutError("test receiver timeout")

        def close(self):
            self.closed = True

    monkeypatch.setattr(p3, "ReceiverActor", Actor)
    row = p3.run_cell(
        config, "two_bit", "noisy", b"x" * 1202, synthetic_frames(), {}, cell_args(), tmp_path, {}
    )
    assert row["file_failed"] and "TimeoutError" in row["receiver_error"]
    assert actors[0].closed and (tmp_path / "cell_01/complete.json").is_file()


def test_noiseless_wrong_user_halts_after_sealing(tmp_path, config, monkeypatch):
    mock_profile(monkeypatch, complete=True)

    class Actor:
        def __init__(self, protocol, timeout_seconds):
            self.server_metadata = {"kind": "ready", "pid": 1234, "start_method": "spawn"}
            self._process = SimpleNamespace(is_alive=lambda: False)

        def receive(self, observations):
            return PoolReconstruction("OK", b"wrong", (), 0, 0, 0, {}, (), (), 1, None)

        def close(self):
            pass

    monkeypatch.setattr(p3, "ReceiverActor", Actor)
    with pytest.raises(RuntimeError, match="NOISELESS_ALL_SENT"):
        p3.run_cell(
            config,
            "two_bit",
            "ideal",
            b"x" * 1202,
            synthetic_frames(),
            {},
            cell_args(),
            tmp_path,
            {},
        )
    assert (tmp_path / "cell_00/complete.json").is_file()
    row = json.loads((tmp_path / "cell_00/summary.json").read_text())
    assert row["file_failed"] and row["undetected_file_error"]


def test_receiver_cleanup_failure_is_fatal_not_retryable(tmp_path, config, monkeypatch):
    mock_profile(monkeypatch, complete=True)

    class Actor:
        def __init__(self, protocol, timeout_seconds):
            self.server_metadata = {"kind": "ready", "pid": 1234, "start_method": "spawn"}
            self._process = SimpleNamespace(is_alive=lambda: False)

        def receive(self, observations):
            raise TimeoutError("receiver timeout")

        def close(self):
            raise p3.ReceiverTerminationUnconfirmedError("cleanup unconfirmed")

    monkeypatch.setattr(p3, "ReceiverActor", Actor)
    with pytest.raises(p3.ReceiverTerminationUnconfirmedError, match="cleanup unconfirmed"):
        p3.run_cell(
            config,
            "two_bit",
            "noisy",
            b"x" * 1202,
            synthetic_frames(),
            {},
            cell_args(),
            tmp_path,
            {},
        )
    assert not (tmp_path / "cell_01/complete.json").exists()
