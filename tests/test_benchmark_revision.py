from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path
from threading import Event

import pytest

import loop_alpha_audit.benchmark_revision as benchmark
from loop_alpha_audit.receiver import reconstruct_pool


def small_config():
    return {
        "schema": benchmark.SCHEMA,
        "phase": "IMPLEMENTATION_PILOT",
        "seeds": [202610230101],
        "files_per_seed": 1,
        "protocol": {"n": 100, "m": 7, "k": 3, "rs_n": 3, "rs_k": 2},
        "methods": ["two_bit"],
        "receiver_timeout_seconds": 20,
        "primary_comparison_family_size": 48,
        "grid": [
            {
                "channel": "mixed",
                "error_scale": 0,
                "coverage": 1,
                "coverage_mode": "fixed",
                "dropout": 0,
                "primary": False,
            }
        ],
    }


def test_pilot_config_is_exactly_24_files_and_plan_derived():
    path = Path(__file__).resolve().parents[1] / "configs/revision/ids_pilot_v1.json"
    config = json.loads(path.read_text())
    cells = benchmark.validate_config(config)
    assert len(cells) == 6
    assert len(cells) * len(config["methods"]) * len(benchmark.expected_keys(config)) == 24
    assert benchmark.expected_keys(config) == [(202610230101, 0)]
    assert not any(cell["grid"]["primary"] for cell in cells)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "duplicate_seed",
        "large_pilot",
        "negative_files",
        "duplicate_cell",
        "bad_fixed_coverage",
        "invalid_mapper",
        "bad_family",
        "hidden_mapper",
        "excess_payload",
    ],
)
def test_bad_or_ambiguous_frozen_configs_rejected(mutation):
    config = small_config()
    if mutation == "unknown":
        config["extra_hidden_truth"] = True
    elif mutation == "duplicate_seed":
        config["seeds"] *= 2
    elif mutation == "large_pilot":
        config["files_per_seed"] = 25
    elif mutation == "negative_files":
        config["files_per_seed"] = -1
    elif mutation == "duplicate_cell":
        config["grid"] *= 2
    elif mutation == "bad_fixed_coverage":
        config["grid"][0]["coverage"] = 1.5
    elif mutation == "invalid_mapper":
        config["methods"] = ["endpoint_m2"]
    elif mutation == "bad_family":
        config["primary_comparison_family_size"] = 0
    elif mutation == "hidden_mapper":
        config["protocol"]["mapper"] = "nonce"
    elif mutation == "excess_payload":
        config["protocol"]["payload_bytes"] = 1000
    with pytest.raises(ValueError):
        benchmark.validate_config(config)


def test_formal_requires_exact_reviewed_freeze_before_directory_creation(tmp_path):
    config = small_config()
    config["phase"] = "FORMAL_FROZEN"
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="acknowledgment"):
        benchmark.run_benchmark(config, destination, reviewed_freeze="not the reviewed SHA")
    assert not destination.exists()


def test_lock_prevents_duplicate_runner(tmp_path):
    with benchmark._run_lock(tmp_path / "run.lock"):
        with pytest.raises(RuntimeError, match="another runner"):
            with benchmark._run_lock(tmp_path / "run.lock"):
                pass


def test_diagnostic_case_keeps_separate_inputs_truth_and_exact_denominators(tmp_path):
    config = small_config()
    cell = benchmark.validate_config(config)[0]
    result = benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {"test_only": "unit"})
    summary = result["summary"]
    assert summary["file_failures"] == 0 and summary["frame_count"] == 3
    assert summary["independent_files"] == summary["planned_files"] == 1
    assert summary["total_synthesized_nt"] == summary["planned_synthesized_nt"] == 300
    folder = next(tmp_path.glob("*/file_*"))
    public_input = json.loads(gzip.decompress((folder / "receiver_input.json.gz").read_bytes()))
    assert set(public_input) == {"public_protocol", "observations", "interface"}
    assert all(set(row) == {"read_id", "sequence"} for row in public_input["observations"])
    private = json.loads(gzip.decompress((folder / "private_truth.json.gz").read_bytes()))
    output = json.loads(gzip.decompress((folder / "receiver_output.json.gz").read_bytes()))
    assert private["expected_user_hex"] == output["result"]["user"]["bytes_hex"]
    assert output["actor_metadata"]["start_method"] == "spawn"
    completed_hash = benchmark.digest(folder / "complete.json")
    resumed = benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {"test_only": "unit"})
    assert resumed["summary"]["file_failures"] == 0
    assert completed_hash == benchmark.digest(folder / "complete.json")
    # A present but incorrect counter cannot be accepted by resume, even
    # though it still looks like a complete file from its directory name.
    (folder / "metrics.json").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {"test_only": "unit"})


def test_timeout_is_a_completed_failure_not_a_missing_or_retried_file(tmp_path, monkeypatch):
    config = small_config()
    config["grid"][0]["error_scale"] = 0.01
    cell = benchmark.validate_config(config)[0]

    class DeadProcess:
        def is_alive(self):
            return False

    class TimeoutActor:
        calls = 0

        def __init__(self, public, **kwargs):
            self._process = DeadProcess()
            self.server_metadata = {"kind": "ready", "start_method": "spawn", "unit_test": True}

        def receive(self, reads):
            type(self).calls += 1
            raise TimeoutError("injected unit-test failure")

        def close(self):
            pass

    monkeypatch.setattr(benchmark, "ReceiverActor", TimeoutActor)
    result = benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {})
    row = result["summary"]
    assert row["file_failures"] == row["receiver_failure_files"] == 1
    assert row["frame_count"] == row["frame_failures"] == 3
    assert row["all_attempts_retained"]
    benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {})
    assert TimeoutActor.calls == 1  # Do not replace a measured failed file on resume.


def test_partial_file_requires_manual_audit_not_silent_retry(tmp_path):
    config = small_config()
    cell = benchmark.validate_config(config)[0]
    folder = tmp_path / f"{cell['cell_id']}_two_bit" / "file_202610230101_00000"
    folder.mkdir(parents=True)
    (folder / "partial.json").write_text("{}")
    with pytest.raises(RuntimeError, match="incomplete file evidence"):
        benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {})


def test_failure_wrapper_cannot_fabricate_a_returned_payload():
    from loop_alpha_audit.framing import PublicProtocol

    public = PublicProtocol(n=100, rs_n=3, rs_k=2)
    failed = benchmark._failed_receiver(public, 7, 123, "timeout")
    assert failed.user is None and len(failed.frames) == 3
    assert failed.status == "RECEIVER_EXCEPTION" and failed.unassigned_read_count == 7
    assert all(
        frame.payload is None and frame.status == "RECEIVER_EXCEPTION" for frame in failed.frames
    )
    assert reconstruct_pool((), public).status == "RS_FAILED"


def test_noiseless_halt_is_enforced_on_resume_too(tmp_path, monkeypatch):
    config = small_config()
    cell = benchmark.validate_config(config)[0]

    class BadNoiselessReceiver:
        calls = 0

        def __init__(self, public, **kwargs):
            self.public = public
            self.server_metadata = {"unit_test": True}

        def receive(self, reads):
            type(self).calls += 1
            return benchmark._failed_receiver(
                self.public, len(reads), 1, "injected control failure"
            )

        def close(self):
            pass

    monkeypatch.setattr(benchmark, "ReceiverActor", BadNoiselessReceiver)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="noiseless all-encoded"):
            benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {})
    assert BadNoiselessReceiver.calls == 1
    assert len(list(tmp_path.glob("*/file_*/complete.json"))) == 1


def test_unconfirmed_constructor_death_halts_before_a_second_file(tmp_path, monkeypatch):
    config = small_config()
    config["files_per_seed"] = 2
    config["grid"][0]["error_scale"] = 0.005
    cell = benchmark.validate_config(config)[0]

    class UnknownDeathReceiver:
        calls = 0

        def __init__(self, *args, **kwargs):
            type(self).calls += 1
            raise benchmark.ReceiverTerminationUnconfirmedError("injected unknown child state")

    monkeypatch.setattr(benchmark, "ReceiverActor", UnknownDeathReceiver)
    with pytest.raises(benchmark.ReceiverTerminationUnconfirmedError):
        benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {})
    assert UnknownDeathReceiver.calls == 1
    assert not list(tmp_path.glob("*/file_*/complete.json"))
    assert len(list(tmp_path.glob("*/file_*"))) == 1


def test_fatal_error_sets_shared_stop_before_next_case(tmp_path, monkeypatch):
    event = Event()
    monkeypatch.setattr(benchmark, "_STOP_EVENT", event)
    config = small_config()
    cell = benchmark.validate_config(config)[0]
    # A partial file is deliberately not relabeled as a fresh experiment.
    folder = tmp_path / f"{cell['cell_id']}_two_bit" / "file_202610230101_00000"
    folder.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="incomplete file"):
        benchmark._case_worker(config, cell, "two_bit", str(tmp_path), {})
    assert event.is_set()
    with pytest.raises(RuntimeError, match="halted before the next file"):
        benchmark._case_worker(config, cell, "nonce", str(tmp_path), {})
    assert not (tmp_path / f"{cell['cell_id']}_nonce").exists()


def test_common_user_and_channel_labels_exclude_scheme():
    # Regression on the actual seed construction, not a claim that differing
    # source sequences experience identical resulting edit distances.
    seed, cell_id, file_id = 51, "aabbcc", 4
    shared = benchmark.derived_seed(seed, "cell-file", cell_id, file_id)
    user_seeds = [benchmark.derived_seed(shared, "common-user") for _ in range(4)]
    assert len(set(user_seeds)) == 1
    assert (
        len(
            {benchmark.derived_seed(shared, "scheme-encoder", kind) for kind in benchmark.MODE_TAGS}
        )
        == 4
    )
    altered = deepcopy(small_config())
    altered["grid"][0]["error_scale"] = 0.005
    assert (
        benchmark.validate_config(altered)[0]["cell_id"]
        != benchmark.validate_config(small_config())[0]["cell_id"]
    )
