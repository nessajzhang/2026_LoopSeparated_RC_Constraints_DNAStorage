"""V2 wrapper/control fixtures; these tests generate no source-profile reads."""

from __future__ import annotations
import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from loop_alpha_audit.framing import EncodedFrame

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT.parent / "literature/profiles/p3_formal_v2"
if not (PRIVATE / "schema.py").is_file():
    PRIVATE = ROOT / "reproducibility/profiles/p3_formal_v2"
if not (PRIVATE / "schema.py").is_file():
    pytest.skip("optional source-profile wrapper not installed", allow_module_level=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p3 = load(ROOT / "code/run_profile_formal_v2.py", "p3_formal_v2_test")
schema = load(PRIVATE / "schema.py", "p3_formal_v2_schema_test")


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/revision/profile_p3_formal_v2.json").read_text())


def test_exact_plan_and_original_768_keys(config):
    p3.validate_config(config)
    all_cells = [row for i in range(96) for row in p3.planned_cells(config, "formal", i)]
    keys = {(x["seed"], x["file_id"], x["method"], x["read_pairs"]) for x in all_cells}
    assert len(all_cells) == len(keys) == 768
    assert {x["seed"] for x in all_cells} == {20261023801, 20261023802, 20261023803}
    assert {x["file_id"] for x in all_cells} == set(range(32))
    smoke = p3.planned_cells(config, "interface_smoke", None)
    assert len(smoke) == 2 and all(x["seed"] == 20261023903 for x in smoke)
    assert not {x["seed"] for x in smoke} & {x["seed"] for x in all_cells}
    for bad in (-1, 96, None, True, 1.5):
        with pytest.raises(ValueError):
            p3.planned_cells(config, "formal", bad)
    with pytest.raises(ValueError):
        p3.planned_cells(config, "interface_smoke", 0)


def test_public_observations_reject_truth_and_empty_is_valid(tmp_path):
    path = tmp_path / "observed.jsonl"
    path.write_text("")
    assert p3.read_public(path) == ()
    for key in (
        "seed",
        "raw_index",
        "origin",
        "strand_id",
        "original_sequence",
        "payload",
        "edit_script",
    ):
        record = {"read_id": "r" + "0" * 32, "sequence": "ACG" * 50, key: "poison"}
        path.write_text(json.dumps(record) + "\n")
        with pytest.raises(ValueError):
            p3.read_public(path)
        with pytest.raises(ValueError):
            schema.validate_observation(record)


def job_from(config):
    physical = config["physical"]
    template = (
        physical["forward_primer"]
        + "ACG" * 50
        + schema.reverse_complement(physical["reverse_primer"])
    )
    return {
        "schema_version": 2,
        "mode": "noisy",
        "body_nt": 150,
        "copies_per_template": 512,
        "read_pairs": 192,
        "seed": 17,
        "shuffle_seed": 19,
        "id_seed": 23,
        "templates": [template] * 63,
        "forward_primer": physical["forward_primer"],
        "reverse_primer": physical["reverse_primer"],
    }


def test_copy_budget_guard_and_stock_bound(config):
    job = job_from(config)
    assert schema.validate_job(job) is job
    for copies in (8, 511, 513, True):
        bad = copy.deepcopy(job)
        bad["copies_per_template"] = copies
        with pytest.raises(ValueError):
            schema.validate_job(bad)
    bad = copy.deepcopy(job)
    bad["templates"].append(bad["templates"][0])
    with pytest.raises(ValueError):
        schema.validate_job(bad)
    for count in range(1, 64):
        for budget in (192, 315):
            stock = 512 * count
            target = int(budget / stock * stock)
            assert 0 < target <= budget < 512
            if count == 47 and budget == 192:
                assert target == 191  # preserve source binary64 underfill


@pytest.mark.parametrize("budget", [192, 315])
def test_complete_actual_planned_and_molecular_costs(config, budget):
    frames = tuple(
        EncodedFrame(
            i, "OK", "ACG" * 50, b"x" * 22, "nonce", 1, False, False, 240, 32, 1, True, True, None
        )
        for i in range(63)
    )
    costs = p3.costs(
        config,
        b"x" * 1202,
        frames,
        {"actual_R1": budget, "actual_R2": budget, "initial_molecules": 32256},
        True,
        "nonce",
        budget,
    )
    assert costs["planned_initial_molecules"] == costs["actual_initial_molecules"] == 32256
    assert costs["planned_molecule_weighted_nt"] == 8773632
    assert (
        costs["actual_sequenced_nt_both_mates"]
        == costs["planned_sequenced_nt_both_mates"]
        == budget * 300
    )
    assert costs["full_template_net_rate_planned"] == 9616 / 17136
    assert costs["body_net_rate_planned"] == 9616 / 9450
    assert costs["R2_received"] is False and costs["assignment_accuracy"] is None


def test_failed_underfilled_and_unknown_counts_are_not_fabricated(tmp_path):
    path = tmp_path / "evaluator_only/initial_pool.csv"
    path.parent.mkdir()
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows([("ACGT" * 68, 512 * 47)])
    with path.with_name("selected_pool.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([("ACGT" * 68, 191)])
    result = p3.failed_generation(
        tmp_path,
        {"exit_code": -9, "watchdog_reason": "fixture"},
        requested_pairs=192,
        molecule_cap=32256,
    )
    assert result["selected_molecules"] == 191 and result["requested_pairs"] == 192
    assert result["actual_R1"] == result["actual_R2"] == 0
    assert result["selected_template_overdraw"] == {}
    assert result["no_automatic_retry"]
    unknown = p3.failed_generation(
        tmp_path / "missing",
        {"exit_code": 1, "watchdog_reason": None},
        requested_pairs=315,
        molecule_cap=32256,
    )
    assert unknown["initial_molecules"] is None and unknown["selected_molecules"] is None


def test_user_and_encoder_streams_do_not_depend_on_read_budget(config):
    rows = p3.planned_cells(config, "formal", 47)
    for method in config["methods"]:
        selected = [x for x in rows if x["method"] == method]
        assert len(selected) == 2
        a, b = selected
        assert p3.seed_for(a["seed"], "encoder", method, a["file_id"]) == p3.seed_for(
            b["seed"], "encoder", method, b["file_id"]
        )
        assert p3.seed_for(
            a["seed"], "profile-v2", a["file_id"], method, "noisy", 192, "source"
        ) != p3.seed_for(b["seed"], "profile-v2", b["file_id"], method, "noisy", 315, "source")


def test_unapproved_formal_phase_fails_before_encoding(tmp_path, config):
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "gate": "P3_V2_ROOT_EXECUTION_ONLY_NOT_RESULT_ACCEPTANCE",
                "approved_phases": ["interface_smoke"],
            }
        )
    )
    args = SimpleNamespace(approval=approval, phase="formal")
    with pytest.raises(ValueError, match="phase"):
        p3.verify_approval(args, ROOT / "configs/revision/profile_p3_formal_v2.json")


def test_no_return_and_wrong_crc_alias_both_keep_failure_denominators():
    sent = [SimpleNamespace(strand_index=i, payload=b"a") for i in range(63)]
    summary, frames = p3.score(b"expected", sent, None)
    assert summary["file_failed"] and summary["frame_erasures"] == len(frames) == 63
    returned = [
        SimpleNamespace(
            strand_index=i, payload=b"b" if i == 2 else b"a", status="OK", failure_reason=None
        )
        for i in range(63)
    ]
    summary, frames = p3.score(
        b"expected", sent, SimpleNamespace(frames=returned, status="OK", user=b"wrong")
    )
    assert summary["undetected_frame_payload_errors"] == 1 and summary["undetected_file_error"]
    assert frames[2]["undetected_payload_error"] and summary["file_failed"]
