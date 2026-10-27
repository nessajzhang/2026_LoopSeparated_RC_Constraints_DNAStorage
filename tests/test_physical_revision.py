"""Physical-arm interface/denominator tests; only two tiny native folds."""

import json
from itertools import product
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import run_physical_revision as physical  # noqa:E402
from matched_dinucleotide import ControlResult  # noqa:E402
from physical_model_helpers import (
    ModelSpec,
    bpp_features,
    fold_sequence,
    model_specs,
    prepare_models,
    structure_features,
)  # noqa:E402
from loop_alpha_audit.model import is_avoiding_full  # noqa:E402
from loop_alpha_audit.statistics_revision import clopper_pearson  # noqa:E402

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/revision/physical_arms_v1.json").read_text()
)


def test_original_reduced_and_independent_reference_all_small_words():
    for n in range(6):
        for digits in product(range(4), repeat=n):
            word = "".join("ACGT"[v] for v in digits)
            for m, k in ((1, 0), (1, 2), (2, 0), (2, 1), (3, 0)):
                expected = is_avoiding_full(digits, m, k, (3, 2, 1, 0))
                assert physical.avoids_windows(word, m, k) == expected
                assert physical.avoids_full(word, m, k) == expected


def test_parent_cap_and_control_failure_never_redraw_target(monkeypatch):
    normal = physical.mechanism_inputs(40, 4, 3, 77, 2, parent_cap=10000, control_cap=100)
    monkeypatch.setattr(
        physical,
        "sample_h_matched",
        lambda target, rng, max_attempts: ControlResult(
            "CONTROL_ATTEMPT_CAP", None, max_attempts, 0, None
        ),
    )
    failed = physical.mechanism_inputs(40, 4, 3, 77, 2, parent_cap=10000, control_cap=100)
    assert [r["sequence"] for r in normal if r["role"] == "target"] == [
        r["sequence"] for r in failed if r["role"] == "target"
    ]
    assert all(
        r["sequence"] is None and r["generation_status"] == "CONTROL_ATTEMPT_CAP"
        for r in failed
        if r["role"] == "control"
    )
    cap = physical.mechanism_inputs(20, 2, 0, 5, 4, parent_cap=1, control_cap=1)
    assert len(cap) == 8 and cap[-1]["parent_attempts_cumulative"] == 1
    assert sum(r["parent_attempts_for_target"] for r in cap if r["role"] == "target") == 1


def test_formal_denominators_and_pre_outcome_selectors():
    count = physical.planned_counts(CONFIG)
    assert count["mechanism_pairs"] == 27648
    assert count["deployment_planned_strands"] == 145152
    assert count["deployment_files"] == 2304 and count["high_alpha_words"] == 4608
    assert count["witness_words"] == 12 and count["formal_observations_generated_by_plan"] == 0
    base = {"arm": "deployment", "n": 150, "m": 7, "k": 3, "file_id": 1, "strand_index": 8}
    assert len(physical.fold_schedule(base, CONFIG)) == 10
    assert len(physical.fold_schedule({**base, "file_id": 2}, CONFIG)) == 1
    assert physical.fold_schedule({**base, "strand_index": 1}, CONFIG)[0][1] is False
    assert (
        len(
            physical.fold_schedule(
                {"arm": "mechanism", "n": 150, "m": 6, "k": 5, "pair_index": 31}, CONFIG
            )
        )
        == 10
    )
    assert (
        len(
            physical.fold_schedule(
                {"arm": "mechanism", "n": 150, "m": 6, "k": 5, "pair_index": 32}, CONFIG
            )
        )
        == 1
    )
    assert [(s.wc_only, pf) for s, pf in physical.fold_schedule({"arm": "high_alpha"}, CONFIG)] == [
        (False, False),
        (True, False),
    ]


def test_actual_endpoint_uniform_bit_rank_and_all_witnesses():
    # Small rank mapper interface, no folding.
    rows = physical.high_alpha_inputs(12, 2, 0.5, 20261023701, 3)
    assert len(rows) == 3 and all(row["generation_status"] == "OK" for row in rows)
    assert all(int(row["uniform_binary_index"]) < 2 ** row["K"] for row in rows)
    assert rows == physical.high_alpha_inputs(12, 2, 0.5, 20261023701, 3)
    for n, m, alpha in product(CONFIG["n_values"], CONFIG["m_values"], CONFIG["alphas"]):
        row = physical.witness_input(n, m, alpha)
        assert len(row["sequence"]) == n and row["full_constraint_valid"]
        assert row["sequence"].startswith("G" * m + "AAA" + "C" * m)


def test_checkpoint_partial_resume_and_changed_source_rejection(tmp_path):
    cp = physical.Checkpoint(tmp_path, {"config": "fixed"})

    def interrupted():
        yield {"target": "AC", "control": None}
        raise RuntimeError("deliberate test interruption")

    with pytest.raises(RuntimeError):
        cp.unit("pair0", interrupted)
    assert list(tmp_path.glob("*.partial1")) and not cp.state["units"]
    expected = [{"target": "AC", "control": None}]
    assert cp.unit("pair0", lambda: expected) == expected
    resumed = physical.Checkpoint(tmp_path, {"config": "fixed"})
    assert (
        resumed.unit("pair0", lambda: pytest.fail("Must not regenerate completed input"))
        == expected
    )
    with pytest.raises(ValueError):
        physical.Checkpoint(tmp_path, {"config": "changed"})
    (tmp_path / "pair0.jsonl.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        resumed.read("pair0")
    with pytest.raises(ValueError):
        physical.Checkpoint(tmp_path, {"config": "fixed"})


def test_formal_gate_and_truth_free_no_sequence_observation(tmp_path):
    with pytest.raises(PermissionError):
        physical.require_approval(None, "a", "b")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "status": "APPROVED_FOR_LOCAL_FORMAL",
                "config_sha256": "a",
                "implementation_sha256": "b",
                "independent_gate_id": "unit-test-fixture-not-real-approval",
            }
        )
    )
    assert physical.require_approval(approval, "a", "b")
    with pytest.raises(PermissionError):
        physical.require_approval(approval, "c", "b")
    result = physical.fold_task(
        (
            {"sequence": None, "generation_status": "ENCODING_FAILED"},
            ModelSpec(37, 1.021, False),
            False,
            "hash",
        )
    )
    assert result["status"] == "NO_SEQUENCE_INPUT_FAILURE"


def test_stem_span_is_not_all_m_subrun_spans_and_probability_layout():
    sequence = "GGGGGAAACCCCC"
    result = structure_features(sequence, "(((((...)))))", 3)
    assert result["stems"][0]["inner_span"] == 3
    assert [r["inner_span"] for r in result["m_subruns"]] == [7, 5, 3]
    matrix = [[0.0] * 5 for _ in range(5)]
    matrix[1][4] = 0.25
    assert bpp_features("ACGT", matrix, wc_only=True)["ensemble_paired_fraction"] == 0.125
    matrix[4][1] = 0.25
    with pytest.raises(AssertionError):
        bpp_features("ACGT", matrix, wc_only=True)
    with pytest.raises(ValueError):
        structure_features("AC", ")(", 1)


def test_native_api_and_two_bounded_folds(tmp_path):
    models = prepare_models(tmp_path / "DNA.par")
    assert len(models["models"]) == 10
    assert all(
        row["status"].startswith("API_FIELDS_EFFECTIVE") for row in models["models"].values()
    )
    assert (
        models["parameter_sha256"]
        == "68e912df500b78bffc6c9fe2c6c81f5c10b0335c848512c9acb6ce14f9728beb"
    )
    for spec in model_specs()[:2]:
        result = fold_sequence("GGGGGGAAACCCCCC", 6, 3, spec, ensemble=True)
        assert result["status"] == "OK", result
        assert result["partition_free_energy"] <= result["mfe"] + 1e-4
        assert 0 <= result["ensemble_paired_fraction"] <= 1
        assert result["effective_model_fields"]["salt"] == 1.021
        if spec.wc_only:
            assert result["GT_pair_probability_mass"] <= 1e-6


def test_deployment_encoder_exception_keeps_all_planned_strands(monkeypatch):
    class Protocol:
        payload_bytes = 1
        maximum_user_bytes = 47

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(physical, "PublicProtocol", Protocol)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic encoder failure")

    monkeypatch.setattr(physical, "encode_file", fail)
    rows = physical.deployed_inputs(100, 6, 3, 20261023601, 0, CONFIG)
    assert len(rows) == 252 and len({r["user_sha256"] for r in rows}) == 1
    assert all(r["sequence"] is None and r["planned_strands"] == 63 for r in rows)
    assert len({r["method"] for r in rows}) == 4


def test_discordant_cp_each_tail_independent_log_binomial_sum():
    # Not beta-CDF inversion: direct binomial probability sums check both tails.
    def probability_sum(n, p, values):
        if p == 0:
            return float(0 in values)
        if p == 1:
            return float(n in values)
        return math.fsum(
            math.exp(
                math.lgamma(n + 1)
                - math.lgamma(j + 1)
                - math.lgamma(n - j + 1)
                + j * math.log(p)
                + (n - j) * math.log1p(-p)
            )
            for j in values
        )

    alpha_cell = 0.05 / 24
    per_tail = alpha_cell / 4
    for n in (1, 2, 10, 1152):
        for x in sorted({0, 1, n // 2, n - 1, n}):
            low, high = clopper_pearson(x, n, alpha=alpha_cell / 2)
            if x:
                assert abs(probability_sum(n, low, range(x, n + 1)) - per_tail) < 1e-10
            else:
                assert low == 0
            if x < n:
                assert abs(probability_sum(n, high, range(x + 1)) - per_tail) < 1e-10
            else:
                assert high == 1


def test_missing_pair_null_ci_fixed_family_and_abstaining_coverage(tmp_path):
    cfg = {
        **CONFIG,
        "n_values": [40],
        "m_values": [4],
        "k_values": [3],
        "mechanism_seeds": [77],
        "pairs_per_seed_cell": 2,
    }
    inputs = physical.mechanism_inputs(40, 4, 3, 77, 2, parent_cap=10000, control_cap=100)
    cp = physical.Checkpoint(tmp_path / "complete", {"unit_fixture": True})
    cp.unit("input_fixture", lambda: inputs)
    observed = [
        {
            **r,
            "model_id": "T37_S1.021_default",
            "status": "OK",
            "has_stem_ge_m": r["role"] == "control",
        }
        for r in inputs
    ]
    cp.unit("fold_fixture", lambda: observed)
    complete = physical.summarize_rows(cp, cfg, "formal")["mechanism_cells"][0]
    assert complete["CI_pp"] is not None and complete["fixed_family_size"] == 24
    assert complete["marginal_CP_each_tail"] == 0.05 / 24 / 4
    assert "no conditional-on-return" in complete["finite_cap_coverage"]
    bad = [
        {**r, "control_result": {"status": "CONTROL_ATTEMPT_CAP"}} if r["pair_index"] == 0 else r
        for r in inputs
    ]
    failed = physical.Checkpoint(tmp_path / "missing", {"unit_fixture": True})
    failed.unit("input_fixture", lambda: bad)
    failed.unit(
        "fold_fixture", lambda: observed
    )  # Even fraudulent OK fold flags cannot conceal cap.
    result = physical.summarize_rows(failed, cfg, "formal")["mechanism_cells"][0]
    assert result["CI_pp"] is None and result["paired_discordance_conservative_interval"] is None
    assert result["comparison_status"] == "INCOMPLETE_NON_EVALUABLE"
    assert result["planned_pairs"] == 2 and result["fixed_family_size"] == 24
    assert result["target_or_control_cap_failed_pairs"] == 1


def test_unplanned_or_duplicate_pair_cannot_replace_a_failed_or_missing_pair(tmp_path):
    cfg = {
        **CONFIG,
        "n_values": [40],
        "m_values": [4],
        "k_values": [3],
        "mechanism_seeds": [77],
        "pairs_per_seed_cell": 2,
    }

    def pair(index, name, status="OK"):
        return [
            {
                "arm": "mechanism",
                "n": 40,
                "m": 4,
                "k": 3,
                "seed": 77,
                "pair_index": index,
                "pair_id": name,
                "row_id": name + "_" + role,
                "role": role,
                "control_result": {"status": status},
            }
            for role in ("target", "control")
        ]

    fixtures = [
        pair(0, "P0", "CONTROL_ATTEMPT_CAP") + pair(1, "P1") + pair(2, "EXTRA"),
        pair(0, "P0") + pair(2, "REPLACEMENT"),
        pair(0, "P0") + pair(0, "DUPLICATE"),
    ]
    for index, inputs in enumerate(fixtures):
        cp = physical.Checkpoint(tmp_path / str(index), {"unit_fixture": True})
        cp.unit("input_fixture", lambda inputs=inputs: inputs)
        cp.unit(
            "fold_fixture",
            lambda inputs=inputs: [
                {
                    **row,
                    "model_id": "T37_S1.021_default",
                    "status": "OK",
                    "has_stem_ge_m": row["role"] == "control",
                }
                for row in inputs
            ],
        )
        result = physical.summarize_rows(cp, cfg, "formal")["mechanism_cells"][0]
        assert result["planned_pairs"] == 2
        assert result["planned_pair_keys_valid"] is False
        assert result["comparison_status"] == "INCOMPLETE_NON_EVALUABLE"
        assert result["CI_pp"] is None
        assert result["paired_discordance_conservative_interval"] is None
