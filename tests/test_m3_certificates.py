from __future__ import annotations

import copy
import importlib.util
import json
import random
import shutil
import signal
import subprocess
import time
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pytest

from loop_alpha_audit.spectral_m3 import (
    PowerBound,
    WindowGraph,
    cardinality_bound,
    extend_search,
    finite_word_count,
    initial_search_certificate,
    n4_identity,
    scc_collatz_certificate,
    structured_candidate,
)

CHECKER_PATH = Path(__file__).resolve().parents[1] / "code" / "check_m3_certificate.py"
SPEC = importlib.util.spec_from_file_location("standalone_checker", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_exact_baseline_and_arbitrary_domain():
    graph = WindowGraph()
    candidate = structured_candidate()
    assert len(candidate["windows"]) == 27
    assert graph.edge_count == 64
    assert graph.state_count == 16
    assert graph.full_mask == 2**64 - 1
    # Both colours may contain members of the same RC orbit: this U is not
    # being restricted to one representative of each of 32 pairs.
    assert graph.mask(((0, 0, 0), (3, 3, 3))) == (1 | (1 << 63))
    data = initial_search_certificate()
    result = CHECKER.check_certificate(data)
    assert result["lower_display"] == 9
    assert result["global_upper"] == {"numerator": 1023, "denominator": 1, "power": 3}
    assert not result["optimality_proved"]


def test_power_bounds_cardinalities_and_finite_counts():
    assert PowerBound(Fraction(9**3), 3) <= PowerBound(Fraction(9))
    assert not PowerBound(Fraction(82), 2) <= PowerBound(Fraction(9))
    assert cardinality_bound(0, 0, 64, 3).value == 1024
    assert cardinality_bound(40, 0, 64, 3).value == 40 * 24
    graph = WindowGraph()
    for mask in (
        0,
        graph.full_mask,
        int(structured_candidate()["mask_hex"], 16),
        0x0123456789ABCDEF,
    ):
        first, second = n4_identity(graph, mask)
        assert finite_word_count(graph, mask, 4) == first
        assert finite_word_count(graph, graph.full_mask ^ mask, 4) == second
        assert finite_word_count(graph, mask, 2) == 16


def test_m3_exact_transpose_similarity_and_all_proved_symmetry_types():
    graph = WindowGraph()
    rng = random.Random(20261023)
    masks = [0, graph.full_mask, int(structured_candidate()["mask_hex"], 16)] + [
        rng.getrandbits(64) for _ in range(5)
    ]
    states = list(product(range(4), repeat=2))
    full = graph.matrix(graph.full_mask)
    for mask in masks:
        old = graph.matrix(mask)
        for permutation in permutations(range(4)):
            for reversal in (False, True):
                mapping = [
                    4 * permutation[(state[::-1] if reversal else state)[0]]
                    + permutation[(state[::-1] if reversal else state)[1]]
                    for state in states
                ]
                for swap in (False, True):
                    new = graph.matrix(graph.transform(mask, permutation, reversal, swap))
                    for i in range(16):
                        for j in range(16):
                            expected = full[i][j] - old[i][j] if swap else old[i][j]
                            a, b = (
                                (mapping[j], mapping[i]) if reversal else (mapping[i], mapping[j])
                            )
                            assert new[a][b] == expected
        assert graph.perp(mask) == graph.full_mask ^ graph.transform(mask, (3, 2, 1, 0), True)


def test_checkpoint_recovery_and_forced_frontier_coverage():
    data = initial_search_certificate()
    extend_search(data, max_nodes=17, max_seconds=10)
    first = CHECKER.check_certificate(data)
    assert first["node_kinds"]["frontier"] > 1
    resumed = copy.deepcopy(data)
    extend_search(resumed, max_nodes=23, max_seconds=10)
    second = CHECKER.check_certificate(resumed)
    assert second["nodes_checked"] > first["nodes_checked"]
    assert second["global_upper_display"] <= first["global_upper_display"]
    zero = initial_search_certificate()
    extend_search(zero, max_nodes=0, max_seconds=0)
    assert CHECKER.check_certificate(zero)["node_kinds"]["frontier"] == 1


def test_missing_sibling_missing_frontier_and_corrupt_saved_bound_rejected():
    data = initial_search_certificate()
    extend_search(data, max_nodes=2, max_seconds=10)
    for mutate in (
        lambda d: d["nodes"].pop(),
        lambda d: d["nodes"][0].update(children=[1]),
        lambda d: d["nodes"][0].update(children=[1, 1]),
        lambda d: d["nodes"][1].update(a="0x0"),
        lambda d: d["nodes"][0]["upper"].update(numerator=1),
        lambda d: d["nodes"][2].update(ref=2),
        lambda d: d["nodes"][2].update(reversal=True, colour_swap=False),
    ):
        bad = copy.deepcopy(data)
        mutate(bad)
        with pytest.raises(ValueError):
            CHECKER.check_certificate(bad)


def test_invalid_collatz_vector_and_mismatched_powers_rejected():
    data = initial_search_certificate()
    graph = WindowGraph()
    left, _ = scc_collatz_certificate(graph.matrix(graph.full_mask), exponent=1, steps=0)
    proof = {"kind": "scc_collatz_product", "left": left, "right": copy.deepcopy(left)}
    data["nodes"][0]["proofs"].append(proof)
    CHECKER.check_certificate(data)
    bad = copy.deepcopy(data)
    bad["nodes"][0]["proofs"][-1]["left"]["components"][0]["vector"][0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        CHECKER.check_certificate(bad)
    bad = copy.deepcopy(data)
    bad["nodes"][0]["proofs"][-1]["right"]["exponent"] = 2
    with pytest.raises(ValueError, match="same exponent"):
        CHECKER.check_certificate(bad)
    bad = copy.deepcopy(data)
    bad["nodes"][1]["kind"] = "pruned"
    with pytest.raises(ValueError, match="pruned without"):
        CHECKER.check_certificate(bad)


def test_m2_entire_65536_domain_rc_transpose_and_complement_objective():
    graph = WindowGraph(q=4, m=2)
    masks = np.arange(65536, dtype=np.uint32)
    matrices = (
        ((masks[:, None] >> np.arange(16, dtype=np.uint32)) & 1).reshape(-1, 4, 4).astype(float)
    )
    # RC is theta-permuted transpose.  All 65536 matrices, not sampled masks.
    theta = np.array((3, 2, 1, 0))
    rc_matrices = matrices.transpose(0, 2, 1)[:, theta][:, :, theta]
    perps = 1 - rc_matrices
    radii = np.max(np.abs(np.linalg.eigvals(matrices)), axis=1)
    perp_radii = np.max(np.abs(np.linalg.eigvals(perps)), axis=1)
    assert np.allclose(perp_radii, radii[65535 - masks], atol=2e-7, rtol=1e-7)
    target = 3 * (1 + np.sqrt(13)) / 2
    assert abs(np.max(radii * perp_radii) - target) < 1e-7
    for mask in range(65536):
        rc_mask = graph.transform(mask, (3, 2, 1, 0), reversal=True)
        expected = sum(
            int(rc_matrices[mask, i, j]) << (4 * i + j) for i in range(4) for j in range(4)
        )
        assert rc_mask == expected
        assert graph.perp(mask) == 65535 ^ rc_mask
    # Eigvals above are cross-checks only.  All final pruning tests use exact
    # certificates; the matrix identity is checked bit-for-bit on the domain.


@pytest.fixture(scope="module")
def native_producer(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++17 compiler is not installed")
    directory = tmp_path_factory.mktemp("m3_native")
    executable = directory / "search_m3"
    source = CHECKER_PATH.with_name("search_m3_certified.cpp")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidate = directory / "candidate.json"
    candidate.write_text(json.dumps(structured_candidate()))
    return executable, candidate, directory


def test_native_checkpoint_recovery_and_actual_sigint(native_producer):
    executable, candidate, directory = native_producer
    first, resumed, interrupted = [
        directory / name for name in ("first.json", "resumed.json", "interrupted.json")
    ]
    common = [str(executable), "--candidate", str(candidate)]
    subprocess.run(
        common + ["--output", str(first), "--max-nodes", "50", "--seconds", "5"],
        check=True,
        capture_output=True,
        text=True,
    )
    a = CHECKER.check_certificate(json.loads(first.read_text()))
    subprocess.run(
        common
        + [
            "--output",
            str(resumed),
            "--resume-state",
            str(first) + ".state",
            "--max-nodes",
            "25",
            "--seconds",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    b = CHECKER.check_certificate(json.loads(resumed.read_text()))
    assert b["nodes_checked"] > a["nodes_checked"]
    process = subprocess.Popen(
        common + ["--output", str(interrupted), "--max-nodes", "500000", "--seconds", "2"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.05)
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 0, stderr
    assert json.loads(stdout)["interrupted"]
    c = CHECKER.check_certificate(json.loads(interrupted.read_text()))
    assert c["node_kinds"]["frontier"] > 0
    assert c["lower_display"] == 9
    assert not c["optimality_proved"]
