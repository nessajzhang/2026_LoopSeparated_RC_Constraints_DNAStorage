from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from loop_alpha_audit.spectral_m3_analytic import ANALYTIC_PROOF, analytic_certificate

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "code/check_m3_analytic_certificate.py"
PRODUCER_PATH = ROOT / "code/certify_m3_analytic.py"


def _load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CHECKER = _load("independent_m3_analytic_checker", CHECKER_PATH)
CLI = _load("m3_analytic_producer_cli", PRODUCER_PATH)


@pytest.fixture(scope="module")
def certificate():
    return analytic_certificate()


def test_exact_analytic_certificate_and_binary_coverage(certificate):
    result = CHECKER.check_certificate(certificate)
    assert result["dna_product_lower"] == result["dna_product_upper"] == 9
    assert result["binary_masks_checked"] == 256
    assert result["rc_single_edge_bases_checked"] == 64
    assert not result["dna_assignments_exhaustively_enumerated"]
    assert not result["formal_proof_assistant"]
    assert [row["mask"] for row in certificate["binary_sanity"]["rows"]] == list(range(256))
    assert certificate == analytic_certificate()


def test_actual_uperp_and_ordinary_complement_are_different_languages(certificate):
    witness = certificate["witness"]
    assert len(witness["u_words"]) == 27
    assert len(witness["uperp_words"]) == len(witness["complement_words"]) == 37
    assert all("A" in word for word in witness["uperp_words"])
    assert all("T" in word for word in witness["complement_words"])
    assert set(witness["uperp_words"]) != set(witness["complement_words"])
    assert witness["uperp_right_positive_eigenvector"][0] == 3  # AA ends in A
    assert witness["complement_right_positive_eigenvector"][0] == 1
    assert witness["uperp_right_positive_eigenvector"][15] == 1  # TT contains no A
    assert witness["complement_right_positive_eigenvector"][15] == 3


def _modified_proof_with_recomputed_hash(data):
    data["proof"]["text"] += "\nThe all-m analogue is also claimed.\n"
    data["proof"]["sha256"] = hashlib.sha256(data["proof"]["text"].encode()).hexdigest()


def _binary_row_missing_with_recomputed_hash(data):
    data["binary_sanity"]["rows"].pop()
    rows = data["binary_sanity"]["rows"]
    data["binary_sanity"]["rows_sha256"] = hashlib.sha256(CHECKER._canonical(rows)).hexdigest()


MUTATIONS = (
    pytest.param(lambda d: d["parameters"].update(q=4.0), id="no-floating-parameters"),
    pytest.param(
        lambda d: d["claim"].update(all_dna_assignments_enumerated=0), id="no-integer-scope-flag"
    ),
    pytest.param(lambda d: d["parameters"].update(m=2), id="wrong-m"),
    pytest.param(lambda d: d["parameters"].update(theta=[0, 1, 2, 3]), id="wrong-involution"),
    pytest.param(lambda d: d["claim"].update(dna_upper=8), id="false-upper"),
    pytest.param(
        lambda d: d["claim"].update(all_dna_assignments_enumerated=True), id="false-enumeration"
    ),
    pytest.param(
        lambda d: d["claim"].update(proof_assistant_formalized=True), id="false-formalization"
    ),
    pytest.param(lambda d: d["claim"].update(all_m_theorem_claimed=True), id="unproved-all-m"),
    pytest.param(_modified_proof_with_recomputed_hash, id="unreviewed-proof-even-with-new-hash"),
    pytest.param(lambda d: d["proof"].update(sha256="0" * 64), id="proof-hash"),
    pytest.param(
        lambda d: d["proof"]["elimination"]["coefficients"][0].__setitem__(3, -3),
        id="polynomial-coefficient",
    ),
    pytest.param(
        lambda d: d["proof"]["attribution"].update(role="direct source theorem"),
        id="false-attribution",
    ),
    pytest.param(lambda d: d["bridge"]["state_permutation"].__setitem__(0, 0), id="rc-permutation"),
    pytest.param(
        lambda d: d["bridge"].update(all_single_edge_bases_checked=63), id="missing-basis"
    ),
    pytest.param(lambda d: d["witness"].update(u_hex="0x0"), id="wrong-u-mask"),
    pytest.param(lambda d: d["witness"]["matrix_u"][0].__setitem__(0, 0), id="matrix-entry"),
    pytest.param(lambda d: d["witness"].update(uperp_required_letter="T"), id="mislabeled-uperp"),
    pytest.param(
        lambda d: d["witness"].update(complement_required_letter="A"), id="mislabeled-complement"
    ),
    pytest.param(
        lambda d: d["witness"].update(
            uperp_right_positive_eigenvector=d["witness"]["complement_right_positive_eigenvector"]
        ),
        id="wrong-right-positive-vector",
    ),
    pytest.param(
        lambda d: d["witness"]["uperp_right_positive_eigenvector"].__setitem__(0, 0),
        id="nonpositive-right-vector",
    ),
    pytest.param(
        lambda d: d["witness"].update(left_nonnegative_eigenvector=[0] * 16), id="zero-left-vector"
    ),
    pytest.param(lambda d: d["witness"]["charpoly_uperp"].__setitem__(1, -2), id="charpoly"),
    pytest.param(_binary_row_missing_with_recomputed_hash, id="binary-coverage-not-hash-only"),
    pytest.param(
        lambda d: d["binary_sanity"]["rows"][17].update(mask=16), id="duplicate-binary-mask"
    ),
    pytest.param(
        lambda d: d["binary_sanity"]["rows"][17]["left_upper"].update(numerator=0),
        id="binary-false-bound",
    ),
    pytest.param(
        lambda d: d["degenerate_sanity"]["zero_inner_product"].update(inner_product=1),
        id="zero-inner-product-branch",
    ),
)


@pytest.mark.parametrize("mutate", MUTATIONS)
def test_tampered_certificates_rejected(certificate, mutate):
    modified = copy.deepcopy(certificate)
    mutate(modified)
    with pytest.raises(ValueError):
        CHECKER.check_certificate(modified)


def test_zero_product_and_reducible_zero_inner_product(certificate):
    data = certificate["degenerate_sanity"]
    CHECKER._degenerate(data)
    assert "No step divides by p^T x" in ANALYTIC_PROOF
    assert "They need not be strictly positive" in ANALYTIC_PROOF
    assert "If lambda*mu=0" in ANALYTIC_PROOF


def test_checker_has_no_producer_or_external_imports(certificate, tmp_path):
    tree = ast.parse(CHECKER_PATH.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "math",
        "fractions",
        "itertools",
        "pathlib",
    }
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(certificate))
    output = tmp_path / "checked.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(CHECKER_PATH),
            "--certificate",
            str(path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == json.loads(output.read_text())
    assert json.loads(output.read_text())["dna_product_upper"] == 9


def test_json_duplicate_key_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema":"one","schema":"two"}')
    completed = subprocess.run(
        [sys.executable, "-I", str(CHECKER_PATH), "--certificate", str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "duplicate JSON" in completed.stderr


def test_one_command_production_and_immutable_output_manifest(tmp_path):
    directory = tmp_path / "attempt-001"
    completed = subprocess.run(
        [sys.executable, str(PRODUCER_PATH), "--output-dir", str(directory)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert json.loads(completed.stdout)["task_state"] == "IMPLEMENTED_UNVERIFIED"
    assert "returncode=0" in (directory / "independent_checker.log").read_text()
    outputs = json.loads((directory / "output_hashes.json").read_text())
    assert "output_hashes.json" not in outputs
    for filename, digest in outputs.items():
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == digest
    sources = json.loads((directory / "source_hashes.json").read_text())
    for filename, digest in sources.items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        CLI.produce_attempt(directory)
