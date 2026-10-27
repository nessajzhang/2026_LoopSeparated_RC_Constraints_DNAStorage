from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from loop_alpha_audit.endpoint_general import EndpointMapper
from loop_alpha_audit.prepared_codec import DecodeResult, EncodeResult

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("codec_benchmark", ROOT / "code/benchmark_codecs.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
ADDRESS = "TCCTCCCCTCTCCTCTTTCTTCTT"


def document(configs, phase="SMOKE_NOT_FORMAL"):
    return {
        "phase": phase,
        "seed": 20261023125,
        "batches": [1, 3],
        "repeats": 2,
        "configs": configs,
    }


def test_exact_m3_adapter_and_actual_perpendicular():
    codec = BENCH.prepare_for_benchmark({"kind": "endpoint_m3", "n": 20, "m": 3, "k": 10})
    assert codec.M == codec._mapper.codebook_size
    assert codec.K == codec.M.bit_length() - 1
    assert len(codec.metadata["left_windows"]) == 27
    assert len(codec.metadata["right_windows"]) == 37
    assert all("A" in word for word in codec.metadata["right_windows"])
    for index in (0, 1, (1 << codec.K) - 1):
        bits = format(index, f"0{codec.K}b")
        encoded = codec.encode_bits(bits)
        assert encoded.status == "OK" and encoded.attempts == 1
        assert BENCH.validate_oligo(codec, encoded.sequence) is None
        assert codec.decode_bits(encoded.sequence).bits == bits
    assert codec.encode_bits("bad").status == "INVALID_INPUT"
    assert codec.decode_bits("N" * codec.n).status == "INVALID_SEQUENCE"
    if codec.M > 1 << codec.K:
        unused = "".join("ACGT"[a] for a in codec._mapper.encode_index(1 << codec.K))
        assert codec.decode_bits(unused).status == "INVALID_SEQUENCE"
    assert BENCH.count_table_bytes(codec) > 0


def test_m3_cache_is_only_benchmark_local():
    BENCH.clear_benchmark_caches()
    config = {"kind": "endpoint_m3", "n": 20, "m": 3, "k": 10}
    left = BENCH.prepare_for_benchmark(config)
    assert BENCH.prepare_for_benchmark(config) is left
    BENCH.clear_benchmark_caches()
    assert BENCH.prepare_for_benchmark(config) is not left
    assert isinstance(left._mapper, EndpointMapper)
    with pytest.raises(ValueError):
        BENCH.prepare_for_benchmark({**config, "prefix": ADDRESS})


def test_formal_config_freezes_complete_predeclared_grid():
    data = json.loads((ROOT / "configs/revision/codec_formal_v1.json").read_text())
    BENCH.validate_document(data)
    assert data["batches"] == [1, 1000, 10000] and data["repeats"] == 5
    assert len(data["configs"]) == 58
    assert sum(data["batches"]) * data["repeats"] * len(data["configs"]) == 3_190_290
    assert data["seed"] != 20261023
    for field, value in (("batches", [1, 100]), ("repeats", 4), ("seed", 20261023)):
        modified = copy.deepcopy(data)
        modified[field] = value
        with pytest.raises(ValueError):
            BENCH.validate_document(modified)
    modified = copy.deepcopy(data)
    modified["configs"][-1]["prefix"] = ADDRESS + "CC"
    with pytest.raises(ValueError):
        BENCH.validate_document(modified)


def test_percentile_and_group_summary_include_failures():
    assert BENCH.percentile95(list(range(1, 101))) == 95
    assert BENCH.percentile95([7]) == 7
    group = BENCH._group_summary(0, 3, 300, [10, 100, 1000], [5], [1, 4, 4], 1, 2)
    assert group["encode_wall_total_ns"] == 1110
    assert group["encode_median_ns"] == 100 and group["encode_p95_ns"] == 1000
    assert group["decode_success_conditional_median_ns"] == 5
    assert group["cold_prepare_amortized_ns_per_input"] == 100
    with pytest.raises(ValueError):
        BENCH.percentile95([])


def test_smoke_all_six_mapper_kinds_and_phase_rows(tmp_path):
    configs = [
        {"kind": "endpoint_m2", "n": 64, "m": 2, "k": 32},
        {"kind": "endpoint_m3", "n": 64, "m": 3, "k": 32},
    ]
    for kind in ("short_core", "nonce", "gc_h_nonce", "two_bit"):
        config = {"kind": kind, "n": 64, "m": 7, "k": 3, "prefix": ADDRESS + BENCH.MODE_TAGS[kind]}
        if kind in ("nonce", "gc_h_nonce"):
            config.update(nonce_nt=4, nonce_budget=64)
        configs.append(config)
    data = document(configs)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    out = tmp_path / "smoke"
    assert BENCH.run(data, config_path, out, 30) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["phase"] == "SMOKE_NOT_FORMAL"
    assert summary["completed_cells"] == 6 and summary["total_samples"] == 48
    assert summary["formal_grid_including_10000_batch_pending_T34"]
    assert summary["codec_source_unchanged"]
    for path in out.glob("config_*/summary.json"):
        cell = json.loads(path.read_text())
        assert cell["phase"] == "SMOKE_NOT_FORMAL"
        with path.with_name("rows.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 8
        for row in rows:
            if row["encode_status"] == "OK":
                assert row["roundtrip_ok"] == "True" and row["validation_status"] == "PASS"
                assert row["input_bits_sha256"] == row["decoded_bits_sha256"]


def test_wrong_noiseless_inverse_retains_row_and_stops_config(tmp_path, monkeypatch):
    class WrongCodec:
        kind, n, m, k, prefix, K, M, metadata = "two_bit", 1, 7, 3, "", 2, 4, {}
        decodes = 0

        def encode_bits(self, bits):
            return EncodeResult("OK", "ACGT"[int(bits, 2)], 1, None, 1)

        def decode_bits(self, sequence):
            self.decodes += 1
            value = "ACGT".index(sequence)
            if self.decodes > 1:
                value ^= 1
            return DecodeResult("OK", format(value, "02b"), 1, None)

    codec = WrongCodec()
    monkeypatch.setattr(BENCH, "prepare_for_benchmark", lambda _: codec)
    data = document([{"kind": "two_bit", "n": 1, "m": 7, "k": 3}])
    assert BENCH.worker(data, 0, tmp_path) == 2
    failure = json.loads((tmp_path / "config_00_two_bit/failure.json").read_text())
    assert failure["partial_rows"] == failure["noiseless_inverse_failure_rows"] == 1
    assert failure["execution_status"] == "INTEGRITY_FAILURE"
    assert (tmp_path / "config_00_two_bit/rows.csv").exists()


def test_timeout_is_bounded_and_not_retried(tmp_path, monkeypatch):
    calls = []

    def timeout(command, **kwargs):
        if command[0] == "sysctl":
            return SimpleNamespace(returncode=0, stdout="test")
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 1, output=b"partial", stderr=b"retained")

    monkeypatch.setattr(BENCH.subprocess, "run", timeout)
    data = document([{"kind": "two_bit", "n": 4, "m": 7, "k": 3}])
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    out = tmp_path / "timeout"
    assert BENCH.run(data, path, out, 1) == 1
    assert len(calls) == 1
    command = json.loads((out / "command_00.json").read_text())
    assert command["exit_code"] == 124 and command["stdout"] == "partial"
    with pytest.raises(ValueError):
        BENCH.run(data, path, tmp_path / "bad_timeout", 301)


def test_full_oligo_validator_checks_transmitted_prefix_and_nonce():
    config = {
        "kind": "nonce",
        "n": 100,
        "m": 7,
        "k": 3,
        "prefix": ADDRESS + "AT",
        "nonce_nt": 4,
        "nonce_budget": 64,
    }
    codec = BENCH.prepare_for_benchmark(config)
    encoded = codec.encode_bits("0" * codec.K)
    assert encoded.status == "OK" and BENCH.validate_oligo(codec, encoded.sequence) is None
    assert BENCH.validate_oligo(codec, "A" + encoded.sequence[1:]) == "fixed_prefix_mismatch"
    assert BENCH.validate_oligo(codec, encoded.sequence[:-1]) == "wrong_length_or_DNA_alphabet"
    assert BENCH.validate_oligo(codec, codec.prefix + "A" * (codec.n - len(codec.prefix)))
