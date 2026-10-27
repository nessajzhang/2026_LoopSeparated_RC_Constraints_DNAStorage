#!/usr/bin/env python3
"""Serial, fresh-process mapper benchmarks with explicit pilot/formal phases.

Only the exact-K public bits interface is timed. RNG, CSV, output hashing,
and additional whole-oligo validation are outside encode/decode timers.
K includes caller framing/filler and is not user net rate. No channel,
error-correction, biochemical benefit, or million-length throughput follows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
from functools import lru_cache
import hashlib
import importlib.metadata
from itertools import product
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
from time import perf_counter_ns
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.encoder import cache_info, clear_caches  # noqa: E402
from loop_alpha_audit.endpoint_general import EndpointMapper  # noqa: E402
from loop_alpha_audit.prepared_codec import DecodeResult, EncodeResult, prepare  # noqa: E402
from loop_alpha_audit.short_loop_codec import avoids_rc_windows, full_oligo_failure  # noqa: E402

FORMAL_PHASE = "FORMAL_MAPPER_BENCHMARK_V1"
PHASES = {FORMAL_PHASE, "PILOT_NOT_FORMAL_GRID", "SMOKE_NOT_FORMAL"}
MODE_TAGS = {"short_core": "CC", "nonce": "AT", "gc_h_nonce": "AG", "two_bit": "GG"}
DNA = "ACGT"


class IntegrityFailure(RuntimeError):
    """An invalid encoded oligo or incorrect noiseless inverse stops the run."""


class EndpointM3Bits:
    """Benchmark-local facade over the unchanged generic all-window mapper.

    The exact product-subcode count is not the complete triangular-language
    count. No fixed TC address can be attached to its ACG left endpoint.
    """

    def __init__(self, config):
        if set(config) - {"kind", "n", "m", "k", "prefix"}:
            raise ValueError("unknown endpoint_m3 benchmark configuration key")
        if config.get("m", 3) != 3 or config.get("prefix", ""):
            raise ValueError("endpoint_m3 requires m=3 and no incompatible fixed TC prefix")
        self._mapper = EndpointMapper(config["n"], 3, config["k"], product(range(3), repeat=3))
        self.kind, self.n, self.m, self.k, self.prefix = (
            "endpoint_m3",
            self._mapper.n,
            3,
            self._mapper.k,
            "",
        )
        self.M, self.K = self._mapper.codebook_size, self._mapper.K
        self.metadata = {
            "constraint_scope": "full_oligo",
            "count_scope": "exact_all_window_endpoint_product_subcode_not_full_language",
            "regime": self._mapper.regime,
            "endpoint_length": self._mapper.endpoint_length,
            "free_length": self._mapper.free_length,
            "left_count": self._mapper.left_count,
            "middle_count": self._mapper.middle_count,
            "right_count": self._mapper.right_count,
            "left_windows": ["".join(DNA[a] for a in w) for w in sorted(self._mapper.allowed)],
            "right_windows": [
                "".join(DNA[a] for a in w) for w in sorted(self._mapper.perpendicular)
            ],
            "right_required_letter": "A",
            "benchmark_local_prepared_object_cache": True,
            "error_correction": False,
            "short_k_codec": False,
        }

    def encode_bits(self, bits):
        start = perf_counter_ns()
        try:
            if not isinstance(bits, str) or len(bits) != self.K or set(bits) - set("01"):
                raise ValueError("payload must contain exactly K binary digits")
            word = self._mapper.encode_payload(int(bits, 2) if bits else 0)
            sequence = "".join(DNA[a] for a in word)
            return EncodeResult("OK", sequence, perf_counter_ns() - start, None, 1)
        except ValueError as error:
            return EncodeResult("INVALID_INPUT", None, perf_counter_ns() - start, str(error), 0)

    def decode_bits(self, sequence):
        start = perf_counter_ns()
        try:
            if not isinstance(sequence, str) or len(sequence) != self.n or set(sequence) - set(DNA):
                raise ValueError("wrong length or invalid DNA symbol")
            value = self._mapper.decode_payload(tuple(DNA.index(a) for a in sequence))
            bits = format(value, f"0{self.K}b") if self.K else ""
            return DecodeResult("OK", bits, perf_counter_ns() - start, None)
        except ValueError as error:
            return DecodeResult("INVALID_SEQUENCE", None, perf_counter_ns() - start, str(error))


@lru_cache(maxsize=16)
def _m3_prepared(serialized):
    return EndpointM3Bits(json.loads(serialized))


def prepare_for_benchmark(config):
    if config.get("kind") == "endpoint_m3":
        return _m3_prepared(json.dumps(config, sort_keys=True))
    return prepare(config)


def clear_benchmark_caches():
    clear_caches()
    _m3_prepared.cache_clear()


def count_table_bytes(codec):
    """Unique reachable tuple/integer bytes of count tables, not process RSS."""
    if codec.kind == "endpoint_m3":
        pending = [side.suffix_counts for side in (codec._mapper.left, codec._mapper.right) if side]
    elif codec.kind == "endpoint_m2":
        pending = [codec._table]
    elif codec.kind == "short_core":
        pending = [codec._membership.table, codec._gc.table]
    else:
        pending = []
    seen, total = set(), 0
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        total += sys.getsizeof(value)
        if isinstance(value, tuple):
            pending.extend(value)
    return total


def percentile95(values):
    if not values:
        raise ValueError("P95 requires at least one observation")
    return sorted(values)[(95 * len(values) + 99) // 100 - 1]


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    names = (
        "encoder.py",
        "prepared_codec.py",
        "short_loop_codec.py",
        "endpoint_general.py",
        "addresses.py",
        "framing.py",
    )
    paths = [ROOT / "loop_alpha_audit" / name for name in names]
    paths += [Path(__file__).resolve(), ROOT / "pyproject.toml", ROOT / "uv.lock"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def validate_document(document):
    if document.get("phase") not in PHASES:
        raise ValueError("explicit supported phase is required")
    batches, repeats = document.get("batches"), document.get("repeats")
    if (
        not isinstance(batches, list)
        or not batches
        or len(set(batches)) != len(batches)
        or any(type(x) is not int or x < 1 for x in batches)
        or type(repeats) is not int
        or repeats < 1
        or type(document.get("seed")) is not int
        or not isinstance(document.get("configs"), list)
        or not document["configs"]
    ):
        raise ValueError("invalid batches, repeats, seed or configurations")
    if document["phase"] == FORMAL_PHASE:
        if batches != [1, 1000, 10000] or repeats != 5 or document["seed"] == 20261023:
            raise ValueError("formal phase requires frozen batches, five repeats, and a new seed")
        from loop_alpha_audit.addresses import make_address_book

        address = make_address_book().sequences[0]
        if document.get("public_address") != address:
            raise ValueError("formal prefix must use the existing actual public 24-nt address")
        for config in document["configs"]:
            kind = config.get("kind")
            if kind in MODE_TAGS and config.get("prefix") != address + MODE_TAGS[kind]:
                raise ValueError("formal operational mapper requires actual address plus mode")
            if kind in {"endpoint_m2", "endpoint_m3"} and config.get("prefix", ""):
                raise ValueError(
                    "theoretical endpoint benchmark cannot silently attach a TC address"
                )


def sample_seed(document, identity, repeat, batch):
    message = f"codec-benchmark-v1|{document['seed']}|{identity}|{repeat}|{batch}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big")


def validate_oligo(codec, sequence):
    if not isinstance(sequence, str) or len(sequence) != codec.n or set(sequence) - set(DNA):
        return "wrong_length_or_DNA_alphabet"
    if not sequence.startswith(codec.prefix):
        return "fixed_prefix_mismatch"
    if codec.kind in {"endpoint_m2", "endpoint_m3"}:
        return None if avoids_rc_windows(sequence, codec.m, codec.k) else "RC_window_violation"
    if codec.kind in {"short_core", "nonce", "gc_h_nonce"}:
        failure = full_oligo_failure(
            sequence, codec.m, codec.k, check_rc=codec.kind != "gc_h_nonce"
        )
        if failure:
            return failure
        if codec.kind == "short_core" and sum(a in "GC" for a in sequence) != codec.n // 2:
            return "exact_total_GC_violation"
    return None


def _group_summary(repeat, batch, cold, encode, decode, attempts, successes, failures):
    encode_total, decode_total = sum(encode), sum(decode)
    return {
        "repeat": repeat,
        "batch_size": batch,
        "successes": successes,
        "failures": failures,
        "encode_wall_total_ns": encode_total,
        "decode_wall_total_ns": decode_total,
        "encode_median_ns": statistics.median(encode),
        "encode_p95_ns": percentile95(encode),
        "decode_success_conditional_median_ns": statistics.median(decode) if decode else None,
        "decode_success_conditional_p95_ns": percentile95(decode) if decode else None,
        "candidate_mean": statistics.mean(attempts),
        "candidate_max": max(attempts),
        "cold_prepare_amortized_ns_per_input": cold / batch,
        "cold_plus_encode_amortized_ns_per_input": (cold + encode_total) / batch,
        # The caller adds exact K-based rates below.
    }


def _measure(document, index, directory):
    config = document["configs"][index]
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    before = source_hashes()
    rows_path = directory / "rows.csv"
    fields = [
        "config_index",
        "phase",
        "repeat",
        "batch_size",
        "sample_id",
        "seed",
        "K",
        "encode_status",
        "encode_ns",
        "encode_wall_ns",
        "attempts",
        "failure_reason",
        "decode_status",
        "decode_ns",
        "decode_wall_ns",
        "decode_failure_reason",
        "roundtrip_ok",
        "n_total",
        "prefix_nt",
        "nonce_nt",
        "validation_status",
        "validation_reason",
        "input_bits_sha256",
        "decoded_bits_sha256",
        "sequence",
    ]
    preparations, groups, reasons = [], [], Counter()
    total_success = total_failed = 0
    with rows_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for repeat in range(document["repeats"]):
            clear_benchmark_caches()
            start = perf_counter_ns()
            codec = prepare_for_benchmark(config)
            cold = perf_counter_ns() - start
            start = perf_counter_ns()
            warmed = prepare_for_benchmark(config)
            warm_prepare = perf_counter_ns() - start
            if (warmed.M, warmed.K) != (codec.M, codec.K):
                raise IntegrityFailure("cold/warm exact count mismatch")
            # Exactly one deterministic warm-up, preserved but not a measured sample.
            warmup = codec.encode_bits("0" * codec.K)
            warmup_inverse = codec.decode_bits(warmup.sequence) if warmup.status == "OK" else None
            if warmup.status == "OK" and (
                validate_oligo(codec, warmup.sequence)
                or warmup_inverse.status != "OK"
                or warmup_inverse.bits != "0" * codec.K
            ):
                raise IntegrityFailure("noiseless deterministic warm-up failed")
            preparations.append(
                {
                    "repeat": repeat,
                    "cold_prepare_ns": cold,
                    "warm_prepare_ns": warm_prepare,
                    "cache": cache_info(),
                    "m3_benchmark_cache": str(_m3_prepared.cache_info()),
                    "retained_count_table_bytes": count_table_bytes(codec),
                    "warmup_encode_status": warmup.status,
                    "warmup_attempts": warmup.attempts,
                    "warmup_failure_reason": warmup.failure_reason,
                    "warmup_decode_status": warmup_inverse.status
                    if warmup_inverse
                    else "NOT_ATTEMPTED",
                }
            )
            for batch in document["batches"]:
                seed = sample_seed(document, identity, repeat, batch)
                generation_start = perf_counter_ns()
                rng = random.Random(seed)
                messages = [
                    format(rng.getrandbits(codec.K), f"0{codec.K}b") if codec.K else ""
                    for _ in range(batch)
                ]
                rng_generation_ns = perf_counter_ns() - generation_start
                encode_times, decode_times, attempts = [], [], []
                failures = successes = 0
                loop_start = perf_counter_ns()
                for sample, bits in enumerate(messages):
                    start = perf_counter_ns()
                    encoded = codec.encode_bits(bits)
                    encode_wall = perf_counter_ns() - start
                    encode_times.append(encode_wall)
                    attempts.append(encoded.attempts)
                    record = {
                        "config_index": index,
                        "phase": document["phase"],
                        "repeat": repeat,
                        "batch_size": batch,
                        "sample_id": sample,
                        "seed": seed,
                        "K": codec.K,
                        "encode_status": encoded.status,
                        "encode_ns": encoded.elapsed_ns,
                        "encode_wall_ns": encode_wall,
                        "attempts": encoded.attempts,
                        "failure_reason": encoded.failure_reason,
                        "decode_status": "NOT_ATTEMPTED",
                        "decode_ns": 0,
                        "decode_wall_ns": 0,
                        "decode_failure_reason": "",
                        "roundtrip_ok": False,
                        "n_total": codec.n,
                        "prefix_nt": len(codec.prefix),
                        "nonce_nt": config.get("nonce_nt", 0),
                        "validation_status": "NOT_ATTEMPTED",
                        "validation_reason": "",
                        "input_bits_sha256": hashlib.sha256(bits.encode()).hexdigest(),
                        "decoded_bits_sha256": "",
                        "sequence": encoded.sequence or "",
                    }
                    integrity_error = None
                    if encoded.status != "OK":
                        failures += 1
                        reasons[encoded.failure_reason] += 1
                        if encoded.status != "ENCODING_FAILED":
                            integrity_error = "exact-K benchmark input unexpectedly rejected"
                    else:
                        reason = validate_oligo(codec, encoded.sequence)
                        record.update(
                            validation_status="FAIL" if reason else "PASS",
                            validation_reason=reason or "",
                        )
                        start = perf_counter_ns()
                        decoded = codec.decode_bits(encoded.sequence)
                        decode_wall = perf_counter_ns() - start
                        decode_times.append(decode_wall)
                        correct = decoded.status == "OK" and decoded.bits == bits
                        record.update(
                            decode_status=decoded.status,
                            decode_ns=decoded.elapsed_ns,
                            decode_wall_ns=decode_wall,
                            decode_failure_reason=decoded.failure_reason,
                            roundtrip_ok=correct,
                            decoded_bits_sha256=hashlib.sha256(decoded.bits.encode()).hexdigest()
                            if decoded.bits is not None
                            else "",
                        )
                        if reason or not correct:
                            integrity_error = "invalid whole oligo or incorrect noiseless inverse"
                        successes += int(not reason and correct)
                    writer.writerow(record)
                    if sample % 64 == 0 or integrity_error:
                        handle.flush()
                    if integrity_error:
                        raise IntegrityFailure(integrity_error + "; raw row retained")
                handle.flush()
                diagnostic_loop_ns = perf_counter_ns() - loop_start
                total_success += successes
                total_failed += failures
                group = _group_summary(
                    repeat, batch, cold, encode_times, decode_times, attempts, successes, failures
                )
                group.update(
                    rng_generation_ns=rng_generation_ns,
                    diagnostic_loop_wall_ns_including_validation_hashing_csv=diagnostic_loop_ns,
                    input_mapper_bits_per_second=batch * codec.K * 1e9 / sum(encode_times),
                    successful_mapper_bits_per_second_including_failures=successes
                    * codec.K
                    * 1e9
                    / sum(encode_times),
                    successful_strands_per_second_including_failures=successes
                    * 1e9
                    / sum(encode_times),
                )
                groups.append(group)
    after = source_hashes()
    summary = {
        "status": "IMPLEMENTED_UNVERIFIED",
        "execution_status": "COMPLETE",
        "phase": document["phase"],
        "config_index": index,
        "config_sha256": identity,
        "config": config,
        "M": codec.M,
        "K": codec.K,
        "M_digits": len(str(codec.M)) if codec.M is not None else None,
        "mapper_rate_K_per_total_nt": codec.K / codec.n if codec.n else 0,
        "mapper_bits_are_not_user_net_bits": True,
        "metadata": dict(codec.metadata),
        "prepare_rows": preparations,
        "groups": groups,
        "sample_count": total_failed + total_success,
        "encode_successes": total_success,
        "encode_failures": total_failed,
        "failure_reason_counts": dict(reasons),
        "roundtrip_failures": 0,
        "invalid_whole_oligo_outputs": 0,
        "process_peak_rss_bytes": peak_rss_bytes(),
        "cache_final": cache_info(),
        "rows_sha256": digest(rows_path),
        "n_includes_prefix_and_nonce": True,
        "timing_scope": "exact-K bits interface only; RNG, CSV, hashing and extra validation excluded",
        "validation_scope": "complete oligo, including prefix and nonce; only claimed constraints",
        "table_memory_scope": "unique tuple/integer object bytes of retained count tables; not RSS",
        "P95_definition": "nearest rank ceil(0.95*N); decode conditional on encode success",
        "count_only_million_length_not_benchmarked": True,
        "codec_source_sha256_before": before,
        "codec_source_sha256_after": after,
        "codec_source_unchanged": before == after,
    }
    if before != after:
        raise IntegrityFailure("source changed during a timed configuration")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "index": index,
                "kind": codec.kind,
                "K": codec.K,
                "samples": summary["sample_count"],
                "failures": total_failed,
                "peak_rss_bytes": summary["process_peak_rss_bytes"],
            }
        ),
        flush=True,
    )
    return 0


def worker(document, index, out):
    config = document["configs"][index]
    directory = out / f"config_{index:02d}_{config['kind']}"
    directory.mkdir(exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    try:
        return _measure(document, index, directory)
    except Exception as error:
        partial_rows = []
        if (directory / "rows.csv").exists():
            with (directory / "rows.csv").open(newline="") as stream:
                partial_rows = list(csv.DictReader(stream))
        failure = {
            "execution_status": "INTEGRITY_FAILURE"
            if isinstance(error, IntegrityFailure)
            else "FAILED",
            "phase": document["phase"],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "process_peak_rss_bytes": peak_rss_bytes(),
            "raw_partial_rows_retained": (directory / "rows.csv").exists(),
            "partial_rows": len(partial_rows),
            "noiseless_inverse_failure_rows": sum(
                row["encode_status"] == "OK" and row["roundtrip_ok"] != "True"
                for row in partial_rows
            ),
            "invalid_whole_oligo_rows": sum(
                row["validation_status"] == "FAIL" for row in partial_rows
            ),
            "source_sha256_after_failure": source_hashes(),
        }
        (directory / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 2 if isinstance(error, IntegrityFailure) else 1


def _as_text(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def run(document, config_path, out, timeout):
    validate_document(document)
    if not 1 <= timeout <= 300:
        raise ValueError("per-configuration timeout must be within 1..300 seconds")
    out.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "UTC": dt.datetime.now(dt.timezone.utc).isoformat(),
        "timed_config_parallelism": 1,
        "other_system_workload_not_controlled": True,
        "background_workload_note": "other lightweight research work may run; machine is not exclusive",
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("pytest", "numpy", "scipy")
        },
        "codec_source_sha256_before": before,
    }
    if sys.platform == "darwin":
        for label in ("machdep.cpu.brand_string", "hw.memsize", "hw.physicalcpu"):
            result = subprocess.run(
                ["sysctl", "-n", label], capture_output=True, text=True, check=False
            )
            environment[label] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    frozen_config = out / "config.json"
    frozen_config.write_text(json.dumps(document, indent=2) + "\n")
    commands, halted = [], False
    for index, config in enumerate(document["configs"]):
        if halted:
            commands.append(
                {
                    "config_index": index,
                    "exit_code": None,
                    "execution_status": "NOT_RUN_AFTER_INTEGRITY_FAILURE",
                }
            )
            continue
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(frozen_config.resolve()),
            "--out",
            str(out.resolve()),
            "--worker",
            str(index),
        ]
        start = perf_counter_ns()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
            code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as error:
            code, stdout, stderr = 124, _as_text(error.stdout), _as_text(error.stderr)
        transcript = {
            "config_index": index,
            "command": command,
            "exit_code": code,
            "execution_status": "COMPLETE" if code == 0 else "TIMEOUT" if code == 124 else "FAILED",
            "elapsed_ns": perf_counter_ns() - start,
            "timeout_seconds": timeout,
            "stdout": stdout,
            "stderr": stderr,
        }
        (out / f"command_{index:02d}.json").write_text(json.dumps(transcript, indent=2) + "\n")
        commands.append(transcript)
        halted = code == 2 or source_hashes() != before
        print(
            f"config {index:02d} {config['kind']}: exit={code} "
            f"elapsed_s={transcript['elapsed_ns'] / 1e9:.3f} {stdout.strip()}",
            flush=True,
        )
    summaries = [json.loads(path.read_text()) for path in sorted(out.glob("config_*/summary.json"))]
    failed_cells = [
        json.loads(path.read_text()) for path in sorted(out.glob("config_*/failure.json"))
    ]
    after = source_hashes()
    complete = len(summaries) == len(document["configs"]) and all(
        row["exit_code"] == 0 for row in commands
    )
    final = {
        "status": "IMPLEMENTED_UNVERIFIED",
        "phase": document["phase"],
        "execution_status": "COMPLETE" if complete and before == after else "INCOMPLETE",
        "config_sha256": digest(config_path),
        "frozen_config_sha256": digest(frozen_config),
        "source_sha256": digest(Path(__file__)),
        "commands": commands,
        "configs": summaries,
        "configured_cells": len(document["configs"]),
        "completed_cells": len(summaries),
        "total_samples": sum(row["sample_count"] for row in summaries),
        "total_encode_failures": sum(row["encode_failures"] for row in summaries),
        "total_roundtrip_failures": sum(row["roundtrip_failures"] for row in summaries)
        + sum(row["noiseless_inverse_failure_rows"] for row in failed_cells),
        "failed_cell_records": failed_cells,
        "integrity_failure_halted_remaining_configs": halted,
        "formal_grid_including_10000_batch_pending_T34": not (
            document["phase"] == FORMAL_PHASE and complete
        ),
        "codec_source_sha256_before": before,
        "codec_source_sha256_after": after,
        "codec_source_unchanged": before == after,
        "artifact_sha256": {
            str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    }
    (out / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
    return 0 if complete and before == after else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--config-timeout", type=int, default=300)
    args = parser.parse_args()
    try:
        document = json.loads(args.config.read_text())
        validate_document(document)
        if args.worker is not None:
            return worker(document, args.worker, args.out)
        return run(document, args.config, args.out, args.config_timeout)
    except (OSError, ValueError) as error:
        parser.exit(1, f"benchmark refused: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
