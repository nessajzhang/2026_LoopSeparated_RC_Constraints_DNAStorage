"""Auditable file-level IDS experiments with a spawned, non-oracle receiver.

Configurations are frozen before any file is encoded. Completed failed files
are immutable results, never retried as successes. Only hash-verified completed
files can be resumed; an interrupted file requires an explicit audit rather
than silently discarding a possibly observed outcome. This module does not
label the diagnostic IDS model as an empirical platform.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
import datetime as dt
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import resource
import sys
from time import perf_counter_ns
from typing import Iterator

from .channel import IDSConfig, derived_seed, simulate_pool
from .evaluation_revision import private_truth_record, score_file
from .framing import MODE_TAGS, PublicProtocol, encode_file
from .receiver import FrameReconstruction, PoolReconstruction
from .receiver_process import ReceiverActor, ReceiverTerminationUnconfirmedError
from .statistics_revision import paired_failure_difference, summarize_independent_files

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "loop-separated-ids-benchmark-v1"
PHASES = ("IMPLEMENTATION_PILOT", "FORMAL_FROZEN")
_STOP_EVENT = None


def _initialize_case_worker(stop_event) -> None:
    global _STOP_EVENT
    _STOP_EVENT = stop_event


def _halt_if_stopped() -> None:
    if _STOP_EVENT is not None and _STOP_EVENT.is_set():
        raise RuntimeError("experiment halted before the next file after a fatal case error")


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: object) -> None:
    """New file only; the manifest is always written last."""
    with path.open("xb") as handle:
        handle.write(canonical(_jsonable(value)))
        handle.flush()
        os.fsync(handle.fileno())


def _write_gzip_json(path: Path, value: object) -> None:
    with path.open("xb") as raw:
        # No wall-clock timestamp or original filename in compressed evidence.
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(canonical(_jsonable(value)))
        raw.flush()
        os.fsync(raw.fileno())


def _load_json(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"JSON object required: {path}")
    return result


def source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "loop_alpha_audit").glob("*.py"))
    paths += [ROOT / "code/run_ids_benchmark.py", ROOT / "uv.lock", ROOT / "pyproject.toml"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def validate_config(config: dict) -> list[dict]:
    required = {
        "schema",
        "phase",
        "seeds",
        "files_per_seed",
        "protocol",
        "methods",
        "grid",
        "receiver_timeout_seconds",
        "primary_comparison_family_size",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("configuration fields must match the declared frozen schema exactly")
    if config["schema"] != SCHEMA or config["phase"] not in PHASES:
        raise ValueError("unknown configuration schema or phase")
    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or type(config["files_per_seed"]) is not int
        or config["files_per_seed"] < 1
    ):
        raise ValueError(
            "unique nonnegative master seeds and positive integer sample count required"
        )
    if not 0 < config["receiver_timeout_seconds"] <= 600:
        raise ValueError("receiver timeout must be in (0,600] seconds")
    methods = config["methods"]
    if (
        not isinstance(methods, list)
        or not methods
        or len(set(methods)) != len(methods)
        or any(method not in MODE_TAGS for method in methods)
    ):
        raise ValueError("methods must be a nonempty unique list of supported public mappers")
    family = config["primary_comparison_family_size"]
    if type(family) is not int or family < 1:
        raise ValueError("positive predeclared primary comparison family size required")
    if not isinstance(config["protocol"], dict) or "mapper" in config["protocol"]:
        raise ValueError("mapper is supplied by the method list, not the shared protocol")
    reference = PublicProtocol(**config["protocol"], mapper="short_core")
    # All schemes carry the same number of useful bytes, not their own maximum.
    for method in methods:
        PublicProtocol(
            **(config["protocol"] | {"payload_bytes": reference.payload_bytes}), mapper=method
        )
    grid = config["grid"]
    if not isinstance(grid, list) or not grid:
        raise ValueError("nonempty explicit grid required")
    cells = []
    seen = set()
    for item in grid:
        if not isinstance(item, dict) or set(item) != {
            "channel",
            "error_scale",
            "coverage",
            "coverage_mode",
            "dropout",
            "primary",
        }:
            raise ValueError(
                "every grid row must declare all channel, sampling and analysis fields"
            )
        IDSConfig.diagnostic(item["error_scale"], item["channel"])
        if (
            item["coverage_mode"] not in ("fixed", "poisson")
            or not 0 <= item["coverage"] <= 100
            or not 0 <= item["dropout"] <= 1
            or (item["coverage_mode"] == "fixed" and int(item["coverage"]) != item["coverage"])
            or type(item["primary"]) is not bool
        ):
            raise ValueError("invalid declared sampling rule or primary-analysis flag")
        identity = hashlib.sha256(canonical(item)).hexdigest()[:16]
        if identity in seen:
            raise ValueError("duplicate grid cell")
        seen.add(identity)
        cells.append({"cell_id": identity, "grid": item})
    primary_count = sum(cell["grid"]["primary"] for cell in cells)
    if primary_count > family:
        raise ValueError("declared primary family cannot be smaller than this frozen grid")
    if config["phase"] == "IMPLEMENTATION_PILOT":
        planned = len(seeds) * config["files_per_seed"] * len(cells) * len(methods)
        if planned > 24:
            raise ValueError("implementation pilot is limited to at most 24 independent files")
    return cells


def expected_keys(config: dict) -> list[tuple[int, int]]:
    """Derived from the frozen plan, never from completed observations."""
    return [
        (seed, file_id) for seed in config["seeds"] for file_id in range(config["files_per_seed"])
    ]


def _failed_receiver(protocol: PublicProtocol, n_reads: int, elapsed: int, reason: str):
    frames = tuple(
        FrameReconstruction(i, "RECEIVER_EXCEPTION", 0, 0, None, None, None, 0, 0, False, 0, reason)
        for i in range(protocol.rs_n)
    )
    return PoolReconstruction(
        "RECEIVER_EXCEPTION",
        None,
        frames,
        n_reads,
        0,
        n_reads,
        {"NOT_OBSERVED_DUE_TO_RECEIVER_FAILURE": n_reads},
        tuple(range(protocol.rs_n)),
        (),
        elapsed,
        reason,
    )


def _checked_complete_file(directory: Path, identity: dict) -> dict:
    manifest = _load_json(directory / "complete.json")
    if manifest["identity"] != identity:
        raise ValueError("completed file identity does not match frozen experiment plan")
    for relative, expected_hash in manifest["files"].items():
        path = directory / relative
        if Path(relative).name != relative or not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f"completed evidence hash mismatch: {path}")
    return _load_json(directory / "metrics.json")


def _check_noiseless_control(row: dict, grid: dict) -> None:
    if (
        grid["error_scale"] == 0
        and grid["dropout"] == 0
        and grid["coverage_mode"] == "fixed"
        and grid["coverage"] >= 1
        and row["all_strands_encoded"]
        and not row["file_correct"]
    ):
        raise RuntimeError(
            "noiseless all-encoded correctness control failed; retained row, halt for audit"
        )


def _case_worker(config: dict, cell: dict, method: str, out: str, hashes: dict) -> dict:
    _halt_if_stopped()
    directory = Path(out) / f"{cell['cell_id']}_{method}"
    directory.mkdir(exist_ok=True)
    public = PublicProtocol(**config["protocol"], mapper=method)
    grid = cell["grid"]
    ids = IDSConfig.diagnostic(grid["error_scale"], grid["channel"])
    keys = expected_keys(config)
    receiver = None
    rows = []
    start_case = perf_counter_ns()
    try:
        for seed, file_id in keys:
            _halt_if_stopped()
            identity = {
                "cell_id": cell["cell_id"],
                "mapper": method,
                "seed": seed,
                "file_id": file_id,
            }
            destination = directory / f"file_{seed}_{file_id:05d}"
            if (destination / "complete.json").is_file():
                row = _checked_complete_file(destination, identity)
                _check_noiseless_control(row, grid)
                rows.append(row)
                continue  # Failed completed files are deliberately NOT retried.
            if destination.exists():
                raise RuntimeError(
                    f"incomplete file evidence requires audit, refusing silent retry: {destination}"
                )
            destination.mkdir()
            shared = derived_seed(seed, "cell-file", cell["cell_id"], file_id)
            user_seed = derived_seed(shared, "common-user")
            channel_seed = derived_seed(shared, "common-channel")
            encoder_seed = derived_seed(shared, "scheme-encoder", method)
            user = random.Random(user_seed).randbytes(public.maximum_user_bytes)
            encoded = encode_file(user, public, seed=encoder_seed)
            sent = {
                frame.strand_index: frame.sequence
                for frame in encoded
                if frame.status == "OK" and frame.sequence is not None
            }
            channel = simulate_pool(
                sent,
                ids,
                coverage=grid["coverage"],
                seed=channel_seed,
                coverage_mode=grid["coverage_mode"],
                dropout=grid["dropout"],
            )
            _write_gzip_json(
                destination / "receiver_input.json.gz",
                {
                    "public_protocol": asdict(public),
                    "observations": [asdict(read) for read in channel.observations],
                    "interface": "public protocol once; opaque IDs/observed bases and transport ID only",
                },
            )
            _write_gzip_json(
                destination / "private_truth.json.gz",
                {
                    **private_truth_record(user, encoded, channel, seed=seed, file_id=file_id),
                    "seed_derivation": {
                        "user": user_seed,
                        "encoder": encoder_seed,
                        "channel": channel_seed,
                    },
                },
            )
            failure = None
            receive_start = perf_counter_ns()
            metadata = None
            try:
                if receiver is None:
                    receiver = ReceiverActor(
                        public, timeout_seconds=config["receiver_timeout_seconds"]
                    )
                metadata = dict(receiver.server_metadata)
                received = receiver.receive(channel.observations)
            except ReceiverTerminationUnconfirmedError:
                # In particular this catches constructor cleanup failures,
                # before assignment to `receiver` has taken place.
                raise
            except (TimeoutError, EOFError, OSError, RuntimeError) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                if receiver is not None:
                    if receiver._process.is_alive():
                        # A transport failure must have invalidated the actor.
                        # Never spawn another receiver while termination is unconfirmed.
                        raise RuntimeError(
                            "receiver failure left a live child; experiment halted"
                        ) from exc
                    receiver.close()
                receiver = None
                received = _failed_receiver(
                    public, len(channel.observations), perf_counter_ns() - receive_start, failure
                )
            _write_gzip_json(
                destination / "receiver_output.json.gz",
                {
                    "result": asdict(received),
                    "actor_metadata": metadata,
                    "receiver_failure": failure,
                    "source_sha256": hashes,
                },
            )
            # No true payload, origin or edit script enters the actor above.
            row, frames = score_file(
                user,
                encoded,
                channel,
                received,
                public,
                seed=seed,
                file_id=file_id,
                receiver_failure=failure,
            )
            row.update(
                cell_id=cell["cell_id"], receiver_boundary_wall_ns=perf_counter_ns() - receive_start
            )
            if row["frames_attempted"] != public.rs_n:
                raise AssertionError("every attempted file must retain all planned RS frames")
            _write_gzip_json(destination / "frames.json.gz", {"frames": frames})
            _write_json(destination / "metrics.json", row)
            file_hashes = {
                path.name: digest(path) for path in sorted(destination.iterdir()) if path.is_file()
            }
            _write_json(destination / "complete.json", {"identity": identity, "files": file_hashes})
            rows.append(row)
            _check_noiseless_control(row, grid)
    except BaseException:
        if _STOP_EVENT is not None:
            _STOP_EVENT.set()
        raise
    finally:
        if receiver is not None:
            try:
                receiver.close()
            except BaseException:
                if _STOP_EVENT is not None:
                    _STOP_EVENT.set()
                raise
    if any(row["frames_attempted"] != public.rs_n for row in rows):
        raise AssertionError("resumed file frame denominator changed")
    summary = summarize_independent_files(rows, expected_keys=keys)
    planned_nt = sum(row["planned_synthesized_nt"] for row in rows)
    summary.update(
        cell_id=cell["cell_id"],
        method=method,
        grid=grid,
        protocol=asdict(public),
        elapsed_wall_ns=perf_counter_ns() - start_case,
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        planned_synthesized_nt=planned_nt,
        R_net_per_planned_nt=summary["user_bits_attempted"] / planned_nt,
        goodput_per_planned_nt=summary["user_bits_correctly_recovered"] / planned_nt,
        primary_mapper_failures=sum(row["primary_mapper_failures"] for row in rows),
        fallback_attempts=sum(row["fallback_attempts"] for row in rows),
        fallback_used=sum(row["fallback_used"] for row in rows),
        terminal_mapper_failures=sum(row["terminal_mapper_failures"] for row in rows),
        receiver_failure_files=sum(
            row["receiver_exception_or_timeout"] is not None for row in rows
        ),
    )
    # Rebuilding a case summary during resume is deterministic except timing;
    # preserve each summary attempt, never overwrite an earlier one.
    serial = len(list(directory.glob("summary_*.json"))) + 1
    _write_json(directory / f"summary_{serial:03d}.json", summary)
    return {
        "summary": summary,
        "file_outcomes": [
            {"seed": r["seed"], "file_id": r["file_id"], "file_failed": not r["file_correct"]}
            for r in rows
        ],
    }


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another runner owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_benchmark(
    config: dict,
    out: Path,
    *,
    jobs: int = 1,
    resume: bool = False,
    reviewed_freeze: str | None = None,
) -> dict:
    cells = validate_config(config)
    if type(jobs) is not int or not 1 <= jobs <= 4:
        raise ValueError("one to four independent case workers required")
    config_hash = hashlib.sha256(canonical(config)).hexdigest()
    if config["phase"] == "FORMAL_FROZEN" and reviewed_freeze != config_hash:
        raise ValueError(
            "formal run requires explicit acknowledgment of the reviewed configuration SHA256"
        )
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=resume)
    hashes = source_hashes()
    with _run_lock(out / ".run.lock"):
        if resume:
            frozen = _load_json(out / "freeze.json")
            if frozen != {"config_sha256": config_hash, "source_sha256": hashes}:
                raise ValueError("config/source drift forbids resuming an immutable experiment")
        else:
            _write_json(out / "config.json", config)
            _write_json(
                out / "freeze.json", {"config_sha256": config_hash, "source_sha256": hashes}
            )
            _write_json(
                out / "planned_keys.json",
                {
                    "keys_per_case": expected_keys(config),
                    "cells": cells,
                    "methods": config["methods"],
                },
            )
            _write_json(
                out / "environment.json",
                {
                    "python": sys.version,
                    "executable": sys.executable,
                    "platform": platform.platform(),
                    "UTC": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "pid": os.getpid(),
                    "logical_cpu_count": os.cpu_count(),
                    "case_workers": jobs,
                    "receiver_start_method": "spawn",
                    "dependencies": {
                        name: importlib.metadata.version(name)
                        for name in ("numpy", "scipy", "reedsolo", "rapidfuzz")
                    },
                    "argv": sys.argv,
                    "hardware_workload_not_exclusively_controlled": True,
                },
            )
        todo = [(cell, method) for cell in cells for method in config["methods"]]
        results, errors = [], []
        # Spawn case workers too; each receiver gets observations through its
        # own spawn boundary rather than inheriting evaluator memory via fork.
        context = mp.get_context("spawn")
        stop_event = context.Event()
        with ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=context,
            initializer=_initialize_case_worker,
            initargs=(stop_event,),
        ) as executor:
            futures = {
                executor.submit(_case_worker, config, cell, method, str(out), hashes): (
                    cell,
                    method,
                )
                for cell, method in todo
            }
            for future in as_completed(futures):
                cell, method = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    point = result["summary"]
                    print(
                        json.dumps(
                            {
                                "cell": cell["cell_id"],
                                "method": method,
                                "files": point["independent_files"],
                                "file_failures": point["file_failures"],
                                "elapsed_s": point["elapsed_wall_ns"] / 1e9,
                            }
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    stop_event.set()
                    for pending in futures:
                        pending.cancel()
                    errors.append(
                        {
                            "cell_id": cell["cell_id"],
                            "method": method,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(json.dumps(errors[-1]), flush=True)
        if hashes != source_hashes():
            errors.append({"error": "source changed during experiment; acceptance forbidden"})
        by_case = {
            (result["summary"]["cell_id"], result["summary"]["method"]): result
            for result in results
        }
        comparisons = []
        if {"nonce", "gc_h_nonce"}.issubset(config["methods"]):
            for cell in cells:
                left = by_case.get((cell["cell_id"], "nonce"))
                right = by_case.get((cell["cell_id"], "gc_h_nonce"))
                if left is None or right is None:
                    continue
                keys = expected_keys(config)
                vectors = []
                for result in (left, right):
                    mapped = {
                        (row["seed"], row["file_id"]): row["file_failed"]
                        for row in result["file_outcomes"]
                    }
                    if set(mapped) != set(keys):
                        raise AssertionError("paired outcome keys do not match the frozen plan")
                    vectors.append([mapped[key] for key in keys])
                alpha = (
                    0.05 / config["primary_comparison_family_size"]
                    if cell["grid"]["primary"]
                    else 0.05
                )
                comparisons.append(
                    {
                        "cell_id": cell["cell_id"],
                        "primary": cell["grid"]["primary"],
                        "difference_direction": "nonce-minus-gc_h_nonce file failure",
                        **paired_failure_difference(*vectors, alpha=alpha),
                    }
                )
        final = {
            "status": "COMPLETE_UNVERIFIED"
            if not errors and len(results) == len(todo)
            else "INCOMPLETE_NOT_ACCEPTED",
            "phase": config["phase"],
            "config_sha256": config_hash,
            "source_sha256": hashes,
            "planned_cases": len(todo),
            "completed_cases": len(results),
            "planned_files": len(todo) * len(expected_keys(config)),
            "observed_files_in_complete_cases": sum(
                result["summary"]["independent_files"] for result in results
            ),
            "case_summaries": sorted(
                (result["summary"] for result in results),
                key=lambda row: (row["cell_id"], row["method"]),
            ),
            "paired_comparisons": comparisons,
            "errors": errors,
            "not_empirical_channel_profile": True,
            "pilot_excluded_from_formal_confidence_intervals": config["phase"]
            == "IMPLEMENTATION_PILOT",
        }
        serial = len(list(out.glob("run_summary_*.json"))) + 1
        _write_json(out / f"run_summary_{serial:03d}.json", final)
        return final
