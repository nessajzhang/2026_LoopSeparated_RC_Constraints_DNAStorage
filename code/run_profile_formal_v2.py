#!/usr/bin/env python3
"""Versioned 512-copy source-profile experiment with explicit execution gates.

The original eight-copy smoke controller and data are unchanged. This separate
controller accepts two fixed interface cells or one of 96 frozen user files,
each with eight formal method/budget cells. A phase-specific root approval is
mandatory before observations; completion is never scientific acceptance.
The source simulator runs in its separate locked environment. Receiving is a
fresh public-only spawn interface, not an OS filesystem security sandbox.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import datetime as dt
import hashlib
import gzip
import json
import os
from pathlib import Path
import platform
import random
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.channel import ReadObservation  # noqa: E402
from loop_alpha_audit.framing import (  # noqa: E402
    PublicProtocol,
    decode_frame,
    encode_file,
    rate_accounting,
)
from loop_alpha_audit.receiver_process import (  # noqa: E402
    ReceiverActor,
    ReceiverTerminationUnconfirmedError,
)

OPAQUE = re.compile(r"r[0-9a-f]{32}\Z")
DNA = set("ACGT")
COMP = str.maketrans("ACGT", "TGCA")
THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
LINEAGE_REASON = "NON_EVALUABLE_NO_LINEAGE_CERTIFICATE"
IMPLEMENTATION_VERSION = "p3_formal_512copy_v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def jsonable(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")


def seed_for(master, *labels):
    message = json.dumps(["profile-p3-v1", master, *labels], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big")


def validate_config(config):
    require(
        config["schema_version"] == 2 and config["phase"] == "P3_FORMAL_V2_FROZEN",
        "versioned formal configuration required; execution still needs phase approval",
    )
    require(
        config["methods"] == ["two_bit", "gc_h_nonce", "short_core", "nonce"],
        "predeclared four methods",
    )
    protocol = config["protocol"]
    require(
        (
            protocol["n"],
            protocol["m"],
            protocol["k"],
            protocol["rs_n"],
            protocol["rs_k"],
            protocol["payload_bytes"],
            protocol["primer_nt"],
        )
        == (150, 7, 3, 63, 55, 22, 0),
        "frozen body protocol",
    )
    smoke = config["interface_smoke"]
    require(
        smoke
        == {
            "seed": 20261023903,
            "file_id": 0,
            "cells": [
                {"method": "two_bit", "mode": "noisy", "read_pairs": 192},
                {"method": "nonce", "mode": "noisy", "read_pairs": 315},
            ],
            "excluded_from_formal_intervals": True,
        },
        "two prospective interface cells, not formal observations",
    )
    future = config["formal"]
    require(
        future["seeds"] == [20261023801, 20261023802, 20261023803]
        and future["files_per_seed"] == 32
        and future["read_pair_budgets"] == [192, 315],
        "previously frozen seed/file/budget declaration",
    )
    require(
        future
        == {
            "seeds": [20261023801, 20261023802, 20261023803],
            "files_per_seed": 32,
            "read_pair_budgets": [192, 315],
            "mode": "noisy",
            "independent_source_calls_not_nested": True,
            "planned_file_method_budget_cells": 768,
            "independent_user_files_per_method_budget": 96,
            "paired_difference": "nonce-minus-gc_h_nonce file failure",
            "paired_comparison_family_size": 2,
            "paired_family_alpha": 0.05,
        },
        "formal design and descriptive/paired statistical family",
    )
    limits = config["limits"]
    require(
        limits
        == {
            "max_initial_molecules_per_cell": 32256,
            "max_read_pairs_per_cell": 315,
            "profile_wall_seconds": 60,
            "profile_rss_watchdog_mib": 800,
            "receiver_timeout_seconds": 60,
            "external_profile_jobs": 1,
            "internal_multiprocessing": False,
            "internal_threads": 1,
        },
        "bounded resource plan",
    )
    physical = config["physical"]
    require(
        len(physical["forward_primer"]) == 58
        and len(physical["reverse_primer"]) == 64
        and physical["copies_per_sent_body"] == 512
        and physical["physical_template_nt"] == 272,
        "physical geometry",
    )
    require(
        physical["constraint_scope"] == "body_only"
        and physical["receiver_extra_trim_nt"] == 0
        and physical["primary_mate"] == "R1"
        and physical["R2_retained_and_costed_not_received"],
        "mate/trim/constraint scope",
    )
    return config


def read_public(path):
    records = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            require(
                isinstance(record, dict) and set(record) == {"read_id", "sequence"},
                "forbidden public observation field",
            )
            require(
                isinstance(record["read_id"], str) and OPAQUE.fullmatch(record["read_id"]),
                "opaque128 read ID required",
            )
            require(
                isinstance(record["sequence"], str)
                and len(record["sequence"]) == 150
                and not (set(record["sequence"]) - DNA),
                "observed body only",
            )
            records.append(record)
            require(len(records) <= 315, "public read cap")
    require(len({r["read_id"] for r in records}) == len(records), "duplicate public read ID")
    return tuple(ReadObservation(r["read_id"], r["sequence"]) for r in records)


def source_snapshot(source):
    return {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}


def tracked_hashes(args, config_path):
    files = list((ROOT / "loop_alpha_audit").glob("*.py"))
    files += [
        ROOT / "code/run_ids_benchmark.py",
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        config_path,
        args.approval,
        args.inputs / "manifest.json",
    ]
    files += sorted(args.worker.parent.glob("*.py"))
    files += sorted(args.inputs.glob("user_*.bin"))
    runtime = args.profile_python.parent.parent.parent
    files += [
        runtime / "pyproject.toml",
        runtime / "uv.lock",
        args.profile_python.parent.parent / "pyvenv.cfg",
        args.profile_python,
    ]
    return {str(p.resolve()): sha(p) for p in files}


def planned_cells(config, phase, file_index):
    if phase == "interface_smoke":
        require(file_index is None, "interface cells have no formal file index")
        item = config["interface_smoke"]
        return [dict(row, seed=item["seed"], file_id=item["file_id"]) for row in item["cells"]]
    require(
        phase == "formal" and type(file_index) is int and 0 <= file_index < 96,
        "formal execution selects one previously frozen file index 0..95",
    )
    formal = config["formal"]
    master = formal["seeds"][file_index // 32]
    file_id = file_index % 32
    return [
        {
            "method": method,
            "mode": "noisy",
            "read_pairs": budget,
            "seed": master,
            "file_id": file_id,
        }
        for method in config["methods"]
        for budget in formal["read_pair_budgets"]
    ]


def verify_frozen_users(directory, config):
    manifest = json.loads((directory / "manifest.json").read_text())
    require(
        manifest["frozen_before_any_formal_profile_observation"] is True,
        "pre-observation user-input manifest required",
    )
    records = manifest["records"]
    expected = {(s, f) for s in config["formal"]["seeds"] for f in range(32)}
    require(
        len(records) == 96 and {(r["seed"], r["file_id"]) for r in records} == expected,
        "exact 96-key frozen user-input plan",
    )
    for row in records:
        require(
            row["path"] == f"user_{row['seed']}_{row['file_id']:02d}.bin",
            "safe deterministic user filename",
        )
        path = directory / row["path"]
        require(
            path.stat().st_size == row["bytes"] == 1202 and sha(path) == row["sha256"],
            "frozen useful-user bytes changed",
        )
        require(
            path.read_bytes()
            == random.Random(seed_for(row["seed"], "user", row["file_id"])).randbytes(1202),
            "frozen uniform-user RNG reconstruction",
        )
    return manifest


def verify_approval(args, config_path):
    approval = json.loads(args.approval.read_text())
    require(
        approval["gate"] == "P3_V2_ROOT_EXECUTION_ONLY_NOT_RESULT_ACCEPTANCE"
        and args.phase in approval["approved_phases"],
        "this phase needs its own explicit root execution approval",
    )
    require(approval["config_sha256"] == sha(config_path), "approval/config drift")
    roles = {
        "driver": Path(__file__).resolve(),
        "worker": args.worker,
        "schema": args.worker.with_name("schema.py"),
    }
    require(
        {k: sha(p) for k, p in roles.items()} == approval["implementation_sha256"],
        "approval/driver or external-wrapper drift",
    )
    for relative, expected in approval["frozen_library_sha256"].items():
        require(sha(ROOT / relative) == expected, f"frozen communication library drift: {relative}")
    runtime = args.profile_python.parent.parent.parent
    require(
        sha(runtime / "uv.lock") == approval["profile_runtime_lock_sha256"],
        "profile runtime lock drift",
    )
    require(
        sha(args.inputs / "manifest.json") == approval["frozen_input_manifest_sha256"],
        "frozen input manifest differs from approved object",
    )
    return approval


def group_snapshot(group):
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid=,rss="], text=True)
    return [
        {"pid": int(pid), "ppid": int(ppid), "pgid": int(pgid), "rss_bytes": int(rss) * 1024}
        for line in output.splitlines()
        for pid, ppid, pgid, rss in [line.split()]
        if int(pgid) == group
    ]


def supervise(command, directory, env, *, seconds=60, rss_mib=800):
    """No retries; a sampled group-RSS watchdog, not a kernel memory cap."""
    started = time.monotonic()
    sampled_peak, samples, reason = 0, 0, None
    errors, process, pid, remaining, cleaned = [], None, None, [], False
    with (
        (directory / "profile_stdout.log").open("x") as stdout,
        (directory / "profile_stderr.log").open("x") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, env=env, start_new_session=True
            )
            pid = process.pid
            while process.poll() is None:
                snapshot = group_snapshot(pid)
                samples += 1
                sampled_peak = max(sampled_peak, sum(row["rss_bytes"] for row in snapshot))
                if time.monotonic() - started >= seconds:
                    reason = "PROFILE_WALL_TIMEOUT"
                elif sum(row["rss_bytes"] for row in snapshot) > rss_mib * 1024**2:
                    reason = "PROFILE_RSS_WATCHDOG"
                if reason:
                    break
                time.sleep(0.1)
        except BaseException as error:
            reason = f"WATCHDOG_INFRASTRUCTURE_ERROR: {type(error).__name__}: {error}"
            errors.append(reason)
        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait(timeout=3)
                except BaseException as error:
                    errors.append(f"WAIT_OR_KILL_FAILED: {type(error).__name__}: {error}")
    if pid is not None:
        try:
            remaining = group_snapshot(pid)
            if remaining:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                time.sleep(0.1)
            cleaned = not group_snapshot(pid)
        except BaseException as error:
            errors.append(f"GROUP_CONFIRMATION_FAILED: {type(error).__name__}: {error}")
    resource = {
        "command": command,
        "pid": pid,
        "process_group": pid,
        "exit_code": process.returncode if process is not None else None,
        "wall_seconds": time.monotonic() - started,
        "watchdog_reason": reason,
        "watchdog_wall_seconds": seconds,
        "watchdog_rss_mib": rss_mib,
        "sampled_group_peak_rss_bytes": sampled_peak,
        "rss_sample_count": samples,
        "sampled_not_kernel_hard_limit": True,
        "remaining_at_worker_exit": remaining,
        "process_group_confirmed_empty": cleaned,
        "automatic_retry": False,
        "cleanup_errors": errors,
    }
    write_json(directory / "resource.json", resource)
    if not cleaned or remaining or errors:
        raise ReceiverTerminationUnconfirmedError(
            "profile descendant cleanup was not clean; halt before receiver"
        )
    return resource


def constraint_record(sequence, m=7, k=3, boundaries=()):
    """Independent string pair checker; report violations, never filter input."""
    require(isinstance(sequence, str) and not (set(sequence) - DNA), "constraint DNA")
    windows, pairs = {}, []
    for start in range(len(sequence) - m + 1):
        window = sequence[start : start + m]
        partner = window.translate(COMP)[::-1]
        for prior in windows.get(partner, ()):
            if start - prior >= m + k:
                cuts = list(boundaries)

                def regions(left):
                    return sorted(
                        {sum(position >= cut for cut in cuts) for position in range(left, left + m)}
                    )

                a, b = regions(prior), regions(start)
                pairs.append(
                    {
                        "left_start": prior,
                        "right_start": start,
                        "stem_nt": m,
                        "loop_nt": start - prior - m,
                        "left_regions": a,
                        "right_regions": b,
                        "cross_region_or_boundary": len(set(a + b)) > 1,
                        "window_straddles_boundary": len(a) > 1 or len(b) > 1,
                    }
                )
        windows.setdefault(window, []).append(start)
    runs, begin = [], 0
    while begin < len(sequence):
        end = begin + 1
        while end < len(sequence) and sequence[end] == sequence[begin]:
            end += 1
        runs.append(
            {
                "start": begin,
                "end_exclusive": end,
                "base": sequence[begin],
                "length": end - begin,
                "cross_boundary": any(begin < cut < end for cut in boundaries),
            }
        )
        begin = end
    gc = sum(base in "GC" for base in sequence)
    longest = max((run["length"] for run in runs), default=0)
    return {
        "n": len(sequence),
        "gc_bases": gc,
        "gc_fraction": gc / len(sequence) if sequence else None,
        "gc_40_60_ok": 2 * len(sequence) <= 5 * gc <= 3 * len(sequence),
        "longest_homopolymer": longest,
        "homopolymer_max4_ok": longest <= 4,
        "rc_violation_count": len(pairs),
        "rc_violations": pairs,
        "cross_region_or_boundary_rc_count": sum(
            pair["cross_region_or_boundary"] for pair in pairs
        ),
        "straddling_window_rc_count": sum(pair["window_straddles_boundary"] for pair in pairs),
        "cross_boundary_homopolymers": [
            run for run in runs if run["cross_boundary"] and run["length"] > 4
        ],
    }


def bit_errors(left, right):
    common = sum((a ^ b).bit_count() for a, b in zip(left, right))
    return common + 8 * abs(len(left) - len(right))


def score(user, encoded, received):
    expected = {frame.strand_index: frame.payload for frame in encoded}
    by_index = {} if received is None else {frame.strand_index: frame for frame in received.frames}
    rows = []
    for index in range(63):
        frame = by_index.get(index)
        returned = frame is not None and frame.status == "OK" and frame.payload is not None
        correct = returned and frame.payload == expected[index]
        rows.append(
            {
                "inferred_public_index": index,
                "returned_crc_ok": returned,
                "payload_correct_posthoc": bool(correct),
                "erasure": not returned,
                "undetected_payload_error": returned and not correct,
                "payload_bit_errors_if_returned": bit_errors(frame.payload, expected[index])
                if returned
                else None,
                "receiver_failure_reason": frame.failure_reason
                if frame is not None
                else "PROFILE_OR_RECEIVER_FAILED",
                "read_origin": None,
                "assignment_accuracy": None,
                "non_evaluable_reason": LINEAGE_REASON,
            }
        )
    user_returned = received is not None and received.status == "OK" and received.user is not None
    user_correct = user_returned and received.user == user
    return {
        "file_correct": bool(user_correct),
        "file_failed": not user_correct,
        "user_returned": user_returned,
        "undetected_file_error": user_returned and not user_correct,
        "user_bit_errors_if_returned": bit_errors(received.user, user) if user_returned else None,
        "frames_attempted": 63,
        "returned_frames": sum(r["returned_crc_ok"] for r in rows),
        "correct_returned_frames": sum(r["payload_correct_posthoc"] for r in rows),
        "frame_erasures": sum(r["erasure"] for r in rows),
        "undetected_frame_payload_errors": sum(r["undetected_payload_error"] for r in rows),
    }, rows


def costs(config, user, encoded, generated, correct, method, read_pairs):
    public = PublicProtocol(**config["protocol"], mapper=method)
    body = rate_accounting(public, len(user), encoded)
    sent = sum(frame.status == "OK" and frame.sequence is not None for frame in encoded)
    actual_physical, planned_physical = sent * 272, 63 * 272
    bits = len(user) * 8
    mate_counts = [generated.get("actual_R1"), generated.get("actual_R2")]
    sequenced_nt = 150 * sum(mate_counts) if all(type(c) is int for c in mate_counts) else None
    initial = generated.get("initial_molecules")
    copies = config["physical"]["copies_per_sent_body"]
    return {
        **body,
        "body_nt_per_sent_design": 150,
        "adapter_primer_nt_per_sent_design": 122,
        "physical_template_nt": 272,
        "constraint_scope": "body_only",
        "planned_body_design_nt": 63 * 150,
        "actual_body_design_nt": sent * 150,
        "planned_full_design_nt": planned_physical,
        "actual_full_design_nt": actual_physical,
        "body_net_rate_actual": bits / (sent * 150) if sent else None,
        "body_net_rate_planned": bits / (63 * 150),
        "full_template_net_rate_actual": bits / actual_physical if sent else None,
        "full_template_net_rate_planned": bits / planned_physical,
        "full_template_goodput_planned": bits * bool(correct) / planned_physical,
        "planned_initial_molecules": 63 * copies,
        "actual_initial_molecules": initial,
        "sent_design_copy_rule_molecules": sent * copies,
        "planned_molecule_weighted_nt": 63 * copies * 272,
        "actual_molecule_weighted_nt": initial * 272 if type(initial) is int else None,
        "sent_design_copy_rule_molecule_nt": sent * copies * 272,
        "actual_R1_reads": mate_counts[0],
        "actual_R2_reads": mate_counts[1],
        "actual_sequenced_nt_both_mates": sequenced_nt,
        "planned_sequenced_nt_both_mates": read_pairs * 2 * 150,
        "primary_receiver_mate": "R1",
        "R2_received": False,
        "read_origin": None,
        "true_edit_script": None,
        "assignment_accuracy": None,
        "non_evaluable_reason": LINEAGE_REASON,
    }


def seal(directory):
    mapping = {
        str(p.relative_to(directory)): sha(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p not in {directory / "manifest.sha256", directory / "complete.json"}
    }
    with (directory / "manifest.sha256").open("x") as stream:
        for name, digest in mapping.items():
            stream.write(f"{digest}  {name}\n")
    write_json(
        directory / "complete.json",
        {"files": mapping, "manifest_sha256": sha(directory / "manifest.sha256")},
    )


def failed_generation(channel, resource, *, requested_pairs, molecule_cap):
    """Retain persisted partial counts; unknown is not fabricated as zero."""
    observed, errors, prefixes = {}, {}, {}
    for mate in (1, 2):
        path = channel / "raw" / f"R{mate}.fq.gz"
        if not path.exists():
            observed[f"actual_R{mate}"] = 0
            prefixes[f"R{mate}"] = 0
            continue
        prefixes[f"R{mate}"] = 0
        try:
            with gzip.open(path, "rt") as stream:
                lines = stream.read().splitlines()
            for position in range(0, len(lines) - 3, 4):
                name, sequence, plus, quality = lines[position : position + 4]
                require(
                    name.startswith("@")
                    and plus == "+"
                    and len(sequence) == len(quality) == 150
                    and not set(sequence) - DNA,
                    "malformed persisted FASTQ record",
                )
                prefixes[f"R{mate}"] += 1
            require(len(lines) % 4 == 0, "partial FASTQ record")
            observed[f"actual_R{mate}"] = len(lines) // 4
        except (OSError, EOFError, ValueError) as error:
            observed[f"actual_R{mate}"] = None
            errors[f"R{mate}"] = f"{type(error).__name__}: {error}"
    pools = {}
    for name in ("initial", "selected"):
        path = channel / "evaluator_only" / f"{name}_pool.csv"
        pools[name] = None
        if path.exists():
            try:
                counts = {}
                # Published SeqPool.save_as_csv uses csv.writer, without a header.
                with path.open(newline="") as stream:
                    for row in csv.reader(stream):
                        require(len(row) == 2, "malformed headerless pool row")
                        sequence, value = row
                        count = int(value)
                        require(
                            count >= 0 and len(sequence) == 272 and not set(sequence) - DNA,
                            "invalid persisted pool count/DNA",
                        )
                        counts[sequence] = counts.get(sequence, 0) + count
                pools[name] = counts
            except (OSError, ValueError) as error:
                errors[f"{name}_pool"] = f"{type(error).__name__}: {error}"
    initial = sum(pools["initial"].values()) if pools["initial"] is not None else None
    selected = sum(pools["selected"].values()) if pools["selected"] is not None else None
    overdraw = None
    if pools["initial"] is not None and pools["selected"] is not None:
        overdraw = {
            hashlib.sha256(sequence.encode()).hexdigest(): count - pools["initial"].get(sequence, 0)
            for sequence, count in pools["selected"].items()
            if count > pools["initial"].get(sequence, 0)
        }
    return {
        "status": "FAILED",
        **observed,
        "initial_molecules": initial,
        "selected_molecules": selected,
        "selected_template_overdraw": overdraw,
        "initial_count_exceeds_cap": initial > molecule_cap if initial is not None else None,
        "selected_count_exceeds_requested": selected > requested_pairs
        if selected is not None
        else None,
        "requested_pairs": requested_pairs,
        "requested_minus_actual_by_mate": {
            f"R{mate}": requested_pairs - observed[f"actual_R{mate}"]
            if type(observed[f"actual_R{mate}"]) is int
            else None
            for mate in (1, 2)
        },
        "actual_read_count_exceeds_requested": {
            f"R{mate}": observed[f"actual_R{mate}"] > requested_pairs
            if type(observed[f"actual_R{mate}"]) is int
            else None
            for mate in (1, 2)
        },
        "actual_counts_scope": "persisted_raw_records_only_after_failed_generation",
        "partial_read_errors": errors,
        "persisted_complete_record_prefix_count": prefixes,
        "generator_exit_code": resource["exit_code"],
        "watchdog_reason": resource["watchdog_reason"],
        "no_automatic_retry": True,
        "worker_failure_evidence_present": (channel / "worker_failure.json").exists(),
    }


def profile_environment(directory):
    env = os.environ.copy()
    env.update({name: "1" for name in THREADS})
    for key, suffix in (
        ("NUMBA_CACHE_DIR", "numba_cache"),
        ("XDG_CACHE_HOME", "xdg_cache"),
        ("TMPDIR", "tmp"),
    ):
        folder = directory / suffix
        folder.mkdir()
        env[key] = str(folder.resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PYTHONPATH", None)
    return env


def run_cell(config, identity, cell_index, user, encoded, sender_record, args, out, hashes):
    method, mode, budget = (identity[key] for key in ("method", "mode", "read_pairs"))
    cell = out / f"cell_{cell_index:02d}"
    cell.mkdir()
    private = cell / "evaluator_only"
    private.mkdir()
    write_json(cell / "config.json", {"identity_evaluator_only": identity, "config": config})
    write_json(
        cell / "environment.json",
        {"python": sys.version, "executable": sys.executable, "source_hashes": hashes},
    )
    write_json(private / "sender.json", sender_record)
    sent = [
        frame.sequence for frame in encoded if frame.status == "OK" and frame.sequence is not None
    ]
    physical = config["physical"]
    templates = [
        physical["forward_primer"] + body + physical["reverse_primer"].translate(COMP)[::-1]
        for body in sent
    ]
    seeds = {
        label: seed_for(
            identity["seed"], "profile-v2", identity["file_id"], method, mode, budget, label
        )
        for label in ("source", "shuffle", "ids")
    }
    job = {
        "schema_version": 2,
        "mode": mode,
        "body_nt": 150,
        "copies_per_template": physical["copies_per_sent_body"],
        "read_pairs": budget,
        "seed": seeds["source"],
        "shuffle_seed": seeds["shuffle"],
        "id_seed": seeds["ids"],
        "templates": templates,
        "forward_primer": physical["forward_primer"],
        "reverse_primer": physical["reverse_primer"],
    }
    write_json(private / "profile_job.json", job)
    command = [
        str(args.profile_python),
        "-B",
        str(args.worker),
        "--job",
        str(private / "profile_job.json"),
        "--source",
        str(args.source),
        "--out",
        str(cell / "channel"),
    ]
    resource = supervise(command, cell, profile_environment(private))
    channel = cell / "channel"
    generated = (
        json.loads((channel / "summary.json").read_text())
        if resource["exit_code"] == 0 and (channel / "summary.json").is_file()
        else failed_generation(
            channel,
            resource,
            requested_pairs=budget,
            molecule_cap=config["limits"]["max_initial_molecules_per_cell"],
        )
    )
    received, receiver_metadata, receiver_error = None, None, None
    public = PublicProtocol(**config["protocol"], mapper=method)
    observations = ()
    receiver_boundary_start = time.perf_counter_ns()
    if generated.get("status") == "COMPLETE":
        observations = read_public(channel / "public/R1.jsonl")
        require(len(observations) == generated["actual_R1"], "public actual read count")
        write_json(
            cell / "receiver_input.json",
            {
                "public_protocol": asdict(public),
                "observations": [asdict(read) for read in observations],
            },
        )
        actor = None
        try:
            # No source paths, seeds, labels or truth are arguments to spawn/receive.
            # multiprocessing preparation also copies sys.argv. Do not let
            # external source/config/output CLI paths cross that side channel.
            saved_argv = sys.argv
            try:
                sys.argv = [str(Path(__file__).resolve())]
                actor = ReceiverActor(public, timeout_seconds=60)
            finally:
                sys.argv = saved_argv
            receiver_metadata = dict(actor.server_metadata)
            received = actor.receive(observations)
        except ReceiverTerminationUnconfirmedError:
            raise
        except (TimeoutError, EOFError, OSError, RuntimeError) as error:
            receiver_error = f"{type(error).__name__}: {error}"
        finally:
            if actor is not None:
                actor.close()
                if actor._process.is_alive():
                    raise ReceiverTerminationUnconfirmedError("receiver child still alive")
    else:
        receiver_error = "NOT_RUN_PROFILE_FAILED"
    receiver_boundary_ns = time.perf_counter_ns() - receiver_boundary_start
    write_json(
        cell / "receiver_output.json",
        {
            "result": asdict(received) if received is not None else None,
            "actor_metadata": receiver_metadata,
            "receiver_error": receiver_error,
            "profile_exited_and_group_empty_before_spawn": resource[
                "process_group_confirmed_empty"
            ],
        },
    )
    metrics, frames = score(user, encoded, received)
    write_json(private / "posthoc_frame_scores.json", frames)
    summary = {
        "status": "IMPLEMENTED_UNVERIFIED",
        **identity,
        **metrics,
        "generator_status": generated.get("status", "FAILED"),
        "generator": generated,
        "receiver_status": received.status if received is not None else "NOT_RUN_OR_FAILED",
        "receiver_error": receiver_error,
        "receiver_actor": receiver_metadata,
        "receiver_boundary_wall_ns_startup_IPC_IO_included": receiver_boundary_ns,
        "receiver_execution_ns": received.elapsed_ns if received is not None else None,
        "receiver_observation_count": len(observations),
        "profile_resource": resource,
        "accounting": costs(
            config, user, encoded, generated, metrics["file_correct"], method, budget
        ),
        "truth_isolation": "fresh spawn public protocol and opaque R1 observations only; not OS filesystem sandbox",
        "read_origin": None,
        "true_edit_script": None,
        "assignment_accuracy": None,
        "non_evaluable_reason": LINEAGE_REASON,
    }
    write_json(cell / "summary.json", summary)
    seal(cell)
    if (channel / "evaluator_only/count_invariant_failure.json").exists():
        raise RuntimeError("SOURCE_COUNT_INVARIANT_FAILURE: sealed cell retained; stop incomplete")
    print(
        json.dumps(
            {
                "cell": cell.name,
                **identity,
                "file_correct": metrics["file_correct"],
                "generator_status": summary["generator_status"],
                "frame_erasures": metrics["frame_erasures"],
            }
        ),
        flush=True,
    )
    return summary


def run(args):
    config_path = args.config.resolve()
    config = validate_config(json.loads(config_path.read_text()))
    # Preserve the venv launcher path: resolving its interpreter symlink would
    # bypass pyvenv.cfg and silently run in the bundled base environment.
    args.profile_python = Path(os.path.abspath(args.profile_python))
    for field in ("worker", "source", "out", "approval", "inputs"):
        setattr(args, field, getattr(args, field).resolve())
    require(
        not args.out.exists(),
        "existing output refused, including partial/failed cells; no implicit resume/retry",
    )
    source_before = source_snapshot(args.source)
    require(
        len(source_before) == 54
        and source_before.get("dt4dds/settings/iseq100.yaml")
        == config["source"]["iseq100_yaml_sha256"],
        "published source tree/profile",
    )
    declared = json.loads(
        (args.source.parent / "dt4dds_v1.0.0_source_inspection_manifest.json").read_text()
    )
    require(
        source_before == {item["path"]: item["sha256"] for item in declared["files"]},
        "full54 published source manifest mismatch",
    )
    cells = planned_cells(config, args.phase, args.file_index)
    input_manifest = verify_frozen_users(args.inputs, config)
    verify_approval(args, config_path)
    hashes = tracked_hashes(args, config_path)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_ONLY_ZERO_ENCODE_OR_PROFILE_CALLS",
                    "phase": args.phase,
                    "file_index": args.file_index,
                    "planned_cells": cells,
                    "frozen_users": len(input_manifest["records"]),
                    "config_sha256": sha(config_path),
                    "implementation_version": IMPLEMENTATION_VERSION,
                }
            )
        )
        return
    args.out.mkdir(parents=True)
    write_json(args.out / "config.json", config)
    write_json(
        args.out / "environment.json",
        {
            "UTC": dt.datetime.now(dt.timezone.utc).isoformat(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "external_profile_jobs": 1,
            "other_IDS_and_lightweight_research_may_run": True,
            "machine_exclusive": False,
            "tracked_hashes": hashes,
            "profile_source_before": source_before,
        },
    )
    private = args.out / "evaluator_only"
    private.mkdir()
    master, file_id = cells[0]["seed"], cells[0]["file_id"]
    require(
        all((x["seed"], x["file_id"]) == (master, file_id) for x in cells),
        "one independent user file per controller",
    )
    user = random.Random(seed_for(master, "user", file_id)).randbytes(1202)
    if args.phase == "formal":
        require(
            user == (args.inputs / f"user_{master}_{file_id:02d}.bin").read_bytes(),
            "formal input must equal original pre-observation bytes",
        )
    user_path = private / "user.bin"
    with user_path.open("xb") as stream:
        stream.write(user)
    user_records = [
        {
            "seed": master,
            "file_id": file_id,
            "bytes": 1202,
            "sha256": sha(user_path),
            "path": str(user_path.relative_to(args.out)),
        }
    ]
    senders = {}
    selected_methods = [m for m in config["methods"] if any(x["method"] == m for x in cells)]
    for method in selected_methods:
        public = PublicProtocol(**config["protocol"], mapper=method)
        encoder_seed = seed_for(master, "encoder", method, file_id)
        encoded = encode_file(user, public, seed=encoder_seed)
        checks = []
        for frame in encoded:
            if frame.status == "OK" and frame.sequence is not None:
                inverse = decode_frame(frame.sequence, frame.strand_index, public)
                require(
                    inverse.status == "OK" and inverse.payload == frame.payload,
                    "sender noiseless inverse failed before profile; halt",
                )
                template = (
                    config["physical"]["forward_primer"]
                    + frame.sequence
                    + config["physical"]["reverse_primer"].translate(COMP)[::-1]
                )
                body, whole = (
                    constraint_record(frame.sequence),
                    constraint_record(template, boundaries=(58, 208)),
                )
                if frame.mode in {"short_core", "nonce"}:
                    require(
                        body["rc_violation_count"] == 0, "claimed short-k body has RC violation"
                    )
                if frame.mode != "two_bit":
                    require(
                        body["gc_40_60_ok"] and body["homopolymer_max4_ok"],
                        "claimed body GC/H violation",
                    )
                checks.append(
                    {"strand_index": frame.strand_index, "body": body, "whole_template": whole}
                )
        record = {
            "user_hex": user.hex(),
            "encoder_seed": encoder_seed,
            "encoded": [asdict(frame) for frame in encoded],
            "protocol": asdict(public),
            "constraint_checks": checks,
            "constraint_scope": "body_only",
            "whole_template_checks_are_diagnostic_not_a_guarantee": True,
        }
        write_json(private / f"sender_{method}.json", record)
        senders[method] = encoded, record
    write_json(
        args.out / "pre_observation_freeze.json",
        {
            "UTC_before_first_profile_observation": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_sha256": sha(config_path),
            "sources": hashes,
            "profile_source": source_before,
            "execution_phase": args.phase,
            "formal_file_index": args.file_index,
            "interface_smoke_excluded_from_formal_intervals": args.phase == "interface_smoke",
            "all_96_user_input_manifest_sha256": sha(args.inputs / "manifest.json"),
            "source_original_pre_observation_freeze_sha256": input_manifest[
                "source_pre_observation_freeze_sha256"
            ],
            "users": user_records,
            "sender_hashes": {
                method: sha(private / f"sender_{method}.json") for method in selected_methods
            },
            "planned_cell_keys": cells,
            "existing_output_or_partial_resume": "REFUSE_NO_AUTOMATIC_RETRY",
        },
    )
    rows = []
    try:
        for index, identity in enumerate(cells):
            encoded, record = senders[identity["method"]]
            require(
                tracked_hashes(args, config_path) == hashes
                and source_snapshot(args.source) == source_before,
                "frozen inputs drifted before cell",
            )
            rows.append(
                run_cell(config, identity, index, user, encoded, record, args, args.out, hashes)
            )
    except BaseException as error:
        write_json(
            args.out / "halt.json",
            {
                "reason": f"{type(error).__name__}: {error}",
                "completed_cells": len(rows),
                "automatic_retry": False,
            },
        )
        raise
    after = tracked_hashes(args, config_path)
    source_after = source_snapshot(args.source)
    require(hashes == after and source_before == source_after, "frozen sources/environment drifted")
    write_json(
        args.out / "source_after.json", {"tracked_hashes": after, "profile_source": source_after}
    )
    summary = {
        "status": "IMPLEMENTED_UNVERIFIED",
        "execution_status": "COMPLETE_INTERFACE_SMOKE_NOT_FORMAL"
        if args.phase == "interface_smoke"
        else "COMPLETE_FORMAL_FILE_UNVERIFIED",
        "execution_phase": args.phase,
        "formal_file_index": args.file_index,
        "implementation_version": IMPLEMENTATION_VERSION,
        "cells": len(rows),
        "planned_cells": len(cells),
        "formal_file_method_budget_cells": len(rows) if args.phase == "formal" else 0,
        "scientific_acceptance": False,
        "file_failures": sum(row["file_failed"] for row in rows),
        "sources_unchanged": True,
        "rows": rows,
        "read_origin": None,
        "true_edit_script": None,
        "assignment_accuracy": None,
        "non_evaluable_reason": LINEAGE_REASON,
    }
    write_json(args.out / "summary.json", summary)
    seal(args.out)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile-python", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--phase", choices=("interface_smoke", "formal"), required=True)
    parser.add_argument("--file-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for variable in THREADS:
        os.environ[variable] = "1"
    run(args)


if __name__ == "__main__":
    main()
