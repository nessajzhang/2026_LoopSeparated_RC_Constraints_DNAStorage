#!/usr/bin/env python3
"""Composition-standardized and actual-message DNA folding arms.

Default mode is a bounded interface smoke, NOT a formal experiment. Formal
execution requires a separate approval record matching config and source
hashes. Targets/users are never redrawn after a control/encoder/fold failure.
All generated data must be outside the source checkout. No IDS is executed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from matched_dinucleotide import DNA, dinucleotide_counts, longest_run, sample_h_matched  # noqa:E402
from physical_model_helpers import (  # noqa: E402
    digest_json,
    fold_sequence,
    model_specs,
    prepare_models,
)
from loop_alpha_audit.endpoint_general import EndpointMapper  # noqa:E402
from loop_alpha_audit.framing import PublicProtocol, encode_file, rate_accounting  # noqa:E402
from loop_alpha_audit.statistics_revision import paired_failure_difference  # noqa:E402

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def rc(word):
    return word.translate(COMPLEMENT)[::-1]


def derived_seed(master, *labels):
    return int.from_bytes(
        hashlib.sha256(json.dumps([master, *labels], separators=(",", ":")).encode()).digest()[:16],
        "big",
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def avoids_windows(sequence, m, k):
    earliest = {}
    for j in range(len(sequence) - m + 1):
        block = sequence[j : j + m]
        opposite = rc(block)
        if opposite in earliest and j - earliest[opposite] >= m + k:
            return False
        earliest.setdefault(block, j)
    return True


def avoids_full(sequence, m, k):
    """Independent direct nested-pair growth, checking original stem/gap domain."""
    n = len(sequence)
    for i in range(n):
        for j in range(i + 2 * m + k - 1, n):
            length = 0
            while i + length < j - length and sequence[i + length] == rc(sequence[j - length]):
                length += 1
                gap = j - i - 2 * length + 1
                if length >= m and gap >= k:
                    return False
                if gap < k:
                    break
    return True


def sequence_fields(sequence, m, k):
    reduced, full = avoids_windows(sequence, m, k), avoids_full(sequence, m, k)
    if reduced != full:
        raise AssertionError("Original/reduced constraint disagreement")
    return {
        "gc_count": sequence.count("G") + sequence.count("C"),
        "gc_fraction": (sequence.count("G") + sequence.count("C")) / len(sequence),
        "longest_homopolymer": longest_run(sequence),
        "dinucleotide_counts_ACGT_rowmajor": dinucleotide_counts(sequence),
        "first": sequence[:1],
        "last": sequence[-1:],
        "full_constraint_valid": full,
        "reduced_constraint_valid": reduced,
    }


def mechanism_inputs(n, m, k, master, count, *, parent_cap, control_cap):
    rng = random.Random(derived_seed(master, "parent", n, m, k))
    attempted, rejected_composition, rejected_rc = 0, 0, 0
    seen_targets, seen_controls = set(), set()
    output = []
    for pair_index in range(count):
        target = None
        previous = attempted
        while attempted < parent_cap:
            attempted += 1
            candidate = "".join(DNA[rng.randrange(4)] for _ in range(n))
            gc = candidate.count("G") + candidate.count("C")
            if not 40 * n <= 100 * gc <= 60 * n or longest_run(candidate) > 4:
                rejected_composition += 1
                continue
            if not avoids_windows(candidate, m, k):
                rejected_rc += 1
                continue
            target = candidate
            break
        common = {
            "arm": "mechanism",
            "n": n,
            "m": m,
            "k": k,
            "seed": master,
            "pair_index": pair_index,
            "pair_id": f"M_n{n}_m{m}_k{k}_s{master}_p{pair_index}",
            "independent_unit": "target/control pair",
            "planned_pairs_in_seed_cell": count,
            "parent_attempts_for_target": attempted - previous,
            "parent_attempts_cumulative": attempted,
            "parent_cap": parent_cap,
            "parent_rejections_composition_cumulative": rejected_composition,
            "parent_rejections_RC_cumulative": rejected_rc,
            "constraint_scope": "whole_unframed_word_no_added_primers",
            "primer_nt": 0,
        }
        if target is None:
            control = None
            result = {"status": "PARENT_ATTEMPT_CAP", "attempts": 0}
        else:
            result = asdict(
                sample_h_matched(
                    target,
                    random.Random(derived_seed(master, "control", n, m, k, pair_index)),
                    max_attempts=control_cap,
                )
            )
            control = result.pop("sequence")
        for role, sequence in (("target", target), ("control", control)):
            row = {
                **common,
                "role": role,
                "row_id": common["pair_id"] + "_" + role,
                "sequence": sequence,
                "generation_status": "OK" if sequence is not None else result["status"],
                "control_result": result,
                "target_present": target is not None,
            }
            if sequence is not None:
                row.update(sequence_fields(sequence, m, k))
            output.append(row)
        if target is not None:
            output[-2]["duplicate_target"] = target in seen_targets
            seen_targets.add(target)
        if control is not None:
            output[-1]["duplicate_control"] = control in seen_controls
            seen_controls.add(control)
            if (
                dinucleotide_counts(target) != dinucleotide_counts(control)
                or target[0] != control[0]
                or target[-1] != control[-1]
                or longest_run(target) != longest_run(control)
            ):
                raise AssertionError("Exact composition/H matching failed")
    return output


def deployed_inputs(n, m, k, master, file_id, config):
    seed = derived_seed(master, "shared-user-file", n, m, k, file_id)
    preparation_failure = None
    try:
        reference = PublicProtocol(n=n, m=m, k=k, mapper="short_core", rs_n=63, rs_k=55)
        user = random.Random(seed).randbytes(reference.maximum_user_bytes)
    except Exception as exc:
        preparation_failure, user = repr(exc), None
    output = []
    for method in config["methods"]:
        common = {
            "arm": "deployment",
            "n": n,
            "m": m,
            "k": k,
            "seed": master,
            "file_id": file_id,
            "method": method,
            "independent_unit": "user file",
            "user_hex": user.hex() if user is not None else None,
            "user_sha256": hashlib.sha256(user).hexdigest() if user is not None else None,
            "user_bits": len(user) * 8 if user is not None else None,
            "file_seed": seed,
            "planned_strands": 63,
            "primer_nt": 0,
            "constraint_scope": "whole_diagnostic_oligo_including_public_header_no_primers",
        }
        public_record = None
        try:
            if preparation_failure:
                raise RuntimeError("Reference protocol preparation failed:" + preparation_failure)
            public = PublicProtocol(
                n=n,
                m=m,
                k=k,
                mapper=method,
                payload_bytes=reference.payload_bytes,
                rs_n=63,
                rs_k=55,
                nonce_budget=config["nonce_budget"],
            )
            frames = encode_file(user, public, seed=seed)
            accounting = rate_accounting(public, len(user), frames)
            assert len(frames) == 63
            public_record = asdict(public)
        except Exception as exc:
            frames, accounting = [None] * 63, {"failure_reason": repr(exc)}
        for strand_index, frame in enumerate(frames):
            frame_record = asdict(frame) if frame is not None else None
            if frame_record is not None:
                frame_record["payload_hex"] = frame_record.pop("payload").hex()
            row = {
                **common,
                "strand_index": strand_index,
                "row_id": f"D_n{n}_m{m}_k{k}_s{master}_f{file_id}_{method}_i{strand_index}",
                "rate_accounting": accounting,
                "frame": frame_record,
                "public_protocol": public_record,
                "sequence": frame.sequence if frame is not None else None,
                "generation_status": frame.status
                if frame is not None
                else "FILE_ENCODING_EXCEPTION_RETAINED",
            }
            if row["sequence"] is not None:
                row.update(sequence_fields(row["sequence"], m, k))
            output.append(row)
    return output


def high_alpha_inputs(n, m, alpha, master, count):
    k = int(alpha * n)
    mapper = EndpointMapper(n, m, k, product(range(3), repeat=m))
    rows = []
    for index in range(count):
        payload = random.Random(
            derived_seed(master, "uniform-bits", n, m, alpha, index)
        ).getrandbits(mapper.K)
        row = {
            "arm": "high_alpha",
            "n": n,
            "m": m,
            "k": k,
            "alpha": alpha,
            "seed": master,
            "word_index": index,
            "row_id": f"H_n{n}_m{m}_a{alpha}_s{master}_i{index}",
            "K": mapper.K,
            "M": str(mapper.codebook_size),
            "uniform_binary_index": str(payload),
            "endpoint_length": mapper.endpoint_length,
            "free_length": mapper.free_length,
            "distribution": "actual mapper of uniform K bits,first2**K ranks;not uniform allM",
            "constraint_scope": "whole_unframed_endpoint_word_no_primers",
            "primer_nt": 0,
        }
        try:
            digits = mapper.encode_index(payload)
            if mapper.decode_index(digits) != payload:
                raise AssertionError("endpoint inverse mismatch")
            sequence = "".join(DNA[d] for d in digits)
            row.update(sequence=sequence, generation_status="OK", **sequence_fields(sequence, m, k))
            if not row["full_constraint_valid"]:
                raise AssertionError("endpoint full constraint failed")
        except Exception as exc:
            row["attempted_sequence_if_produced"] = row.get("sequence")
            row.update(
                sequence=None,
                generation_status="ENDPOINT_FAILURE_RETAINED",
                failure_reason=repr(exc),
            )
        rows.append(row)
    return rows


def witness_input(n, m, alpha):
    k = int(alpha * n)
    ell, free = n - m - k, 2 * (m + k) - n
    motif = "G" * m + "AAA" + "C" * m
    if free < 0 or ell < len(motif):
        raise ValueError("witness not supported in this endpoint geometry")
    sequence = motif + "A" * (ell - len(motif)) + "A" * free + "A" * ell
    fields = sequence_fields(sequence, m, k)
    if not fields["full_constraint_valid"]:
        raise AssertionError("high-alpha witness not avoiding")
    return {
        "arm": "witness",
        "n": n,
        "m": m,
        "k": k,
        "alpha": alpha,
        "row_id": f"W_n{n}_m{m}_a{alpha}",
        "sequence": sequence,
        "generation_status": "OK",
        "deterministic_motif_inner_span": 3,
        "motif_start": 0,
        "constraint_scope": "whole_unframed_endpoint_word_no_primers",
        "primer_nt": 0,
        **fields,
    }


def planned_counts(config):
    cells = len(config["n_values"]) * len(config["m_values"]) * len(config["k_values"])
    pairs = cells * len(config["mechanism_seeds"]) * config["pairs_per_seed_cell"]
    files = (
        cells
        * len(config["deployment_seeds"])
        * config["files_per_seed_cell_method"]
        * len(config["methods"])
    )
    high = (
        len(config["n_values"])
        * len(config["m_values"])
        * len(config["alphas"])
        * len(config["high_alpha_seeds"])
        * config["high_words_per_seed_cell"]
    )
    return {
        "mechanism_pairs": pairs,
        "deployment_files": files,
        "deployment_planned_strands": 63 * files,
        "high_alpha_words": high,
        "high_alpha_MFE_models_per_word": 2,
        "witness_words": len(config["n_values"]) * len(config["m_values"]) * len(config["alphas"]),
        "primary_comparisons": cells,
        "formal_observations_generated_by_plan": 0,
    }


def validate_config(config):
    """Version1 has no silently ignored design/model options or reduced grid."""
    required = {
        "n_values": [100, 150, 200],
        "m_values": [6, 7],
        "k_values": [0, 3, 4, 5],
        "pairs_per_seed_cell": 384,
        "parent_candidate_cap_per_seed_cell": 10000000,
        "control_attempt_cap": 100000,
        "files_per_seed_cell_method": 8,
        "outer_RS": [63, 55],
        "alphas": [0.5, 0.75],
        "high_words_per_seed_cell": 128,
        "primary_model": "T37_S1.021_default",
        "high_alpha_models": ["T37_S1.021_default", "T37_S1.021_WC"],
        "high_alpha_ensemble": False,
        "sensitivity_n": 150,
        "sensitivity_k": [0, 3, 5],
        "sensitivity_pairs_per_seed_cell": 32,
        "sensitivity_files_per_seed_cell_method": 2,
        "primary_family_size": 24,
        "family_alpha": 0.05,
        "sensitivity_settings_temperature_C_salt_M": [
            [37, 1.021],
            [25, 1.021],
            [50, 1.021],
            [37, 0.05],
            [37, 0.15],
        ],
        "pairing_rules": ["default", "WC_noGU"],
        "maximum_folding_workers": 2,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Unsupported/changed physical v1 design;new reviewed version required")
    if (
        set(config["methods"]) != {"short_core", "nonce", "gc_h_nonce", "two_bit"}
        or len(config["methods"]) != 4
    ):
        raise ValueError("All four practical methods must remain")
    indices = config["ensemble_strand_indices"]
    if (
        len(indices) != 8
        or len(set(indices)) != 8
        or any(type(i) is not int or not 0 <= i < 63 for i in indices)
    ):
        raise ValueError("Declare exactly8 valid predetermined strand indices")
    if (
        not 1 <= config["smoke_max_sequences_per_model"] <= 12
        or not 1 <= config["fold_shard_records"] <= 64
    ):
        raise ValueError("Unbounded smoke/shard size")
    seeds = config["mechanism_seeds"] + config["deployment_seeds"] + config["high_alpha_seeds"]
    if len(seeds) != 9 or len(set(seeds)) != 9 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Need three distinct master seeds per arm")


def fold_schedule(row, config, *, smoke=False):
    models = model_specs()
    if smoke:
        return [(spec, True) for spec in models]
    if row["arm"] in {"high_alpha", "witness"}:
        return [(spec, False) for spec in models[:2]]
    sensitivity_cell = row["n"] == config["sensitivity_n"] and row["k"] in config["sensitivity_k"]
    if row["arm"] == "mechanism":
        return [
            (spec, True)
            for spec in (
                models
                if sensitivity_cell
                and row["pair_index"] < config["sensitivity_pairs_per_seed_cell"]
                else models[:1]
            )
        ]
    selected = row["strand_index"] in config["ensemble_strand_indices"]
    sensitive = (
        sensitivity_cell
        and selected
        and row["file_id"] < config["sensitivity_files_per_seed_cell_method"]
    )
    return [(spec, selected) for spec in (models if sensitive else models[:1])]


def _init_worker():
    import RNA

    if RNA.__version__ != "2.7.2" or not RNA.params_load_DNA_Mathews2004():
        raise RuntimeError("Pinned DNA folding worker initialization failed")


def fold_task(task):
    row, spec, ensemble, parameter_hash = task
    if row["sequence"] is None:
        return {
            **row,
            "model_id": spec.model_id,
            "status": "NO_SEQUENCE_INPUT_FAILURE",
            "ensemble_requested": ensemble,
        }
    result = fold_sequence(row["sequence"], row["m"], row["k"], spec, ensemble=ensemble)
    if (
        result["status"] == "OK"
        and row["full_constraint_valid"]
        and result["wc_m_subruns_at_or_above_k"]
    ):
        result.update(
            status="INVARIANT_FAILED_RETAINED",
            failure_reason="WC m-subrun contradicts full sequence constraint",
        )
    return {
        **row,
        **result,
        "parameter_sha256": parameter_hash,
        "worker_pid": os.getpid(),
        "worker_peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


class Checkpoint:
    """Atomic bounded shards; partial attempts retained, completed hashes checked."""

    def __init__(self, folder, context):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.context_hash = digest_json(context)
        self.path = self.folder / "checkpoint.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["context_sha256"] != self.context_hash:
                raise ValueError("Refuse resume with changed config/source/environment")
            for entry in self.state["units"].values():
                if sha(self.folder / entry["path"]) != entry["sha256"]:
                    raise ValueError("Completed checkpoint shard hash changed")
        else:
            self.state = {"context_sha256": self.context_hash, "context": context, "units": {}}
            write_json(self.path, self.state)

    def unit(self, name, producer):
        if any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in name
        ):
            raise ValueError("Unsafe checkpoint key")
        path = self.folder / (name + ".jsonl.gz")
        if name in self.state["units"]:
            return self.read(name)
        if path.exists():
            # A completed rename can precede checkpoint commit on interruption.
            rows = self._read_path(path, name)
        else:
            attempt = 1
            partial = path.with_name(path.name + f".partial{attempt}")
            while partial.exists():
                attempt += 1
                partial = path.with_name(path.name + f".partial{attempt}")
            count = 0
            with partial.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                    stream.write(
                        (json.dumps({"_context": self.context_hash, "_unit": name}) + "\n").encode()
                    )
                    for row in producer():
                        stream.write((json.dumps(row, allow_nan=False) + "\n").encode())
                        count += 1
                    stream.write(
                        (json.dumps({"_complete": True, "records": count}) + "\n").encode()
                    )
                raw.flush()
                os.fsync(raw.fileno())
            partial.replace(path)
            rows = self._read_path(path, name)
        self.state["units"][name] = {"path": path.name, "sha256": sha(path), "records": len(rows)}
        write_json(self.path, self.state)
        return rows

    def _read_path(self, path, name):
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        if (
            len(rows) < 2
            or rows[0] != {"_context": self.context_hash, "_unit": name}
            or rows[-1] != {"_complete": True, "records": len(rows) - 2}
        ):
            raise ValueError("Incomplete/orphan shard cannot be accepted as complete")
        return rows[1:-1]

    def read(self, name):
        entry = self.state["units"][name]
        path = self.folder / entry["path"]
        if sha(path) != entry["sha256"]:
            raise ValueError("Checkpoint shard changed after initialization")
        return self._read_path(path, name)


def source_identity():
    paths = [
        Path(__file__),
        Path(__file__).with_name("matched_dinucleotide.py"),
        Path(__file__).with_name("physical_model_helpers.py"),
        CODE_ROOT / "pyproject.toml",
        CODE_ROOT / "uv.lock",
        *sorted((CODE_ROOT / "loop_alpha_audit").glob("*.py")),
    ]
    return {str(p.relative_to(CODE_ROOT)): sha(p) for p in paths}


def require_approval(path, config_hash, source_hash):
    if path is None:
        raise PermissionError(
            "Formal grid not authorized: fresh proof/code/design approval required"
        )
    approval = json.loads(Path(path).read_text())
    if (
        approval.get("status") != "APPROVED_FOR_LOCAL_FORMAL"
        or approval.get("config_sha256") != config_hash
        or approval.get("implementation_sha256") != source_hash
        or not approval.get("independent_gate_id")
    ):
        raise PermissionError("Approval does not match this exact config/source")
    return approval


def input_units(config):
    for n, m, k in product(config["n_values"], config["m_values"], config["k_values"]):
        for master in config["mechanism_seeds"]:
            yield (
                f"input_mechanism_n{n}_m{m}_k{k}_s{master}",
                lambda n=n, m=m, k=k, s=master: mechanism_inputs(
                    n,
                    m,
                    k,
                    s,
                    config["pairs_per_seed_cell"],
                    parent_cap=config["parent_candidate_cap_per_seed_cell"],
                    control_cap=config["control_attempt_cap"],
                ),
            )
        for master in config["deployment_seeds"]:
            for file_id in range(config["files_per_seed_cell_method"]):
                yield (
                    f"input_deployment_n{n}_m{m}_k{k}_s{master}_f{file_id}",
                    lambda n=n, m=m, k=k, s=master, f=file_id: deployed_inputs(
                        n, m, k, s, f, config
                    ),
                )
    for n, m, alpha in product(config["n_values"], config["m_values"], config["alphas"]):
        for master in config["high_alpha_seeds"]:
            yield (
                f"input_high_n{n}_m{m}_a{alpha}_s{master}",
                lambda n=n, m=m, a=alpha, s=master: high_alpha_inputs(
                    n, m, a, s, config["high_words_per_seed_cell"]
                ),
            )
        yield (
            f"input_witness_n{n}_m{m}_a{alpha}",
            lambda n=n, m=m, a=alpha: [witness_input(n, m, a)],
        )


def smoke_inputs(config):
    records = mechanism_inputs(
        100,
        6,
        3,
        config["mechanism_seeds"][0],
        1,
        parent_cap=config["parent_candidate_cap_per_seed_cell"],
        control_cap=config["control_attempt_cap"],
    )
    full_file = deployed_inputs(100, 6, 3, config["deployment_seeds"][0], 0, config)
    # Fixed index0, never choose the first successful/favorable sequence.
    records += [row for row in full_file if row["strand_index"] == 0]
    for n, m, alpha in ((100, 6, 0.5), (150, 7, 0.75)):
        records += high_alpha_inputs(n, m, alpha, config["high_alpha_seeds"][0], 1)
        records.append(witness_input(n, m, alpha))
    assert len(records) <= 10
    return records, full_file


def summarize_rows(checkpoint, config, mode):
    generation, observations = {}, []
    for name in checkpoint.state["units"]:
        if name.startswith("input_") or name.startswith("audit_smoke_"):
            for row in checkpoint.read(name):
                generation[row["row_id"]] = row
        elif name.startswith("fold_"):
            for row in checkpoint.read(name):
                observations.append(
                    {
                        key: row.get(key)
                        for key in (
                            "row_id",
                            "arm",
                            "n",
                            "m",
                            "k",
                            "seed",
                            "pair_id",
                            "role",
                            "file_id",
                            "method",
                            "model_id",
                            "status",
                            "has_stem_ge_m",
                            "generation_status",
                        )
                    }
                )
    primary = {
        row["row_id"]: row for row in observations if row["model_id"] == "T37_S1.021_default"
    }
    mechanism = []
    for n, m, k in product(config["n_values"], config["m_values"], config["k_values"]):
        pairs = [
            r
            for r in generation.values()
            if r["arm"] == "mechanism"
            and r["role"] == "target"
            and (r["n"], r["m"], r["k"]) == (n, m, k)
        ]
        target, control = [], []
        cap_failures = sum(row.get("control_result", {}).get("status") != "OK" for row in pairs)
        for row in pairs:
            a, b = primary.get(row["row_id"]), primary.get(row["pair_id"] + "_control")
            if (
                a
                and b
                and a["status"] == b["status"] == "OK"
                and row.get("control_result", {}).get("status") == "OK"
            ):
                target.append(a["has_stem_ge_m"])
                control.append(b["has_stem_ge_m"])
        planned = (
            len(config["mechanism_seeds"]) * config["pairs_per_seed_cell"]
            if mode == "formal"
            else len(pairs)
        )
        planned_keys = {
            (seed, index)
            for seed in config["mechanism_seeds"]
            for index in range(config["pairs_per_seed_cell"])
        }
        observed_keys = [(row["seed"], row["pair_index"]) for row in pairs]
        plan_keys_valid = mode != "formal" or (
            len(pairs) == planned
            and len(set(observed_keys)) == len(observed_keys)
            and set(observed_keys) == planned_keys
        )
        complete_cell = bool(
            planned and plan_keys_valid and len(target) == planned and cap_failures == 0
        )
        result = {
            "n": n,
            "m": m,
            "k": k,
            "planned_pairs": planned,
            "attempted_pair_rows": len(pairs),
            "complete_pairs": len(target),
            "planned_pair_keys_valid": plan_keys_valid,
            "target_or_control_cap_failed_pairs": cap_failures,
            "paired_discordance_conservative_interval": None,
            "CI_pp": None,
            "alpha_cell": config["family_alpha"] / config["primary_family_size"],
            "marginal_two_sided_CP_alpha": config["family_alpha"]
            / config["primary_family_size"]
            / 2,
            "marginal_CP_each_tail": config["family_alpha"] / config["primary_family_size"] / 4,
            "fixed_family_size": config["primary_family_size"],
            "finite_cap_coverage": "Under potential uncapped iid-pair coupling,P(returnCI and noncoverage)<=alpha_cell;no conditional-on-return coverage or iid-given-all-success claim",
            "comparison_status": "COMPLETE" if complete_cell else "INCOMPLETE_NON_EVALUABLE",
        }
        if complete_cell and mode == "formal":
            interval = paired_failure_difference(
                control, target, alpha=config["family_alpha"] / config["primary_family_size"]
            )
            interval["independent_pairs"] = interval.pop("independent_files")
            interval["control_only_stem"] = interval.pop("left_only_failures")
            interval["target_only_stem"] = interval.pop("right_only_failures")
            interval["difference_pp"] = interval["difference"] * 100
            interval["confidence_interval_pp"] = [x * 100 for x in interval["confidence_interval"]]
            interval["direction"] = (
                "control minus target,percentage points;not biochemical causal effect"
            )
            result["paired_discordance_conservative_interval"] = interval
            result["CI_pp"] = interval["confidence_interval_pp"]
        mechanism.append(result)
    deployment_files = []
    grouped = {}
    for row in generation.values():
        if row["arm"] == "deployment":
            key = tuple(row[k] for k in ("n", "m", "k", "seed", "file_id", "method"))
            grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        folded = [primary.get(row["row_id"]) for row in rows]
        okay = [row for row in folded if row and row["status"] == "OK"]
        qualifying = sum(row["has_stem_ge_m"] for row in okay)
        deployment_files.append(
            {
                "key_n_m_k_seed_file_method": key,
                "planned_strands": 63,
                "attempted_strands": len(rows),
                "physical_observations": len(okay),
                "physical_missing": 63 - len(okay),
                "not_selected_for_interface_smoke": 63 - len(okay) if mode == "smoke" else 0,
                "file_physical_status": "COMPLETE"
                if len(okay) == 63
                else "SMOKE_SUBSET_ONLY"
                if mode == "smoke"
                else "INCOMPLETE_NON_EVALUABLE",
                "has_stem_count": qualifying,
                "file_fraction_if_complete": qualifying / 63 if len(okay) == 63 else None,
                "planned_fraction_identification_bounds": [
                    qualifying / 63,
                    (qualifying + 63 - len(okay)) / 63,
                ],
                "uncertainty_unit": "file;no independent-strand binomial CI",
            }
        )
    counts_by_model = {}
    for row in observations:
        counts_by_model[row["model_id"]] = counts_by_model.get(row["model_id"], 0) + 1
    return {
        "status": "SMOKE_ONLY_NOT_FORMAL_EVIDENCE"
        if mode == "smoke"
        else "FORMAL_OUTPUT_PENDING_INDEPENDENT_ACCEPTANCE",
        "mode": mode,
        "planned_formal_denominators": planned_counts(config),
        "generated_input_rows": len(generation),
        "fold_attempts_by_model": counts_by_model,
        "fold_failure_rows": sum(row["status"] != "OK" for row in observations),
        "mechanism_cells": mechanism,
        "deployment_file_metrics": deployment_files,
        "boundary": "Prediction only;composition-standardized association and generator distributions,not FER/wet-lab or isolated RC causal effect.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=CODE_ROOT / "configs/revision/physical_arms_v1.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "smoke", "formal"), default="smoke")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--approval-file", type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("Only1..2 folding workers permitted")
    out = args.out.resolve()
    if out.is_relative_to(CODE_ROOT.resolve()):
        parser.error("Generated physical data must be outside source checkout")
    config = json.loads(args.config.read_text())
    validate_config(config)
    identity = source_identity()
    versions = {
        name: importlib.metadata.version(name)
        for name in ("ViennaRNA", "numpy", "scipy", "reedsolo", "rapidfuzz")
    }
    context = {
        "config_sha256": sha(args.config),
        "implementation_sha256": digest_json(identity),
        "source_files": identity,
        "python": sys.version,
        "mode": args.mode,
        "package_versions": versions,
        "platform": platform.platform(),
    }
    if args.mode == "formal":
        context["approval"] = require_approval(
            args.approval_file, context["config_sha256"], context["implementation_sha256"]
        )
    checkpoint = Checkpoint(out, context)
    write_json(out / "config.json", config)
    write_json(
        out / "environment.json",
        {
            **context,
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "cpu_count": os.cpu_count(),
            "workers": args.workers,
            "process_start": "spawn",
            "versions": versions,
            "peakRSS_unit": "bytes_on_macOS,kilobytes_on_Linux",
        },
    )
    if args.mode == "plan":
        write_json(out / "summary.json", planned_counts(config))
        print(json.dumps(planned_counts(config)))
        return
    started = time.monotonic()
    models = prepare_models(out / "parameters/active_DNA_Mathews2004.par")
    write_json(out / "model_api.json", models)
    if any(value["status"] == "UNSUPPORTED_NOT_SUBSTITUTED" for value in models["models"].values()):
        write_json(
            out / "summary.json",
            {
                "status": "BLOCKED_UNSUPPORTED_MODELS_NOT_SUBSTITUTED",
                "mode": args.mode,
                "planned_denominators": planned_counts(config),
                "fold_observations": 0,
            },
        )
        raise RuntimeError(
            "One or more requested models unsupported;see model_api.json;no substitution"
        )
    if args.mode == "smoke":

        def smoke_producer():
            selected, full = smoke_inputs(config)
            checkpoint.unit("audit_smoke_all252_frame_attempts", lambda: full)
            return selected

        units = [("input_smoke", smoke_producer)]
    else:
        units = input_units(config)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker
    ) as executor:
        for name, producer in units:
            records = checkpoint.unit(name, producer)
            tasks = [
                (row, spec, ensemble, models["parameter_sha256"])
                for row in records
                for spec, ensemble in fold_schedule(row, config, smoke=args.mode == "smoke")
            ]
            if args.mode == "smoke":
                for spec in model_specs():
                    if (
                        sum(task[1] == spec for task in tasks)
                        > config["smoke_max_sequences_per_model"]
                    ):
                        raise RuntimeError(
                            "Smoke per-model sequence budget exceeded before folding"
                        )
            for offset in range(0, len(tasks), config["fold_shard_records"]):
                batch = tasks[offset : offset + config["fold_shard_records"]]
                checkpoint.unit(
                    f"fold_{name}_{offset}",
                    lambda batch=batch: executor.map(fold_task, batch, chunksize=1),
                )
    summary = summarize_rows(checkpoint, config, args.mode)
    summary.update(
        elapsed_seconds=time.monotonic() - started,
        peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_unchanged=identity == source_identity(),
        models_parameter_sha256=models["parameter_sha256"],
    )
    if not summary["source_unchanged"]:
        summary["status"] = "INVALID_SOURCE_CHANGED_DURING_RUN_RETAINED_NOT_ACCEPTED"
        write_json(out / "summary.json", summary)
        raise RuntimeError("Source changed during run;results retained,acceptance refused")
    write_json(out / "summary.json", summary)
    paths = sorted(
        p
        for p in out.rglob("*")
        if p.is_file() and p.name not in {"manifest.sha256", "manifest.sha256.tmp"}
    )
    (out / "manifest.sha256").write_text(
        "".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in paths)
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "fold_attempts_by_model",
                    "fold_failure_rows",
                    "elapsed_seconds",
                    "source_unchanged",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
