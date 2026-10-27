#!/usr/bin/env python3
"""Bounded source-faithful iSeq-only generator in the separate profile venv.

No codec/receiver imports and no mutation or sampling algorithm is copied.
Observational wrappers consume no RNG and retain only private diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import dataclasses
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
import random
import resource
import signal
import sys
import time
import traceback

from schema import reverse_complement, validate_job, validate_observations


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(source):
    return {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}


def read_fastq(path):
    with gzip.open(path, "rt", encoding="ascii") as stream:
        lines = stream.read().splitlines()
    if len(lines) % 4:
        raise ValueError("truncated source FASTQ")
    records = [tuple(lines[i : i + 4]) for i in range(0, len(lines), 4)]
    for header, sequence, plus, quality in records:
        if not header.startswith("@") or plus != "+" or len(sequence) != len(quality):
            raise ValueError("invalid source FASTQ record")
        if len(sequence) != 150 or set(sequence) - set("ACGT"):
            raise ValueError("source read outside declared body interface")
    return records


def write_fastq(path, records):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="ascii", newline="\n") as stream:
                for record in records:
                    stream.write("\n".join(record) + "\n")


def publish(out, job):
    mates = [read_fastq(out / "raw" / f"R{i}.fq.gz") for i in (1, 2)]
    if len(mates[0]) != len(mates[1]) or len(mates[0]) > 315:
        raise ValueError("unequal mates or source read-pair cap violation")
    for a, b in zip(*mates, strict=True):
        if a[0] != b[0]:
            raise ValueError("source positional mate headers differ")
    order = list(range(len(mates[0])))
    random.Random(job["shuffle_seed"]).shuffle(order)
    names = random.Random(job["id_seed"])
    mapping, public_mates, observations = [], [[], []], [[], []]
    for raw_index in order:
        opaque = f"r{names.getrandbits(128):032x}"
        mapping.append(
            {"read_id": opaque, "raw_index": raw_index, "raw_header": mates[0][raw_index][0]}
        )
        for mate in range(2):
            record = mates[mate][raw_index]
            public_mates[mate].append((f"@{opaque}/{mate + 1}", record[1], "+", record[3]))
            observations[mate].append({"read_id": opaque, "sequence": record[1]})
    for rows in observations:
        validate_observations(rows)
    assert sorted(order) == list(range(len(order)))
    raw_multiset = Counter((a[1], a[3], b[1], b[3]) for a, b in zip(*mates, strict=True))
    public_multiset = Counter((a[1], a[3], b[1], b[3]) for a, b in zip(*public_mates, strict=True))
    assert raw_multiset == public_multiset
    for mate in range(2):
        write_fastq(out / "public" / f"R{mate + 1}.fq.gz", public_mates[mate])
        with (out / "public" / f"R{mate + 1}.jsonl").open("w") as stream:
            for record in observations[mate]:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    write_json(
        out / "evaluator_only" / "shuffle_map.json",
        {
            "mapping": mapping,
            "whole_file_uniform_permutation": order,
            "shuffle_seed": job["shuffle_seed"],
            "id_seed": job["id_seed"],
            "positional_mates_are_not_persistent_molecule_lineage": True,
            "paired_tuple_multiset_preserved": True,
        },
    )
    return len(order)


def generate(job, source, out):
    before = source_snapshot(source)
    if (
        len(before) != 54
        or before.get("dt4dds/settings/iseq100.yaml")
        != "4cc87cf4244857f25f1b7544a03a9d61cae55913fe419d50b69ad7dc3f3389f9"
    ):
        raise ValueError("unexpected published source tree/profile")
    write_json(out / "evaluator_only/source_before.json", before)
    sys.path.insert(0, str(source))
    imports = []
    for name in ("numpy", "numba", "dt4dds"):
        start = time.monotonic()
        write_json(out / "evaluator_only/import_progress.json", {"done": imports, "current": name})
        importlib.import_module(name)
        imports.append({"module": name, "seconds": time.monotonic() - start})
    import dt4dds as dt
    import numba
    import numpy as np

    if (
        Path(dt.__file__).resolve() != (source / "dt4dds/__init__.py").resolve()
        or dt.__VERSION__ != "1.0.0"
    ):
        raise ValueError("wrong simulator import/version")
    dt.config.enable_multiprocessing = False
    dt.config.n_processes = 1
    dt.config.show_progressbars = False
    numba.set_num_threads(1)
    random.seed(job["seed"])
    np.random.seed(job["seed"] % 2**32)
    dt.config.set_random_seed(job["seed"])
    dt.properties.AmplificationEfficiency._efficiency_cache.clear()
    write_json(
        out / "evaluator_only/environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "pandas",
                    "biopython",
                    "edlib",
                    "numba",
                    "llvmlite",
                    "PyYAML",
                    "rapidfuzz",
                )
            },
            "imports": imports,
            "internal_multiprocessing": False,
            "internal_threads": 1,
            "seed": job["seed"],
            "numpy_legacy_seed_mod_2_32": job["seed"] % 2**32,
        },
    )
    pool = dt.datastructures.SeqPool()
    # The guard is evaluated before this first pool.add_sequences call.
    validate_job(job)
    pool.add_sequences(job["templates"], [8] * len(job["templates"]))
    initial = {str(sequence): int(count) for sequence, count in pool}
    assert sum(initial.values()) == 8 * len(job["templates"]) <= 504
    pool.save_as_csv(out / "evaluator_only/initial_pool.csv")
    settings = dt.settings.defaults.iSeq100(
        n_reads=job["read_pairs"], output_directory=str(out / "raw")
    )
    settings_dict = dataclasses.asdict(settings)
    baseline = dataclasses.asdict(dt.settings.defaults.iSeq100())
    for key in set(baseline) - {"n_reads", "output_directory"}:
        assert settings_dict[key] == baseline[key], f"forbidden source setting override: {key}"
    assert settings.read_length == 150 and settings.read_mode.lower() == "paired-end"
    assert settings.insertion_rates == settings.deletion_rates == [0.0, 0.0]
    assert settings.substitution_rates == [0.001123, 0.0025]
    ideal = importlib.import_module("dt4dds.processes.sbs_sequencing").IdealSBSSequencing
    sequencer = (ideal if job["mode"] == "ideal" else dt.processes.SBSSequencing)(settings)
    groups, selected = [], {}
    original_process = sequencer._process_reads
    original_read = sequencer._read

    def observed_process(selected_pool):
        selected.update({str(sequence): int(count) for sequence, count in selected_pool})
        selected_pool.save_as_csv(out / "evaluator_only/selected_pool.csv")
        return original_process(selected_pool)

    def observed_read(arguments):
        sequence, count = arguments
        crops = []
        for mate, read in enumerate((sequence, sequence.reverse_complement())):
            position = read.find_primer(
                settings.primers_read[mate], min_overlap=settings.primer_minoverlap
            )
            clean = str(read[position : position + 150]) if position else ""
            expected = str(sequence)[58:208]
            if mate:
                expected = reverse_complement(expected)
            crops.append(
                {
                    "mate": mate + 1,
                    "primer_suffix_end": position,
                    "clean_before_errors": clean,
                    "padding_before_errors": 150 - len(clean),
                    "geometry_matches_body": clean == expected,
                }
            )
        result = original_read(arguments)
        groups.append(
            {
                "template": str(sequence),
                "selected_count": int(count),
                "crops": crops,
                "returned_counts": [len(values) for values in result],
            }
        )
        return result

    sequencer._process_reads = observed_process
    sequencer._read = observed_read
    write_json(
        out / "evaluator_only/settings.json",
        {
            "settings": settings_dict,
            "only_setting_overrides": ["n_reads", "output_directory"],
            "sequencer_class": type(sequencer).__name__,
            "mode": job["mode"],
            "ideal_uses_source_IdealSBS_not_modified_error_rates": job["mode"] == "ideal",
            "ideal_pool_no_synthesis_PCR_or_aging": True,
        },
    )
    sequencer.process(pool)
    write_json(out / "evaluator_only/read_groups.json", groups)
    after = source_snapshot(source)
    write_json(out / "evaluator_only/source_after.json", after)
    if before != after:
        raise RuntimeError("published source files changed")
    actual = publish(out, job)
    overdraw = {
        hashlib.sha256(sequence.encode()).hexdigest(): count - initial.get(sequence, 0)
        for sequence, count in selected.items()
        if count > initial.get(sequence, 0)
    }
    geometry_ok = all(
        crop["geometry_matches_body"] and crop["padding_before_errors"] == 0
        for group in groups
        for crop in group["crops"]
    )
    if not geometry_ok:
        raise RuntimeError("source crop geometry/padding differs; no oracle trimming applied")
    return {
        "status": "COMPLETE",
        "mode": job["mode"],
        "initial_molecules": sum(initial.values()),
        "input_designs": len(job["templates"]),
        "unique_templates": len(initial),
        "duplicate_designs_merged": len(job["templates"]) - len(initial),
        "selected_molecules": sum(selected.values()),
        "requested_pairs": job["read_pairs"],
        "actual_R1": actual,
        "actual_R2": actual,
        "requested_minus_actual_pairs": job["read_pairs"] - actual,
        "source_float_truncation_target": int(
            job["read_pairs"] / sum(initial.values()) * sum(initial.values())
        )
        if initial
        else 0,
        "selected_template_overdraw": overdraw,
        "overdraw_not_corrected": True,
        "source_before_after_equal": True,
        "crop_geometry_all_body": geometry_ok,
        "receiver_extra_trim_nt": 0,
        "R1_primary_R2_retained": True,
        "read_origin": None,
        "true_edit_script": None,
        "assignment_accuracy": None,
        "non_evaluable_reason": "NON_EVALUABLE_NO_LINEAGE_CERTIFICATE",
        "constant_source_FASTQ_quality_is_not_calibrated": True,
        "raw_fastq_decompressed_sha256": {
            f"R{i}": hashlib.sha256(
                gzip.decompress((out / "raw" / f"R{i}.fq.gz").read_bytes())
            ).hexdigest()
            for i in (1, 2)
        },
    }


def alarm_handler(signum, frame):
    raise TimeoutError("profile worker own 60s wall alarm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    job = validate_job(json.loads(args.job.read_text()))
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "evaluator_only", "public"):
        (args.out / name).mkdir()
    write_json(args.out / "evaluator_only/job.json", job)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, 60)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    started = time.monotonic()
    try:
        result = generate(job, args.source.resolve(), args.out)
    except BaseException as error:
        write_json(
            args.out / "worker_failure.json",
            {
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "wall_seconds": time.monotonic() - started,
                "traceback": traceback.format_exc(),
            },
        )
        raise
    else:
        result["worker_wall_seconds"] = time.monotonic() - started
        write_json(args.out / "summary.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
