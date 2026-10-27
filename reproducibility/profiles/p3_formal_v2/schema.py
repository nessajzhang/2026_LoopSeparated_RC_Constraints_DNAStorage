"""Strict 512-copy v2 profile schemas; original eight-copy schemas retained."""

from __future__ import annotations

import re

DNA = frozenset("ACGT")
OPAQUE = re.compile(r"r[0-9a-f]{32}\Z")
JOB_FIELDS = {
    "schema_version",
    "mode",
    "body_nt",
    "copies_per_template",
    "read_pairs",
    "seed",
    "shuffle_seed",
    "id_seed",
    "templates",
    "forward_primer",
    "reverse_primer",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def validate_job(job):
    require(
        isinstance(job, dict) and set(job) == JOB_FIELDS, "profile job schema mismatch"
    )
    require(
        job["schema_version"] == 2 and job["mode"] in {"ideal", "noisy"},
        "job version/mode",
    )
    require(
        job["body_nt"] == 150 and job["copies_per_template"] == 512,
        "bounded body/copy plan",
    )
    require(
        type(job["read_pairs"]) is int and job["read_pairs"] in {192, 315},
        "bounded read-pair plan",
    )
    require(
        all(
            type(job[name]) is int and job[name] >= 0
            for name in ("seed", "shuffle_seed", "id_seed")
        ),
        "integer private seeds",
    )
    forward, reverse = job["forward_primer"], job["reverse_primer"]
    require(isinstance(forward, str) and isinstance(reverse, str), "primer type")
    require(
        len(forward) == 58
        and len(reverse) == 64
        and not (set(forward + reverse) - DNA),
        "primer geometry",
    )
    templates = job["templates"]
    require(isinstance(templates, list) and len(templates) <= 63, "template count cap")
    for sequence in templates:
        require(
            isinstance(sequence, str)
            and len(sequence) == 272
            and not (set(sequence) - DNA),
            "template DNA/length",
        )
        require(
            sequence.startswith(forward)
            and sequence.endswith(reverse_complement(reverse)),
            "template public adapters",
        )
    require(512 * len(templates) <= 32256, "molecule cap before SeqPool.add")
    return job


def validate_observation(record):
    require(
        isinstance(record, dict) and set(record) == {"read_id", "sequence"},
        "public observation has forbidden or missing field",
    )
    require(
        isinstance(record["read_id"], str) and OPAQUE.fullmatch(record["read_id"]),
        "opaque128 read ID required",
    )
    require(
        isinstance(record["sequence"], str)
        and len(record["sequence"]) == 150
        and not (set(record["sequence"]) - DNA),
        "observed body DNA/length required",
    )
    return record


def validate_observations(records):
    require(isinstance(records, list) and len(records) <= 315, "public read count cap")
    for record in records:
        validate_observation(record)
    require(len({r["read_id"] for r in records}) == len(records), "duplicate opaque ID")
    return records
