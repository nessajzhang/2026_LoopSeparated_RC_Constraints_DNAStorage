#!/usr/bin/env python3
"""Audit qualifying base-pair runs in the row-level ViennaRNA screen.

The combinatorial constraint uses Watson--Crick reverse complements, whereas
ViennaRNA's DNA parameterization can also form G--T/T--G wobble pairs. This
script reconstructs maximal contiguous nested runs from dot-bracket structures
and reports whether each qualifying run is Watson--Crick-only or contains a
wobble pair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


WATSON_CRICK = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
WOBBLE = {("G", "T"), ("T", "G")}
EXPECTED = {
    "screened_rows": 6000,
    "flagged_sequences": 2114,
    "qualifying_runs": 2635,
    "runs_with_wobble": 2635,
    "watson_crick_only_runs": 0,
    "runs_with_other_pairs": 0,
}


def dot_bracket_pairs(structure: str) -> list[tuple[int, int]]:
    """Return zero-based paired coordinates from a dot-bracket string."""
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for index, symbol in enumerate(structure):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced closing parenthesis")
            pairs.append((stack.pop(), index))
        elif symbol != ".":
            raise ValueError(f"unsupported dot-bracket symbol: {symbol!r}")
    if stack:
        raise ValueError("unbalanced opening parenthesis")
    return sorted(pairs)


def maximal_nested_runs(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Partition contiguous nested pairs into maximal stem runs."""
    pair_set = set(pairs)
    runs: list[list[tuple[int, int]]] = []
    for left, right in sorted(pair_set):
        if (left - 1, right + 1) in pair_set:
            continue
        run: list[tuple[int, int]] = []
        current = (left, right)
        while current in pair_set:
            run.append(current)
            current = (current[0] + 1, current[1] - 1)
        runs.append(run)
    return runs


def run_pair_classes(sequence: str, run: list[tuple[int, int]]) -> set[str]:
    classes: set[str] = set()
    for left, right in run:
        pair = (sequence[left], sequence[right])
        if pair in WATSON_CRICK:
            classes.add("watson_crick")
        elif pair in WOBBLE:
            classes.add("wobble")
        else:
            classes.add("other")
    return classes


def audit(path: Path) -> dict[str, object]:
    screened_rows = 0
    flagged_sequences = 0
    qualifying_runs = 0
    runs_with_wobble = 0
    watson_crick_only_runs = 0
    runs_with_other_pairs = 0
    flagged_by_group: Counter[str] = Counter()
    runs_by_group: Counter[str] = Counter()

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["group"] != "mk_ssa_gc_homopolymer" or int(row["k"]) != 0:
                continue
            screened_rows += 1
            if row["mk_ssa_valid"] != "True":
                raise AssertionError("k=0 constrained row failed its exact sequence check")
            sequence = row["sequence"]
            structure = row["mfe_structure"]
            if len(sequence) != len(structure):
                raise AssertionError("sequence and structure lengths differ")
            m = int(row["m"])
            all_runs = maximal_nested_runs(dot_bracket_pairs(structure))
            runs = [run for run in all_runs if len(run) >= m]
            observed_max = max((len(run) for run in all_runs), default=0)
            if observed_max != int(row["mfe_max_stem"]):
                raise AssertionError(
                    "stored maximum stem length does not match dot-bracket structure"
                )
            if not runs:
                continue
            flagged_sequences += 1
            group_key = f"length={row['length']},m={row['m']}"
            flagged_by_group[group_key] += 1
            for run in runs:
                qualifying_runs += 1
                runs_by_group[group_key] += 1
                classes = run_pair_classes(sequence, run)
                if "wobble" in classes:
                    runs_with_wobble += 1
                if classes == {"watson_crick"}:
                    watson_crick_only_runs += 1
                if "other" in classes:
                    runs_with_other_pairs += 1

    result: dict[str, object] = {
        "status": "PASS",
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "screened_rows": screened_rows,
        "flagged_sequences": flagged_sequences,
        "qualifying_runs": qualifying_runs,
        "runs_with_wobble": runs_with_wobble,
        "watson_crick_only_runs": watson_crick_only_runs,
        "runs_with_other_pairs": runs_with_other_pairs,
        "flagged_sequences_by_group": dict(sorted(flagged_by_group.items())),
        "qualifying_runs_by_group": dict(sorted(runs_by_group.items())),
        "evidence_boundary": (
            "This is a deterministic audit of supplied ViennaRNA predictions. "
            "It is not a folding experiment or a capacity calculation."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-manuscript-counts", action="store_true")
    args = parser.parse_args()

    result = audit(args.input_csv)
    if args.assert_manuscript_counts:
        observed = {key: result[key] for key in EXPECTED}
        if observed != EXPECTED:
            raise AssertionError({"expected": EXPECTED, "observed": observed})
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
