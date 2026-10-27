#!/usr/bin/env python3
"""Extract audited structural details without materializing huge BPP arrays.

The full raw auditor has already checked every structure/subrun/probability.
This pass retains the structural fields needed for complete negative-boundary
tables. Every output key is reconciled with the full auditor's slim table.
It performs no new folding and creates no additional scientific samples.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

NAMES = {
    "row_id",
    "model_id",
    "sequence",
    "mfe_structure",
    "stems",
    "m_subruns",
    "full_constraint_valid",
    "wc_m_subruns_at_or_above_k",
}
PATTERN = re.compile(r'"(' + "|".join(sorted(NAMES)) + r')"\s*:\s*')
DECODER = json.JSONDecoder()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def extract(line):
    result = {}
    for match in PATTERN.finditer(line):
        value, _ = DECODER.raw_decode(line, match.end())
        key = match.group(1)
        if key in result:
            assert value == result[key], key
        result[key] = value
    assert set(result) == NAMES
    return result


def shard(task):
    path, expected, context, unit, records, scalar_check = task
    assert sha(path) == expected
    rows = []
    with gzip.open(path, "rt") as stream:
        assert json.loads(next(stream)) == {"_context": context, "_unit": unit}
        for index, line in enumerate(stream):
            if line.startswith('{"_complete"'):
                assert json.loads(line) == {"_complete": True, "records": records}
                assert len(rows) == records and stream.read() == ""
                break
            row = extract(line)
            if scalar_check and index == 0:
                full = json.loads(line)
                assert row == {k: full[k] for k in NAMES}
            rows.append(row)
        else:
            raise AssertionError("missing completion marker")
    return unit, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--audited", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    checkpoint = json.loads((a.run / "checkpoint.json").read_text())
    expected = {}
    with gzip.open(a.audited / "fold_metrics.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["row_id"], row["model_id"]
            assert key not in expected and row["status"] == "OK"
            expected[key] = row
    assert len(expected) == 230424
    tasks = [
        (
            a.run / e["path"],
            e["sha256"],
            checkpoint["context_sha256"],
            name,
            e["records"],
            i % 97 == 0,
        )
        for i, (name, e) in enumerate(checkpoint["units"].items())
        if name.startswith("fold_")
    ]
    seen, checked, scalar_checks = set(), 0, sum(t[-1] for t in tasks)
    output = a.out / "fold_details.jsonl.gz"
    with gzip.open(output, "xt", compresslevel=1) as stream:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futures = [pool.submit(shard, t) for t in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                _, rows = future.result()
                for row in rows:
                    key = row["row_id"], row["model_id"]
                    assert key not in seen and key in expected
                    seen.add(key)
                    slim = expected[key]
                    n, m, k = slim["n"], slim["m"], slim["k"]
                    sequence, structure = row["sequence"], row["mfe_structure"]
                    assert len(sequence) == len(structure) == n
                    assert slim["has_stem_ge_m"] == any(
                        s["length"] >= m for s in row["stems"]
                    )
                    wc_at_k = sum(
                        r["wc_only"] and r["inner_span"] >= k for r in row["m_subruns"]
                    )
                    assert wc_at_k == row["wc_m_subruns_at_or_above_k"]
                    if row["full_constraint_valid"]:
                        assert wc_at_k == 0
                    qualifying = [s for s in row["stems"] if s["length"] >= m]
                    short_terminal = [
                        s
                        for s in qualifying
                        if 3 <= s["inner_span"] <= 12
                        and all(
                            c == "."
                            for c in structure[
                                s["pairs"][-1][0] + 1 : s["pairs"][-1][1]
                            ]
                        )
                    ]
                    value = {
                        **slim,
                        "sequence": sequence,
                        "mfe_structure": structure,
                        "GC_fraction": (sequence.count("G") + sequence.count("C")) / n,
                        "full_constraint_valid": row["full_constraint_valid"],
                        "WC_m_subruns_at_or_above_k": wc_at_k,
                        "has_WC_m_subrun": any(r["wc_only"] for r in row["m_subruns"]),
                        "has_GT_m_subrun": any(r["has_GT"] for r in row["m_subruns"]),
                        "has_WC_m_subrun_below_k": any(
                            r["wc_only"] and r["inner_span"] < k
                            for r in row["m_subruns"]
                        ),
                        "has_qualifying_true_hairpin_span_3to12": bool(short_terminal),
                        "has_WC_qualifying_true_hairpin_span_3to12": any(
                            s["wc_only"] for s in short_terminal
                        ),
                        "qualifying_stem_terminal_spans": [
                            s["inner_span"] for s in qualifying
                        ],
                        "qualifying_stem_true_hairpin": [
                            all(
                                c == "."
                                for c in structure[
                                    s["pairs"][-1][0] + 1 : s["pairs"][-1][1]
                                ]
                            )
                            for s in qualifying
                        ],
                        "m_subrun_gaps": [r["inner_span"] for r in row["m_subruns"]],
                        "m_subrun_WC_flags": [r["wc_only"] for r in row["m_subruns"]],
                    }
                    stream.write(
                        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                    )
                    checked += 1
                if i % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "shards": i,
                                "rows": checked,
                                "elapsed_s": round(time.monotonic() - start, 2),
                            }
                        ),
                        flush=True,
                    )
    assert seen == set(expected)
    result = {
        "status": "ALL_STRUCTURAL_DETAILS_RECONCILED_WITH_FULL_RAW_AUDIT",
        "rows": checked,
        "shards": len(tasks),
        "scalar_full_JSON_crosschecks": scalar_checks,
        "context_sha256": checkpoint["context_sha256"],
        "fold_details_sha256": sha(output),
        "new_native_folding_calls": 0,
        "new_scientific_samples": 0,
        "elapsed_seconds": time.monotonic() - start,
        "input_slim_sha256": sha(a.audited / "fold_metrics.jsonl.gz"),
        "source_sha256": sha(Path(__file__)),
    }
    (a.out / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
