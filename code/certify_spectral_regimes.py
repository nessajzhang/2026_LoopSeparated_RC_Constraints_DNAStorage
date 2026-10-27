"""Produce or independently check the finite spectral-regime base cases.

The producer uses Kosaraju certificates.  The checker reconstructs every
overlap matrix and uses reflexive transitive closure, not the producer's SCCs.
All finite masks and the orbit order are retained in the JSON certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loop_alpha_audit.regimes import (  # noqa: E402
    classify_graph,
    overlap_matrix,
    verify_graph_certificate,
)

CASES = (
    (2, (0, 1), 1, (1, 0, 0)),
    (2, (0, 1), 2, (2, 0, 0)),
    (2, (0, 1), 3, (4, 0, 0)),
    (2, (0, 1), 4, (64, 0, 0)),
    (2, (0, 1), 5, (3968, 128, 0)),
    (2, (1, 0), 1, (0, 2, 0)),
    (2, (1, 0), 2, (0, 2, 0)),
    (2, (1, 0), 3, (0, 14, 2)),
    (3, (0, 1, 2), 1, (1, 0, 0)),
    (3, (0, 1, 2), 2, (6, 2, 0)),
    (3, (0, 1, 2), 3, (270, 120, 122)),
    (3, (1, 0, 2), 1, (0, 2, 0)),
    (3, (1, 0, 2), 2, (0, 6, 2)),
    (4, (0, 1, 2, 3), 2, (24, 16, 24)),
)

WITNESSES = (
    (2, (1, 0), 3, "000 001 010 100", 2),
    (3, (1, 0, 2), 2, "11 12 21", 2),
    (4, (0, 1, 2, 3), 2, "01 12 20 13 30", 2),
    (3, (0, 1, 2), 3, "001 011 012 112 120 122 200 201 220", 2),
    (2, (0, 1), 5, "00101 01011 10110 01100 11001 10010", 1),
    (2, (0, 1), 6, "001101 010011 010100 011010 100110 101001 101010 110100 110101", 2),
)


def rc_pairs(q, theta, m):
    return [
        (w, partner)
        for w in product(range(q), repeat=m)
        if w < (partner := tuple(theta[a] for a in w[::-1]))
    ]


def closure_regime(words, q, m):
    """Independent direct word graph and Boolean-reachability classification."""

    vertices = list(product(range(q), repeat=m - 1))
    indices = {v: i for i, v in enumerate(vertices)}
    size = len(vertices)
    matrix = [[0] * size for _ in vertices]
    for word in words:
        matrix[indices[word[:-1]]][indices[word[1:]]] += 1
    reachable = [
        (1 << i) | sum((1 << j) for j, a in enumerate(row) if a) for i, row in enumerate(matrix)
    ]
    for k in range(size):
        bit = 1 << k
        for i in range(size):
            if reachable[i] & bit:
                reachable[i] |= reachable[k]
    cyclic = False
    for v in range(size):
        members = [w for w in range(size) if reachable[v] >> w & 1 and reachable[w] >> v & 1]
        if len(members) == 1 and not matrix[v][v]:
            continue
        cyclic = True
        if sum(matrix[v][w] for w in members) > 1:
            return 2
    return int(cyclic)


def elementary_cycles(m):
    """All vertex-simple cycles in the binary nonpalindromic overlap graph."""

    vertices = list(product(range(2), repeat=m - 1))
    indices = {w: i for i, w in enumerate(vertices)}
    adjacency = [[] for _ in vertices]
    for word in product(range(2), repeat=m):
        if word != word[::-1]:
            adjacency[indices[word[:-1]]].append((indices[word[1:]], word))
    cycles = []
    for start in range(len(vertices)):

        def visit(v, seen, words):
            for w, edge in adjacency[v]:
                if w == start:
                    cycle = words + [edge]
                    chosen = set(cycle)
                    cycles.append(
                        {
                            "words": [list(word) for word in cycle],
                            "rc_free": all(word[::-1] not in chosen for word in chosen),
                        }
                    )
                elif w > start and w not in seen:
                    visit(w, seen | {w}, words + [edge])

        visit(start, {start}, [])
    return cycles


def integer_charpoly(matrix):
    """Independent integer trace recurrence, not the producer's SymPy call."""
    n = len(matrix)
    previous = [[int(i == j) for j in range(n)] for i in range(n)]
    coefficients = [1]
    for k in range(1, n + 1):
        image = [
            [sum(matrix[i][s] * previous[s][j] for s in range(n)) for j in range(n)]
            for i in range(n)
        ]
        coefficient, remainder = divmod(-sum(image[i][i] for i in range(n)), k)
        if remainder:
            raise ValueError("noninteger characteristic polynomial recurrence")
        coefficients.append(coefficient)
        for i in range(n):
            image[i][i] += coefficient
        previous = image
    return coefficients


def produce():
    cases = []
    for q, theta, m, expected in CASES:
        pairs = rc_pairs(q, theta, m)
        counts = [0, 0, 0]
        regimes = []
        for mask in range(1 << len(pairs)):
            core = [pair[mask >> i & 1] for i, pair in enumerate(pairs)]
            certificate = classify_graph(overlap_matrix(q, m, core))
            verify_graph_certificate(certificate)
            regime = certificate["regime_index"]
            counts[regime] += 1
            regimes.append(str(regime))
        if tuple(counts) != expected:
            raise AssertionError((q, theta, m, counts, expected))
        cases.append(
            {
                "q": q,
                "theta": list(theta),
                "m": m,
                "orbit_pairs": [[list(w) for w in pair] for pair in pairs],
                "mask_order": "ascending integer; bit i selects member 0 or 1 of orbit_pairs[i]",
                "regimes_by_mask": "".join(regimes),
                "counts": counts,
            }
        )
    witnesses = []
    for q, theta, m, words, expected in WITNESSES:
        core = sorted(tuple(map(int, word)) for word in words.split())
        graph = classify_graph(overlap_matrix(q, m, core))
        if graph["regime_index"] != expected:
            raise AssertionError("wrong witness regime")
        witnesses.append({"q": q, "theta": list(theta), "m": m, "core": core, "graph": graph})
    from sympy import Matrix, Symbol

    t = Symbol("t")
    m6_poly = Matrix(witnesses[-1]["graph"]["adjacency"]).charpoly(t).as_expr()
    if m6_poly.expand() != (t**24 * (t**8 - t**2 - 1)).expand():
        raise AssertionError("binary m=6 characteristic polynomial changed")
    cycle_data = {str(m): elementary_cycles(m) for m in (4, 5)}
    return {
        "schema": "spectral-regime-finite-certificate-v1",
        "word_order": "lexicographic, symbols 0,...,q-1; no singleton RC orbit selected",
        "maximal_cores_checked": sum(sum(row["counts"]) for row in cases),
        "cases": cases,
        "witnesses": witnesses,
        "binary_m6_characteristic_polynomial": "t^24*(t^8-t^2-1)",
        "binary_identity_elementary_cycles": cycle_data,
    }


def check(data):
    if data.get("schema") != "spectral-regime-finite-certificate-v1":
        raise ValueError("unknown finite certificate schema")
    if len(data["cases"]) != len(CASES):
        raise ValueError("missing or duplicate finite case")
    checked = 0
    for row, (q, theta, m, expected) in zip(data["cases"], CASES):
        if (row["q"], tuple(row["theta"]), row["m"]) != (q, theta, m):
            raise ValueError("incorrect finite parameter coverage")
        pairs = rc_pairs(q, theta, m)
        if row["orbit_pairs"] != [[list(w) for w in pair] for pair in pairs]:
            raise ValueError("orbit order mismatch")
        regimes = row["regimes_by_mask"]
        if len(regimes) != 1 << len(pairs):
            raise ValueError("missing maximal core mask")
        counts = [0, 0, 0]
        for mask, regime in enumerate(regimes):
            core = [pair[(mask // (2**i)) % 2] for i, pair in enumerate(pairs)]
            actual = closure_regime(core, q, m)
            if str(actual) != regime:
                raise ValueError(f"wrong regime at q={q},m={m},mask={mask}")
            counts[actual] += 1
            checked += 1
        if counts != row["counts"] or tuple(counts) != expected:
            raise ValueError("incorrect finite case total")
    if len(data["witnesses"]) != len(WITNESSES):
        raise ValueError("missing witness")
    for row, (q, theta, m, words, expected) in zip(data["witnesses"], WITNESSES):
        wanted = sorted([list(map(int, word)) for word in words.split()])
        if (row["q"], tuple(row["theta"]), row["m"], row["core"]) != (q, theta, m, wanted):
            raise ValueError("witness changed")
        chosen = {tuple(w) for w in row["core"]}
        if any(tuple(theta[a] for a in w[::-1]) in chosen for w in chosen):
            raise ValueError("witness is not RC-free")
        if closure_regime(chosen, q, m) != expected:
            raise ValueError("incorrect independent witness regime")
        if row["graph"]["adjacency"] != [list(r) for r in overlap_matrix(q, m, chosen)]:
            raise ValueError("wrong witness matrix")
        verify_graph_certificate(row["graph"])
    for m, expected_count, expected_free in ((4, 6, 0), (5, 24, 2)):
        actual = elementary_cycles(m)
        if data["binary_identity_elementary_cycles"][str(m)] != actual:
            raise ValueError("elementary cycle certificate changed")
        if len(actual) != expected_count or sum(c["rc_free"] for c in actual) != expected_free:
            raise ValueError("wrong elementary cycle count")
    m5_free = [
        set(tuple(w) for w in c["words"])
        for c in data["binary_identity_elementary_cycles"]["5"]
        if c["rc_free"]
    ]
    if {w[::-1] for w in m5_free[0]} != m5_free[1]:
        raise ValueError("the two m=5 free cycles must be reversals of one another")
    expected_m6 = [1, 0, 0, 0, 0, 0, -1, 0, -1] + [0] * 24
    if integer_charpoly(data["witnesses"][-1]["graph"]["adjacency"]) != expected_m6:
        raise ValueError("independent m=6 characteristic polynomial failed")
    if data["binary_m6_characteristic_polynomial"] != "t^24*(t^8-t^2-1)":
        raise ValueError("incorrect polynomial label")
    if checked != 4782 or data["maximal_cores_checked"] != checked:
        raise ValueError("incomplete 4782-base-case coverage")
    return {
        "status": "PASS",
        "independent_method": "direct word graph plus bitset transitive closure",
        "maximal_cores_checked": checked,
        "witnesses_checked": len(WITNESSES),
        "binary_m4_counts": [64, 0, 0],
        "binary_m5_counts": [3968, 128, 0],
        "binary_m4_cycles": 6,
        "binary_m5_cycles": 24,
        "binary_m5_rc_free_cycles": 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    start = time.monotonic()
    if args.check:
        data = json.loads(args.check.read_text())
        result = check(data)
        result["certificate_sha256"] = hashlib.sha256(args.check.read_bytes()).hexdigest()
    else:
        if args.output is None:
            parser.error("--output or --check is required")
        data = produce()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
        result = check(json.loads(args.output.read_text()))
        result["certificate_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    result.update({"wall_seconds": time.monotonic() - start, "python": platform.python_version()})
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
