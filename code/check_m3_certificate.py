"""Standalone exact checker for arbitrary-window search-tree certificates.

No producer module, NumPy, saved floating bound, search-process memory, or
hash-only coverage argument is used.  Root coverage is reconstructed through
both children of every binary split and explicitly verified graph symmetries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path


def _bound(value, power=1):
    if value < 0 or not isinstance(power, int) or not 1 <= power <= 128:
        raise ValueError("invalid exact power bound")
    return Fraction(value), power


def _read_bound(data):
    n, d, p = data["numerator"], data["denominator"], data["power"]
    if not isinstance(n, int) or not isinstance(d, int) or d <= 0:
        raise ValueError("invalid rational bound")
    return _bound(Fraction(n, d), p)


def _le(left, right):
    power = math.lcm(left[1], right[1])
    return left[0] ** (power // left[1]) <= right[0] ** (power // right[1])


def _minimum(bounds):
    result = bounds[0]
    for bound in bounds[1:]:
        if _le(bound, result):
            result = bound
    return result


def _maximum(bounds):
    result = bounds[0]
    for bound in bounds[1:]:
        if _le(result, bound):
            result = bound
    return result


def _serialize(bound):
    return {"numerator": bound[0].numerator, "denominator": bound[0].denominator, "power": bound[1]}


def _matrix(mask, q, m):
    n = q ** (m - 1)
    matrix = [[0] * n for _ in range(n)]
    # Independent little-endian bit extraction, with base-q lexicographic ids.
    for edge in range(q**m):
        if mask & (1 << edge):
            prefix = edge // q
            suffix = edge - (edge // n) * n
            matrix[prefix][suffix] += 1
    return matrix


def _apply(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _components(matrix):
    n = len(matrix)
    reach = [(1 << i) | sum(1 << j for j, a in enumerate(row) if a) for i, row in enumerate(matrix)]
    for k in range(n):
        for i in range(n):
            if reach[i] & (1 << k):
                reach[i] |= reach[k]
    remaining = set(range(n))
    components = []
    while remaining:
        i = min(remaining)
        comp = [j for j in sorted(remaining) if (reach[i] >> j & 1) and (reach[j] >> i & 1)]
        remaining.difference_update(comp)
        components.append(comp)
    return components


def _collatz(matrix, certificate):
    exponent = certificate["exponent"]
    if not isinstance(exponent, int) or not 1 <= exponent <= 128:
        raise ValueError("unsupported Collatz exponent")
    cyclic = [c for c in _components(matrix) if len(c) > 1 or matrix[c[0]][c[0]]]
    if [row["vertices"] for row in certificate["components"]] != cyclic:
        raise ValueError("missing or incorrect cyclic SCC coverage")
    maximum = Fraction(0)
    for component, row in zip(cyclic, certificate["components"]):
        vector = row["vector"]
        if len(vector) != len(component) or any(not isinstance(x, int) or x <= 0 for x in vector):
            raise ValueError("Collatz vector must be strictly positive on its entire SCC")
        block = [[matrix[i][j] for j in component] for i in component]
        image = list(vector)
        for _ in range(exponent):
            image = _apply(block, image)
        maximum = max(maximum, *(Fraction(y, x) for x, y in zip(vector, image)))
    return maximum, exponent


def _charpoly(matrix):
    """Integer Faddeev-LeVerrier, independent of symbolic producer software."""
    n = len(matrix)
    b = [[int(i == j) for j in range(n)] for i in range(n)]
    coefficients = [1]
    for k in range(1, n + 1):
        ab = [[sum(matrix[i][s] * b[s][j] for s in range(n)) for j in range(n)] for i in range(n)]
        trace = sum(ab[i][i] for i in range(n))
        coefficient, remainder = divmod(-trace, k)
        if remainder:
            raise ValueError("nonintegral characteristic polynomial coefficient")
        coefficients.append(coefficient)
        for i in range(n):
            ab[i][i] += coefficient
        b = ab
    return coefficients


def _candidate(candidate, q, m, full):
    mask = int(candidate["mask_hex"], 16)
    if not 0 <= mask <= full:
        raise ValueError("invalid incumbent mask")
    stated = candidate["product_lower"]
    lower = Fraction(stated["numerator"], stated["denominator"])
    if lower < 0:
        raise ValueError("negative incumbent")
    left, right = _matrix(mask, q, m), _matrix(full ^ mask, q, m)
    if candidate["kind"] == "structured_ternary_cube":
        if (q, m) != (4, 3):
            raise ValueError("structured witness has incorrect parameters")
        wanted = sum(1 << (16 * a + 4 * b + c) for a, b, c in product(range(3), repeat=3))
        words = [[a, b, c] for a, b, c in product(range(3), repeat=3)]
        if mask != wanted or candidate["windows"] != words:
            raise ValueError("incorrect structured window set")
        if candidate["matrix"] != left or candidate["complement_matrix"] != right:
            raise ValueError("incorrect incumbent matrix")
        vector = candidate["complement_positive_eigenvector"]
        if len(vector) != 16 or any(not isinstance(x, int) or x <= 0 for x in vector):
            raise ValueError("invalid incumbent positive eigenvector")
        if _apply(right, vector) != [3 * x for x in vector]:
            raise ValueError("incumbent eigenvalue identity fails")
        vertices = [4 * a + b for a, b in product(range(3), repeat=2)]
        if any(sum(left[i][j] for j in vertices) != 3 for i in vertices):
            raise ValueError("left radius-three closed block is absent")
        if any(sum(row) > 3 for row in left):
            raise ValueError("left row-sum upper bound fails")
        if lower != 9 or candidate["radius_each"] != 3:
            raise ValueError("incorrect exact candidate lower bound")
        if _charpoly(left) != [1, -3] + [0] * 15:
            raise ValueError("left characteristic polynomial fails")
        if _charpoly(right) != [1, -1, -3, -9] + [0] * 13:
            raise ValueError("complement characteristic polynomial fails")
        digest = hashlib.sha256(
            json.dumps(
                {"windows": words, "matrix": left, "complement_matrix": right},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if candidate["sha256"] != digest:
            raise ValueError("incumbent manifest mismatch")
    elif candidate["kind"] == "rational_collatz":
        factors = []
        for side, matrix in (("left", left), ("right", right)):
            vertices = candidate[side + "_component"]
            vector = candidate[side + "_vector"]
            if (
                not vertices
                or len(set(vertices)) != len(vertices)
                or any(not 0 <= i < len(matrix) for i in vertices)
            ):
                raise ValueError("invalid candidate principal block")
            if len(vector) != len(vertices) or any(
                not isinstance(x, int) or x <= 0 for x in vector
            ):
                raise ValueError("invalid candidate lower vector")
            block = [[matrix[i][j] for j in vertices] for i in vertices]
            factors.append(min(Fraction(y, x) for x, y in zip(vector, _apply(block, vector))))
        if lower > factors[0] * factors[1]:
            raise ValueError("candidate lower bound is not certified")
    else:
        raise ValueError("unknown candidate certificate")
    return _bound(lower)


def check_certificate(data):
    if data.get("schema") != "arbitrary-window-frontier-v1":
        raise ValueError("unknown search certificate schema")
    q, m = data["q"], data["m"]
    if not isinstance(q, int) or not isinstance(m, int) or q < 2 or m < 1 or q**m > 256:
        raise ValueError("unsupported graph dimensions")
    edges = q**m
    full = (1 << edges) - 1
    nodes = {}
    for node in data["nodes"]:
        i = node["id"]
        if not isinstance(i, int) or i in nodes:
            raise ValueError("duplicate or invalid node id")
        nodes[i] = node
    root = data["root"]
    if root != 0 or root not in nodes:
        raise ValueError("missing full-domain root")
    lower = _candidate(data["incumbent"], q, m, full)
    threshold_data = data["prune_threshold"]
    threshold = _bound(Fraction(threshold_data["numerator"], threshold_data["denominator"]))
    if not _le(lower, threshold):
        raise ValueError("prune threshold lies below the certified incumbent")
    masks, paths, reached = {}, {}, set()

    def visit(i, a, b, ancestors):
        if i not in nodes or i in reached:
            raise ValueError("missing child, repeated child, or cyclic split")
        reached.add(i)
        node = nodes[i]
        actual_a, actual_b = int(node["a"], 16), int(node["b"], 16)
        if (a, b) != (actual_a, actual_b) or a & b or (a | b) & ~full:
            raise ValueError("node cylinder does not match its parent assignment")
        masks[i], paths[i] = (a, b), ancestors
        kind = node["kind"]
        if kind == "split":
            edge = node["edge"]
            children = node["children"]
            if not isinstance(edge, int) or not 0 <= edge < edges or (a | b) >> edge & 1:
                raise ValueError("split edge is already assigned or outside the domain")
            if len(children) != 2 or children[0] == children[1]:
                raise ValueError("a split must cover both distinct children")
            bit = 1 << edge
            visit(children[0], a | bit, b, ancestors + (i,))
            visit(children[1], a, b | bit, ancestors + (i,))
        elif kind in ("frontier", "pruned", "complete_assignment", "symmetry"):
            if "children" in node:
                raise ValueError("leaf unexpectedly contains children")
            if kind == "complete_assignment" and a | b != full:
                raise ValueError("incomplete assignment marked complete")
        else:
            raise ValueError("unknown leaf or split kind")

    visit(root, 0, 0, ())
    if reached != set(nodes):
        raise ValueError("unreachable extra nodes or omitted root coverage")

    def mapped(mask, permutation, reversal):
        result = 0
        for edge in range(edges):
            if not mask >> edge & 1:
                continue
            digits, value = [], edge
            for _ in range(m):
                value, a = divmod(value, q)
                digits.append(a)
            if not reversal:
                digits.reverse()
            value = 0
            for a in digits:
                value = q * value + permutation[a]
            result |= 1 << value
        return result

    for i, node in nodes.items():
        if node["kind"] != "symmetry":
            continue
        ref = node["ref"]
        p, reversal, swap = node["permutation"], node["reversal"], node["colour_swap"]
        if ref not in nodes or ref >= i or sorted(p) != list(range(q)):
            raise ValueError("invalid or circular symmetry canonical reference")
        if not isinstance(reversal, bool) or not isinstance(swap, bool):
            raise ValueError("invalid symmetry flags")
        a, b = masks[ref]
        want_a, want_b = mapped(a, p, reversal), mapped(b, p, reversal)
        if swap:
            want_a, want_b = want_b, want_a
        if masks[i] != (want_a, want_b):
            raise ValueError("explicit symmetry does not map the entire cylinder")

    own_cache = {}

    def own_bound(i):
        if i in own_cache:
            return own_cache[i]
        node = nodes[i]
        if node["kind"] == "symmetry":
            raise ValueError("symmetry leaves cannot be used as inherited bound proofs")
        a, b = masks[i]
        proofs = node.get("proofs", [])
        if not proofs:
            raise ValueError("node has no independently checkable upper bound")
        bounds = []
        for proof in proofs:
            kind = proof["kind"]
            if kind == "cardinality":
                low, high = a.bit_count(), edges - b.bit_count()
                s = min(high, max(low, edges // 2))
                bounds.append(_bound(s * (edges - s), m))
            elif kind == "dna_m3_integer_gap":
                if (q, m) != (4, 3):
                    raise ValueError("DNA integer-gap lemma used outside its proved domain")
                bounds.append(_bound(1023, 3))
            elif kind == "scc_collatz_product":
                left = _collatz(_matrix(full ^ b, q, m), proof["left"])
                right = _collatz(_matrix(full ^ a, q, m), proof["right"])
                if left[1] != right[1]:
                    raise ValueError("product powers must have the same exponent")
                bounds.append(_bound(left[0] * right[0], left[1]))
            elif kind == "finite_block_product":
                length = proof["length"]
                if not isinstance(length, int) or not m <= length <= 128:
                    raise ValueError("invalid finite-block length")
                values = []
                for mask in (full ^ b, full ^ a):
                    matrix = _matrix(mask, q, m)
                    vector = [1] * len(matrix)
                    for _ in range(length - m + 1):
                        vector = _apply(matrix, vector)
                    values.append(sum(vector))
                bounds.append(_bound(values[0] * values[1], length))
            elif kind == "inherit":
                ref = proof["ref"]
                if ref not in paths[i]:
                    raise ValueError("bound inheritance is not from a strict ancestor")
                bounds.append(own_bound(ref))
            else:
                raise ValueError("unsupported upper-bound proof")
        result = _minimum(bounds)
        stored = _read_bound(node["upper"])
        if not _le(result, stored) or not _le(stored, result):
            raise ValueError("producer's saved bound differs from independently recomputed minimum")
        own_cache[i] = result
        return result

    covered_cache, active = {}, set()

    def covered_bound(i):
        if i in covered_cache:
            return covered_cache[i]
        if i in active:
            raise ValueError("circular canonical subtree reference")
        active.add(i)
        node = nodes[i]
        if node["kind"] == "symmetry":
            result = covered_bound(node["ref"])
        else:
            result = own_bound(i)
            if node["kind"] == "pruned" and not _le(result, threshold):
                raise ValueError("leaf was pruned without an exact threshold comparison")
            if node["kind"] == "split":
                result = _minimum([result, _maximum([covered_bound(c) for c in node["children"]])])
        active.remove(i)
        covered_cache[i] = result
        return result

    upper = _maximum([lower, covered_bound(root)])
    counts = {
        kind: sum(n["kind"] == kind for n in nodes.values())
        for kind in ("split", "pruned", "frontier", "symmetry", "complete_assignment")
    }
    lo = float(lower[0])
    hi = float(upper[0]) ** (1 / upper[1])
    grade = (
        "A_exact"
        if _le(upper, lower)
        else "B_target_interval"
        if _le(upper, _bound(Fraction(2253, 250)))
        else "C_gap_not_substantively_closed"
    )
    return {
        "status": "PASS",
        "coverage": "full arbitrary 2^(q^m) domain, recursively reconstructed",
        "q": q,
        "m": m,
        "independent_edge_bits": edges,
        "nodes_checked": len(nodes),
        "node_kinds": counts,
        "lower": _serialize(lower),
        "global_upper": _serialize(upper),
        "lower_display": lo,
        "global_upper_display": hi,
        "optimality_proved": _le(upper, lower),
        "worst_capacity_gap_bits_per_nt_display": 0.5 * math.log2(hi / lo) if lo else None,
        "evidence_grade": grade,
        "floating_values_used_only_for_display": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = check_certificate(json.loads(args.certificate.read_text()))
    result["certificate_sha256"] = hashlib.sha256(args.certificate.read_bytes()).hexdigest()
    output = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
