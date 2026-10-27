"""Independent standard-library checker for the analytic DNA m=3 evidence.

No producer import, external library, eigensolver, DNA exhaustive search, or
proof-assistant formalization is used. The quantified global theorem must
be assessed from the locked human-readable proof, not from these finite tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

KNOWN_PROOF_SHA256 = "fdde4a01b4640a6b79d759a5668a16bb9544048c8bd966058fc11ad005f11498"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _exact_tree(value, key=None):
    """Reject floating input and bool-as-int substitutions before arithmetic."""
    flags = {
        "all_dna_assignments_enumerated",
        "proof_assistant_formalized",
        "all_m_theorem_claimed",
        "legacy_search_frontiers_closed",
        "symmetry_reduction",
    }
    if key in flags:
        require(type(value) is bool, "scope flags must be Boolean")
    elif isinstance(value, dict):
        for child_key, child in value.items():
            require(type(child_key) is str, "JSON keys must be strings")
            _exact_tree(child, child_key)
    elif isinstance(value, list):
        for child in value:
            _exact_tree(child)
    else:
        require(type(value) in (str, int), "only exact integer/string evidence is supported")
        if type(value) is int:
            require(value.bit_length() <= 1024, "unbounded certificate integer")


def _matrix(q, mask):
    states = list(product(range(q), repeat=2))
    indices = {word: i for i, word in enumerate(states)}
    matrix = [[0 for _ in states] for _ in states]
    for edge, word in enumerate(product(range(q), repeat=3)):
        if mask & (1 << edge):
            matrix[indices[word[:2]]][indices[word[1:]]] += 1
    return matrix


def _apply(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def _multiply(left, right):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*right)] for row in left]


def _checked_matrix(saved, expected):
    require(isinstance(saved, list) and len(saved) == len(expected), "wrong matrix dimensions")
    for saved_row, expected_row in zip(saved, expected):
        require(
            isinstance(saved_row, list)
            and len(saved_row) == len(expected_row)
            and all(type(x) is int for x in saved_row)
            and saved_row == expected_row,
            "incorrect integer matrix",
        )


def _vector(vector, size, *, positive):
    require(
        isinstance(vector, list)
        and len(vector) == size
        and all(type(x) is int and 0 <= x < 2**256 for x in vector)
        and any(vector)
        and (not positive or min(vector) > 0),
        "invalid nonnegative or strictly positive integer eigenvector",
    )
    return vector


def _characteristic_polynomial(matrix):
    size = len(matrix)
    previous = [[int(i == j) for j in range(size)] for i in range(size)]
    coefficients = [1]
    for k in range(1, size + 1):
        previous = _multiply(matrix, previous)
        coefficient, remainder = divmod(-sum(previous[i][i] for i in range(size)), k)
        require(remainder == 0, "noninteger characteristic-polynomial recurrence")
        coefficients.append(coefficient)
        for i in range(size):
            previous[i][i] += coefficient
    return coefficients


def _components(matrix):
    size = len(matrix)
    reach = [
        (1 << i) | sum(1 << j for j, value in enumerate(row) if value)
        for i, row in enumerate(matrix)
    ]
    for k in range(size):
        for i in range(size):
            if reach[i] & (1 << k):
                reach[i] |= reach[k]
    remaining, result = set(range(size)), []
    while remaining:
        i = min(remaining)
        component = [j for j in sorted(remaining) if reach[i] >> j & 1 and reach[j] >> i & 1]
        remaining.difference_update(component)
        result.append(component)
    return result


def _radius_upper(matrix):
    upper = Fraction(0)
    for component in _components(matrix):
        if len(component) == 1 and not matrix[component[0]][component[0]]:
            continue
        block = [[matrix[i][j] for j in component] for i in component]
        vector = [1] * len(component)
        for _ in range(64):
            image = _apply(block, vector)
            vector = [x + y for x, y in zip(vector, image)]
        # Exact Collatz upper bound on each cyclic diagonal block.
        upper = max(upper, *(Fraction(y, x) for x, y in zip(vector, _apply(block, vector))))
    return upper


def _fraction(data):
    require(isinstance(data, dict) and set(data) == {"numerator", "denominator"}, "bad rational")
    n, d = data["numerator"], data["denominator"]
    require(
        type(n) is int
        and type(d) is int
        and n >= 0
        and d > 0
        and n.bit_length() <= 512
        and d.bit_length() <= 512
        and math.gcd(n, d) == 1,
        "invalid or noncanonical rational bound",
    )
    return Fraction(n, d)


def _poly_add(left, right, scale=1):
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0) + scale * value
    return {key: value for key, value in result.items() if value}


def _poly_multiply(left, right):
    result = {}
    for powers_a, a in left.items():
        for powers_b, b in right.items():
            key = tuple(x + y for x, y in zip(powers_a, powers_b))
            result[key] = result.get(key, 0) + a * b
    return {key: value for key, value in result.items() if value}


def _elimination(data):
    lam, mu, q = {(1, 0, 0): 1}, {(0, 1, 0): 1}, {(0, 0, 1): 1}
    s = _poly_add(lam, mu)
    z, square_s = _poly_multiply(lam, mu), _poly_multiply(s, s)
    twice_s_minus_q = _poly_add(_poly_add(s, s), q, -1)
    first = _poly_add(
        _poly_multiply(_poly_add(s, q, -1), square_s),
        _poly_multiply(twice_s_minus_q, z),
        -1,
    )
    lhs = {key: 4 * value for key, value in first.items()}
    difference = _poly_add(lam, mu, -1)
    rhs = _poly_add(
        _poly_multiply(square_s, _poly_add(_poly_add(s, s), q, -3)),
        _poly_multiply(twice_s_minus_q, _poly_multiply(difference, difference)),
    )
    require(lhs == rhs, "internal elimination identity failed")
    expected = {
        "variables": ["lambda", "mu", "q"],
        "identity": (
            "4*((S-q)*S^2-(2*S-q)*lambda*mu)=S^2*(2*S-3*q)+(2*S-q)*(lambda-mu)^2; S=lambda+mu"
        ),
        "coefficients": [list(key) + [value] for key, value in sorted(lhs.items())],
    }
    require(data == expected, "incorrect elimination polynomial or coefficients")


def _binary(data):
    require(
        {
            key: data[key]
            for key in ("q", "m", "all_arbitrary_masks", "symmetry_reduction", "vector_steps")
        }
        == {
            "q": 2,
            "m": 3,
            "all_arbitrary_masks": 256,
            "symmetry_reduction": False,
            "vector_steps": 64,
        },
        "incorrect binary finite-domain specification",
    )
    rows = data["rows"]
    require(isinstance(rows, list) and len(rows) == 256, "incomplete binary mask coverage")
    for mask, row in enumerate(rows):
        require(type(row["mask"]) is int and row["mask"] == mask, "binary mask order changed")
        left, right = _fraction(row["left_upper"]), _fraction(row["right_upper"])
        require(left == _radius_upper(_matrix(2, mask)), "incorrect binary left upper bound")
        require(
            right == _radius_upper(_matrix(2, 255 ^ mask)), "incorrect binary right upper bound"
        )
        require(left + right <= 3 and left * right <= Fraction(9, 4), "binary sanity bound failed")
    require(hashlib.sha256(_canonical(rows)).hexdigest() == data["rows_sha256"], "binary rows hash")


def _degenerate(data):
    require(data["zero_radius_masks_q2"] == [0, 255], "missing zero-radius cases")
    for mask in (0, 255):
        require(
            _radius_upper(_matrix(2, mask)) * _radius_upper(_matrix(2, 255 ^ mask)) == 0,
            "zero-radius case failed",
        )
    case = data["zero_inner_product"]
    require(
        case
        == {
            "q": 2,
            "u_hex": "0x45",
            "complement_hex": "0xba",
            "left_right_eigenvector": [1, 0, 0, 0],
            "complement_left_eigenvector": [0, 0, 0, 1],
            "radii": [1, 1],
            "inner_product": 0,
        },
        "zero-inner-product case altered",
    )
    a, b = _matrix(2, 0x45), _matrix(2, 0xBA)
    x, p = case["left_right_eigenvector"], case["complement_left_eigenvector"]
    require(_apply(a, x) == x and _apply(_transpose(b), p) == p, "reducible eigenvector check")
    require(_radius_upper(a) == _radius_upper(b) == 1, "reducible exact radii check")
    require(sum(v * w for v, w in zip(x, p)) == 0, "p^T x must vanish in this sanity case")


def _check(data):
    require(data["schema"] == "dna-m3-analytic-certificate-v1", "unknown analytic schema")
    require(
        data["parameters"]
        == {
            "q": 4,
            "m": 3,
            "alphabet": ["A", "C", "G", "T"],
            "theta": [3, 2, 1, 0],
            "word_order": "lexicographic base-q integer; bit q^2*a+q*b+c selects (a,b,c)",
        },
        "wrong DNA parameters, alphabet labels, or involution",
    )
    require(
        data["claim"]
        == {
            "optimization_domain": "all 64 independent triple-edge bits; arbitrary U",
            "objective": "rho(U)*rho(Uperp), Uperp=Sigma^3\\RC_theta(U)",
            "global_method": "human-readable analytic proof plus exact integer equality witness",
            "general_sum_upper": "3*q/2",
            "general_product_upper": "9*q^2/16",
            "dna_lower": 9,
            "dna_upper": 9,
            "all_dna_assignments_enumerated": False,
            "proof_assistant_formalized": False,
            "all_m_theorem_claimed": False,
            "legacy_search_frontiers_closed": False,
            "capacity_scope": "exact asymptotic input-constraint capacity, not finite counts or FER",
        },
        "incorrect theorem claim or evidence scope",
    )
    proof = data["proof"]
    require(isinstance(proof["text"], str), "missing human-readable proof")
    require(
        hashlib.sha256(proof["text"].encode()).hexdigest() == proof["sha256"] == KNOWN_PROOF_SHA256,
        "unrecognized or modified analytic proof text; re-review required",
    )
    require(
        proof["attribution"]
        == {
            "title": "Generalized Nordhaus--Gaddum Inequalities for Eigenvalues",
            "authors": [
                "Sahil Agarwal",
                "Carter Antley",
                "Joseph Aulenbacher",
                "George Brooks",
                "Ian Gonzalez",
                "Luke Hawranick",
                "William Linz",
                "Linyuan Lu",
                "Aiden Williams",
            ],
            "year": 2026,
            "arxiv_id": "2607.15941v2",
            "url": "https://arxiv.org/abs/2607.15941",
            "location": "Lemma 12, PDF pages 5-6",
            "role": "deficit-vector technique adaptation, not a direct source theorem",
        },
        "incorrect attribution or source theorem scope",
    )
    _elimination(proof["elimination"])
    words = list(product(range(4), repeat=3))
    states = list(product(range(4), repeat=2))
    state_index = {state: i for i, state in enumerate(states)}
    word_index = {word: i for i, word in enumerate(words)}
    theta = (3, 2, 1, 0)
    permutation = [state_index[(theta[b], theta[a])] for a, b in states]
    edge_permutation = [word_index[tuple(theta[a] for a in w[::-1])] for w in words]
    require(
        data["bridge"]
        == {
            "identity": "M_RC(S)=R*M_S^T*R^-1; Uperp=RC(complement U)",
            "state_permutation": permutation,
            "edge_permutation": edge_permutation,
            "all_single_edge_bases_checked": 64,
        },
        "incorrect RC transpose bridge",
    )
    for word, rc_edge in zip(words, edge_permutation):
        i, j = state_index[word[:2]], state_index[word[1:]]
        rc_word = words[rc_edge]
        require(
            state_index[rc_word[:2]] == permutation[j]
            and state_index[rc_word[1:]] == permutation[i],
            "RC transpose identity fails on a single-edge basis",
        )
    selected = sum(1 << i for i, word in enumerate(words) if all(a != 3 for a in word))
    complement = ((1 << 64) - 1) ^ selected
    rc_selected = sum(1 << edge_permutation[i] for i in range(64) if selected >> i & 1)
    perp = ((1 << 64) - 1) ^ rc_selected
    expected_perp = sum(1 << i for i, word in enumerate(words) if 0 in word)
    require(perp == expected_perp, "actual Uperp must contain A, not T")
    witness = data["witness"]
    masks = {"u": selected, "complement": complement, "rc_u": rc_selected, "uperp": perp}
    for name, mask in masks.items():
        require(witness[name + "_hex"] == hex(mask), "incorrect witness mask")
    alphabet = "ACGT"
    for name in ("u", "complement", "uperp"):
        expected = [
            "".join(alphabet[a] for a in word)
            for i, word in enumerate(words)
            if masks[name] >> i & 1
        ]
        require(witness[name + "_words"] == expected, "incorrect labelled witness windows")
    d, a, b, c = [_matrix(4, mask) for mask in ((1 << 64) - 1, selected, complement, perp)]
    for name, matrix in (("full", d), ("u", a), ("complement", b), ("uperp", c)):
        _checked_matrix(witness["matrix_" + name], matrix)
    require(_multiply(d, d) == [[1] * 16 for _ in range(16)], "D^2=J failed")
    for i in range(16):
        for j in range(16):
            require(c[permutation[j]][permutation[i]] == b[i][j], "witness transpose bridge")
    x = _vector(witness["left_nonnegative_eigenvector"], 16, positive=False)
    ordinary = _vector(witness["complement_right_positive_eigenvector"], 16, positive=True)
    actual = _vector(witness["uperp_right_positive_eigenvector"], 16, positive=True)
    require(witness["complement_required_letter"] == "T", "ordinary complement label must be T")
    require(witness["uperp_required_letter"] == "A", "actual Uperp label must be A")
    require(
        max(map(sum, a)) == 3 and _apply(a, x) == [3 * value for value in x],
        "left radius-three eigenvector or row upper bound failed",
    )
    require(
        _apply(b, ordinary) == [3 * value for value in ordinary],
        "ordinary complement right eigenvector failed",
    )
    require(
        _apply(c, actual) == [3 * value for value in actual],
        "actual Uperp right eigenvector failed",
    )
    require(witness["radius_each"] == 3 and witness["product"] == 9, "wrong exact product")
    for name, matrix in (("u", a), ("complement", b), ("uperp", c)):
        actual_polynomial = _characteristic_polynomial(matrix)
        expected = [1, -3] + [0] * 15 if name == "u" else [1, -1, -3, -9] + [0] * 13
        require(
            witness["charpoly_" + name] == actual_polynomial == expected,
            "incorrect characteristic polynomial",
        )
    _binary(data["binary_sanity"])
    _degenerate(data["degenerate_sanity"])
    return {
        "status": "EXACT_FINITE_CHECKS_PASS_ANALYTIC_PROOF_REQUIRED",
        "global_conclusion_basis": "human-readable analytic theorem plus exact integer witness",
        "dna_product_lower": 9,
        "dna_product_upper": 9,
        "binary_masks_checked": 256,
        "rc_single_edge_bases_checked": 64,
        "dna_assignments_exhaustively_enumerated": False,
        "formal_proof_assistant": False,
        "actual_uperp_required_letter": "A",
        "ordinary_complement_required_letter": "T",
        "proof_sha256": KNOWN_PROOF_SHA256,
        "canonical_certificate_sha256": hashlib.sha256(_canonical(data)).hexdigest(),
    }


def check_certificate(data):
    """Audit finite evidence; does not mechanically prove a quantified theorem."""
    try:
        _exact_tree(data)
        return _check(data)
    except (KeyError, TypeError, IndexError, AttributeError, ZeroDivisionError) as error:
        raise ValueError("malformed analytic certificate") from error


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        require(args.certificate.stat().st_size <= 4 * 1024**2, "certificate exceeds 4 MiB")
        payload = args.certificate.read_bytes()
        result = check_certificate(json.loads(payload, object_pairs_hook=_unique_pairs))
        result["certificate_file_sha256"] = hashlib.sha256(payload).hexdigest()
        text = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.output:
            with args.output.open("x") as stream:
                stream.write(text)
        print(text, end="")
    except (ValueError, OSError) as error:
        parser.exit(1, f"analytic certificate refused: {error}\n")


if __name__ == "__main__":
    main()
