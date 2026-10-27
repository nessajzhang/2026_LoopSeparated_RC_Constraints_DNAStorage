"""Analytic m=3 certificate producer; no exhaustive DNA-domain search.

The global conclusion uses the human-readable proof below. Exact finite
identities, a DNA equality witness, and an exhaustive binary sanity check
are accompanying audits, not a proof-assistant formalization.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from itertools import product

SCHEMA = "dna-m3-analytic-certificate-v1"
ALPHABET = ("A", "C", "G", "T")
THETA = (3, 2, 1, 0)
WORD_ORDER = "lexicographic base-q integer; bit q^2*a+q*b+c selects (a,b,c)"

ANALYTIC_PROOF = """Analytic m=3 two-colour overlap theorem

Let D be the row-to-column overlap matrix on ordered pairs over q>=2
letters: D[(a,b),(b,c)]=1. For any nonnegative matrices A+B=D, write
lambda=rho(A), mu=rho(B). Then lambda+mu<=3q/2 and
lambda*mu<=9q^2/16. Fractional as well as binary edge partitions are covered.

Every pair of states has exactly one two-step path, so D^2=J. Each row of
D sums to q. If lambda*mu=0, monotonicity gives lambda+mu<=q. Otherwise
choose nonnegative nonzero Perron vectors Ax=lambda*x and p^T B=mu*p^T.
They need not be strictly positive. Put X=sum x_ab>0, Y=sum p_ab>0,
h_b=sum_c x_bc, k_b=sum_a p_ab, S=lambda+mu, z=lambda*mu, and
T=p^T D x/(XY)=sum_b (h_b/X)(k_b/Y).

Since lambda*x_ab<=h_b, summing over b gives lambda*h_a<=X. Similarly
mu*p_bc<=k_b gives mu*k_c<=Y. Thus u_b=lambda*h_b/X and
v_b=mu*k_b/Y belong to [0,1], with sums lambda and mu. Summing
u_b*v_b>=u_b+v_b-1 gives T>=(S-q)/z.

Set d=Dx/lambda-x=Bx/lambda>=0. We have p^T d=mu*p^T x/lambda and
p^T D x=S*p^T x. The componentwise inequality mu*p^T<=p^T D gives
mu*p^T d<=p^T Dd=XY/lambda-p^T D x. Substitution yields
T<=S/(lambda^2+lambda*mu+mu^2)=S/(S^2-z).
No step divides by p^T x, which may be zero for reducible matrices.

If S<=q the sum bound is immediate. If S>q, combining the two T bounds
gives (S-q)S^2<=(2S-q)z. Since z<=S^2/4 and 2S-q>0, it follows that
S-q<=(2S-q)/4, hence S<=3q/2. Then z<=S^2/4<=9q^2/16.
The elimination can also be checked by the polynomial identity
4*((S-q)*S^2-(2*S-q)*lambda*mu)
=S^2*(2*S-3*q)+(2*S-q)*(lambda-mu)^2.

For any alphabet involution theta, reverse complementation maps an edge
i->j to R(j)->R(i), where R(a,b)=(theta(b),theta(a)). Consequently
M_RC(S)=R M_S^T R^-1 and rho(Uperp)=rho(complement U), because
Uperp=RC(complement U). This transfers the theorem to the original objective.

For DNA in A,C,G,T order, take U={A,C,G}^3. Its matrix has maximum row
sum 3 and a nonnegative nonzero eigenvector of eigenvalue 3. The actual
Uperp consists of all triples containing A. Its strictly positive right
eigenvector is 3 on states ending in A, 2 on states with first letter A
and last letter not A, and 1 otherwise. Direct multiplication gives
eigenvalue 3. The ordinary complement instead consists of triples
containing T and uses the same right-vector rule with T in place of A.
Thus the original DNA maximum over all 64 independent edge bits equals 9.

Attribution: the deficit-vector technique is adapted from Agarwal et al.,
Generalized Nordhaus--Gaddum Inequalities for Eigenvalues,
arXiv:2607.15941v2 (2026), Lemma 12, PDF pages 5-6. That source treats
symmetric matrices summing to J; it does not state the directed overlap
theorem proved here. The left Perron vector, D^2=J, and the one-letter
marginal caps are the additional ingredients.

Scope: this is a human-readable analytic proof, not enumeration of 2^64
DNA assignments and not a formal proof-assistant certificate. The separate
binary exhaustive audit and integer DNA witness check finite identities.
The all-m analogue is not claimed. Exact asymptotic constrained capacity
does not imply a full finite codebook count, IDS reliability, or a
biochemical advantage.
"""


def canonical_bytes(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _matrix(q, mask):
    n = q * q
    matrix = [[0] * n for _ in range(n)]
    for edge in range(q**3):
        if mask >> edge & 1:
            matrix[edge // q][edge % n] += 1
    return matrix


def _apply(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def _components(matrix):
    """Kosaraju traversal; the checker instead uses Boolean closure."""
    n = len(matrix)
    seen, order = set(), []

    def forward(v):
        seen.add(v)
        for w, value in enumerate(matrix[v]):
            if value and w not in seen:
                forward(w)
        order.append(v)

    for vertex in range(n):
        if vertex not in seen:
            forward(vertex)
    seen.clear()
    components = []

    def backward(v, component):
        seen.add(v)
        component.append(v)
        for w in range(n):
            if matrix[w][v] and w not in seen:
                backward(w, component)

    for vertex in reversed(order):
        if vertex not in seen:
            component = []
            backward(vertex, component)
            components.append(sorted(component))
    return components


def _radius_upper(matrix, steps=64):
    maximum = Fraction(0)
    for component in _components(matrix):
        if len(component) == 1 and not matrix[component[0]][component[0]]:
            continue
        block = [[matrix[i][j] for j in component] for i in component]
        vector = [1] * len(component)
        for _ in range(steps):
            vector = [x + y for x, y in zip(vector, _apply(block, vector))]
        maximum = max(maximum, max(Fraction(y, x) for x, y in zip(vector, _apply(block, vector))))
    return maximum


def _fraction(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def _rc_mask(q, theta, mask):
    result = 0
    for a, b, c in product(range(q), repeat=3):
        if mask >> (q * q * a + q * b + c) & 1:
            result |= 1 << (q * q * theta[c] + q * theta[b] + theta[a])
    return result


def _words(mask):
    return [
        "".join(ALPHABET[a] for a in word)
        for edge, word in enumerate(product(range(4), repeat=3))
        if mask >> edge & 1
    ]


def _polynomial_identity():
    # Monomials are [lambda exponent, mu exponent, q exponent, coefficient].
    return {
        "variables": ["lambda", "mu", "q"],
        "identity": (
            "4*((S-q)*S^2-(2*S-q)*lambda*mu)=S^2*(2*S-3*q)+(2*S-q)*(lambda-mu)^2; S=lambda+mu"
        ),
        "coefficients": [
            [0, 2, 1, -4],
            [0, 3, 0, 4],
            [1, 1, 1, -4],
            [1, 2, 0, 4],
            [2, 0, 1, -4],
            [2, 1, 0, 4],
            [3, 0, 0, 4],
        ],
    }


def _binary_sanity():
    rows = []
    for mask in range(256):
        left = _radius_upper(_matrix(2, mask))
        right = _radius_upper(_matrix(2, 255 ^ mask))
        if left + right > 3 or left * right > Fraction(9, 4):
            raise ArithmeticError("exact binary sanity check failed")
        rows.append({"mask": mask, "left_upper": _fraction(left), "right_upper": _fraction(right)})
    return {
        "q": 2,
        "m": 3,
        "all_arbitrary_masks": 256,
        "symmetry_reduction": False,
        "vector_steps": 64,
        "rows": rows,
        "rows_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
    }


def analytic_certificate():
    """Return deterministic JSON-ready analytic evidence for DNA m=3.

    Run ``code/check_m3_analytic_certificate.py`` independently before use.
    The checker audits finite identities; the general bound is the proof
    above, which must be read and assessed as mathematics.
    """
    full = (1 << 64) - 1
    selected = sum(1 << (16 * a + 4 * b + c) for a, b, c in product(range(3), repeat=3))
    complement = full ^ selected
    rc_selected = _rc_mask(4, THETA, selected)
    perp = full ^ rc_selected
    states = list(product(range(4), repeat=2))

    def right_vector(letter):
        return [3 if b == letter else 2 if a == letter else 1 for a, b in states]

    return {
        "schema": SCHEMA,
        "parameters": {
            "q": 4,
            "m": 3,
            "alphabet": list(ALPHABET),
            "theta": list(THETA),
            "word_order": WORD_ORDER,
        },
        "claim": {
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
        "proof": {
            "text": ANALYTIC_PROOF,
            "sha256": hashlib.sha256(ANALYTIC_PROOF.encode()).hexdigest(),
            "elimination": _polynomial_identity(),
            "attribution": {
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
        },
        "bridge": {
            "identity": "M_RC(S)=R*M_S^T*R^-1; Uperp=RC(complement U)",
            "state_permutation": [4 * THETA[b] + THETA[a] for a, b in states],
            "edge_permutation": [
                16 * THETA[c] + 4 * THETA[b] + THETA[a] for a, b, c in product(range(4), repeat=3)
            ],
            "all_single_edge_bases_checked": 64,
        },
        "witness": {
            "u_hex": hex(selected),
            "complement_hex": hex(complement),
            "rc_u_hex": hex(rc_selected),
            "uperp_hex": hex(perp),
            "u_words": _words(selected),
            "complement_words": _words(complement),
            "uperp_words": _words(perp),
            "matrix_full": _matrix(4, full),
            "matrix_u": _matrix(4, selected),
            "matrix_complement": _matrix(4, complement),
            "matrix_uperp": _matrix(4, perp),
            "left_nonnegative_eigenvector": [int(a < 3 and b < 3) for a, b in states],
            "complement_right_positive_eigenvector": right_vector(3),
            "uperp_right_positive_eigenvector": right_vector(0),
            "complement_required_letter": "T",
            "uperp_required_letter": "A",
            "radius_each": 3,
            "product": 9,
            "charpoly_u": [1, -3] + [0] * 15,
            "charpoly_complement": [1, -1, -3, -9] + [0] * 13,
            "charpoly_uperp": [1, -1, -3, -9] + [0] * 13,
        },
        "degenerate_sanity": {
            "zero_radius_masks_q2": [0, 255],
            "zero_inner_product": {
                "q": 2,
                "u_hex": "0x45",
                "complement_hex": "0xba",
                "left_right_eigenvector": [1, 0, 0, 0],
                "complement_left_eigenvector": [0, 0, 0, 1],
                "radii": [1, 1],
                "inner_product": 0,
            },
        },
        "binary_sanity": _binary_sanity(),
    }
