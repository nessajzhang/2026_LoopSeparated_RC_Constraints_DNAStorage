"""Exact finite certificates for arbitrary two-colour overlap graphs.

The DNA m=3 domain has 64 independent edge bits, not 32 RC-orbit choices.
Floating-point values are never used to discard a search cylinder.  A bound
stores ``rho(U) rho(complement U) <= value**(1/power)`` with exact integers.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import random
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations, product

from .regimes import strongly_connected_components

DNA_THETA = (3, 2, 1, 0)


@dataclass(frozen=True)
class PowerBound:
    value: Fraction
    power: int = 1

    def __post_init__(self):
        if self.value < 0 or self.power < 1:
            raise ValueError("a power bound must be nonnegative with positive exponent")

    def __lt__(self, other):
        common = math.lcm(self.power, other.power)
        return self.value ** (common // self.power) < other.value ** (common // other.power)

    def __le__(self, other):
        return self == other or self < other or not other < self

    def as_dict(self):
        return {
            "numerator": self.value.numerator,
            "denominator": self.value.denominator,
            "power": self.power,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(Fraction(data["numerator"], data["denominator"]), data["power"])

    def approximate(self):
        """Display only; never a pruning input."""
        return float(self.value) ** (1 / self.power)


@dataclass(frozen=True)
class WindowGraph:
    q: int = 4
    m: int = 3

    def __post_init__(self):
        if self.q < 2 or self.m < 1:
            raise ValueError("require q>=2, m>=1")

    @property
    def edge_count(self):
        return self.q**self.m

    @property
    def state_count(self):
        return self.q ** (self.m - 1)

    @property
    def full_mask(self):
        return (1 << self.edge_count) - 1

    @property
    def words(self):
        return tuple(product(range(self.q), repeat=self.m))

    def check_mask(self, mask):
        if not isinstance(mask, int) or not 0 <= mask <= self.full_mask:
            raise ValueError("mask is outside the full arbitrary-window domain")
        return mask

    def matrix(self, mask):
        self.check_mask(mask)
        rows = [[0] * self.state_count for _ in range(self.state_count)]
        for edge in range(self.edge_count):
            if mask >> edge & 1:
                rows[edge // self.q][edge % self.state_count] += 1
        return tuple(map(tuple, rows))

    def mask(self, words):
        mask = 0
        for word in words:
            if len(word) != self.m:
                raise ValueError("wrong word length")
            edge = 0
            for a in word:
                if not isinstance(a, int) or not 0 <= a < self.q:
                    raise ValueError("invalid alphabet symbol")
                edge = self.q * edge + a
            mask |= 1 << edge
        return mask

    def selected_words(self, mask):
        self.check_mask(mask)
        return tuple(word for edge, word in enumerate(self.words) if mask >> edge & 1)

    def transform(self, mask, permutation=None, reversal=False, colour_swap=False):
        self.check_mask(mask)
        permutation = tuple(range(self.q)) if permutation is None else tuple(permutation)
        if sorted(permutation) != list(range(self.q)):
            raise ValueError("not an alphabet permutation")
        words = self.selected_words(mask)
        transformed = self.mask(
            [tuple(permutation[a] for a in (w[::-1] if reversal else w)) for w in words]
        )
        return self.full_mask ^ transformed if colour_swap else transformed

    def perp(self, mask, theta=DNA_THETA):
        theta = tuple(theta)
        if len(theta) != self.q or any(theta[theta[a]] != a for a in range(self.q)):
            raise ValueError("invalid involution")
        return self.transform(mask, theta, reversal=True, colour_swap=True)


def matrix_apply(matrix, vector):
    return tuple(sum(a * vector[j] for j, a in enumerate(row) if a) for row in matrix)


def power_apply(matrix, vector, exponent):
    result = tuple(vector)
    for _ in range(exponent):
        result = matrix_apply(matrix, result)
    return result


def cardinality_bound(a_count, b_count, edge_count, m):
    if a_count < 0 or b_count < 0 or a_count + b_count > edge_count:
        raise ValueError("infeasible assigned cardinalities")
    low, high = a_count, edge_count - b_count
    s = min(high, max(low, edge_count // 2))
    return PowerBound(Fraction(s * (edge_count - s)), m)


def scc_collatz_certificate(matrix, exponent=4, steps=8):
    """Positive integer vectors on every cyclic SCC; all arithmetic exact."""

    if exponent < 1 or steps < 0:
        raise ValueError("invalid Collatz parameters")
    blocks = []
    maximum = Fraction(0)
    for component in strongly_connected_components(matrix):
        if len(component) == 1 and not matrix[component[0]][component[0]]:
            continue
        block = tuple(tuple(matrix[i][j] for j in component) for i in component)
        vector = (1,) * len(component)
        for _ in range(steps):
            image = matrix_apply(block, vector)
            vector = tuple(x + y for x, y in zip(vector, image))
        image = power_apply(block, vector, exponent)
        bound = max(Fraction(y, x) for x, y in zip(vector, image))
        maximum = max(maximum, bound)
        blocks.append({"vertices": list(component), "vector": list(vector)})
    return {"exponent": exponent, "components": blocks}, maximum


def finite_word_count(graph, mask, length):
    """Number N_length of all-window-valid words, including short lengths."""

    if length < 0:
        raise ValueError("negative word length")
    if length < graph.m:
        return graph.q**length
    vector = (1,) * graph.state_count
    matrix = graph.matrix(mask)
    return sum(power_apply(matrix, vector, length - graph.m + 1))


def n4_identity(graph, mask):
    if graph.q != 4 or graph.m != 3:
        raise ValueError("the N4 identity is for q=4,m=3")
    matrix = graph.matrix(mask)
    indegrees = [sum(row[j] for row in matrix) for j in range(16)]
    outdegrees = list(map(sum, matrix))
    value = sum(i * o for i, o in zip(indegrees, outdegrees))
    return value, value + 256 - 8 * mask.bit_count()


def node_proofs(graph, a, b, *, exponent=4, steps=8, include_collatz=True):
    graph.check_mask(a)
    graph.check_mask(b)
    if a & b:
        raise ValueError("A and B must be disjoint")
    bound = cardinality_bound(a.bit_count(), b.bit_count(), graph.edge_count, graph.m)
    proofs = [{"kind": "cardinality"}]
    if graph.q == 4 and graph.m == 3:
        # The strictly improved universal integer bound is proved separately;
        # it is deliberately not advertised as substantive m=3 optimization.
        integer_bound = PowerBound(Fraction(1023), 3)
        proofs.append({"kind": "dna_m3_integer_gap"})
        bound = min(bound, integer_bound)
    if include_collatz:
        left, lval = scc_collatz_certificate(graph.matrix(graph.full_mask ^ b), exponent, steps)
        right, rval = scc_collatz_certificate(graph.matrix(graph.full_mask ^ a), exponent, steps)
        candidate = PowerBound(lval * rval, exponent)
        if candidate < bound:
            proofs.append({"kind": "scc_collatz_product", "left": left, "right": right})
            bound = candidate
    return proofs, bound


def structured_candidate(graph=WindowGraph()):
    if graph.q != 4 or graph.m != 3:
        raise ValueError("this explicit algebraic witness is DNA m=3")
    mask = graph.mask(product(range(3), repeat=3))
    complement = graph.full_mask ^ mask
    # For the graph of words containing symbol 3, the suffix run of non-3
    # symbols has weights 3,2,1.  This vector is strictly positive on all 16
    # states, and Mx=3x exactly.
    vector = [3 if b == 3 else 2 if a == 3 else 1 for a, b in product(range(4), repeat=2)]
    if matrix_apply(graph.matrix(complement), vector) != tuple(3 * x for x in vector):
        raise AssertionError("structured positive eigenvector is invalid")
    selected = [list(w) for w in graph.selected_words(mask)]
    matrix = [list(row) for row in graph.matrix(mask)]
    other = [list(row) for row in graph.matrix(complement)]
    digest = hashlib.sha256(
        json.dumps(
            {"windows": selected, "matrix": matrix, "complement_matrix": other},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "kind": "structured_ternary_cube",
        "mask_hex": hex(mask),
        "windows": selected,
        "matrix": matrix,
        "complement_matrix": other,
        "complement_positive_eigenvector": vector,
        "radius_each": 3,
        "product_lower": {"numerator": 9, "denominator": 1},
        "left_characteristic_polynomial": "t^15*(t-3)",
        "complement_characteristic_polynomial": "t^13*(t^3-t^2-3*t-9)",
        "cubic_factorization": "(t-3)*(t^2+2*t+3)",
        "sha256": digest,
    }


def rational_candidate_certificate(graph, mask, steps=64):
    """Exact principal-block Collatz lower bounds for an arbitrary candidate."""
    if steps < 0:
        raise ValueError("negative candidate-vector iteration count")
    certificate = {"kind": "rational_collatz", "mask_hex": hex(mask)}
    factors = []
    for side, selected in (("left", mask), ("right", graph.full_mask ^ mask)):
        matrix = graph.matrix(selected)
        best, best_component, best_vector = Fraction(0), (0,), (1,)
        for component in strongly_connected_components(matrix):
            block = tuple(tuple(matrix[i][j] for j in component) for i in component)
            vector = (1,) * len(component)
            for _ in range(steps):
                image = matrix_apply(block, vector)
                vector = tuple(x + y for x, y in zip(vector, image))
            image = matrix_apply(block, vector)
            lower = min(Fraction(y, x) for x, y in zip(vector, image))
            if lower >= best:
                best, best_component, best_vector = lower, component, vector
        factors.append(best)
        certificate[side + "_component"] = list(best_component)
        certificate[side + "_vector"] = list(best_vector)
    lower = factors[0] * factors[1]
    certificate["product_lower"] = {"numerator": lower.numerator, "denominator": lower.denominator}
    certificate["windows"] = [list(w) for w in graph.selected_words(mask)]
    certificate["matrix"] = [list(row) for row in graph.matrix(mask)]
    certificate["complement_matrix"] = [list(row) for row in graph.matrix(graph.full_mask ^ mask)]
    return certificate


def explore_candidates(seed=20261023, restarts=8, steps=256):
    """Small multi-start hill/annealing exploration; floats order candidates only.

    The return object labels all floating scores as exploratory.  Every retained
    candidate also has an integer-vector rational lower certificate.  No search
    cylinder is discarded by this routine and it makes no global-optimum claim.
    """
    import numpy as np

    graph = WindowGraph()
    rng = random.Random(seed)
    baseline = structured_candidate(graph)
    baseline_mask = int(baseline["mask_hex"], 16)
    scores = {baseline_mask: 9.0}
    evaluations = 0
    trajectories = []

    def score(mask):
        nonlocal evaluations
        if mask not in scores:
            left = np.asarray(graph.matrix(mask), dtype=float)
            right = np.asarray(graph.matrix(graph.full_mask ^ mask), dtype=float)
            scores[mask] = float(
                max(abs(np.linalg.eigvals(left))) * max(abs(np.linalg.eigvals(right)))
            )
            evaluations += 1
        return scores[mask]

    for restart in range(restarts):
        strategy = "hill_climb" if restart % 2 == 0 else "annealing"
        mask = baseline_mask if restart < 2 else rng.getrandbits(64)
        current = score(mask)
        best = current
        for step in range(steps):
            proposal = mask ^ (1 << rng.randrange(64))
            value = score(proposal)
            temperature = 0.1 * (1 - step / steps)
            if value > current or (
                strategy == "annealing"
                and rng.random() < math.exp(min(0, (value - current) / max(temperature, 1e-12)))
            ):
                mask, current = proposal, value
            best = max(best, current)
        trajectories.append(
            {
                "restart": restart,
                "strategy": strategy,
                "best_float_exploratory": best,
                "final_mask_hex": hex(mask),
            }
        )
    retained = sorted(scores, key=lambda mask: scores[mask], reverse=True)[:8]
    if baseline_mask not in retained:
        retained.append(baseline_mask)
    candidates = []
    certified_best = baseline
    certified_lower = Fraction(9)
    for mask in retained:
        candidate = (
            baseline if mask == baseline_mask else rational_candidate_certificate(graph, mask)
        )
        lower = Fraction(**candidate["product_lower"])
        candidates.append(
            {
                "mask_hex": hex(mask),
                "floating_score_exploratory_only": scores[mask],
                "lower_numerator": lower.numerator,
                "lower_denominator": lower.denominator,
                "certificate": candidate,
            }
        )
        if lower > certified_lower:
            certified_best, certified_lower = candidate, lower
    return {
        "seed": seed,
        "restarts": restarts,
        "steps_each": steps,
        "evaluations": evaluations,
        "strategies": ["hill_climb", "annealing"],
        "trajectories": trajectories,
        "retained_candidates": candidates,
        "best_certified": certified_best,
        "global_optimality_evidence": False,
        "warning": "floating spectral scores are candidate exploration only, never pruning or proof",
    }


def initial_search_certificate(q=4, m=3, threshold=Fraction(2253, 250), symmetry=True):
    graph = WindowGraph(q, m)
    if q != 4 or m != 3:
        lower = {
            "kind": "rational_collatz",
            "mask_hex": hex(1),
            "left_component": [0],
            "left_vector": [1],
            "right_component": [graph.state_count - 1],
            "right_vector": [1],
            "product_lower": {"numerator": 1, "denominator": 1},
        }
    else:
        lower = structured_candidate(graph)
    root_proofs, bound = node_proofs(graph, 0, 0, include_collatz=False)
    root = {
        "id": 0,
        "a": "0x0",
        "b": "0x0",
        "kind": "frontier",
        "proofs": root_proofs,
        "upper": bound.as_dict(),
    }
    certificate = {
        "schema": "arbitrary-window-frontier-v1",
        "q": q,
        "m": m,
        "word_order": "lexicographic base-q integer, bit i is the independent i-th window",
        "root": 0,
        "nodes": [root],
        "incumbent": lower,
        "prune_threshold": {"numerator": threshold.numerator, "denominator": threshold.denominator},
        "stats": {"nodes_processed": 0, "wall_seconds": 0.0},
    }
    if symmetry:
        root.update({"kind": "split", "edge": 0, "children": [1, 2]})
        certificate["nodes"].extend(
            [
                {
                    "id": 1,
                    "a": "0x1",
                    "b": "0x0",
                    "kind": "frontier",
                    "proofs": [{"kind": "inherit", "ref": 0}, {"kind": "cardinality"}],
                    "upper": bound.as_dict(),
                },
                {
                    "id": 2,
                    "a": "0x0",
                    "b": "0x1",
                    "kind": "symmetry",
                    "ref": 1,
                    "permutation": list(range(q)),
                    "reversal": False,
                    "colour_swap": True,
                },
            ]
        )
    return certificate


def _branch_edge(graph, a, b):
    unassigned = graph.full_mask ^ (a | b)
    left = graph.matrix(graph.full_mask ^ b)
    right = graph.matrix(graph.full_mask ^ a)
    left_out, right_out = list(map(sum, left)), list(map(sum, right))
    left_in = [sum(row[j] for row in left) for j in range(graph.state_count)]
    right_in = [sum(row[j] for row in right) for j in range(graph.state_count)]
    return max(
        (e for e in range(graph.edge_count) if unassigned >> e & 1),
        key=lambda e: (
            left_in[e // graph.q] * left_out[e % graph.state_count]
            + right_in[e // graph.q] * right_out[e % graph.state_count],
            e // graph.q == e % graph.state_count,
            -e,
        ),
    )


def extend_search(certificate, *, max_nodes=1000, max_seconds=30.0, exponent=4, steps=8):
    """Bounded DFS with every queued sibling serialized as a frontier node.

    A node remains in the frontier until its bound is computed and both
    children exist.  KeyboardInterrupt returns a complete checkpoint too.
    This is the transparent reference producer, not a claim of rapid closure.
    """

    graph = WindowGraph(certificate["q"], certificate["m"])
    nodes = certificate["nodes"]
    if [node["id"] for node in nodes] != list(range(len(nodes))):
        raise ValueError("producer requires contiguous, unique node ids")
    threshold = PowerBound(Fraction(**certificate["prune_threshold"]), 1)
    queue = sorted((node["id"] for node in nodes if node["kind"] == "frontier"), reverse=True)
    start = time.monotonic()
    count = 0
    try:
        while queue and count < max_nodes and time.monotonic() - start < max_seconds:
            if count % 64 == 0:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_bytes = rss if sys.platform == "darwin" else rss * 1024
                if rss_bytes > 3 * 1024**3:
                    certificate["memory_guard_stopped"] = True
                    break
            node_id = queue.pop()
            node = nodes[node_id]
            a, b = int(node["a"], 16), int(node["b"], 16)
            proofs, bound = node_proofs(graph, a, b, exponent=exponent, steps=steps)
            old_bound = PowerBound.from_dict(node["upper"])
            if old_bound < bound:
                proofs = node["proofs"]
                bound = old_bound
            node["proofs"], node["upper"] = proofs, bound.as_dict()
            count += 1
            if bound <= threshold:
                node["kind"] = "pruned"
                continue
            if (a | b) == graph.full_mask:
                node["kind"] = "complete_assignment"
                continue
            edge = _branch_edge(graph, a, b)
            bit = 1 << edge
            first_id = len(nodes)
            children = [
                {
                    "id": first_id,
                    "a": hex(a | bit),
                    "b": hex(b),
                    "kind": "frontier",
                    "proofs": [{"kind": "inherit", "ref": node_id}, {"kind": "cardinality"}],
                    "upper": min(
                        bound,
                        cardinality_bound(
                            (a | bit).bit_count(), b.bit_count(), graph.edge_count, graph.m
                        ),
                    ).as_dict(),
                },
                {
                    "id": first_id + 1,
                    "a": hex(a),
                    "b": hex(b | bit),
                    "kind": "frontier",
                    "proofs": [{"kind": "inherit", "ref": node_id}, {"kind": "cardinality"}],
                    "upper": min(
                        bound,
                        cardinality_bound(
                            a.bit_count(), (b | bit).bit_count(), graph.edge_count, graph.m
                        ),
                    ).as_dict(),
                },
            ]
            # Both children are created before the parent ceases being frontier.
            nodes.extend(children)
            node.update({"kind": "split", "edge": edge, "children": [first_id, first_id + 1]})
            queue.extend([first_id + 1, first_id])
    except KeyboardInterrupt:
        certificate["interrupted"] = True
    stats = certificate["stats"]
    stats["nodes_processed"] += count
    stats["wall_seconds"] += time.monotonic() - start
    stats["last_run_nodes"] = count
    stats["last_run_seconds"] = time.monotonic() - start
    stats["frontier_nodes"] = sum(node["kind"] == "frontier" for node in nodes)
    stats["pruned_nodes"] = sum(node["kind"] == "pruned" for node in nodes)
    stats["stored_nodes"] = len(nodes)
    return certificate


def symmetry_group(graph):
    """The proved group only: global alphabet permutations, reversal, colour swap."""
    return tuple(
        (p, rev, swap)
        for p in permutations(range(graph.q))
        for rev in (False, True)
        for swap in (False, True)
    )
