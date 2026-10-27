"""Exact four-letter dinucleotide/end-point randomization, not a new method.

Last-exit trees are the Euler randomization construction used by uShuffle
(Jiang et al. 2008, doi:10.1186/1471-2105-9-192).  Here the at most 4**3
parent assignments are enumerated explicitly, with integer parallel-edge
weights.  No Wilson walk cutoff, floating-point probability or MCMC is used.

For root t, tree weight W(T)=prod_{v!=t} C[v,p(v)].  A valid word fixes T
and each outgoing neighbor order.  Its probability is

  prod_{v,b} C[v,b]! / (Z * d[t]! * prod_{v!=t}(d[v]-1)!),

where Z=sum_T W(T).  Each factor C[v,p(v)] cancels the one reserved edge
in the inverse multiset-order count.  The last-exit tree guarantees the
ordered traversal consumes every edge: if it stopped early at t, vertices
with unused edges would have unused reserved edges leading eventually to t,
contradicting balance of unused in/out degrees.  Thus local orders and words
are bijective.  Loops cannot be nonroot tree edges; repeated edges DO affect
weights.  Singleton/empty words have a single possible output.

Independent rejection on exact longest-run H is uniform on its conditional
class, also conditional on success before a fixed cap.  A cap failure is a
missing control, never permission to redraw the target.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby, product
from math import factorial, prod

DNA = "ACGT"


def validate_dna(word: str) -> None:
    if not isinstance(word, str) or set(word) - set(DNA):
        raise ValueError("sequence must be an uppercase ACGT string")


def longest_run(word: str) -> int:
    validate_dna(word)
    return max((sum(1 for _ in g) for _, g in groupby(word)), default=0)


def dinucleotide_counts(word: str) -> tuple[int, ...]:
    validate_dna(word)
    result = [0] * 16
    for a, b in zip(word, word[1:]):
        result[4 * DNA.index(a) + DNA.index(b)] += 1
    return tuple(result)


def signature(word: str) -> tuple[str, str, tuple[int, ...]]:
    return word[:1], word[-1:], dinucleotide_counts(word)


@dataclass(frozen=True)
class LastExitTree:
    parents: tuple[int, ...]  # -1 at root and inactive letters
    weight: int


@dataclass(frozen=True)
class ControlResult:
    status: str
    sequence: str | None
    attempts: int
    candidates_equal_target: int
    accepted_equal_target: bool | None


class EulerSampler:
    """Prepared composition class from a valid DNA word; exact integer draws."""

    def __init__(self, word: str):
        validate_dna(word)
        self.source = word
        self.counts = dinucleotide_counts(word)
        self.n = len(word)
        self.start = DNA.index(word[0]) if word else None
        self.root = DNA.index(word[-1]) if word else None
        self.degrees = tuple(sum(self.counts[4 * a : 4 * a + 4]) for a in range(4))
        self.active = tuple(sorted({DNA.index(c) for c in word}))
        nonroot = tuple(v for v in self.active if v != self.root)
        choices = [
            tuple(b for b in self.active if b != v and self.counts[4 * v + b]) for v in nonroot
        ]
        trees = []
        for selected in product(*choices):
            parents = [-1] * 4
            for v, b in zip(nonroot, selected, strict=True):
                parents[v] = b
            valid = True
            for v in nonroot:
                seen = set()
                while v != self.root:
                    if v in seen or v == -1:
                        valid = False
                        break
                    seen.add(v)
                    v = parents[v]
                if not valid:
                    break
            if valid:
                weight = prod(self.counts[4 * v + parents[v]] for v in nonroot)
                trees.append(LastExitTree(tuple(parents), weight))
        self.trees = tuple(trees)
        self.total_weight = sum(t.weight for t in trees)
        if not self.total_weight:
            raise AssertionError("valid source word has no root-reaching last-exit tree")

    def word_probability(self) -> Fraction:
        if self.n <= 1:
            return Fraction(1)
        numerator = prod(factorial(c) for c in self.counts)
        denominator = self.total_weight * factorial(self.degrees[self.root])
        denominator *= prod(factorial(self.degrees[v] - 1) for v in self.active if v != self.root)
        return Fraction(numerator, denominator)

    def sample(self, rng: random.Random) -> str:
        if self.n <= 1:
            return self.source
        ticket = rng.randrange(self.total_weight)
        for tree in self.trees:
            if ticket < tree.weight:
                break
            ticket -= tree.weight
        else:
            raise AssertionError("integer tree ticket exhausted weights")
        orders = []
        for v in range(4):
            row = [self.counts[4 * v + b] - (tree.parents[v] == b) for b in range(4)]
            neighbors = [b for b in range(4) for _ in range(row[b])]
            # Fisher-Yates on a labeled list induces a uniform multiset order:
            # every distinct order has exactly prod_b row[b]! preimages.
            for i in range(len(neighbors) - 1, 0, -1):
                j = rng.randrange(i + 1)
                neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
            if tree.parents[v] != -1:
                neighbors.append(tree.parents[v])
            orders.append(neighbors)
        used = [0] * 4
        current = self.start
        result = [DNA[current]]
        for _ in range(self.n - 1):
            if used[current] == len(orders[current]):
                raise AssertionError("last-exit traversal stopped before consuming all edges")
            following = orders[current][used[current]]
            used[current] += 1
            current = following
            result.append(DNA[current])
        output = "".join(result)
        if (
            current != self.root
            or used != list(self.degrees)
            or signature(output) != signature(self.source)
        ):
            raise AssertionError("Euler traversal violated endpoint/edge-count invariant")
        return output


def sample_h_matched(
    target: str, rng: random.Random, *, max_attempts: int = 100_000
) -> ControlResult:
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    sampler = EulerSampler(target)
    wanted = longest_run(target)
    equal = 0
    for attempt in range(1, max_attempts + 1):
        candidate = sampler.sample(rng)
        equal += candidate == target
        if longest_run(candidate) == wanted:
            return ControlResult("OK", candidate, attempt, equal, candidate == target)
    return ControlResult("CONTROL_ATTEMPT_CAP", None, max_attempts, equal, None)
