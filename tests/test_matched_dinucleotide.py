"""Independent Fraction enumeration, not random histograms, audits uniformity."""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations, product
from math import prod
from pathlib import Path
import json
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from matched_dinucleotide import (
    EulerSampler,
    dinucleotide_counts,
    longest_run,
    sample_h_matched,
    signature,
)  # noqa:E402


def independent_distribution(word):
    if len(word) <= 1:
        return {word: Fraction(1)}
    letters = sorted(set(word))
    counts = Counter(zip(word, word[1:]))
    root = word[-1]
    vertices = [v for v in letters if v != root]
    choices = [[b for b in letters if b != a and counts[a, b] > 0] for a in vertices]
    trees = []
    for selected in product(*choices):
        parent = dict(zip(vertices, selected))
        good = True
        for start in vertices:
            u, seen = start, set()
            while u != root:
                if u in seen or u not in parent:
                    good = False
                    break
                seen.add(u)
                u = parent[u]
        if good:
            trees.append((parent, prod(counts[a, b] for a, b in parent.items())))
    total = sum(weight for _, weight in trees)
    result = defaultdict(Fraction)
    for parent, weight in trees:
        options = []
        for a in letters:
            remaining = [b for b in letters for _ in range(counts[a, b] - (parent.get(a) == b))]
            orders = sorted(set(permutations(remaining)))
            options.append([order + ((parent[a],) if a in parent else ()) for order in orders])
        probability = Fraction(weight, total * prod(len(o) for o in options))
        for orders in product(*options):
            queues = {a: list(order) for a, order in zip(letters, orders)}
            output = word[0]
            for _ in range(len(word) - 1):
                assert queues[output[-1]], "Independent last-exit traversal stranded"
                output += queues[output[-1]].pop(0)
            assert output[-1] == root and not any(queues.values())
            result[output] += probability
    return dict(result)


def exact_certificate(max_length=6):
    classes = defaultdict(list)
    # Reference signatures are independent of the production count routine.
    for n in range(max_length + 1):
        for chars in product("ACGT", repeat=n):
            word = "".join(chars)
            key = (word[:1], word[-1:], tuple(sorted(Counter(zip(word, word[1:])).items())))
            classes[key].append(word)
    checked = 0
    for words in classes.values():
        outcomes = independent_distribution(words[0])
        assert set(outcomes) == set(words)
        assert sum(outcomes.values()) == 1
        assert set(outcomes.values()) == {Fraction(1, len(words))}
        assert EulerSampler(words[0]).word_probability() == Fraction(1, len(words))
        assert production_distribution(words[0]) == outcomes
        checked += len(outcomes)
    return {
        "status": "EXACT_FRACTION_ALL_WORDS_ALL_CLASSES_PASS_NOT_ACCEPTANCE",
        "alphabet": "ACGT",
        "max_length": max_length,
        "composition_classes": len(classes),
        "words_checked": checked,
        "production_randrange_branches_exactly_enumerated": True,
        "random_histograms_used": False,
    }


def production_distribution(word):
    """Enumerate every actual production randrange branch with exact mass."""

    class NeedBranch(Exception):
        pass

    class BranchRNG:
        def __init__(self, prefix):
            self.prefix, self.position = prefix, 0

        def randrange(self, bound):
            assert type(bound) is int and bound > 0
            if self.position == len(self.prefix):
                raise NeedBranch(bound)
            value = self.prefix[self.position]
            self.position += 1
            assert 0 <= value < bound
            return value

    sampler = EulerSampler(word)
    pending = [((), Fraction(1))]
    result = defaultdict(Fraction)
    while pending:
        prefix, mass = pending.pop()
        try:
            output = sampler.sample(BranchRNG(prefix))
        except NeedBranch as exc:
            bound = exc.args[0]
            pending.extend((prefix + (i,), mass / bound) for i in range(bound))
        else:
            result[output] += mass
    assert sum(result.values()) == 1
    return dict(result)


def test_all_short_composition_classes_exact_probability():
    assert exact_certificate()["words_checked"] == sum(4**n for n in range(7))


@pytest.mark.parametrize(
    "word", ["", "A", "AAAAA", "AC", "ACACA", "AACACA", "ACACAGCGA", "ACGTACGTT"]
)
def test_fraction_boundaries_and_weighted_parallel_edges(word):
    outcomes = independent_distribution(word)
    assert sum(outcomes.values()) == 1
    assert len(set(outcomes.values())) == 1
    sampler = EulerSampler(word)
    for seed in range(24):
        assert sampler.sample(random.Random(seed)) in outcomes
    assert sampler.word_probability() == next(iter(outcomes.values()))


def test_nonboolean_tree_weights():
    sampler = EulerSampler("ACACAGCGA")
    assert sorted(t.weight for t in sampler.trees) == [1, 2, 2]
    assert sampler.total_weight == 5


def test_fixed_cap_h_rejection_probability_and_failure(monkeypatch):
    target = "AAACACA"
    distribution = independent_distribution(target)
    eligible = {w: p for w, p in distribution.items() if longest_run(w) == longest_run(target)}
    q = sum(eligible.values())
    assert 0 < q < 1
    cap = 3
    mass = {w: p * sum((1 - q) ** j for j in range(cap)) for w, p in eligible.items()}
    assert len(set(mass.values())) == 1
    assert sum(mass.values()) == 1 - (1 - q) ** cap
    rejected = next(w for w in distribution if w not in eligible)
    monkeypatch.setattr(EulerSampler, "sample", lambda self, rng: rejected)
    result = sample_h_matched(target, random.Random(8), max_attempts=cap)
    assert (
        result.status == "CONTROL_ATTEMPT_CAP"
        and result.sequence is None
        and result.attempts == cap
    )


def test_h_invariants_duplicates_allowed_and_invalid_inputs():
    for target in ("", "A", "AAA", "AACACAGTACGT", "ACACAGCGA"):
        result = sample_h_matched(target, random.Random(5))
        assert result.status == "OK" and signature(result.sequence) == signature(target)
        assert longest_run(result.sequence) == longest_run(target)
        assert result.sequence.count("G") + result.sequence.count("C") == target.count(
            "G"
        ) + target.count("C")
    assert sample_h_matched("AAAA", random.Random(1)).accepted_equal_target
    with pytest.raises(ValueError):
        EulerSampler("N")
    with pytest.raises(ValueError):
        sample_h_matched("AC", random.Random(1), max_attempts=0)
    assert sum(dinucleotide_counts("ACGT")) == 3


if __name__ == "__main__":
    payload = exact_certificate()
    print(json.dumps(payload, indent=2))
