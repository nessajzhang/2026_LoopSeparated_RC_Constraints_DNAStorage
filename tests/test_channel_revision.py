from __future__ import annotations

import dataclasses
import math
import random
from collections import Counter

import pytest

from loop_alpha_audit.channel import (
    IDSConfig,
    ReadObservation,
    derived_seed,
    simulate_pool,
    simulate_read,
)


def test_no_noise_and_empty_sequence() -> None:
    for sequence in ("", "A", "ACGTACGT"):
        read, events, survivors = simulate_read(sequence, IDSConfig(), random.Random(3))
        assert read == sequence
        assert events == ()
        assert survivors == len(sequence)


def test_each_pure_event_including_boundary_gaps() -> None:
    read, events, survivors = simulate_read("AC", IDSConfig(p_insert=1), random.Random(1))
    assert len(read) == 5
    assert read[1] == "A" and read[3] == "C"
    assert [event.original_coordinate for event in events] == [0, 1, 2]
    assert survivors == 2
    read, events, survivors = simulate_read("ACGT", IDSConfig(p_delete=1), random.Random(1))
    assert read == "" and len(events) == 4 and survivors == 0
    read, events, survivors = simulate_read("ACGT", IDSConfig(p_substitute=1), random.Random(1))
    assert all(a != b for a, b in zip(read, "ACGT"))
    assert len(events) == 4 and survivors == 4
    read, events, survivors = simulate_read("", IDSConfig(p_insert=1), random.Random(1))
    assert len(read) == 1 and len(events) == 1 and survivors == 0


def test_declared_event_order() -> None:
    read, events, survivors = simulate_read("ACGT", IDSConfig(1, 1, 1), random.Random(11))
    assert len(read) == 5  # Inserted symbols survive; all original bases deleted.
    assert Counter(e.operation for e in events) == {"insertion": 5, "deletion": 4}
    assert survivors == 0


@pytest.mark.parametrize("value", [-1, 1.1, math.nan, math.inf])
def test_invalid_parameters(value: float) -> None:
    with pytest.raises(ValueError):
        IDSConfig(p_insert=value)
    with pytest.raises(ValueError):
        simulate_pool({0: "A"}, IDSConfig(), coverage=value, seed=0)


def test_substitution_matrix_and_invalid_sequences() -> None:
    matrix = ((0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 0, 0, 0))
    read, events, _ = simulate_read(
        "ACGT", IDSConfig(p_substitute=1, substitution_matrix=matrix), random.Random(8)
    )
    assert read == "CGTA" and len(events) == 4
    with pytest.raises(ValueError):
        IDSConfig(substitution_matrix=((1, 0, 0, 0),) * 4)
    for invalid in ("acgt", "N", " A", "A\n"):
        with pytest.raises(ValueError):
            simulate_read(invalid, IDSConfig(), random.Random())


def test_observations_have_no_sender_truth() -> None:
    assert [field.name for field in dataclasses.fields(ReadObservation)] == ["read_id", "sequence"]
    result = simulate_pool({0: "ACGT", 1: "TGCA"}, IDSConfig(), coverage=3, seed=13)
    assert len(result.observations) == len(result.truth) == 6
    assert len({read.read_id for read in result.observations}) == 6
    assert all(not hasattr(read, "strand_id") for read in result.observations)
    assert all(not hasattr(read, "events") for read in result.observations)
    assert all(not hasattr(read, "original_sequence") for read in result.observations)
    assert result == simulate_pool({1: "TGCA", 0: "ACGT"}, IDSConfig(), coverage=3, seed=13)


def test_fixed_poisson_and_dropout_are_separate() -> None:
    strands = {i: "ACGT" for i in range(4000)}
    fixed = simulate_pool(strands, IDSConfig(), coverage=3, seed=20261023)
    assert len(fixed.observations) == 12000
    poisson = simulate_pool(
        strands, IDSConfig(), coverage=3, seed=20261023, coverage_mode="poisson"
    )
    assert abs(len(poisson.observations) - 12000) < 6 * math.sqrt(12000)
    dropped = simulate_pool(strands, IDSConfig(), coverage=3, seed=1, dropout=1)
    assert dropped.observations == () and len(dropped.dropped_strands) == 4000
    assert simulate_pool(strands, IDSConfig(), coverage=0, seed=1).observations == ()
    with pytest.raises(ValueError):
        simulate_pool(strands, IDSConfig(), coverage=1.5, seed=1)


def test_independent_opportunity_event_statistics() -> None:
    config = IDSConfig(0.13, 0.21, 0.17)
    rng = random.Random(20261023)
    counts: Counter[str] = Counter()
    surviving = 0
    repetitions, n = 20000, 8
    for _ in range(repetitions):
        _, events, survivors = simulate_read("ACGTACGT", config, rng)
        counts.update(event.operation for event in events)
        surviving += survivors
    for operation, opportunities, probability in (
        ("insertion", repetitions * (n + 1), config.p_insert),
        ("deletion", repetitions * n, config.p_delete),
        ("substitution", surviving, config.p_substitute),
    ):
        expected = opportunities * probability
        sigma = math.sqrt(opportunities * probability * (1 - probability))
        assert abs(counts[operation] - expected) < 6 * sigma


def test_diagnostic_labels_and_seed_reproducibility() -> None:
    assert IDSConfig.diagnostic(0.01) == IDSConfig(0.0025, 0.0025, 0.005)
    assert IDSConfig.diagnostic(0.01, "substitution") == IDSConfig(p_substitute=0.01)
    assert IDSConfig.diagnostic(0.01, "indel") == IDSConfig(0.005, 0.005, 0)
    with pytest.raises(ValueError):
        IDSConfig.diagnostic(0.01, "realistic")
    assert derived_seed(9, "x", 2) == derived_seed(9, "x", 2)
    assert derived_seed(9, "x", 2) != derived_seed(9, "x", 3)
