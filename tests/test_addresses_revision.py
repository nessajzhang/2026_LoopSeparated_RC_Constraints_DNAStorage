from __future__ import annotations

import itertools
import random

import pytest
from rapidfuzz.distance import Levenshtein

from loop_alpha_audit.addresses import (
    AddressBook,
    assign_address,
    make_address_book,
    reference_edit_distance,
)


def test_fast_distance_against_independent_dynamic_program() -> None:
    words = ["".join(word) for n in range(5) for word in itertools.product("AC", repeat=n)]
    for left in words:
        for right in words:
            assert Levenshtein.distance(left, right) == reference_edit_distance(left, right)
    rng = random.Random(20261023)
    for _ in range(200):
        left = "".join(rng.choices("ACGT", k=rng.randrange(30)))
        right = "".join(rng.choices("ACGT", k=rng.randrange(30)))
        assert Levenshtein.distance(left, right) == reference_edit_distance(left, right)


def test_public_address_code_and_independent_certificate() -> None:
    book = make_address_book()
    certificate = book.certificate()
    assert certificate["address_count"] == 63
    assert certificate["pairs_checked"] == 1953
    assert certificate["minimum_levenshtein_distance"] >= 5
    assert certificate["reference_minimum_levenshtein_distance"] >= 5
    assert certificate["gc_counts"] == [12] * 63
    assert max(certificate["longest_runs"]) <= 4
    assert all(address.endswith("TT") for address in book.sequences)
    assert make_address_book() is book


def test_assignment_uses_observed_address_not_read_order() -> None:
    book = make_address_book(7)
    for index, address in enumerate(book.sequences):
        result = assign_address(address + "CCACGT", book)
        assert result.status == "OK" and result.strand_index == index
        for position in (0, 8, 23):
            replacement = "A" if address[position] != "A" else "G"
            edited = address[:position] + replacement + address[position + 1 :]
            result = assign_address(edited + "CCACGT", book)
            assert result.status == "OK" and result.strand_index == index
        for edited in ("A" + address, address[:9] + address[10:]):
            result = assign_address(edited + "CCACGT", book)
            assert result.status == "OK" and result.strand_index == index


def test_invalid_distance_and_ambiguous_reads_are_rejected() -> None:
    with pytest.raises(ValueError):
        AddressBook(("AAAA", "AAAT"), 2)
    with pytest.raises(ValueError):
        make_address_book(300)
    with pytest.raises(RuntimeError):
        make_address_book(63, max_trials=63)
    book = AddressBook(("AAAA", "TTTT"), 4)
    assert assign_address("AATT", book, boundary_slack=0).status == "AMBIGUOUS"
    assert assign_address("N", book).status == "INVALID_READ"
    assert assign_address("", book).status == "TOO_SHORT"
