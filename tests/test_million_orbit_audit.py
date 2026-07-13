from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "check_orbit_interval_million_audit.py"
SPEC = importlib.util.spec_from_file_location("million_orbit_crosscheck", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Case = MODULE.Case
audit = MODULE.audit
direct_valid = MODULE.direct_valid


def test_pair_boundary_is_strict() -> None:
    accepted = Case(m=2, starts=100, k=3, singleton=False, selected=(40,), noncore=(36, 44))
    rejected = Case(m=2, starts=100, k=3, singleton=False, selected=(40,), noncore=(45,))
    assert direct_valid(accepted)
    assert not direct_valid(rejected)
    assert audit(accepted)[0]
    assert not audit(rejected)[0]


def test_pair_cover_bound_is_attained() -> None:
    case = Case(m=4, starts=100, k=2, singleton=False, selected=(50,), noncore=(45, 55))
    assert case.distance == 6
    assert max(case.noncore) - min(case.noncore) + 1 == 2 * case.distance - 1
    assert audit(case)[0]


def test_self_rc_cover_bound_is_attained() -> None:
    case = Case(m=4, starts=100, k=2, singleton=True, selected=(20, 25), noncore=())
    assert max(case.selected) - min(case.selected) + 1 == case.distance
    assert audit(case)[0]


def test_vacuous_and_empty_cases() -> None:
    vacuous = Case(m=64, starts=8, k=0, singleton=False, selected=(0, 7), noncore=(0, 7))
    empty = Case(m=64, starts=0, k=0, singleton=True, selected=(), noncore=())
    assert audit(vacuous)[0]
    assert audit(empty)[0]
