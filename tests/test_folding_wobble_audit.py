import runpy
from pathlib import Path


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "code" / "audit_folding_wobble_runs.py"
)
dot_bracket_pairs = MODULE["dot_bracket_pairs"]
maximal_nested_runs = MODULE["maximal_nested_runs"]
run_pair_classes = MODULE["run_pair_classes"]
contiguous_subruns = MODULE["contiguous_subruns"]


def test_nested_run_reconstruction() -> None:
    pairs = dot_bracket_pairs("(((...)))..((..))")
    assert maximal_nested_runs(pairs) == [
        [(0, 8), (1, 7), (2, 6)],
        [(11, 16), (12, 15)],
    ]


def test_pair_classification() -> None:
    sequence = "AGTTT"
    run = [(0, 4), (1, 3)]
    assert run_pair_classes(sequence, run) == {"watson_crick", "wobble"}


def test_every_length_m_subrun_must_be_checked() -> None:
    sequence = "AAGTTT"
    run = [(0, 5), (1, 4), (2, 3)]
    subruns = contiguous_subruns(run, 2)
    assert run_pair_classes(sequence, run) == {"watson_crick", "wobble"}
    assert run_pair_classes(sequence, subruns[0]) == {"watson_crick"}
    assert run_pair_classes(sequence, subruns[1]) == {"watson_crick", "wobble"}
