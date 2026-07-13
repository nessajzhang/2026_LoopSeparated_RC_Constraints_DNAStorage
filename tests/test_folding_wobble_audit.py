import runpy
from pathlib import Path


MODULE = runpy.run_path(
    Path(__file__).resolve().parents[1] / "code" / "audit_folding_wobble_runs.py"
)
dot_bracket_pairs = MODULE["dot_bracket_pairs"]
maximal_nested_runs = MODULE["maximal_nested_runs"]
run_pair_classes = MODULE["run_pair_classes"]


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
