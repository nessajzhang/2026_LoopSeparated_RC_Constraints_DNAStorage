"""ViennaRNA-based folding validation for loop-aware SSA screens.

Generate DNA sequences under configurable screens, fold them with ViennaRNA,
and report minimum-free-energy (MFE) summaries.

The sampler is not claimed to be uniform over the constrained language. It is
intended for diagnostic comparisons rather than uniform sampling of the
constrained language.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "analysis_tools" / "_vendor_viennarna"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

try:
    import RNA  # type: ignore
except Exception as exc:  # pragma: no cover - this is an environment check
    raise SystemExit(
        "ViennaRNA is not importable. Install it locally with:\n"
        f"  python -m pip install --target {VENDOR} ViennaRNA\n"
        f"Original import error: {exc}"
    )

if hasattr(RNA, "params_load_DNA_Mathews2004"):
    RNA.params_load_DNA_Mathews2004()
    ENERGY_MODEL = "ViennaRNA_DNA_Mathews2004"
elif hasattr(RNA, "params_load_DNA_Mathews1999"):
    RNA.params_load_DNA_Mathews1999()
    ENERGY_MODEL = "ViennaRNA_DNA_Mathews1999"
else:  # pragma: no cover - retained for older ViennaRNA builds
    ENERGY_MODEL = "ViennaRNA_default_parameters"


ALPHABET = "ACGT"
COMP = str.maketrans("ACGT", "TGCA")


def rc(word: str) -> str:
    return word.translate(COMP)[::-1]


def gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def homopolymer_ok(seq: str, max_run: int | None) -> bool:
    if max_run is None:
        return True
    run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            run += 1
            if run > max_run:
                return False
        else:
            run = 1
    return True


def is_mk_ssa(seq: str, m: int, k: int) -> bool:
    """Return True iff seq satisfies the length-m window form of (m,k)-SSA."""
    d = m + k
    windows: dict[str, list[int]] = {}
    for i in range(0, len(seq) - m + 1):
        word = seq[i : i + m]
        for j in windows.get(rc(word), []):
            if i - j >= d:
                return False
        windows.setdefault(word, []).append(i)
    return True


def random_screened_sequence(
    rng: random.Random,
    length: int,
    gc_low: float,
    gc_high: float,
    max_homopolymer: int | None,
    max_tries: int,
) -> str:
    for _ in range(max_tries):
        seq = "".join(rng.choice(ALPHABET) for _ in range(length))
        if gc_low <= gc_fraction(seq) <= gc_high and homopolymer_ok(seq, max_homopolymer):
            return seq
    raise RuntimeError("failed to generate random screened sequence")


def greedy_mk_sequence(
    rng: random.Random,
    length: int,
    m: int,
    k: int,
    gc_low: float,
    gc_high: float,
    max_homopolymer: int | None,
    max_restarts: int,
) -> str:
    """Generate one valid sequence by randomized greedy construction.

    The state update mirrors exact_counts.py: old contains windows far enough
    in the past to conflict with the reverse complement of the next window,
    while recent stores windows still protected by the loop threshold.
    """
    d = m + k
    for _ in range(max_restarts):
        seq: list[str] = []
        suffix = ""
        recent: deque[str] = deque()
        old: set[str] = set()

        while len(seq) < length:
            bases = list(ALPHABET)
            rng.shuffle(bases)
            chosen: str | None = None
            for base in bases:
                candidate = "".join(seq) + base
                if not homopolymer_ok(candidate, max_homopolymer):
                    continue
                if len(candidate) < m:
                    chosen = base
                    break
                word = (suffix + base)[-m:]
                if rc(word) not in old:
                    chosen = base
                    break

            if chosen is None:
                break

            seq.append(chosen)
            current = "".join(seq)
            if len(current) >= m:
                word = (suffix + chosen)[-m:]
                recent.append(word)
                if len(recent) == d:
                    old.add(recent.popleft())
                suffix = word[-(m - 1) :] if m > 1 else ""
            else:
                suffix = current[-(m - 1) :] if m > 1 else ""

        if len(seq) == length:
            out = "".join(seq)
            if gc_low <= gc_fraction(out) <= gc_high and is_mk_ssa(out, m, k):
                return out
    raise RuntimeError(f"failed to generate (m,k)-SSA sequence for m={m}, k={k}")


def mfe_stem_summary(structure: str) -> tuple[int, int]:
    """Return pair count and longest contiguous nested stem in dot-bracket form."""
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            pairs.append((stack.pop(), i))
    pair_set = set(pairs)
    max_stem = 0
    for i, j in pairs:
        if (i - 1, j + 1) in pair_set:
            continue
        run = 1
        while (i + run, j - run) in pair_set:
            run += 1
        max_stem = max(max_stem, run)
    return len(pairs), max_stem


def fold_sequence(seq: str) -> tuple[str, float, int, int]:
    structure, mfe = RNA.fold(seq)
    pair_count, max_stem = mfe_stem_summary(structure)
    return structure, float(mfe), pair_count, max_stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--m", type=int, default=6)
    parser.add_argument("--k-values", default="0,4")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--gc-low", type=float, default=0.40)
    parser.add_argument("--gc-high", type=float, default=0.60)
    parser.add_argument("--max-homopolymer", type=int, default=4)
    parser.add_argument("--max-tries", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]

    rows: list[dict[str, object]] = []

    for idx in range(args.samples):
        seq = random_screened_sequence(
            rng,
            args.length,
            args.gc_low,
            args.gc_high,
            args.max_homopolymer,
            args.max_tries,
        )
        structure, mfe, pair_count, max_stem = fold_sequence(seq)
        rows.append(
            {
                "group": "random_gc_homopolymer",
                "sample": idx,
                "m": "",
                "k": "",
                "length": args.length,
                "gc": f"{gc_fraction(seq):.4f}",
                "mfe_kcal_mol": f"{mfe:.4f}",
                "mfe_pair_count": pair_count,
                "mfe_max_stem": max_stem,
                "energy_model": ENERGY_MODEL,
                "mk_ssa_valid": "",
                "sequence": seq,
                "mfe_structure": structure,
            }
        )

    for k in k_values:
        for idx in range(args.samples):
            seq = greedy_mk_sequence(
                rng,
                args.length,
                args.m,
                k,
                args.gc_low,
                args.gc_high,
                args.max_homopolymer,
                args.max_tries,
            )
            structure, mfe, pair_count, max_stem = fold_sequence(seq)
            rows.append(
                {
                    "group": "mk_ssa_gc_homopolymer",
                    "sample": idx,
                    "m": args.m,
                    "k": k,
                    "length": args.length,
                    "gc": f"{gc_fraction(seq):.4f}",
                    "mfe_kcal_mol": f"{mfe:.4f}",
                    "mfe_pair_count": pair_count,
                    "mfe_max_stem": max_stem,
                    "energy_model": ENERGY_MODEL,
                    "mk_ssa_valid": is_mk_ssa(seq, args.m, k),
                    "sequence": seq,
                    "mfe_structure": structure,
                }
            )

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")
    for group in sorted({str(r["group"]) + "_k" + str(r["k"]) for r in rows}):
        subset = [r for r in rows if str(r["group"]) + "_k" + str(r["k"]) == group]
        mfes = [float(r["mfe_kcal_mol"]) for r in subset]
        stems = [int(r["mfe_max_stem"]) for r in subset]
        print(
            f"{group}: n={len(subset)} "
            f"mean_mfe={sum(mfes) / len(mfes):.3f} "
            f"mean_max_stem={sum(stems) / len(stems):.3f}"
        )


if __name__ == "__main__":
    main()
