"""Exact enumerator for finite-length (m,k)-SSA sequence counts.

The state records (i) the suffix needed to form the next m-mer, (ii) the
recent m-mers whose start positions are too close to the next m-mer to form a
forbidden pair, and (iii) the set of m-mers that already occurred far enough
in the past.  This avoids enumerating all 4**n sequences.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def reverse_complement(word: int, m: int) -> int:
    """Return the base-4 code of the reverse complement (A,C,G,T = 0,1,2,3)."""
    rc = 0
    for _ in range(m):
        base = word & 3
        word >>= 2
        rc = (rc << 2) | (3 - base)
    return rc


def exact_counts(m: int, k: int, n_max: int) -> list[int]:
    if m < 1 or k < 0 or n_max < 0:
        raise ValueError("require m >= 1, k >= 0, and n_max >= 0")

    # State: (current suffix code, suffix length, recent m-mers, old-seen mask)
    states: dict[tuple[int, int, tuple[int, ...], int], int] = {(0, 0, (), 0): 1}
    counts = [1]
    min_start_separation = m + k
    suffix_mask = (1 << (2 * (m - 1))) - 1 if m > 1 else 0

    for length in range(n_max):
        nxt: dict[tuple[int, int, tuple[int, ...], int], int] = defaultdict(int)
        for (suffix, suffix_len, recent, old_seen), multiplicity in states.items():
            for base in range(4):
                new_len = length + 1
                if new_len < m:
                    new_suffix = (suffix << 2) | base
                    key = (new_suffix, suffix_len + 1, recent, old_seen)
                    nxt[key] += multiplicity
                    continue

                word = ((suffix << 2) | base) & ((1 << (2 * m)) - 1)
                rc = reverse_complement(word, m)
                if old_seen & (1 << rc):
                    continue

                updated_recent = recent + (word,)
                updated_old = old_seen
                if len(updated_recent) == min_start_separation:
                    expired = updated_recent[0]
                    updated_recent = updated_recent[1:]
                    updated_old |= 1 << expired

                new_suffix = word & suffix_mask if m > 1 else 0
                key = (new_suffix, min(m - 1, suffix_len + 1), updated_recent, updated_old)
                nxt[key] += multiplicity

        states = nxt
        counts.append(sum(states.values()))
        print(
            f"m={m} k={k} n={new_len:3d} N={counts[-1]} "
            f"states={len(states)} rate={math.log2(counts[-1]) / new_len:.9f}",
            flush=True,
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n-max", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    counts = exact_counts(args.m, args.k, args.n_max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "N", "finite_block_rate_bits_per_nt", "valid_fraction"])
        for n, count in enumerate(counts):
            rate = math.log2(count) / n if n else 2.0
            writer.writerow([n, count, f"{rate:.12f}", f"{count / (4**n):.12g}"])


if __name__ == "__main__":
    main()
