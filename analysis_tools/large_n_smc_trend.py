#!/usr/bin/env python3
"""Large-n sequential Monte Carlo checks for loop-separated RC counts.

The exact enumerator in ``exact_counts.py`` is deterministic but its state
space grows rapidly with the loop threshold.  This script gives a separate
large-n sanity check: it estimates finite-block rates by sequential importance
resampling.  At each prefix length it estimates the count ratio

    A(n+1) / A(n) = E[# valid one-symbol extensions of a uniform valid prefix].

Particles are resampled proportional to their number of valid extensions and
then extended uniformly.  The resulting log-normalizing-constant estimate is a
standard particle-filter estimate of log A(n).  It is not used as theorem
evidence; it is meant to show whether the finite-block rates seen at short
lengths are still trending toward the proven capacity at much larger n.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DNA_ALPHABET_SIZE = 4
CLASSICAL_C2 = 1.167987028946


def reverse_complement(word: int, m: int) -> int:
    """Return the base-4 code of the reverse complement (A,C,G,T = 0,1,2,3)."""
    rc = 0
    for _ in range(m):
        base = word & 3
        word >>= 2
        rc = (rc << 2) | (3 - base)
    return rc


@dataclass(frozen=True)
class Model:
    m: int
    k: int
    q: int = DNA_ALPHABET_SIZE

    @property
    def min_start_separation(self) -> int:
        return self.m + self.k

    @property
    def suffix_mask(self) -> int:
        return (1 << (2 * (self.m - 1))) - 1 if self.m > 1 else 0

    @property
    def word_mask(self) -> int:
        return (1 << (2 * self.m)) - 1


State = tuple[int, int, tuple[int, ...], int]


class TransitionCache:
    def __init__(self, model: Model, max_entries: int | None = None):
        self.model = model
        self.max_entries = max_entries
        self.cache: dict[State, tuple[State, ...]] = {}

    def next_states(self, state: State) -> tuple[State, ...]:
        if self.max_entries != 0:
            cached = self.cache.get(state)
            if cached is not None:
                return cached

        model = self.model
        suffix, suffix_len, recent, old_seen = state
        out: list[State] = []

        for base in range(model.q):
            if suffix_len < model.m - 1:
                new_suffix = (suffix << 2) | base
                out.append((new_suffix, suffix_len + 1, recent, old_seen))
                continue

            word = ((suffix << 2) | base) & model.word_mask
            rc = reverse_complement(word, model.m)
            if old_seen & (1 << rc):
                continue

            updated_recent = recent + (word,)
            updated_old = old_seen
            if len(updated_recent) == model.min_start_separation:
                expired = updated_recent[0]
                updated_recent = updated_recent[1:]
                updated_old |= 1 << expired

            new_suffix = word & model.suffix_mask if model.m > 1 else 0
            new_suffix_len = min(model.m - 1, suffix_len + 1)
            out.append((new_suffix, new_suffix_len, updated_recent, updated_old))

        result = tuple(out)
        if self.max_entries is None or len(self.cache) < self.max_entries:
            self.cache[state] = result
        return result


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split() if part]


def systematic_resample_and_extend(
    transitions: list[tuple[State, ...]],
    weights: list[int],
    total_weight: int,
    particles: int,
    rng: random.Random,
) -> list[State]:
    """Sample a near-uniform cloud of next valid prefixes."""
    if total_weight <= 0:
        raise ValueError("cannot resample with zero total weight")

    spacing = total_weight / particles
    threshold = rng.random() * spacing
    cumulative = weights[0]
    parent = 0
    new_states: list[State] = []

    for _ in range(particles):
        while threshold >= cumulative and parent + 1 < particles:
            parent += 1
            cumulative += weights[parent]
        choices = transitions[parent]
        new_states.append(choices[rng.randrange(len(choices))])
        threshold += spacing

    return new_states


def run_one_window_queue(args: tuple) -> list[dict[str, str]]:
    if len(args) == 6:
        m, k, particles, rep, n_values_tuple, seed = args
        cache_max = None
    else:
        m, k, particles, rep, n_values_tuple, seed, cache_max = args
    n_values = set(n_values_tuple)
    max_n = max(n_values)
    rng = random.Random(seed)
    model = Model(m=m, k=k)
    cache = TransitionCache(model, max_entries=cache_max)
    init: State = (0, 0, (), 0)
    states = [init] * particles
    log2_count = 0.0
    records: list[dict[str, str]] = []
    t0 = time.time()

    for n in range(1, max_n + 1):
        transitions: list[tuple[State, ...]] = []
        weights: list[int] = []
        total_weight = 0
        for state in states:
            nexts = cache.next_states(state)
            weight = len(nexts)
            transitions.append(nexts)
            weights.append(weight)
            total_weight += weight

        if total_weight == 0:
            # Particle extinction is an estimator failure, not evidence that the
            # constrained language itself is empty.  Emit an explicit row for
            # every requested length that can no longer be reached so a failed
            # replicate cannot silently disappear from the output grid.
            elapsed = f"{time.time() - t0:.3f}"
            for requested_n in sorted(target for target in n_values if target >= n):
                records.append(
                    {
                        "m": str(m),
                        "k": str(k),
                        "n": str(requested_n),
                        "rep": str(rep),
                        "seed": str(seed),
                        "particles": str(particles),
                        "log2_count_est": "-inf",
                        "rate_est_bits_per_nt": "-inf",
                        "mean_branching_est": "0",
                        "unique_particle_states": "0",
                        "state_representation": "window_queue",
                        "transition_cache_size": str(len(cache.cache)),
                        "transition_cache_limit": "unbounded"
                        if cache_max is None
                        else str(cache_max),
                        "run_status": "particle_extinction",
                        "extinction_n": str(n),
                        "elapsed_sec": elapsed,
                    }
                )
            break

        mean_branching = total_weight / particles
        log2_count += math.log2(mean_branching)

        states = systematic_resample_and_extend(transitions, weights, total_weight, particles, rng)

        if n in n_values:
            records.append(
                {
                    "m": str(m),
                    "k": str(k),
                    "n": str(n),
                    "rep": str(rep),
                    "seed": str(seed),
                    "particles": str(particles),
                    "log2_count_est": f"{log2_count:.12f}",
                    "rate_est_bits_per_nt": f"{log2_count / n:.12f}",
                    "mean_branching_est": f"{mean_branching:.12f}",
                    "unique_particle_states": str(len(set(states))),
                    "state_representation": "window_queue",
                    "transition_cache_size": str(len(cache.cache)),
                    "transition_cache_limit": "unbounded" if cache_max is None else str(cache_max),
                    "run_status": "ok",
                    "extinction_n": "",
                    "elapsed_sec": f"{time.time() - t0:.3f}",
                }
            )

    return records


def run_one_first_seen(args: tuple) -> list[dict[str, str]]:
    """Run one SMC replicate using a first-seen state compression.

    The older implementation kept the full recent window of length ``m+k`` in
    each particle state and moved expired words into ``old_seen``.  That is
    correct, but it makes growing-k regimes unnecessarily slow because every
    transition copies an O(k)-length tuple.

    For this one-sided scan, the same validity test can be made from the first
    occurrence position of each m-word.  When a new word starts at position
    ``s``, it is invalid exactly when the first occurrence of its reverse
    complement is at most ``s-(m+k)``.  This is equivalent to the monotone
    ``old_seen`` set used above, but the particle state has size O(4^m), not
    O(k).  The transition order and branching counts are unchanged.
    """

    if len(args) == 6:
        m, k, particles, rep, n_values_tuple, seed = args
        cache_max = None
    else:
        m, k, particles, rep, n_values_tuple, seed, cache_max = args
    del cache_max  # The compressed transition depends on time; caching is not useful here.

    n_values = set(n_values_tuple)
    max_n = max(n_values)
    rng = random.Random(seed)
    q = DNA_ALPHABET_SIZE
    word_count = q**m
    word_mask = (1 << (2 * m)) - 1
    suffix_mask = (1 << (2 * (m - 1))) - 1 if m > 1 else 0
    min_start_separation = m + k
    inf_pos = max_n + min_start_separation + 1
    rc_table = tuple(reverse_complement(word, m) for word in range(word_count))

    # State: suffix, suffix_len, first_seen_positions.
    init_first_seen = (inf_pos,) * word_count
    init = (0, 0, init_first_seen)
    states = [init] * particles
    log2_count = 0.0
    records: list[dict[str, str]] = []
    t0 = time.time()

    for n in range(1, max_n + 1):
        start_pos = n - m
        old_cutoff = start_pos - min_start_separation
        transitions: list[tuple[tuple[int, int, tuple[int, ...]], ...]] = []
        weights: list[int] = []
        total_weight = 0

        for suffix, suffix_len, first_seen in states:
            out: list[tuple[int, int, tuple[int, ...]]] = []
            for base in range(q):
                if suffix_len < m - 1:
                    new_suffix = (suffix << 2) | base
                    out.append((new_suffix, suffix_len + 1, first_seen))
                    continue

                word = ((suffix << 2) | base) & word_mask
                rc = rc_table[word]
                if first_seen[rc] <= old_cutoff:
                    continue

                if first_seen[word] == inf_pos:
                    updated = list(first_seen)
                    updated[word] = start_pos
                    new_first_seen = tuple(updated)
                else:
                    new_first_seen = first_seen

                new_suffix = word & suffix_mask if m > 1 else 0
                new_suffix_len = min(m - 1, suffix_len + 1)
                out.append((new_suffix, new_suffix_len, new_first_seen))

            result = tuple(out)
            transitions.append(result)
            weight = len(result)
            weights.append(weight)
            total_weight += weight

        if total_weight == 0:
            elapsed = f"{time.time() - t0:.3f}"
            for requested_n in sorted(target for target in n_values if target >= n):
                records.append(
                    {
                        "m": str(m),
                        "k": str(k),
                        "n": str(requested_n),
                        "rep": str(rep),
                        "seed": str(seed),
                        "particles": str(particles),
                        "log2_count_est": "-inf",
                        "rate_est_bits_per_nt": "-inf",
                        "mean_branching_est": "0",
                        "unique_particle_states": "0",
                        "state_representation": "first_seen",
                        "transition_cache_size": "0",
                        "transition_cache_limit": "not_applicable",
                        "run_status": "particle_extinction",
                        "extinction_n": str(n),
                        "elapsed_sec": elapsed,
                    }
                )
            break

        mean_branching = total_weight / particles
        log2_count += math.log2(mean_branching)

        states = systematic_resample_and_extend(transitions, weights, total_weight, particles, rng)

        if n in n_values:
            records.append(
                {
                    "m": str(m),
                    "k": str(k),
                    "n": str(n),
                    "rep": str(rep),
                    "seed": str(seed),
                    "particles": str(particles),
                    "log2_count_est": f"{log2_count:.12f}",
                    "rate_est_bits_per_nt": f"{log2_count / n:.12f}",
                    "mean_branching_est": f"{mean_branching:.12f}",
                    "unique_particle_states": str(len(set(states))),
                    "state_representation": "first_seen",
                    "transition_cache_size": "0",
                    "transition_cache_limit": "not_applicable",
                    "run_status": "ok",
                    "extinction_n": "",
                    "elapsed_sec": f"{time.time() - t0:.3f}",
                }
            )

    return records


def run_one(args: tuple) -> list[dict[str, str]]:
    """Run one SMC replicate with the original window-queue state.

    This remains the default because the released large-n CSV artifacts were
    generated with this representation.  ``run_one_first_seen`` is an
    equivalent state-compressed implementation used for development checks, but
    changing the default would make regenerated raw metadata differ from the
    archived tables.
    """

    return run_one_window_queue(args)


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], capacity: float) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["m"], row["k"], row["n"])
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, str]] = []
    for key in sorted(grouped, key=lambda item: (int(item[0]), int(item[1]), int(item[2]))):
        group_rows = grouped[key]
        particle_values = {row["particles"] for row in group_rows}
        if len(particle_values) != 1:
            raise ValueError(
                f"mixed particle counts for SMC summary group {key}: {sorted(particle_values)}"
            )
        finite_vals = []
        for row in group_rows:
            rate = float(row["rate_est_bits_per_nt"])
            if row.get("run_status", "ok") == "ok" and math.isfinite(rate):
                finite_vals.append(rate)
        failed_reps = len(group_rows) - len(finite_vals)
        if failed_reps:
            mean_text = sd_text = sem_text = excess_text = ""
            summary_status = "particle_extinction"
        else:
            mean = statistics.fmean(finite_vals)
            sd = statistics.stdev(finite_vals) if len(finite_vals) > 1 else 0.0
            sem = sd / math.sqrt(len(finite_vals))
            mean_text = f"{mean:.12f}"
            sd_text = f"{sd:.12f}"
            sem_text = f"{sem:.12f}"
            excess_text = f"{mean - capacity:.12f}"
            summary_status = "ok"
        m, k, n = key
        out.append(
            {
                "m": m,
                "k": k,
                "n": n,
                "reps": str(len(group_rows)),
                "finite_reps": str(len(finite_vals)),
                "failed_reps": str(failed_reps),
                "particles": next(iter(particle_values)),
                "mean_rate_est_bits_per_nt": mean_text,
                "sd_rate": sd_text,
                "sem_rate": sem_text,
                "excess_over_capacity": excess_text,
                "capacity_reference": f"{capacity:.12f}",
                "summary_status": summary_status,
            }
        )
    return out


def load_exact_rates(paths: list[Path]) -> dict[tuple[int, int, int], float]:
    exact: dict[tuple[int, int, int], float] = {}
    for path in paths:
        name = path.name
        if not name.startswith("counts_m") or "_k" not in name:
            continue
        stem = path.stem
        try:
            m_part, k_part = stem.replace("counts_m", "").split("_k", 1)
            m = int(m_part)
            k = int(k_part.split("_", 1)[0])
        except ValueError:
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                n = int(row["n"])
                if n == 0:
                    continue
                exact[(m, k, n)] = float(row["finite_block_rate_bits_per_nt"])
    return exact


def validate(
    summary_rows: list[dict[str, str]], exact: dict[tuple[int, int, int], float]
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in summary_rows:
        key = (int(row["m"]), int(row["k"]), int(row["n"]))
        if key not in exact or row.get("summary_status", "ok") != "ok":
            continue
        est = float(row["mean_rate_est_bits_per_nt"])
        truth = exact[key]
        out.append(
            {
                "m": row["m"],
                "k": row["k"],
                "n": row["n"],
                "exact_rate_bits_per_nt": f"{truth:.12f}",
                "smc_mean_rate_bits_per_nt": f"{est:.12f}",
                "smc_minus_exact": f"{est - truth:.12f}",
                "sd_rate": row["sd_rate"],
                "reps": row["reps"],
                "particles": row["particles"],
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--ks", default="0,1,2,5")
    parser.add_argument(
        "--n-values",
        default="10,20,30,40,50,100,200,400,600,1000,2000,5000,10000",
    )
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument("--reps", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=max(1, min(4, mp.cpu_count() - 1)))
    parser.add_argument(
        "--cache-max",
        type=int,
        default=-1,
        help="Transition cache cap per worker. -1=unbounded, 0=disabled, positive=entry cap.",
    )
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--capacity-reference", type=float, default=CLASSICAL_C2)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exact-dir", type=Path)
    args = parser.parse_args()

    ks = parse_int_list(args.ks)
    n_values = tuple(sorted(set(parse_int_list(args.n_values))))
    tasks = []
    cache_max = None if args.cache_max < 0 else args.cache_max
    for k in ks:
        for rep in range(args.reps):
            tasks.append(
                (args.m, k, args.particles, rep, n_values, args.seed + 1009 * k + rep, cache_max)
            )

    print(
        f"Running SMC: m={args.m}, ks={ks}, n_max={max(n_values)}, "
        f"particles={args.particles}, reps={args.reps}, jobs={args.jobs}",
        flush=True,
    )
    t0 = time.time()
    if args.jobs == 1:
        nested = [run_one(task) for task in tasks]
    else:
        with mp.Pool(processes=args.jobs) as pool:
            nested = list(pool.imap_unordered(run_one, tasks))

    rows = [row for group in nested for row in group]
    rows.sort(key=lambda row: (int(row["m"]), int(row["k"]), int(row["rep"]), int(row["n"])))
    summary_rows = summarize(rows, args.capacity_reference)

    raw_path = args.out_dir / f"smc_m{args.m}_raw.csv"
    summary_path = args.out_dir / f"smc_m{args.m}_summary.csv"
    write_csv(raw_path, rows)
    write_csv(summary_path, summary_rows)
    print(raw_path)
    print(summary_path)

    if args.exact_dir is not None:
        exact_paths = sorted(args.exact_dir.glob(f"counts_m{args.m}_k*.csv"))
        exact = load_exact_rates(exact_paths)
        validation_rows = validate(summary_rows, exact)
        if validation_rows:
            validation_path = args.out_dir / f"smc_m{args.m}_validation_against_exact.csv"
            write_csv(validation_path, validation_rows)
            print(validation_path)

    failed_rows = sum(row.get("run_status", "ok") != "ok" for row in rows)
    if failed_rows:
        print(
            f"SMC_PARTICLE_EXTINCTION failed_rows={failed_rows}; "
            "failure rows and non-numeric summaries were written",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    print(f"SMC_DONE elapsed_sec={time.time() - t0:.3f}", flush=True)


if __name__ == "__main__":
    main()
