"""Brute-force cross-checks for exact_counts.py on small instances."""

from __future__ import annotations

import itertools

from exact_counts import exact_counts, reverse_complement


def valid(word: tuple[int, ...], m: int, k: int) -> bool:
    n = len(word)
    for i in range(n - m + 1):
        left = 0
        for base in word[i : i + m]:
            left = (left << 2) | base
        for j in range(i + m + k, n - m + 1):
            right = 0
            for base in word[j : j + m]:
                right = (right << 2) | base
            if left == reverse_complement(right, m):
                return False
    return True


def brute_count(n: int, m: int, k: int) -> int:
    return sum(valid(word, m, k) for word in itertools.product(range(4), repeat=n))


def main() -> None:
    for m, k, n_max in (
        (2, 0, 10),
        (2, 1, 10),
        (2, 2, 10),
        (2, 3, 10),
        (2, 5, 10),
        (3, 0, 7),
        (3, 2, 7),
    ):
        dp = exact_counts(m, k, n_max)
        for n in range(n_max + 1):
            brute = brute_count(n, m, k)
            assert dp[n] == brute, (m, k, n, dp[n], brute)
        print(f"PASS m={m}, k={k}, n<= {n_max}")


if __name__ == "__main__":
    main()
