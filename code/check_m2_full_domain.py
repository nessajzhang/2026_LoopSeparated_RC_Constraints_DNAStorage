"""Independent integer audit of all 65536 dinucleotide objective values.

This does not import either spectral producer.  Each graph is reconstructed
from its 16 independent bits.  Exact positive-vector bounds are paired with
the set complement, and equality cases use an integer trace charpoly routine.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import time
from pathlib import Path


def matrix(mask):
    return [[int(bool(mask & (1 << (4 * i + j)))) for j in range(4)] for i in range(4)]


def apply(rows, vector):
    return [sum(vector[j] for j in row) for row in rows]


def collatz(mask):
    rows = [[j for j in range(4) if mask & (1 << (4 * i + j))] for i in range(4)]
    x = [1, 1, 1, 1]
    for _ in range(50):
        x = [old + image for old, image in zip(x, apply(rows, x))]
    y = apply(rows, x)
    numerator, denominator = 0, 1
    for a, b in zip(y, x):
        if a * denominator > numerator * b:
            numerator, denominator = a, b
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def charpoly(mat):
    n = len(mat)
    previous = [[int(i == j) for j in range(n)] for i in range(n)]
    coeffs = [1]
    for k in range(1, n + 1):
        image = [
            [sum(mat[i][r] * previous[r][j] for r in range(n)) for j in range(n)] for i in range(n)
        ]
        coefficient, remainder = divmod(-sum(image[i][i] for i in range(n)), k)
        if remainder:
            raise AssertionError("integer characteristic polynomial recurrence failed")
        coeffs.append(coefficient)
        for i in range(n):
            image[i][i] += coefficient
        previous = image
    return coeffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh independent m2 attempt directory")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    upper = [collatz(mask) for mask in range(65536)]
    unresolved, strict = [], 0
    digest = hashlib.sha256()
    rows_path = args.output / "m2_bounds.tsv.gz"
    with gzip.open(rows_path, "wt", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "mask",
                "left_num",
                "left_den",
                "complement_num",
                "complement_den",
                "product_num",
                "product_den",
                "status",
            ]
        )
        for mask in range(65536):
            ln, ld = upper[mask]
            rn, rd = upper[65535 ^ mask]
            numerator, denominator = ln * rn, ld * rd
            divisor = math.gcd(numerator, denominator)
            numerator //= divisor
            denominator //= divisor
            # Compare u with 3(1+sqrt(13))/2 without computing a square root.
            shifted = 2 * numerator - 3 * denominator
            below = shifted <= 0 or shifted * shifted < 117 * denominator * denominator
            if below:
                strict += 1
                status = "strict_rational_upper"
            else:
                left, right = charpoly(matrix(mask)), charpoly(matrix(65535 ^ mask))
                if (left, right) not in (
                    ([1, -3, 0, 0, 0], [1, -1, -3, 0, 0]),
                    ([1, -1, -3, 0, 0], [1, -3, 0, 0, 0]),
                ):
                    raise AssertionError((mask, left, right))
                status = "exact_polynomial_equality"
                unresolved.append(
                    {
                        "mask": mask,
                        "left_charpoly_coefficients": left,
                        "complement_charpoly_coefficients": right,
                    }
                )
            row = [mask, ln, ld, rn, rd, numerator, denominator, status]
            writer.writerow(row)
            digest.update(("\t".join(map(str, row)) + "\n").encode())
    if strict != 65528 or [r["mask"] for r in unresolved] != [
        1911,
        4383,
        8946,
        20292,
        45243,
        56589,
        61152,
        63624,
    ]:
        raise AssertionError("unexpected full-domain extremizers")
    summary = {
        "status": "PASS",
        "arbitrary_subsets_checked": 65536,
        "integer_strict_bounds": strict,
        "exact_maximizers": unresolved,
        "exact_maximum": "3*(1+sqrt(13))/2",
        "collatz_steps": 50,
        "word_order": "base-4 lexicographic; all 16 bits independent",
        "objective": "rho(U)*rho(complement U)",
        "independent_method": "new bit-row matrix reconstruction, integer (I+M)^50 vector, exact squared comparison, integer trace charpoly",
        "uses_floating_eigenvalues": False,
        "row_digest_sha256": digest.hexdigest(),
        "wall_seconds": time.monotonic() - start,
        "python": platform.python_version(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    sources = [Path(__file__).resolve(), rows_path, args.output / "summary.json"]
    (args.output / "manifest.sha256").write_text(
        "\n".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p) for p in sources)
        + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
