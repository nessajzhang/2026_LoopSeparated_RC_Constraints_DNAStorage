#!/usr/bin/env python3
"""Certify the global DNA m=2 paired spectral maximum.

For every one of the 65,536 subsets A of DNA dinucleotides, the script first
builds exact rational Collatz--Wielandt upper bounds for rho(A) and
rho(A^perp).  All strict comparisons with the proposed maximum are performed
using integer arithmetic.  The few unresolved equality candidates are then
checked by exact symbolic characteristic polynomials.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loop_alpha_audit.spectral import (  # noqa: E402
    collatz_rational_upper,
    dna_m2_matrix,
    dna_m2_perp_matrix,
    target_lambda_decimal,
    target_lambda_squared_comparison,
)


OUTPUT = ROOT / "output"
CERTIFICATE = OUTPUT / "dna_m2_spectral_collatz_certificate.tsv.gz"
SUMMARY = OUTPUT / "dna_m2_spectral_certificate_summary.json"
STEPS = 50
EXPECTED_MAXIMIZERS = (1911, 4383, 8946, 20292, 45243, 56589, 61152, 63624)


def characteristic_polynomial(matrix: tuple[tuple[int, ...], ...]) -> str:
    variable = sp.symbols("x")
    expression = sp.factor(sp.Matrix(matrix).charpoly(variable).as_expr())
    return str(expression)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    unresolved: list[dict[str, object]] = []
    digest = hashlib.sha256()
    resolved = 0

    # Fix the gzip header timestamp so rerunning the certificate produces the
    # same compressed artifact, not merely the same decompressed rows.
    with CERTIFICATE.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(
                    [
                        "mask",
                        "rho_A_upper_num",
                        "rho_A_upper_den",
                        "rho_Aperp_upper_num",
                        "rho_Aperp_upper_den",
                        "product_upper_num",
                        "product_upper_den",
                        "status",
                    ]
                )
                for mask in range(2**16):
                    matrix = dna_m2_matrix(mask)
                    perpendicular = dna_m2_perp_matrix(matrix)
                    upper_a = collatz_rational_upper(matrix, STEPS)
                    upper_b = collatz_rational_upper(perpendicular, STEPS)
                    product_upper = upper_a * upper_b
                    comparison = target_lambda_squared_comparison(product_upper)
                    status = (
                        "strictly_below_target" if comparison < 0 else "symbolic_check_required"
                    )
                    if comparison < 0:
                        resolved += 1
                    else:
                        unresolved.append(
                            {
                                "mask": mask,
                                "charpoly_A": characteristic_polynomial(matrix),
                                "charpoly_Aperp": characteristic_polynomial(perpendicular),
                            }
                        )
                    row = [
                        mask,
                        upper_a.numerator,
                        upper_a.denominator,
                        upper_b.numerator,
                        upper_b.denominator,
                        product_upper.numerator,
                        product_upper.denominator,
                        status,
                    ]
                    encoded = ("\t".join(map(str, row)) + "\n").encode()
                    digest.update(encoded)
                    writer.writerow(row)

    unresolved_masks = tuple(item["mask"] for item in unresolved)
    if unresolved_masks != EXPECTED_MAXIMIZERS:
        raise AssertionError(f"unexpected unresolved masks: {unresolved_masks}")

    expected_polynomials = {
        ("x**3*(x - 3)", "x**2*(x**2 - x - 3)"),
        ("x**2*(x**2 - x - 3)", "x**3*(x - 3)"),
    }
    for item in unresolved:
        pair = (item["charpoly_A"], item["charpoly_Aperp"])
        if pair not in expected_polynomials:
            raise AssertionError(f"unexpected symbolic equality candidate: {item}")
        item["certified_rho_product"] = target_lambda_decimal()

    summary = {
        "status": "COMPUTE_CERTIFIED",
        "alphabet": ["A", "C", "G", "T"],
        "m": 2,
        "subsets_checked": 2**16,
        "collatz_steps": STEPS,
        "strictly_resolved_by_exact_rational_upper": resolved,
        "exact_symbolic_maximizers": unresolved,
        "global_maximum": target_lambda_decimal(),
        "certificate_rows_sha256": digest.hexdigest(),
        "certificate_file": CERTIFICATE.name,
        "proof_note": (
            "For every strict row, rho(A)rho(Aperp) is bounded by the product "
            "of two exact Collatz rational bounds, and that rational product is "
            "proved smaller than 3(1+sqrt(13))/2 by an exact squared-integer "
            "comparison. The eight remaining matrices have the displayed exact "
            "characteristic-polynomial pair, hence product exactly equal to the target."
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
