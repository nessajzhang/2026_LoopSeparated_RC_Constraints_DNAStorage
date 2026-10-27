"""Pinned ViennaRNA DNA conditions and auditable folding observations.

Predictions are not measured physical error probabilities. A maximal stem's
inner span differs from the spans of all its length-m contiguous subruns;
both are recorded explicitly, with zero-based inclusive pair coordinates.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from matched_dinucleotide import validate_dna

WC = {"AT", "TA", "CG", "GC"}
GT = {"GT", "TG"}


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    temperature_c: float
    salt_molar: float
    wc_only: bool

    @property
    def model_id(self) -> str:
        return f"T{self.temperature_c:g}_S{self.salt_molar:g}_{'WC' if self.wc_only else 'default'}"


def model_specs() -> tuple[ModelSpec, ...]:
    return tuple(
        ModelSpec(t, salt, wc)
        for t, salt in ((37, 1.021), (25, 1.021), (50, 1.021), (37, 0.05), (37, 0.15))
        for wc in (False, True)
    )


def effective_fields(md) -> dict:
    fields = {}
    for key in dir(md):
        if key.startswith("_") or key in {"this", "thisown"}:
            continue
        value = getattr(md, key)
        if callable(value):
            continue
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            # The alias short* is not exposed as a Python-indexable array.
            fields[key] = {"binding_type": type(value).__name__, "serialized": False}
        else:
            fields[key] = value
    return fields


def make_md(spec: ModelSpec):
    import RNA

    md = RNA.md()
    wanted = {
        "temperature": spec.temperature_c,
        "salt": spec.salt_molar,
        "dangles": 2,
        "noGU": int(spec.wc_only),
        "noGUclosure": 0,
        "noLP": 0,
        "compute_bpp": 1,
        "min_loop_size": 3,
    }
    for key, value in wanted.items():
        if not hasattr(md, key):
            raise NotImplementedError(f"UNSUPPORTED_MODEL_FIELD:{key}")
        setattr(md, key, value)
        actual = getattr(md, key)
        if not math.isclose(actual, value, rel_tol=0, abs_tol=1e-12):
            raise NotImplementedError(f"UNEFFECTIVE_MODEL_FIELD:{key}={actual},wanted={value}")
    return md


def check_native_fields(requested: dict, actual: dict, n: int) -> dict:
    differences = {
        key: [value, actual.get(key)]
        for key, value in requested.items()
        if value != actual.get(key)
    }
    if any(
        key not in {"max_bp_span", "window_size"} or values != [-1, n]
        for key, values in differences.items()
    ):
        raise AssertionError(f"Unexplained native model difference:{differences}")
    return differences


def prepare_models(parameter_file: Path) -> dict:
    import RNA

    if RNA.__version__ != "2.7.2":
        raise RuntimeError(f"Requires ViennaRNA2.7.2, found{RNA.__version__}")
    parameter_file.parent.mkdir(parents=True, exist_ok=True)
    if not RNA.params_load_DNA_Mathews2004():
        raise RuntimeError("DNA_Mathews2004 parameter loader failed")
    if not RNA.params_save(str(parameter_file)):
        raise RuntimeError("Active parameter export failed")
    records = {}
    for spec in model_specs():
        try:
            md = make_md(spec)
            fields = effective_fields(md)
            # Construction probes that the C compound really receives options;
            # no MFE/PF is evaluated here (zero folding observations).
            fc = RNA.fold_compound("ACGT", md)
            compound_fields = effective_fields(fc.params.model_details)
            normalizations = check_native_fields(fields, compound_fields, 4)
            records[spec.model_id] = {
                "status": "API_FIELDS_EFFECTIVE_NOT_PHYSICAL_CALIBRATION",
                "requested": asdict(spec),
                "effective": fields,
                "effective_sha256": digest_json(fields),
                "native_probe_fields": compound_fields,
                "native_probe_normalization": normalizations,
            }
        except Exception as exc:
            records[spec.model_id] = {
                "status": "UNSUPPORTED_NOT_SUBSTITUTED",
                "requested": asdict(spec),
                "error": repr(exc),
            }
    return {
        "RNA_version": RNA.__version__,
        "RNA_module": RNA.__file__,
        "parameter_loader": "RNA.params_load_DNA_Mathews2004",
        "parameter_file": str(parameter_file),
        "parameter_sha256": hashlib.sha256(parameter_file.read_bytes()).hexdigest(),
        "models": records,
        "binding_notes": "alias short* recorded as nonserialized binding field; scalar fields, pair/rtype matrices retained. Salt/temperature are software sensitivities, not measured laboratory conditions.",
    }


def structure_features(sequence: str, structure: str, m: int) -> dict:
    validate_dna(sequence)
    if len(sequence) != len(structure) or type(m) is not int or m < 1:
        raise ValueError("structure length/m mismatch")
    stack, pairs = [], []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            pairs.append((stack.pop(), i))
        elif c != ".":
            raise ValueError("unsupported or unbalanced dot-bracket")
    if stack:
        raise ValueError("unclosed dot-bracket")
    pairset = set(pairs)
    stems, subruns = [], []
    for i, j in sorted(pairset):
        if (i - 1, j + 1) in pairset:
            continue
        run = []
        while (i, j) in pairset:
            run.append([i, j])
            i, j = i + 1, j - 1
        classes = [sequence[a] + sequence[b] for a, b in run]
        record = {
            "pairs": run,
            "length": len(run),
            "inner_span": run[-1][1] - run[-1][0] - 1,
            "pair_types": classes,
            "wc_only": all(c in WC for c in classes),
            "has_GT": any(c in GT for c in classes),
            "has_other": any(c not in WC | GT for c in classes),
        }
        stems.append(record)
        for begin in range(len(run) - m + 1):
            selected = run[begin : begin + m]
            types = classes[begin : begin + m]
            subruns.append(
                {
                    "stem_index": len(stems) - 1,
                    "offset": begin,
                    "pairs": selected,
                    "inner_span": selected[-1][1] - selected[-1][0] - 1,
                    "pair_types": types,
                    "wc_only": all(c in WC for c in types),
                    "has_GT": any(c in GT for c in types),
                }
            )
    return {
        "mfe_pairs": sorted([list(p) for p in pairs]),
        "stems": stems,
        "m_subruns": subruns,
        "max_stem": max((s["length"] for s in stems), default=0),
        "has_stem_ge_m": any(s["length"] >= m for s in stems),
    }


def bpp_features(sequence: str, matrix, *, wc_only: bool, tolerance: float = 1e-6) -> dict:
    n = len(sequence)
    if len(matrix) != n + 1 or any(len(row) != n + 1 for row in matrix):
        raise AssertionError("Unexpected BPP API layout: require(n+1)x(n+1)")
    paired = [0.0] * n
    sparse, gt_mass = [], 0.0
    for i in range(n + 1):
        for j in range(n + 1):
            value = float(matrix[i][j])
            if not math.isfinite(value) or not -tolerance <= value <= 1 + tolerance:
                raise AssertionError("Nonfinite/out-of-range BPP")
            if i == 0 or j <= i:
                if abs(value) > tolerance:
                    raise AssertionError("BPP is not 1-based strict upper-triangular")
                continue
            paired[i - 1] += value
            paired[j - 1] += value
            if sequence[i - 1] + sequence[j - 1] in GT:
                gt_mass += value
            if value != 0:
                sparse.append([i, j, value])
    if any(not -tolerance <= p <= 1 + tolerance for p in paired):
        raise AssertionError("Per-base pair probability violates[0,1]")
    fraction = 2 * sum(row[2] for row in sparse) / n
    if not -tolerance <= fraction <= 1 + tolerance or (wc_only and gt_mass > tolerance):
        raise AssertionError("Paired fraction or WC-only BPP invariant failed")
    return {
        "bpp_layout": "1-based(n+1)x(n+1),strict_upper_triangle;raw_nonzero_entries_saved",
        "bpp_nonzero": sparse,
        "ensemble_paired_fraction": fraction,
        "per_base_paired_probabilities": paired,
        "GT_pair_probability_mass": gt_mass,
        "bpp_probability_tolerance": tolerance,
    }


def fold_sequence(sequence: str, m: int, k: int, spec: ModelSpec, *, ensemble: bool) -> dict:
    import RNA

    validate_dna(sequence)
    if not sequence:
        raise ValueError("Folding requires a nonempty sequence")
    started = time.perf_counter_ns()
    output = {
        "sequence": sequence,
        "n": len(sequence),
        "model_id": spec.model_id,
        "m": m,
        "k": k,
        "ensemble_requested": ensemble,
        "energy_unit": "kcal/mol",
    }
    try:
        md = make_md(spec)
        fc = RNA.fold_compound(sequence, md)
        fields = effective_fields(fc.params.model_details)
        normalization = check_native_fields(effective_fields(md), fields, len(sequence))
        output["effective_model_sha256"] = digest_json(fields)
        output["effective_model_fields"] = fields
        output["native_model_normalization"] = normalization
        structure, mfe = fc.mfe()
        evaluated = float(fc.eval_structure(structure))
        if not math.isfinite(mfe) or abs(mfe - evaluated) > 1e-4 or mfe > 1e-4:
            raise AssertionError("MFE/eval/unpaired baseline invariant failed")
        features = structure_features(sequence, structure, m)
        if any(s["has_other"] for s in features["stems"]) or (
            spec.wc_only and any(s["has_GT"] for s in features["stems"])
        ):
            raise AssertionError("Unexpected pair type in predicted MFE")
        output.update(
            status="OK",
            mfe=float(mfe),
            mfe_structure=structure,
            evaluated_mfe=evaluated,
            **features,
        )
        output["mfe_ns"] = time.perf_counter_ns() - started
        if ensemble:
            pf_start = time.perf_counter_ns()
            fc.exp_params_rescale(mfe)
            pf_structure, free_energy = fc.pf()
            if not math.isfinite(free_energy) or free_energy > mfe + 1e-4:
                raise AssertionError("PF free energy must be finite and <=MFE")
            output.update(
                partition_free_energy=float(free_energy),
                pf_annotation=pf_structure,
                **bpp_features(sequence, fc.bpp(), wc_only=spec.wc_only),
            )
            output["pf_ns"] = time.perf_counter_ns() - pf_start
        output["wc_m_subruns_at_or_above_k"] = sum(
            r["wc_only"] and r["inner_span"] >= k for r in features["m_subruns"]
        )
    except Exception as exc:
        output.update(status="FOLD_FAILED_RETAINED", failure_reason=repr(exc))
    output["elapsed_ns"] = time.perf_counter_ns() - started
    return output
