#!/usr/bin/env python3
"""Sample numerical audit of the complete streamed ppnn check mapping."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import opt_einsum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from verify_code import check_basis_vs_original as audit
from verify_code.ppnn_state_quotient_audit import _operators
from kernels.wick_reducer import expression_output, operator_model


def _mapping(path: Path, sample_count: int):
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ppnn_origen_wick_to_independent["):
                continue
            slot_text, literal = line.split(" = ", 1)
            slot = int(slot_text.split("[", 1)[1].split("]", 1)[0])
            if slot % max(1, 514102 // sample_count) == 0:
                rows[slot] = ast.literal_eval(literal)
    if len(rows) < sample_count // 2:
        raise AssertionError(f"only found {len(rows)} sampled dependency rows")
    return rows


def _term_tables(operators):
    fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    pairings = {}
    signs = {}
    for wick_id, term in enumerate(core.enumerate_wicks(quarks, antiquarks), 1):
        pairings[wick_id] = tuple(term.pairing)
        signs[wick_id] = int(term.sign)
    return fields, quarks, antiquarks, pairings, signs


def run(path: Path, color_dimension: int, sample_count: int, seed: int):
    operators = tuple(_operators())
    fields, quarks, antiquarks, pairings, signs = _term_tables(operators)
    rows = _mapping(path, sample_count)
    values, projections = audit._random_values(operators, color_dimension, seed)
    residuals = []
    for slot, relation in rows.items():
        wick_id = int(relation["wick_id"])
        direct = signs[wick_id] * audit._projected_contract(
            operators, pairings[wick_id], values, projections,
        )
        reconstructed = 0.0 + 0.0j
        for term in relation["terms"]:
            basis_id = int(term["basis_wick_id"])
            coefficient = complex(term["coefficient"])
            permutation = tuple(term["transpose"])
            basis_value = audit._projected_contract(
                operators, pairings[basis_id], values, projections,
                projection_permutation=permutation,
            )
            reconstructed += coefficient * signs[basis_id] * basis_value
        absolute = abs(direct - reconstructed)
        relative = absolute / max(abs(direct), abs(reconstructed), 1.0e-30)
        residuals.append((absolute, relative, slot, wick_id))
    max_absolute, max_relative, slot, wick_id = max(residuals)
    return {
        "mode": "sampled_check",
        "color_dimension": color_dimension,
        "seed": seed,
        "sample_count": len(residuals),
        "max_absolute_residual": max_absolute,
        "max_relative_residual": max_relative,
        "worst_slot": slot,
        "worst_wick_id": wick_id,
        "passed": bool(max_relative < 1.0e-10),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path, default=Path("verify_output/check/ppnn_check.py"))
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--color-dimension", type=int, action="append", default=None)
    args = parser.parse_args(argv)
    dimensions = tuple(args.color_dimension or (3, 6))
    records = [run(args.check, dimension, args.sample_count, 20261011 + index)
               for index, dimension in enumerate(dimensions)]
    print(json.dumps({"records": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
