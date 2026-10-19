#!/usr/bin/env python3
"""Stream a ppnn ``check`` document without retaining 518400 graph records.

The public detailed report is convenient for small fixtures but stores one
large dictionary per raw Wick.  ppnn needs the same output objects without
that memory cost.  This writer enumerates the raw contractions directly,
uses the generic reducer's streaming relation callback, and writes the final
``basis_contract``, ``basis``, ``origen_wick_raw`` and ``origen_wick`` layers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from verify_code.ppnn_state_quotient_audit import _operators
from kernels.wick_reducer import expression_output, operator_model
from kernels.wick_reducer.reducer import ReductionOptions, reduce_operators


def _call(equation: str, operands: Sequence[str]) -> str:
    return f"contract({repr(equation)}, {', '.join(operands)})"


def _complex(value) -> str:
    return expression_output._python_complex(value)


def write(path: Path) -> Dict[str, object]:
    operators = tuple(_operators())
    fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    axes = expression_output._bare_axes(operators)
    safe_name = "ppnn"
    temp_path = path.with_suffix(path.suffix + ".relations.tmp")
    relation_weights: Dict[Tuple[int, Tuple[int, ...]], complex] = defaultdict(complex)
    relation_views = set()
    relation_count = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        output.write("# Auto-generated ppnn check document (streamed).\n")
        output.write("from opt_einsum import contract\n\n")
        output.write("# Original Wick IDs start at 1; output slots start at 0.\n")
        output.write("ppnn_coefficients = []\n")
        output.write("ppnn_origen_wick_raw = {}\nppnn_origen_wick_sign = {}\n")
        output.write("ppnn_origen_wick = {}\nppnn_origen_wick_to_independent = {}\n")
        output.write("\n")
        # Raw direct contracts are independent of reduction and can be emitted
        # while the exact Wick generator is traversed once.
        raw_count = 0
        for raw_count, term in enumerate(core.enumerate_wicks(quarks, antiquarks), 1):
            equation, operands_names, _ = expression_output._contraction_expression_from_fields(
                operators, fields, quarks, antiquarks, term.pairing,
                isospin_degenerate=True,
            )
            output.write(
                f"ppnn_origen_wick_raw[{raw_count - 1}] = "
                f"{_call(equation, operands_names)}\n"
            )
            output.write(f"ppnn_origen_wick_sign[{raw_count - 1}] = {term.sign}\n")
        if raw_count != 518400:
            raise AssertionError(f"expected 518400 raw Wicks, got {raw_count}")

    relation_stream = temp_path.open("w", encoding="utf-8")

    def relation_callback(relation, candidate_state, basis, _term):
        nonlocal relation_count
        relation_count += 1
        basis_maps = {
            int(entry.wick_index): entry.state.bare_axis_map
            for entry in basis
        }
        terms = []
        for entry, coefficient in zip(basis, relation["coefficients"]):
            basis_id = int(entry.wick_index)
            permutation = expression_output._transpose_for_relation(
                candidate_state.bare_axis_map,
                basis_maps.get(basis_id, {}), axes,
            )
            coefficient = complex(float(coefficient[0]), float(coefficient[1]))
            terms.append((basis_id, coefficient, tuple(permutation)))
            view = (basis_id, tuple(permutation))
            relation_views.add(view)
            relation_weights[view] += coefficient
        term_source = ", ".join(
            "{" + f"'basis_wick_id': {basis_id}, 'coefficient': {_complex(coefficient)}, "
            f"'transpose': {permutation}" + "}"
            for basis_id, coefficient, permutation in terms
        )
        slot = int(relation["wick_index"]) - 1
        relation_stream.write(
            f"ppnn_origen_wick[{slot}] = "
            + " + ".join(
                f"({_complex(coefficient)} * ppnn_basis[{basis_id, permutation}])"
                for basis_id, coefficient, permutation in terms
            )
            + "\n"
        )
        relation_stream.write(
            f"ppnn_origen_wick_to_independent[{slot}] = "
            f"{{'wick_id': {relation['wick_index']}, 'terms': [{term_source}]}}\n"
        )

    report = reduce_operators(
        operators,
        label="ppnn",
        mode="independent",
        options=ReductionOptions(isospin_degenerate=True),
        _relation_term_callback=relation_callback,
    )
    relation_stream.close()

    independent = report["independent_wicks"]
    identity = tuple(range(len(axes)))
    basis_views = {(int(item["wick_index"]), identity) for item in independent}
    basis_views.update(relation_views)
    basis_views = sorted(
        basis_views,
        key=lambda item: (int(item[0]), tuple(item[1])),
    )
    basis_position = {view: index for index, view in enumerate(basis_views)}

    with path.open("a", encoding="utf-8") as output:
        output.write("\nppnn_basis_contract = {}\n")
        for item in independent:
            wick_id = int(item["wick_index"])
            equation, operands_names, _ = expression_output._contraction_expression_from_fields(
                operators, fields, quarks, antiquarks, item["pairing"],
                isospin_degenerate=True,
            )
            output.write(
                f"ppnn_basis_contract[{wick_id}] = "
                f"{_call(equation, operands_names)}\n"
            )
        output.write("\nppnn_basis = {}\n")
        for index, (wick_id, permutation) in enumerate(basis_views):
            sign = next(int(item["wick_sign"]) for item in independent
                        if int(item["wick_index"]) == wick_id)
            expression = f"ppnn_basis_contract[{wick_id}]"
            if sign == -1:
                expression = f"(-{expression})"
            if permutation != identity:
                expression += f".transpose({', '.join(map(str, permutation))})"
            output.write(f"ppnn_basis[{wick_id, permutation}] = {expression}\n")

        coefficients = []
        for view in basis_views:
            coefficient = relation_weights.get(view, 0.0 + 0.0j)
            if view[1] == identity:
                coefficient += 1.0 + 0.0j
            coefficients.append(coefficient)
        output.write(
            "ppnn_coefficients = [" + ", ".join(_complex(value) for value in coefficients) + "]\n"
        )
        output.write("ppnn_result = sum(coefficient * ppnn_basis[basis_key] "
                     "for basis_key, coefficient in zip(ppnn_basis, ppnn_coefficients) if coefficient)\n")
        output.write("\n# Definition: ppnn_origen_wick[slot] = "
                     "ppnn_origen_wick_sign[slot] * ppnn_origen_wick_raw[slot].\n")
        output.write("# The following expressions reconstruct the signed raw Wick from ppnn_basis.\n")
        output.write("# Independent Wick expressions and identity basis views.\n")
        for item in independent:
            wick_id = int(item["wick_index"])
            sign = int(item["wick_sign"])
            view = (wick_id, identity)
            expression = f"ppnn_basis[{view}]"
            if sign == -1:
                expression = f"(-{expression})"
            output.write(f"ppnn_origen_wick[{wick_id - 1}] = {expression}\n")
            output.write(
                f"ppnn_origen_wick_to_independent[{wick_id - 1}] = "
                f"{{'wick_id': {wick_id}, 'terms': [{{'basis_wick_id': {wick_id}, "
                f"'coefficient': 1, 'transpose': {identity}}}]}}\n"
            )
        output.write("\n# Streamed dependency expressions.\n")
        with temp_path.open("r", encoding="utf-8") as relation_input:
            shutil.copyfileobj(relation_input, output)
        output.write("\n")

    temp_path.unlink(missing_ok=True)
    return {
        "name": "ppnn", "mode": "check", "raw_wick_count": raw_count,
        "factor_buckets": report["peram_incidence_buckets"],
        "basis_contract_count": report["independent_count"],
        "dependent_count": report["dependent_count"],
        "relation_callback_count": relation_count,
        "basis_view_count": len(basis_views), "path": str(path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("verify_output/check/ppnn_check.py"))
    args = parser.parse_args(argv)
    print(write(args.output))


if __name__ == "__main__":
    main()
