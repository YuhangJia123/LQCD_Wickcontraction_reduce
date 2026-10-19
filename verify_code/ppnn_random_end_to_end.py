#!/usr/bin/env python3
"""End-to-end random-peram audit for the generic factor-graph reducer.

The ppnn output has eight open spin axes, so materializing one dense
``(4,)*8`` tensor for all 518,400 raw terms is unnecessarily large.  This
audit contracts every raw expression with eight fixed random spin vectors,
which gives an exact scalar projection of the same open-index identity.  It
then reconstructs that scalar from the reducer's independent basis terms and
all streamed dependency relations.

The raw side evaluates all 518,400 Wick contractions.  The reduced side
evaluates each basis contraction once; relation coefficients and axis maps are
then applied without additional tensor contractions.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import opt_einsum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from kernels.wick_reducer import expression_output
from kernels.wick_reducer import operator_model
from kernels.wick_reducer.reducer import ReductionOptions, reduce_operators
from verify_code.ppnn_state_quotient_audit import _operators


def _random_inputs(seed: int, axes: Sequence[str], color_dimension: int):
    rng = np.random.default_rng(seed)

    def random_complex(shape):
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ) / np.sqrt(2.0)

    registry = core._euclidean_gamma_matrices()
    values = {
        name.replace("Cgamma", "C_g").replace("gamma", "g"):
        np.asarray(matrix, dtype=np.complex128)
        for name, matrix in registry.items()
    }
    # The ppnn fixture uses only sin_sou, but all four directions are supplied
    # so the same evaluator remains useful for arbitrary operator input.
    for direction in ("sin_sin", "sin_sou", "sou_sin", "sou_sou"):
        values[f"peram_u_{direction}"] = random_complex(
            (4, color_dimension, color_dimension, 4)
        )
    for name in ("VVV_sink", "conj_VVV_source"):
        raw_vvv = random_complex((color_dimension,) * 3)
        vvv = np.zeros_like(raw_vvv)
        for permutation in itertools.permutations(range(3)):
            inversions = sum(
                permutation[i] > permutation[j]
                for i in range(3) for j in range(i + 1, 3)
            )
            vvv += (-1 if inversions % 2 else 1) * raw_vvv.transpose(permutation)
        values[name] = vvv / 6.0
    projections = tuple(random_complex((4,)) for _axis in axes)
    return values, projections


def _scalar_contract(
    operators,
    pairing: Sequence[int],
    values: Mapping[str, np.ndarray],
    projections: Sequence[np.ndarray],
    fields=None,
    quarks=None,
    antiquarks=None,
) -> complex:
    if fields is None or quarks is None or antiquarks is None:
        fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    equation, operands, axes = expression_output._contraction_expression_from_fields(
        operators, fields, quarks, antiquarks, pairing,
        isospin_degenerate=True,
    )
    if tuple(axes) != tuple("ABCDEFGH"):
        raise AssertionError(f"unexpected ppnn open-axis order: {axes}")
    lhs, output = equation.split("->")
    projected_equation = lhs + "," + ",".join(output) + "->"
    arguments = [values[name] for name in operands]
    arguments.extend(projections)
    return complex(opt_einsum.contract(projected_equation, *arguments, optimize="greedy"))


def _pairing_table(operators):
    _fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    pairings = np.empty((518400, len(quarks)), dtype=np.int16)
    signs = np.empty((518400,), dtype=np.int8)
    count = 0
    for term in core.enumerate_wicks(quarks, antiquarks):
        if count >= len(pairings):
            raise AssertionError("ppnn raw Wick count exceeded the expected 518,400")
        pairings[count] = term.pairing
        signs[count] = term.sign
        count += 1
    if count != len(pairings):
        raise AssertionError(f"expected 518,400 raw Wicks, got {count}")
    return pairings, signs


def _projected_basis_scalar(
    operators,
    pairing: Sequence[int],
    values: Mapping[str, np.ndarray],
    projections: Sequence[np.ndarray],
    permutation: Sequence[int] | None = None,
    fields=None,
    quarks=None,
    antiquarks=None,
) -> complex:
    if permutation is None:
        return _scalar_contract(
            operators, pairing, values, projections,
            fields, quarks, antiquarks,
        )
    inverse = {target: source for source, target in enumerate(permutation)}
    reordered = tuple(projections[inverse[index]] for index in range(len(projections)))
    return _scalar_contract(
        operators, pairing, values, reordered, fields, quarks, antiquarks,
    )


def _safe_literal(node):
    """Read only the literal forms emitted in the generated check mapping."""
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (int, float, complex, str, type(None))
    ):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [_safe_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _safe_literal(key): _safe_literal(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_literal(node.operand)
        if isinstance(value, (int, float, complex)):
            return value if isinstance(node.op, ast.UAdd) else -value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "complex"
        and not node.keywords
        and len(node.args) in (1, 2)
    ):
        args = tuple(_safe_literal(arg) for arg in node.args)
        if all(isinstance(value, (int, float)) for value in args):
            return complex(*args)
    raise ValueError("check mapping contains a non-literal expression")


def _iter_check_mappings(path: Path):
    prefix = "ppnn_origen_wick_to_independent["
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.startswith(prefix):
                continue
            statement = ast.parse(line, filename=str(path)).body
            if len(statement) != 1 or not isinstance(statement[0], ast.Assign):
                raise ValueError(f"invalid mapping assignment at line {line_number}")
            assignment = statement[0]
            target = assignment.targets[0]
            if (
                not isinstance(target, ast.Subscript)
                or not isinstance(target.value, ast.Name)
                or target.value.id != "ppnn_origen_wick_to_independent"
                or not isinstance(target.slice, ast.Constant)
                or not isinstance(target.slice.value, int)
            ):
                raise ValueError(f"invalid mapping slot at line {line_number}")
            value = _safe_literal(assignment.value)
            if not isinstance(value, dict):
                raise ValueError(f"invalid mapping value at line {line_number}")
            yield int(target.slice.value), value


def _verify_check_file(
    path: Path,
    *,
    raw_values: np.ndarray,
    signs: np.ndarray,
    basis_view_values: Mapping[Tuple[int, Tuple[int, ...]], complex],
    independent_ids: set[int],
) -> Dict[str, object]:
    seen = bytearray(len(raw_values))
    count = 0
    max_absolute = 0.0
    max_relative = 0.0
    worst_slot = None
    for slot, relation in _iter_check_mappings(path):
        if slot < 0 or slot >= len(raw_values) or seen[slot]:
            raise AssertionError(f"duplicate or invalid check slot {slot}")
        seen[slot] = 1
        wick_id = int(relation["wick_id"])
        if wick_id != slot + 1:
            raise AssertionError(f"slot {slot} contains Wick ID {wick_id}")
        direct = int(signs[wick_id - 1]) * raw_values[wick_id - 1]
        reconstructed = 0.0 + 0.0j
        terms = relation.get("terms")
        if not isinstance(terms, list) or not terms:
            raise AssertionError(f"Wick ID {wick_id} has no basis terms")
        for term in terms:
            basis_id = int(term["basis_wick_id"])
            if basis_id not in independent_ids:
                raise AssertionError(f"Wick ID {wick_id} refers to non-basis {basis_id}")
            permutation = tuple(int(value) for value in term["transpose"])
            key = (basis_id, permutation)
            if key not in basis_view_values:
                raise AssertionError(f"missing evaluated basis view {key}")
            reconstructed += (
                complex(term["coefficient"])
                * int(signs[basis_id - 1])
                * basis_view_values[key]
            )
        absolute = abs(direct - reconstructed)
        relative = absolute / max(abs(direct), abs(reconstructed), 1.0e-30)
        if absolute > max_absolute:
            max_absolute = absolute
            worst_slot = slot
        max_relative = max(max_relative, relative)
        count += 1
    if count != len(raw_values) or not all(seen):
        raise AssertionError(
            f"check file covers {count} of {len(raw_values)} raw Wick slots"
        )
    return {
        "check_mapping_count": count,
        "check_max_absolute_residual": float(max_absolute),
        "check_max_relative_residual": float(max_relative),
        "check_worst_slot": worst_slot,
        "check_passed": bool(max_relative < 1.0e-10),
    }


def run_seed(
    seed: int,
    *,
    skip_reducer: bool = False,
    raw_cache: str | None = None,
    expected_raw_sum: complex | None = None,
    color_dimension: int = 3,
    check_path: Path | None = None,
) -> Dict[str, object]:
    operators = _operators()
    pairings, signs = _pairing_table(operators)
    axes = expression_output._bare_axes(operators)
    fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    values, projections = _random_inputs(seed, axes, color_dimension)

    cache_path = Path(raw_cache) if raw_cache else None
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        cached_pairings = np.asarray(cached["pairings"], dtype=np.int16)
        cached_signs = np.asarray(cached["signs"], dtype=np.int8)
        if not np.array_equal(cached_pairings, pairings) or not np.array_equal(cached_signs, signs):
            raise AssertionError("raw cache does not match the ppnn Wick enumeration")
        pairings = cached_pairings
        signs = cached_signs
        raw_values = np.asarray(cached["raw_values"], dtype=np.complex128)
        projections = tuple(np.asarray(item) for item in cached["projections"])
        raw_sum = complex(np.sum(signs.astype(np.complex128) * raw_values))
        raw_seconds = 0.0
    elif expected_raw_sum is not None:
        raw_values = None
        raw_sum = complex(expected_raw_sum)
        raw_seconds = None
    else:
        raw_values = np.empty((len(pairings),), dtype=np.complex128)
        raw_sum = 0.0 + 0.0j
        started = time.perf_counter()
        for index, pairing in enumerate(pairings):
            value = _scalar_contract(
                operators, tuple(int(item) for item in pairing), values, projections,
                fields, quarks, antiquarks,
            )
            raw_values[index] = value
            raw_sum += int(signs[index]) * value
        raw_seconds = time.perf_counter() - started
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                pairings=pairings,
                signs=signs,
                raw_values=raw_values,
                projections=np.stack(projections),
            )

    if skip_reducer:
        return {
            "seed": seed,
            "raw_wick_count": len(pairings),
            "raw_sum": [raw_sum.real, raw_sum.imag],
            "raw_contract_seconds": raw_seconds,
        }

    if raw_values is None:
        raise ValueError("full ppnn check requires direct raw contractions")
    basis_values: Dict[int, complex] = {}
    basis_view_values: Dict[Tuple[int, Tuple[int, ...]], complex] = {}
    relation_count = 0
    relation_residuals = []
    reconstructed = 0.0 + 0.0j

    def relation_callback(relation, candidate_state, basis, term):
        nonlocal relation_count, reconstructed
        relation_count += 1
        basis_maps = {
            int(entry.wick_index): entry.state.bare_axis_map
            for entry in basis
        }
        relation_value = 0.0 + 0.0j
        for entry, coefficient in zip(basis, relation["coefficients"]):
            basis_id = int(entry.wick_index)
            permutation = expression_output._transpose_for_relation(
                candidate_state.bare_axis_map,
                basis_maps.get(basis_id, {}),
                axes,
            )
            view_key = (basis_id, tuple(permutation))
            if view_key not in basis_view_values:
                basis_view_values[view_key] = _projected_basis_scalar(
                    operators,
                    tuple(int(value) for value in pairings[basis_id - 1]),
                    values,
                    projections,
                    permutation,
                    fields,
                    quarks,
                    antiquarks,
                )
            factor = complex(float(coefficient[0]), float(coefficient[1]))
            relation_value = relation_value + (
                factor * int(signs[basis_id - 1]) * basis_view_values[view_key]
            )
        direct_signed = int(term.sign) * raw_values[int(relation["wick_index"]) - 1]
        relation_residuals.append((
            abs(direct_signed - relation_value),
            abs(direct_signed - relation_value)
            / max(abs(direct_signed), abs(relation_value), 1.0e-30),
        ))
        reconstructed += relation_value

    reducer_started = time.perf_counter()
    report = reduce_operators(
        operators,
        label="ppnn",
        mode="independent",
        options=ReductionOptions(isospin_degenerate=True),
        _relation_term_callback=relation_callback,
    )
    reducer_seconds = time.perf_counter() - reducer_started

    independent_ids = set()
    for item in report["independent_wicks"]:
        wick_id = int(item["wick_index"])
        independent_ids.add(wick_id)
        identity = tuple(range(len(axes)))
        key = (wick_id, identity)
        if key not in basis_view_values:
            basis_view_values[key] = _projected_basis_scalar(
                operators, tuple(int(value) for value in pairings[wick_id - 1]),
                values, projections, None, fields, quarks, antiquarks,
            )
        basis_values[wick_id] = basis_view_values[key]
        reconstructed += int(signs[wick_id - 1]) * basis_values[wick_id]

    check_result = {}
    if check_path is not None:
        check_result = _verify_check_file(
            check_path,
            raw_values=raw_values,
            signs=signs,
            basis_view_values=basis_view_values,
            independent_ids=independent_ids,
        )
        if not check_result["check_passed"]:
            raise AssertionError(
                "generated check mappings failed random-peram verification: "
                f"max relative residual={check_result['check_max_relative_residual']}"
            )

    total_residual = abs(raw_sum - reconstructed)
    total_scale = max(abs(raw_sum), abs(reconstructed), 1.0e-30)
    return {
        "seed": seed,
        "color_dimension": color_dimension,
        "raw_wick_count": len(pairings),
        "factor_buckets": report["peram_incidence_buckets"],
        "basis_contract_count": report["independent_count"],
        "dependent_count": report["dependent_count"],
        "relation_count": relation_count,
        "basis_evaluations": len(basis_values),
        "basis_view_evaluations": len(basis_view_values),
        **check_result,
        "raw_contract_seconds": raw_seconds,
        "reducer_seconds": reducer_seconds,
        "raw_sum": [raw_sum.real, raw_sum.imag],
        "reconstructed_sum": [reconstructed.real, reconstructed.imag],
        "max_relation_absolute_residual": max((item[0] for item in relation_residuals), default=0.0),
        "max_relation_relative_residual": max((item[1] for item in relation_residuals), default=0.0),
        "total_absolute_residual": float(total_residual),
        "total_relative_residual": float(total_residual / total_scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--skip-reducer", action="store_true")
    parser.add_argument("--raw-cache", default=None)
    parser.add_argument("--check", type=Path, default=None,
                        help="verify every mapping in a generated ppnn check file")
    parser.add_argument("--expected-raw-real", type=float, default=None)
    parser.add_argument("--expected-raw-imag", type=float, default=None)
    parser.add_argument("--color-dimension", type=int, action="append", default=None)
    args = parser.parse_args()
    if args.check is not None and not args.check.is_file():
        parser.error(f"check file does not exist: {args.check}")
    seeds = tuple(args.seed or (20261011,))
    started = time.perf_counter()
    expected_raw_sum = None
    if args.expected_raw_real is not None or args.expected_raw_imag is not None:
        if args.expected_raw_real is None or args.expected_raw_imag is None:
            parser.error("--expected-raw-real and --expected-raw-imag must be supplied together")
        expected_raw_sum = complex(args.expected_raw_real, args.expected_raw_imag)
    dimensions = tuple(args.color_dimension or (3, 6))
    if any(dimension < 3 for dimension in dimensions):
        parser.error("--color-dimension must be >= 3")
    results = [run_seed(
        seed,
        skip_reducer=args.skip_reducer,
        raw_cache=args.raw_cache,
        expected_raw_sum=expected_raw_sum,
        color_dimension=dimension,
        check_path=args.check,
    ) for dimension in dimensions for seed in seeds]
    print(json.dumps({
        "seeds": results,
        "total_seconds": time.perf_counter() - started,
        "note": "raw side covers all 518,400 Wick terms; open indices are tested by a fixed random scalar projection",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
