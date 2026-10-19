#!/usr/bin/env python3
"""Numerically audit ``check`` output for arbitrary operator fixtures.

For one input fixture this script performs the same two evaluations exposed by
``expression_output``:

* ``origen_wick_raw``: every exact raw Wick contraction is evaluated directly;
* ``basis_contract``/``basis``: every independent representative is evaluated
  once and each raw Wick is reconstructed with the recorded coefficients and
  bare-axis permutation.

The comparison uses a random linear functional on the complete open-spin
tensor.  This avoids materializing the roughly 4^8-dimensional ppnn basis for
every one of its 518,400 raw terms, while still testing the recorded
coefficients and every reported axis permutation.  Distillation eigenvector
dimensions 3 and 6 are supported; normal mode always uses three colors,
Levi-Civita epsilon and identity delta.  Distillation VVV elementals are
generated as random fully antisymmetric rank-three tensors; VDV elementals
use random rank-two tensors.  The same random arrays and spin projection
vectors are shared by both paths, and u/d use one peram array in
isospin-degenerate mode.

The optional ``opt_einsum`` package is used when available.  NumPy's greedy
einsum planner is a correctness-preserving fallback, useful for small cases.
The ppnn fixture is intentionally opt-in because its 518,400 raw terms can
require substantial time and disk space in check mode.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from kernels.wick_reducer import expression_output, operator_model
from kernels.wick_reducer.reducer import ReductionOptions, reduce_operators

try:  # opt_einsum is faster for the large ppnn fixture.
    import opt_einsum  # type: ignore
except ImportError:  # pragma: no cover - exercised only in minimal runtimes
    opt_einsum = None


def ppnn_operators():
    """Load the ppnn fixture without making it part of the generic library."""
    from verify_code.ppnn_state_quotient_audit import _operators

    return tuple(_operators())


def _fully_antisymmetric_vvv(dimension: int, rng: np.random.Generator) -> np.ndarray:
    raw = (
        rng.standard_normal((dimension,) * 3)
        + 1j * rng.standard_normal((dimension,) * 3)
    )
    result = np.zeros_like(raw)
    for permutation in itertools.permutations(range(3)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )
        result += (-1 if inversions % 2 else 1) * raw.transpose(permutation)
    return result / 6.0


def _random_complex(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)


def _epsilon_color() -> np.ndarray:
    """Return the actual three-color Levi-Civita tensor used by normal mode."""
    tensor = np.zeros((3, 3, 3))
    for permutation in itertools.permutations(range(3)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(3) for j in range(i + 1, 3)
        )
        tensor[permutation] = -1 if inversions % 2 else 1
    return tensor


def _contract(equation: str, operands: Sequence[np.ndarray]) -> np.ndarray:
    if opt_einsum is not None:
        return np.asarray(opt_einsum.contract(equation, *operands, optimize="greedy"))
    return np.asarray(np.einsum(equation, *operands, optimize="greedy"))


class RandomTensorBank:
    """Lazily create one shared random tensor for each symbolic operand."""

    def __init__(self, seed: int, color_dimension: int, *, mode: str | None = None):
        self.rng = np.random.default_rng(seed)
        self.color_dimension = int(color_dimension)
        # ``None`` keeps the small standalone helper's historical name-based
        # behavior; audit_one passes the operator mode explicitly.
        self.mode = None if mode is None else str(mode)
        if self.mode not in (None, "normal", "distillation"):
            raise ValueError("RandomTensorBank mode must be normal or distillation")
        self.values: Dict[str, np.ndarray] = {}
        registry = core._euclidean_gamma_matrices()
        for name, matrix in registry.items():
            normalized = name.replace("Cgamma", "C_g").replace("gamma", "g")
            self.values[normalized] = np.asarray(matrix, dtype=np.complex128)

    def get(self, name: str) -> np.ndarray:
        if name in self.values:
            return self.values[name]
        if name == "epsilon_color":
            value = _epsilon_color()
        elif name == "delta_color":
            value = np.eye(3)
        elif name.startswith("peram_"):
            # Normal fields use color, independently of the requested
            # distillation eigenvector dimension.
            dimension = 3 if "_x_x" in name else self.color_dimension
            # Every mode uses the project-wide (V_sink,D_sink,V_source,D_source)
            # layout.
            value = _random_complex(self.rng, (dimension, 4, dimension, 4))
        elif name.startswith("VVV_") or name.startswith("conj_VVV_"):
            value = _fully_antisymmetric_vvv(self.color_dimension, self.rng)
        elif name.startswith("VDV_") or name.startswith("conj_VDV_"):
            value = _random_complex(self.rng, (self.color_dimension, self.color_dimension))
        else:
            raise KeyError(
                f"no random tensor rule for operand {name!r}; "
                "provide a concrete Gamma matrix in the fixture"
            )
        self.values[name] = value
        return value


def _evaluate_scalar(
    operators: Sequence[core.Operator],
    pairing: Sequence[int],
    bank: RandomTensorBank,
    *,
    isospin_degenerate: bool,
    projections: Sequence[np.ndarray],
    transpose: Optional[Sequence[int]] = None,
) -> complex:
    equation, operands, axes = expression_output.contraction_expression(
        operators, pairing, isospin_degenerate=isospin_degenerate,
    )
    values = [bank.get(name) for name in operands]
    if not axes:
        return complex(_contract(equation, values))
    if transpose is None:
        projected_vectors = tuple(projections)
    else:
        inverse = {
            target: source
            for source, target in enumerate(transpose)
        }
        projected_vectors = tuple(
            projections[inverse[index]] for index in range(len(transpose))
        )
    lhs = equation.split("->", 1)[0]
    projected_equation = lhs + "," + ",".join(axes) + "->"
    return complex(_contract(
        projected_equation,
        [*values, *projected_vectors],
    ))


def _residual(left: np.ndarray, right: np.ndarray) -> Tuple[float, float]:
    absolute = float(np.max(np.abs(left - right)))
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.0e-30)
    return absolute, absolute / scale


def audit_one(
    label: str,
    operators: Sequence[core.Operator],
    *,
    seed: int,
    color_dimension: int,
    write_check: Optional[Path] = None,
    isospin_degenerate: bool = True,
) -> Dict[str, object]:
    started = time.perf_counter()
    operators = tuple(operators)
    # Detailed mode is the in-memory form of expression_output's check mode.
    report = reduce_operators(
        operators, label=label, mode="detailed",
        options=ReductionOptions(isospin_degenerate=isospin_degenerate),
    )
    if write_check is not None:
        write_check.parent.mkdir(parents=True, exist_ok=True)
        write_check.write_text(
            expression_output.render_python_output(
                report, operators, name=label, mode="check",
            ),
            encoding="utf-8",
        )

    _fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    # Do not retain every ppnn WickTerm or open tensor.  The detailed report
    # already contains the exact pairing and sign for every raw Wick.
    exact_count = sum(1 for _term in core.enumerate_wicks(quarks, antiquarks))
    if exact_count != int(report["wick_count"]):
        raise AssertionError("detailed report and exact raw Wick enumeration disagree")
    bank = RandomTensorBank(seed, color_dimension, mode=operators[0].mode)
    axis_order = expression_output._bare_axes(operators)
    projection_rng = np.random.default_rng(seed + 1_000_003)
    projections = tuple(
        _random_complex(projection_rng, (4,))
        for _axis in axis_order
    )

    independent_items = report["independent_wicks"]
    independent_position = {
        int(item["wick_index"]): position
        for position, item in enumerate(independent_items)
    }
    independent_by_id = {
        int(item["wick_index"]): item
        for item in independent_items
    }
    # ``basis`` includes the representative Wick sign.  It is deliberately
    # kept separate from the unsigned ``basis_contract`` scalar.
    basis_contract: Dict[int, complex] = {}
    basis_sign: Dict[int, int] = {}
    for item in independent_items:
        wick_index = int(item["wick_index"])
        basis_contract[wick_index] = _evaluate_scalar(
            operators, item["pairing"], bank, isospin_degenerate=isospin_degenerate,
            projections=projections,
        )
        basis_sign[wick_index] = int(item["wick_sign"])
    basis_views: Dict[Tuple[int, Tuple[int, ...]], complex] = {}

    relations = {
        int(item["wick_index"]): item
        for item in report["dependent_relations"]
    }
    residuals = []
    for graph in report["graph_records"]:
        wick_index = int(graph["wick_index"])
        relation = relations.get(wick_index)
        terms_for_graph = expression_output._relation_terms_for_graph(
            graph, relation, independent_position, axis_order,
        )
        reconstructed = None
        for _position, basis_id, coefficient, permutation in terms_for_graph:
            view_key = (basis_id, tuple(permutation))
            if view_key not in basis_views:
                view = basis_sign[basis_id] * _evaluate_scalar(
                    operators,
                    independent_by_id[basis_id]["pairing"],
                    bank,
                    isospin_degenerate=isospin_degenerate,
                    projections=projections,
                    transpose=permutation,
                )
                basis_views[view_key] = view
            view = basis_views[view_key]
            contribution = coefficient * view
            reconstructed = contribution if reconstructed is None else reconstructed + contribution
        if reconstructed is None:
            raise AssertionError(f"Wick {wick_index} has no basis reconstruction terms")
        direct_signed = int(graph["wick_sign"]) * _evaluate_scalar(
            operators, graph["pairing"], bank, isospin_degenerate=isospin_degenerate,
            projections=projections,
        )
        absolute, relative = _residual(direct_signed, reconstructed)
        residuals.append((absolute, relative))

    max_absolute = max((item[0] for item in residuals), default=0.0)
    max_relative = max((item[1] for item in residuals), default=0.0)
    return {
        "label": label,
        "mode": "check",
        "operator_mode": report["operator_mode"],
        "isospin_symmetric": bool(isospin_degenerate),
        "seed": int(seed),
        "color_dimension": 3 if report["operator_mode"] == "normal" else int(color_dimension),
        "eigenvector_dimension": int(color_dimension) if report["operator_mode"] == "distillation" else None,
        "raw_wick_count": int(report["wick_count"]),
        "factor_buckets": int(report["peram_incidence_buckets"]),
        "basis_contract_count": int(report["independent_count"]),
        "dependent_count": int(report["dependent_count"]),
        "max_absolute_residual": max_absolute,
        "max_relative_residual": max_relative,
        "check_output": str(write_check) if write_check is not None else None,
        "elapsed_seconds": time.perf_counter() - started,
    }


def fixtures(names: Sequence[str]):
    library = operator_model.operator_library()
    for name in names:
        if name == "ppnn":
            yield name, ppnn_operators()
            continue
        if name not in library:
            raise ValueError(f"unknown fixture {name!r}; available: {sorted(library)} + ppnn")
        yield name, library[name]


def main(argv=None) -> None:
    library_names = tuple(operator_model.operator_library())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples", nargs="+", default=None,
        help="fixture names; ppnn is explicit because it has 518400 raw Wicks",
    )
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--color-dimension", type=int, nargs="+", default=(3, 6))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "verify_code" / "numeric_check")
    parser.add_argument("--no-check-files", action="store_true")
    args = parser.parse_args(argv)
    names = tuple(args.examples or library_names)
    if any(dimension < 1 for dimension in args.color_dimension):
        parser.error("color dimensions must be positive")
    results = []
    for name, operators in fixtures(names):
        for dimension in args.color_dimension:
            check_path = None if args.no_check_files else args.output_dir / f"{name}_check_C{dimension}.py"
            item = audit_one(
                name, operators, seed=args.seed, color_dimension=dimension,
                write_check=check_path,
            )
            results.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

