#!/usr/bin/env python3
"""Check generated basis output against the direct original Wick sum.

The reducer's ``check`` representation has three layers:

``basis_contract``
    one explicit ``contract`` call for each independent Wick representative;
``basis``
    the representative sign and any open-axis transpose applied to that call;
``origen_wick``
    the signed original Wick expression reconstructed from the basis views.

This script evaluates those layers with the same random input.  It uses a
random complex perambulator of shape ``(4, N, N, 4)`` and random *antisymmetric*
VVV tensors of shape ``(N, N, N)`` in distillation mode. The antisymmetry is
required because the reducer's VVV transporter sign is the algebraic sign
for exchanging two VVV legs. Meson VDV elementals are dense ``(N, N)`` tensors.
Normal mode uses actual three-color epsilon and delta tensors and propagators
of shape ``(3, 4, 3, 4)`` in ``(V_sink, D_sink, V_source, D_source)`` order. Open spin axes are
projected with fixed random spin vectors, so the ppnn run does not need to
materialize a dense ``4**8`` tensor for every one of its 518,400 raw terms.

The historical seven fixtures are checked by default.  ``--include-eighth``
adds ``pn_omega_pi+``.  The ppnn fixture is always included unless
``--without-ppnn`` is passed.  Small check files are written by default; the
518,400-line ppnn file is opt-in through ``--write-ppnn-check``.

Run from this project's root, for example::

    PYTHONPATH=. python3 verify_code/check_basis_vs_original.py --no-run
    PYTHONPATH=. python3 verify_code/check_basis_vs_original.py --examples pn delta

The first command only renders the small check documents.  A full ppnn
numeric run is intentionally explicit because it evaluates every raw Wick
term for both eigenvector dimensions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
try:
    import opt_einsum
except ImportError:  # pragma: no cover - NumPy is the lightweight fallback
    _EINSUM_PATHS = {}
    _MAX_EINSUM_PATH_CACHE = 32768

    def _numpy_contract(equation, *arguments, **kwargs):
        # NumPy's optimize=True recomputes the contraction path on every call;
        # ppn/ppnn have thousands of different pairings.  Reusing the path is
        # the difference between a practical audit and a multi-hour run.
        kwargs.pop("optimize", None)
        key = (equation, tuple(tuple(argument.shape) for argument in arguments))
        path = _EINSUM_PATHS.get(key)
        if path is None:
            path = np.einsum_path(equation, *arguments, optimize="greedy")[0]
            # ppnn has hundreds of thousands of distinct pairings.  Keep a
            # bounded cache so path strings cannot consume the audit's memory;
            # small fixtures still reuse every path.
            if len(_EINSUM_PATHS) < _MAX_EINSUM_PATH_CACHE:
                _EINSUM_PATHS[key] = path
        return np.einsum(equation, *arguments, optimize=path, **kwargs)

    class _NumpyEinsum:
        contract = staticmethod(_numpy_contract)

    opt_einsum = _NumpyEinsum()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from verify_code.ppnn_state_quotient_audit import _operators as ppnn_operators
from verify_code.generic_check_numeric import _epsilon_color
from kernels.wick_reducer import expression_output, operator_model
from kernels.wick_reducer.reducer import ReductionOptions, reduce_operators


HISTORICAL_EXAMPLES = (
    "pipi", "pn", "ppn", "delta", "delta++delta-", "pip", "pip-p",
)
EIGHTH_EXAMPLE = "pn_omega_pi+"
DEFAULT_DIMENSIONS = (3, 6)


def _gamma_symbol(name: str) -> str:
    """Match expression_output's public operand naming convention."""
    dagger = name.endswith("_dagger")
    base = name[:-7] if dagger else name
    if base.startswith("Cgamma"):
        base = "C_g" + base[len("Cgamma"):]
    elif base.startswith("gamma"):
        base = "g" + base[len("gamma"):]
    return base + ("_dagger" if dagger else "")


def _epsilon_sign(permutation: Sequence[int]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _random_antisymmetric_vvv(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a dense rank-three antisymmetric tensor for any N >= 3."""
    tensor = np.zeros((size, size, size), dtype=np.complex128)
    for i in range(size):
        for j in range(i + 1, size):
            for k in range(j + 1, size):
                value = (
                    rng.standard_normal() + 1j * rng.standard_normal()
                ) / np.sqrt(2.0)
                for permutation in (
                    (i, j, k), (i, k, j), (j, i, k),
                    (j, k, i), (k, i, j), (k, j, i),
                ):
                    tensor[permutation] = _epsilon_sign(
                        tuple((i, j, k).index(item) for item in permutation)
                    ) * value
    return tensor


def _random_values(
    operators: Sequence[core.Operator],
    size: int,
    seed: int,
    *,
    isospin_degenerate: bool = True,
) -> Tuple[Dict[str, np.ndarray], Tuple[np.ndarray, ...]]:
    """Return gamma, peram, VVV/VDV, and open-spin projection values."""
    rng = np.random.default_rng(seed)

    def complex_array(shape):
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ) / np.sqrt(2.0)

    gamma_values = core._euclidean_gamma_matrices()
    values: Dict[str, np.ndarray] = {
        _gamma_symbol(name): np.asarray(matrix, dtype=np.complex128)
        for name, matrix in gamma_values.items()
    }

    fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    # Use the writer's endpoint-aware names, including p labels and the
    # declared isospin policy, for every legal exact-flavor pairing.
    peram_symbols = {
        expression_output._peram_symbol(
            quark, anti, isospin_degenerate=isospin_degenerate,
        )
        for quark in quarks for anti in antiquarks
        if quark.flavor == anti.flavor
    }
    for symbol in sorted(peram_symbols):
        dimension = 3 if "_x_x" in symbol else size
        # Every mode uses (V_sink,D_sink,V_source,D_source).
        values[symbol] = complex_array((dimension, 4, dimension, 4))

    elemental_symbols = {
        expression_output._elemental_symbol(operator)
        for operator in operators
    }
    for symbol in sorted(elemental_symbols):
        if symbol == "epsilon_color":
            values[symbol] = _epsilon_color()
        elif symbol == "delta_color":
            values[symbol] = np.eye(3)
        elif symbol.startswith("VVV_") or symbol.startswith("conj_VVV_"):
            values[symbol] = _random_antisymmetric_vvv(size, rng)
        else:
            values[symbol] = complex_array((size, size))

    projections = tuple(
        complex_array((4,))
        for _field in fields if _field.bare
    )
    return values, projections


def _projected_contract(
    operators: Sequence[core.Operator],
    pairing: Sequence[int],
    values: Mapping[str, np.ndarray],
    projections: Sequence[np.ndarray],
    *,
    projection_permutation: Sequence[int] | None = None,
    isospin_degenerate: bool = True,
) -> complex:
    """Evaluate one Wick expression after projecting its open spin axes."""
    equation, operands, axes = expression_output.contraction_expression(
        operators, pairing, isospin_degenerate=isospin_degenerate,
    )
    arguments = [values[name] for name in operands]
    if axes:
        if projection_permutation is None:
            reordered = tuple(projections)
        else:
            inverse = {
                target: source
                for source, target in enumerate(projection_permutation)
            }
            reordered = tuple(projections[inverse[index]] for index in range(len(projections)))
        lhs, output = equation.split("->")
        equation = lhs + "," + ",".join(output) + "->"
        arguments.extend(reordered)
    return complex(opt_einsum.contract(equation, *arguments, optimize="greedy"))


def _raw_pairings(report: Mapping[str, object]) -> Dict[int, Tuple[int, ...]]:
    return {
        int(item["wick_index"]): tuple(int(value) for value in item["pairing"])
        for item in report["graph_records"]
    }


def _check_one_dimension(
    operators: Sequence[core.Operator],
    report: Mapping[str, object],
    *,
    size: int,
    seed: int,
) -> Dict[str, object]:
    """Compare every signed original Wick with the basis reconstruction."""
    isospin_degenerate = bool(report["isospin_symmetric"])
    values, projections = _random_values(
        operators, size, seed, isospin_degenerate=isospin_degenerate,
    )
    axis_order = expression_output._bare_axes(operators)
    relation_map = {
        int(item["wick_index"]): item
        for item in report["dependent_relations"]
    }
    independent = report["independent_wicks"]
    independent_position = {
        int(item["wick_index"]): position
        for position, item in enumerate(independent)
    }
    graph_records = sorted(
        report["graph_records"], key=lambda item: int(item["wick_index"])
    )
    pairings = _raw_pairings(report)

    started = time.perf_counter()
    raw_values: Dict[int, complex] = {}
    for graph in graph_records:
        wick_id = int(graph["wick_index"])
        raw_values[wick_id] = _projected_contract(
            operators, pairings[wick_id], values, projections,
            isospin_degenerate=isospin_degenerate,
        )

    # basis_contract is intentionally unsigned.  The sign and transpose live
    # in basis/origen_wick, matching expression_output.render_python_output.
    basis_values: Dict[Tuple[int, Tuple[int, ...]], complex] = {}
    identity = tuple(range(len(axis_order)))

    def basis_view(wick_id: int, permutation: Tuple[int, ...]) -> complex:
        key = (wick_id, permutation)
        if key not in basis_values:
            basis_values[key] = _projected_contract(
                operators, pairings[wick_id], values, projections,
                projection_permutation=permutation,
                isospin_degenerate=isospin_degenerate,
            )
        return basis_values[key]

    residuals = []
    relative_residuals = []
    max_record = None
    direct_total = 0.0 + 0.0j
    for graph in graph_records:
        wick_id = int(graph["wick_index"])
        relation = relation_map.get(wick_id)
        terms = expression_output._relation_terms_for_graph(
            graph, relation, independent_position, axis_order,
        )
        reconstructed = 0.0 + 0.0j
        for _position, basis_id, coefficient, permutation in terms:
            permutation = tuple(permutation)
            basis_sign = int(
                next(item["wick_sign"] for item in independent if int(item["wick_index"]) == basis_id)
            )
            reconstructed += (
                coefficient * basis_sign * basis_view(basis_id, permutation)
            )
        direct_signed = int(graph["wick_sign"]) * raw_values[wick_id]
        direct_total += direct_signed
        residual = abs(direct_signed - reconstructed)
        residuals.append(residual)
        # Match the per-Wick relative error used by the full ppnn check.
        relative_residuals.append(
            residual / max(abs(direct_signed), abs(reconstructed), 1.0e-30)
        )
        if max_record is None or residual > max_record["residual"]:
            max_record = {
                "wick_id": wick_id,
                "direct_signed": [direct_signed.real, direct_signed.imag],
                "basis_signed": [reconstructed.real, reconstructed.imag],
                "residual": residual,
            }

    max_abs = float(max(residuals, default=0.0))
    max_relative = float(max(relative_residuals, default=0.0))
    relation_coefficients = expression_output._collect_report_relations(
        report, axis_order,
    )
    reduced_terms = expression_output._build_reduced_terms(
        independent, relation_coefficients, axis_order,
    )
    basis_total = 0.0 + 0.0j
    for reduced_term in reduced_terms:
        basis_position = int(reduced_term["basis_position"])
        basis_id = int(independent[basis_position]["wick_index"])
        basis_sign = int(independent[basis_position]["wick_sign"])
        permutation = tuple(reduced_term["transpose"])
        basis_total += (
            complex(reduced_term["coefficient"])
            * basis_sign * basis_view(basis_id, permutation)
        )
    total_residual = abs(direct_total - basis_total)
    total_scale = max(abs(direct_total), abs(basis_total), 1.0e-30)
    return {
        "size": size,
        "seed": seed,
        "raw_wick_count": int(report["wick_count"]),
        "basis_contract_count": int(report["independent_count"]),
        "dependent_count": int(report["dependent_count"]),
        "basis_view_count": len(basis_values),
        "max_absolute_residual": max_abs,
        "max_relative_residual": max_relative,
        "direct_total": [direct_total.real, direct_total.imag],
        "basis_total": [basis_total.real, basis_total.imag],
        "total_absolute_residual": float(total_residual),
        "total_relative_residual": float(total_residual / total_scale),
        "worst_case": max_record,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": bool(
            max_relative <= 1.0e-9
            and total_residual <= 1.0e-9 * total_scale
        ),
    }


def _render_check_file(
    operators: Sequence[core.Operator],
    label: str,
    report: Mapping[str, object],
    path: Path,
) -> None:
    """Write the explicit check document for one detailed report."""
    axis_order = expression_output._bare_axes(operators)
    relation_coefficients = expression_output._collect_report_relations(report, axis_order)
    reduced_terms = expression_output._build_reduced_terms(
        report["independent_wicks"], relation_coefficients, axis_order,
    )
    text = expression_output.render_python_output(
        report, operators, name=label, mode="check", reduced_terms=reduced_terms,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixture_map(include_eighth: bool, include_ppnn: bool):
    library = operator_model.operator_library()
    names = list(HISTORICAL_EXAMPLES)
    if include_eighth:
        names.append(EIGHTH_EXAMPLE)
    result = [(name, tuple(library[name])) for name in names]
    if include_ppnn:
        result.append(("ppnn", tuple(ppnn_operators())))
    return result


def run(
    *,
    names: Sequence[str] | None = None,
    include_eighth: bool = False,
    include_ppnn: bool = True,
    dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
    seed: int = 20261011,
    output_dir: Path | None = None,
    write_ppnn_check: bool = False,
    numeric: bool = True,
) -> Dict[str, object]:
    fixtures = _fixture_map(include_eighth, include_ppnn)
    if names is not None:
        requested = set(names)
        fixtures = [item for item in fixtures if item[0] in requested]
        missing = requested - {name for name, _ in fixtures}
        if missing:
            raise ValueError(f"unknown fixtures: {sorted(missing)}")

    result = {"mode": "check", "fixtures": []}
    for label, operators in fixtures:
        started = time.perf_counter()
        check_path = None
        if output_dir is not None and (label != "ppnn" or write_ppnn_check):
            check_path = output_dir / f"{label.replace('+', '_plus')}_check.py"
            # Use the public writer for check-mode artifacts.  It performs the
            # same generic detailed reduction and keeps the generated file's
            # basis_contract/origen_wick layout identical to normal users.
            if label != "ppnn":
                written = expression_output.write_case_output(
                    label, check_path, mode="check", isospin_degenerate=True,
                )
            else:
                written = expression_output.write_output(
                    expression_output.split_operators(operators), check_path,
                    name=label, mode="check", isospin_degenerate=True,
                )
            report = written["report"]
        else:
            report = reduce_operators(
                operators,
                label=label,
                mode="detailed",
                options=ReductionOptions(isospin_degenerate=True),
            )

        numeric_records = []
        if numeric:
            if label == "ppnn" and check_path is not None:
                # The streamed check writer deliberately returns no graph
                # records.  Reusing the dedicated sampler keeps this audit
                # bounded in memory while still evaluating random peram/VVV
                # contractions at each requested color dimension.
                from verify_code.ppnn_sample_numeric import run as sample_ppnn

                for index, dimension in enumerate(dimensions):
                    numeric_records.append(sample_ppnn(
                        check_path,
                        int(dimension),
                        sample_count=128,
                        seed=int(seed + index),
                    ))
            else:
                if report.get("graph_records") is None:
                    # Small streamed check files are byte-for-byte compatible
                    # with the former detailed output, but numeric auditing
                    # needs the in-memory pairing table.  Obtain that table
                    # only for these explicitly requested small fixtures.
                    report = reduce_operators(
                        operators,
                        label=label,
                        mode="detailed",
                        options=ReductionOptions(isospin_degenerate=True),
                    )
                for index, dimension in enumerate(dimensions):
                    numeric_records.append(_check_one_dimension(
                        operators, report,
                        size=int(dimension), seed=int(seed + index),
                    ))
        item = {
            "name": label,
            "mode": report["mode"],
            "wick_count": report["wick_count"],
            "basis_contract_count": report["independent_count"],
            "dependent_count": report["dependent_count"],
            "factor_graph_buckets": report.get("peram_incidence_buckets"),
            "check_output": str(check_path) if check_path else None,
            "numeric": numeric_records,
            "elapsed_seconds": time.perf_counter() - started,
        }
        result["fixtures"].append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", nargs="*", help="subset of fixtures; default is seven plus ppnn")
    parser.add_argument("--include-eighth", action="store_true", help=f"include {EIGHTH_EXAMPLE}")
    parser.add_argument("--without-ppnn", action="store_true")
    parser.add_argument("--dimensions", default="3,6", help="comma-separated VVV eigen dimensions")
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output-dir", type=Path, default=Path("verify_output/check"))
    parser.add_argument("--write-ppnn-check", action="store_true", help="write the very large ppnn check Python file")
    parser.add_argument("--no-numeric", action="store_true", help="only render check files and skip random evaluation")
    parser.add_argument("--no-run", action="store_true", help="alias for --no-numeric")
    parser.add_argument("--json", type=Path, help="also write the aggregate JSON report")
    args = parser.parse_args(argv)

    dimensions = tuple(int(value) for value in args.dimensions.split(",") if value.strip())
    if not dimensions or any(value < 3 for value in dimensions):
        parser.error("--dimensions must contain positive values >= 3")
    summary = run(
        names=args.examples or None,
        include_eighth=args.include_eighth,
        include_ppnn=not args.without_ppnn,
        dimensions=dimensions,
        seed=args.seed,
        output_dir=args.output_dir,
        write_ppnn_check=args.write_ppnn_check,
        numeric=not (args.no_numeric or args.no_run),
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

