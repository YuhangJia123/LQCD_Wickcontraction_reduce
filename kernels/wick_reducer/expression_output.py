"""Write reduced Wick contractions as explicit opt_einsum Python expressions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import string
import sys
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernels.wick_reducer import operator_model as model
from kernels.wick_reducer.reducer import ReductionOptions, reduce_operators


def normalize_correlator(correlator: Sequence[Sequence[model.core.Operator]]):
    """Return a flat operator tuple from ``[O1_sink, O2_source]``."""
    if isinstance(correlator, (str, bytes)) or len(correlator) != 2:
        raise ValueError("correlator must be [O1_sink, O2_source]")
    sink, source = tuple(correlator[0]), tuple(correlator[1])
    if not sink or not source:
        raise ValueError("O1_sink and O2_source must each contain at least one operator")
    if any(operator.side != "sink" for operator in sink):
        raise ValueError("every operator in O1_sink must have side='sink'")
    if any(operator.side != "source" for operator in source):
        raise ValueError("every operator in O2_source must have side='source'")
    operators = sink + source
    model.core.validate_operator_modes(operators)
    return operators


def split_operators(operators: Sequence[model.core.Operator]):
    sink = tuple(operator for operator in operators if operator.side == "sink")
    source = tuple(operator for operator in operators if operator.side == "source")
    return sink, source


def _identifier(value: str) -> str:
    result = re.sub(r"\W", "_", value, flags=re.UNICODE)
    if not result or result[0].isdigit():
        result = "x_" + result
    return result


def default_block_output_path(name: str = "correlator") -> Path:
    """Return the legacy API destination for a normal-wick-block document."""
    return Path("result") / "normal_wick_block" / f"{_identifier(name)}_normal_wick_block.py"


def _gamma_symbol(name: str) -> str:
    dagger = name.endswith("_dagger")
    base = name[:-7] if dagger else name
    if base.startswith("Cgamma"):
        base = "C_g" + base[len("Cgamma"):]
    elif base.startswith("gamma"):
        base = "g" + base[len("gamma"):]
    symbol = _identifier(base)
    return symbol + ("_dagger" if dagger else "")


def _elemental_symbol(operator: model.core.Operator) -> str:
    if operator.mode == "normal":
        return "epsilon_color" if operator.kind == "baryon" else "delta_color"
    family = "VVV" if operator.kind == "baryon" else "VDV"
    tag = re.sub(r"\W", "_", operator.external_tag, flags=re.UNICODE)
    suffix = "_" + tag if tag else ""
    metadata = model.core.contraction_metadata_dict(operator.contraction_metadata)
    if metadata:
        serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        suffix += "_meta_" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if operator.side == "sink":
        return f"{family}_sink{suffix}"
    return f"conj_{family}_source{suffix}"


def _elemental_declarations(operators: Sequence[model.core.Operator]) -> Tuple[str, ...]:
    declarations = []
    seen = set()
    for operator in operators:
        symbol = _elemental_symbol(operator)
        if symbol in seen:
            continue
        seen.add(symbol)
        if operator.mode == "normal":
            description = (
                "supplied Levi-Civita color tensor, shape (3, 3, 3); epsilon[0, 1, 2] = 1"
                if operator.kind == "baryon"
                else "supplied color identity matrix, shape (3, 3)"
            )
            declarations.append(f"#   {symbol}: {description}")
            continue
        momentum = (
            "shared default momentum (zero for built-in examples)"
            if not operator.external_tag
            else f"external_tag={operator.external_tag!r}"
        )
        metadata = model.core.contraction_metadata_dict(operator.contraction_metadata)
        if metadata:
            momentum += f", contraction_metadata={metadata!r}"
        declarations.append(f"#   {symbol}: {momentum}")
    return tuple(declarations)


@lru_cache(maxsize=8192)
def _endpoint_label_hash(descriptors: Tuple[Tuple[object, ...], ...]) -> str:
    """Hash ordered endpoint labels once when rendering many Wick graphs."""
    serialized = json.dumps(
        model.core._metadata_json_value(descriptors),
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _peram_symbol(
    q_field: model.core.Field,
    anti_field: model.core.Field,
    *,
    isospin_degenerate: bool = True,
) -> str:
    """Name one directed propagator using its flavor and endpoint labels."""
    q_flavor = model.flavor_class(q_field.flavor, isospin_degenerate)
    anti_flavor = model.flavor_class(anti_field.flavor, isospin_degenerate)
    if q_flavor != anti_flavor:
        raise ValueError("Wick pairing connects different peram flavor classes")
    if q_field.mode != anti_field.mode:
        raise ValueError("peram endpoints must use the same operator mode")
    peram_flavor = "u" if q_flavor == "ud" else q_flavor
    symbol = f"peram_{_identifier(peram_flavor)}_{model.core.peram_direction(q_field, anti_field)}"
    if q_field.mode == "normal":
        symbol += "_x_x"
        descriptors = (q_field.endpoint_descriptor, anti_field.endpoint_descriptor)
        if any(descriptor[1] or descriptor[2] for descriptor in descriptors):
            symbol += "_p_" + _endpoint_label_hash(descriptors)
    elif q_field.mode != "distillation":
        raise ValueError("operator mode must be 'distillation' or 'normal'")
    return symbol


def _peram_declarations(
    operators: Sequence[model.core.Operator], *, isospin_degenerate: bool,
) -> Tuple[str, ...]:
    """Declare normal propagators without enumerating Wick permutations."""
    _fields, quarks, antiquarks = model.fields_for_operators(operators)
    declarations = []
    seen = set()
    for q_field in quarks:
        for anti_field in antiquarks:
            if model.flavor_class(q_field.flavor, isospin_degenerate) != model.flavor_class(
                anti_field.flavor, isospin_degenerate
            ):
                continue
            symbol = _peram_symbol(
                q_field, anti_field, isospin_degenerate=isospin_degenerate,
            )
            if symbol in seen:
                continue
            seen.add(symbol)
            endpoints = {
                "quark": model.core.field_endpoint_metadata(q_field),
                "antiquark": model.core.field_endpoint_metadata(anti_field),
            }
            labels = json.dumps(endpoints, sort_keys=True, ensure_ascii=False)
            declarations.append(f"#   {symbol}: {labels}")
    return tuple(declarations)


def _output_header(
    operators: Sequence[model.core.Operator],
    axes: Sequence[str],
    *,
    isospin_degenerate: bool,
) -> list[str]:
    operator_mode = model.core.validate_operator_modes(operators)
    lines = [
        "# Auto-generated by wick_diss_reduce.wick_reducer.expression_output.",
        "# Correlator input is [O1_sink, O2_source].",
        "from opt_einsum import contract",
        "",
    ]
    lines.extend((
        "# Peram axes for every mode: (V_sink, D_sink, V_source, D_source).",
        "# The first port is the quark endpoint (V_sink, D_sink); the second is the antiquark endpoint (V_source, D_source).",
        "# _x_x means both endpoints are unsmeared; p labels add no coordinate axes.",
        "# _p_<sha256> hashes the ordered (quark, antiquark) endpoint descriptors.",
        "# Peram endpoint label mapping:",
        *_peram_declarations(operators, isospin_degenerate=isospin_degenerate),
    ))
    if operator_mode == "distillation":
        lines.extend((
            "# Elemental momentum labels; species does not distinguish VVV/VDV tensors.",
            "# No external_tag means a shared default momentum (zero for built-in examples).",
            *_elemental_declarations(operators),
            "# Use distinct external_tag values for distinct momentum tensors.",
        ))
    else:
        lines.extend((
            "# Operator mode: normal. All tensor arrays are supplied by the caller.",
            "# Shared real color tensors are used at both sink and source.",
            *_elemental_declarations(operators),
        ))
    lines.extend((
        "# Original Wick IDs start at 1; basis view keys start at 0.",
        f"# Output indices: {''.join(axes) if axes else '(scalar)'}.",
    ))
    return lines


class _IndexLabels:
    def __init__(self, fields: Sequence[model.core.Field]):
        self._alphabet = string.ascii_lowercase + string.ascii_uppercase
        self._assigned: Dict[Tuple[str, int], str] = {}
        self._used = set()
        self._cursor = 0
        self.free_fields = tuple(field for field in fields if field.bare)
        free_symbols = string.ascii_uppercase + string.ascii_lowercase
        if len(self.free_fields) > len(free_symbols):
            raise ValueError("more than 52 open bare indices cannot be written in opt_einsum notation")
        for position, field in enumerate(self.free_fields):
            symbol = free_symbols[position]
            self._assigned[("spin", field.index)] = symbol
            self._used.add(symbol)

    def get(self, kind: str, field: model.core.Field) -> str:
        key = (kind, field.index)
        if key not in self._assigned:
            while self._cursor < len(self._alphabet) and self._alphabet[self._cursor] in self._used:
                self._cursor += 1
            if self._cursor >= len(self._alphabet):
                raise ValueError("more than 52 distinct tensor indices cannot be written in opt_einsum notation")
            symbol = self._alphabet[self._cursor]
            self._cursor += 1
            self._assigned[key] = symbol
            self._used.add(symbol)
        return self._assigned[key]

    @property
    def output(self) -> str:
        return "".join(self.get("spin", field) for field in self.free_fields)


def contraction_expression(
    operators: Sequence[model.core.Operator],
    pairing: Sequence[int],
    *,
    isospin_degenerate: bool = True,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """Build an explicit einsum equation, operand list, and open-axis list."""
    model.core.validate_operator_modes(operators)
    fields, quarks, antiquarks = model.fields_for_operators(operators)
    return _contraction_expression_from_fields(
        operators, fields, quarks, antiquarks, pairing,
        isospin_degenerate=isospin_degenerate,
    )


def _contraction_expression_from_fields(
    operators: Sequence[model.core.Operator],
    fields: Sequence[model.core.Field],
    quarks: Sequence[model.core.Field],
    antiquarks: Sequence[model.core.Field],
    pairing: Sequence[int],
    *,
    isospin_degenerate: bool = True,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    """Build one equation from precomputed fields.

    Rendering a large check document visits every raw pairing.  The operator
    fields are invariant across those pairings, so callers that render many
    equations can avoid rebuilding the same field objects on every iteration.
    """
    by_operator: Dict[int, Dict[int, model.core.Field]] = {}
    for field in fields:
        by_operator.setdefault(field.operator, {})[field.slot] = field

    for op_index, operator in enumerate(operators):
        expected = 3 if operator.kind == "baryon" else 2
        slots = by_operator.get(op_index, {})
        if set(slots) != set(range(expected)):
            raise ValueError(f"operator {operator.label!r} does not have all {expected} field slots")
        covered = set(operator.gamma_slots)
        if operator.open_slot is not None:
            covered.add(operator.open_slot)
        if covered != set(range(expected)):
            raise ValueError(
                f"operator {operator.label!r} must assign every spin slot to its Gamma or an open bare index"
            )

    if len(pairing) != len(quarks):
        raise ValueError("Wick pairing length does not match the quark field count")
    if any(index < 0 or index >= len(antiquarks) for index in pairing):
        raise ValueError("Wick pairing contains an invalid antiquark index")

    labels = _IndexLabels(fields)
    spin = {field.index: labels.get("spin", field) for field in fields}
    spatial_kind = "color" if operators and operators[0].mode == "normal" else "eigen"
    spatial = {field.index: labels.get(spatial_kind, field) for field in fields}

    factor_indices = []
    operands = []
    for op_index, operator in enumerate(operators):
        if not operator.gamma:
            continue
        factor_indices.append("".join(
            spin[by_operator[op_index][slot].index] for slot in operator.gamma_slots
        ))
        operands.append(_gamma_symbol(operator.gamma))

    for q_index, anti_index in enumerate(pairing):
        q_field = quarks[q_index]
        anti_field = antiquarks[anti_index]
        # Every mode uses the same peram layout: (V_sink,D_sink,V_source,D_source).
        indices = (
            spatial[q_field.index], spin[q_field.index],
            spatial[anti_field.index], spin[anti_field.index],
        )
        factor_indices.append("".join(indices))
        operands.append(_peram_symbol(
            q_field, anti_field, isospin_degenerate=isospin_degenerate,
        ))

    for op_index, operator in enumerate(operators):
        slots = by_operator[op_index]
        ordered = [slots[index] for index in range(3 if operator.kind == "baryon" else 2)]
        factor_indices.append("".join(spatial[field.index] for field in ordered))
        operands.append(_elemental_symbol(operator))

    output_indices = labels.output
    counts = defaultdict(int)
    for indices in factor_indices:
        for symbol in indices:
            counts[symbol] += 1
    output_counts = defaultdict(int)
    for symbol in output_indices:
        output_counts[symbol] += 1
    if any(counts[symbol] != 1 for symbol in output_counts):
        raise ValueError("each open bare index must occur once in the contraction operands")
    if any(count != 2 for symbol, count in counts.items() if symbol not in output_counts):
        raise ValueError("every dummy index must occur exactly twice in the contraction operands")
    if len(output_counts) != len(output_indices):
        raise ValueError("open output indices must be distinct")
    equation = ",".join(factor_indices) + "->" + output_indices
    return equation, tuple(operands), tuple(output_indices)


def _python_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _python_complex(value) -> str:
    if isinstance(value, (list, tuple)):
        real, imag = float(value[0]), float(value[1])
    else:
        real, imag = complex(value).real, complex(value).imag
    if imag == 0:
        return _python_number(real)
    return f"complex({_python_number(real)}, {_python_number(imag)})"


def _scalar_metadata_factor(value: object) -> Optional[complex]:
    if isinstance(value, tuple) and len(value) == 3 and value[0] == "complex":
        return complex(float(value[1]), float(value[2]))
    if isinstance(value, (int, float, complex)):
        return complex(value)
    return None


def _operator_scalar_weight(operators: Sequence[model.core.Operator]) -> complex:
    weight = 1.0 + 0.0j
    for operator in operators:
        metadata = dict(operator.contraction_metadata)
        for name in ("clebsch", "phase"):
            if name in metadata:
                factor = _scalar_metadata_factor(metadata[name])
                if factor is not None:
                    weight *= factor
    return weight


def _contract_call(
    equation: str,
    operands: Sequence[str],
    indent: str = "    ",
    scalar_weight: complex = 1.0 + 0.0j,
) -> str:
    arguments = ", ".join((repr(equation), *operands))
    expression = f"contract({arguments})"
    if scalar_weight != 1.0 + 0.0j:
        expression = f"({_python_complex(scalar_weight)} * {expression})"
    return f"{indent}{expression},"


def _render_contracts(
    report_items: Sequence[Mapping[str, object]],
    operators: Sequence[model.core.Operator],
    *,
    isospin_degenerate: bool,
) -> Tuple[list[str], list[Tuple[str, ...]]]:
    calls = []
    outputs = []
    fields, quarks, antiquarks = model.fields_for_operators(operators)
    scalar_weight = _operator_scalar_weight(operators)
    for item in report_items:
        equation, operands, output = _contraction_expression_from_fields(
            operators, fields, quarks, antiquarks, item["pairing"],
            isospin_degenerate=isospin_degenerate,
        )
        calls.append(_contract_call(equation, operands, scalar_weight=scalar_weight))
        outputs.append(output)
    return calls, outputs


def _bare_axes(operators: Sequence[model.core.Operator]) -> Tuple[str, ...]:
    fields, _quarks, _antiquarks = model.fields_for_operators(operators)
    return tuple(field.axis for field in fields if field.bare and field.axis is not None)


def _transpose_for_relation(
    candidate_map: Mapping[str, str],
    basis_map: Mapping[str, str],
    axis_order: Sequence[str],
) -> Tuple[int, ...]:
    inverse_basis = {root: basis for basis, root in basis_map.items()}
    positions = {axis: index for index, axis in enumerate(axis_order)}
    permutation = []
    for candidate_axis in axis_order:
        root_axis = candidate_map.get(candidate_axis, candidate_axis)
        basis_axis = inverse_basis.get(root_axis, root_axis)
        if basis_axis not in positions:
            raise ValueError("transporter bare-axis map does not match the correlator output axes")
        permutation.append(positions[basis_axis])
    if sorted(permutation) != list(range(len(axis_order))):
        raise ValueError("transporter does not induce a permutation of the open bare axes")
    return tuple(permutation)


def _collect_report_relations(report, axis_order):
    accumulated = defaultdict(complex)
    for relation in report.get("dependent_relations", ()):
        solutions = relation.get("transporter_solutions", ())
        solution_index = int(relation["transporter_solution"])
        if solution_index >= len(solutions):
            raise ValueError(f"missing selected transporter for Wick {relation['wick_index']}")
        candidate_map = solutions[solution_index].get("bare_axis_map", {})
        basis_maps = {
            int(item["wick_index"]): item.get("bare_axis_map", {})
            for item in relation.get("basis_transports", ())
        }
        for basis_index, coefficient in zip(
            relation["basis_wick_indices"], relation["coefficients"]
        ):
            basis_index = int(basis_index)
            permutation = _transpose_for_relation(
                candidate_map, basis_maps.get(basis_index, {}), axis_order,
            )
            accumulated[(basis_index, permutation)] += complex(
                float(coefficient[0]), float(coefficient[1])
            )
    return accumulated


def _build_reduced_terms(independent_items, relation_coefficients, axis_order):
    identity = tuple(range(len(axis_order)))
    accumulated = defaultdict(complex)
    for item in independent_items:
        accumulated[(int(item["wick_index"]), identity)] += 1.0 + 0.0j
    for key, coefficient in relation_coefficients.items():
        accumulated[key] += coefficient

    positions = {
        int(item["wick_index"]): position
        for position, item in enumerate(independent_items)
    }
    terms = []
    for (wick_index, permutation), coefficient in sorted(
        accumulated.items(), key=lambda item: (positions[item[0][0]], item[0][1])
    ):
        if abs(coefficient) <= 1.0e-12:
            continue
        terms.append({
            "basis_position": positions[wick_index],
            "coefficient": coefficient,
            "transpose": permutation,
        })
    return terms


def _relation_terms_for_graph(graph, relation, independent_position, axis_order):
    wick_index = int(graph["wick_index"])
    if relation is None:
        position = independent_position[wick_index]
        return [(position, wick_index, 1.0 + 0.0j, tuple(range(len(axis_order))))]

    solution_index = int(relation["transporter_solution"])
    solutions = relation.get("transporter_solutions", ())
    if solution_index >= len(solutions):
        raise ValueError(f"missing selected transporter for Wick {wick_index}")
    candidate_map = solutions[solution_index].get("bare_axis_map", {})
    basis_maps = {
        int(item["wick_index"]): item.get("bare_axis_map", {})
        for item in relation.get("basis_transports", ())
    }
    basis_indices = relation["basis_wick_indices"]
    coefficients = relation["coefficients"]
    if len(basis_indices) != len(coefficients):
        raise ValueError(f"Wick {wick_index} has inconsistent basis coefficients")

    terms = []
    for basis_index, coefficient in zip(basis_indices, coefficients):
        basis_index = int(basis_index)
        if basis_index not in independent_position:
            raise ValueError(f"Wick {wick_index} references a non-independent basis graph")
        transpose = _transpose_for_relation(
            candidate_map, basis_maps.get(basis_index, {}), axis_order,
        )
        terms.append((
            independent_position[basis_index], basis_index,
            complex(float(coefficient[0]), float(coefficient[1])), transpose,
        ))
    return terms


def _collect_basis_views(independent_items, reduced_terms, graph_relations, axis_order):
    identity = tuple(range(len(axis_order)))
    independent_position = {
        int(item["wick_index"]): position
        for position, item in enumerate(independent_items)
    }
    views = {(int(item["wick_index"]), identity) for item in independent_items}
    for term in reduced_terms:
        item = independent_items[int(term["basis_position"])]
        views.add((int(item["wick_index"]), tuple(term["transpose"])))
    for graph, relation in graph_relations:
        for _, wick_index, _, transpose in _relation_terms_for_graph(
            graph, relation, independent_position, axis_order,
        ):
            views.add((wick_index, transpose))
    return sorted(views, key=lambda view: (independent_position[view[0]], view[1]))


def _render_relation_data(
    report: Mapping[str, object],
    independent_items: Sequence[Mapping[str, object]],
    axis_order: Sequence[str],
    name: str,
    basis_view_position: Mapping[Tuple[int, Tuple[int, ...]], int],
) -> Tuple[str, list[str]]:
    independent_position = {
        int(item["wick_index"]): position
        for position, item in enumerate(independent_items)
    }
    relations = {int(item["wick_index"]): item for item in report["dependent_relations"]}
    rows = []
    expressions = []
    for graph in sorted(report["graph_records"], key=lambda item: int(item["wick_index"])):
        wick_index = int(graph["wick_index"])
        relation = relations.get(wick_index)
        terms = _relation_terms_for_graph(
            graph, relation, independent_position, axis_order,
        )
        term_source = ", ".join(
            "{" +
            f"'basis_index': {basis_view_position[(basis_wick_id, permutation)]}, "
            f"'coefficient': {_python_complex(coefficient)}, "
            f"'basis_wick_id': {basis_wick_id}, "
            f"'transpose': {permutation!r}" +
            "}"
            for _, basis_wick_id, coefficient, permutation in terms
        )
        slot = wick_index - 1
        rows.append(f"    {slot}: {{'wick_id': {wick_index}, 'terms': [{term_source}]}},")
        expression_terms = []
        for _, basis_wick_id, coefficient, permutation in terms:
            basis_index = basis_view_position[(basis_wick_id, permutation)]
            term = f"({_python_complex(coefficient)} * {name}_basis[{basis_index}])"
            expression_terms.append(term)
        expression = " + ".join(expression_terms) if expression_terms else "0"
        expressions.append(f"{name}_origen_wick[{slot}] = {expression}")

    relation_literal = "{\n" + "\n".join(rows) + "\n}"
    return relation_literal, expressions


def render_python_output(
    report: Mapping[str, object],
    operators: Sequence[model.core.Operator],
    *,
    name: str,
    mode: str = "independent",
    reduced_terms=None,
) -> str:
    """Render a reducer report as a standalone Python contraction document."""
    model.core.validate_operator_modes(operators)
    if mode not in ("independent", "check"):
        raise ValueError("output mode must be 'independent' or 'check'")
    if mode == "independent" and report.get("mode") != "independent":
        raise ValueError("independent output requires an independent reduction report")
    if mode == "check" and report.get("mode") != "detailed":
        raise ValueError("check output requires a detailed reduction report")

    safe_name = _identifier(name)
    isospin_degenerate = bool(report.get("isospin_symmetric", True))
    independent_items = report["independent_wicks"]
    independent_calls, output_axes = _render_contracts(
        independent_items, operators, isospin_degenerate=isospin_degenerate,
    )
    if output_axes and any(axes != output_axes[0] for axes in output_axes[1:]):
        raise ValueError("independent Wick expressions disagree on output-axis order")
    axes = output_axes[0] if output_axes else _bare_axes(operators)
    axis_order = _bare_axes(operators)
    if mode == "check":
        relation_coefficients = _collect_report_relations(report, axis_order)
    else:
        if reduced_terms is None:
            raise ValueError("independent output needs transporter-aggregated reduced terms; use write_output")
        relation_coefficients = None
    if mode == "check":
        if reduced_terms is None:
            reduced_terms = _build_reduced_terms(
                independent_items, relation_coefficients, axis_order,
            )
        graph_records = sorted(report["graph_records"], key=lambda item: int(item["wick_index"]))
        relations = {
            int(item["wick_index"]): item
            for item in report["dependent_relations"]
        }
        graph_relations = [
            (graph, relations.get(int(graph["wick_index"])))
            for graph in graph_records
        ]
    else:
        graph_records = []
        graph_relations = []

    basis_views = _collect_basis_views(
        independent_items, reduced_terms, graph_relations, axis_order,
    )
    basis_view_position = {view: position for position, view in enumerate(basis_views)}
    independent_by_id = {
        int(item["wick_index"]): item
        for item in independent_items
    }
    coefficients_by_view = defaultdict(complex)
    for term in reduced_terms:
        representative = independent_items[int(term["basis_position"])]
        view = (int(representative["wick_index"]), tuple(term["transpose"]))
        coefficients_by_view[basis_view_position[view]] += complex(term["coefficient"])
    coefficient_values = [
        coefficients_by_view.get(basis_index, 0.0 + 0.0j)
        for basis_index in range(len(basis_views))
    ]

    lines = _output_header(
        operators, axes, isospin_degenerate=isospin_degenerate,
    )
    lines.extend((
        "",
        "# 最终系数数组只保存一次；coefficients[i] 对应 basis[i]。",
        f"{safe_name}_coefficients = [{', '.join(_python_complex(value) for value in coefficient_values)}]",
    ))
    if mode == "check":
        graph_records = sorted(report["graph_records"], key=lambda item: int(item["wick_index"]))
        raw_calls, raw_axes = _render_contracts(
            graph_records, operators, isospin_degenerate=isospin_degenerate,
        )
        if raw_axes and any(value != axes for value in raw_axes):
            raise ValueError("raw Wick expressions disagree on output-axis order")
        lines.extend((
            "# 原始逐图收缩；输出槽位从 0 起，注释中的 Wick ID 从 1 起。",
            f"{safe_name}_origen_wick_raw = {{}}",
            f"{safe_name}_origen_wick_sign = {{}}",
        ))
        for item, call in zip(graph_records, raw_calls):
            wick_id = int(item["wick_index"])
            slot = wick_id - 1
            call = call.strip()
            if call.endswith(","):
                call = call[:-1]
            lines.append(f"{safe_name}_origen_wick_raw[{slot}] = {call}")
            lines.append(
                f"{safe_name}_origen_wick_sign[{slot}] = {int(item['wick_sign'])}"
            )
        lines.append("")

    lines.extend((
        "",
        "# 第一步：每个独立 Wick 代表只做一次原始 contract 收缩；字典键是 1 起始 Wick 编号。",
        f"{safe_name}_basis_contract = {{}}",
    ))
    for item, call in zip(independent_items, independent_calls):
        wick_id = int(item["wick_index"])
        expression = call.strip()
        if expression.endswith(","):
            expression = expression[:-1]
        lines.append(f"{safe_name}_basis_contract[{wick_id}] = {expression}")

    lines.extend((
        "",
        "# 第二步：构造真正参与线性组合的基底；已包含代表 Wick 符号和裸轴转置。",
        f"{safe_name}_basis = {{}}",
    ))
    identity = tuple(range(len(axis_order)))
    for basis_index, (wick_id, permutation) in enumerate(basis_views):
        sign = int(independent_by_id[wick_id]["wick_sign"])
        expression = f"{safe_name}_basis_contract[{wick_id}]"
        if sign == -1:
            expression = f"(-{expression})"
        elif sign != 1:
            expression = f"({sign} * {expression})"
        if permutation != identity:
            expression += f".transpose({', '.join(map(str, permutation))})"
        lines.append(
            f"# basis[{basis_index}] derives from Wick {wick_id}; output-axis permutation {permutation}."
        )
        lines.append(f"{safe_name}_basis[{basis_index}] = {expression}")

    lines.extend((
        "",
        "# 用顶部的一维系数数组与同下标 basis 相乘并求和。",
        f"{safe_name}_result = sum("
        f"coefficient * {safe_name}_basis[basis_index] "
        f"for basis_index, coefficient in enumerate({safe_name}_coefficients) if coefficient"
        ")",
    ))

    if mode == "check":
        relation_literal, expressions = _render_relation_data(
            report, independent_items, axis_order, safe_name, basis_view_position,
        )
        lines.extend((
            "",
            f"# Definition: {safe_name}_origen_wick[slot] = "
            f"{safe_name}_origen_wick_sign[slot] * {safe_name}_origen_wick_raw[slot].",
            f"# The following right-hand sides reconstruct the signed raw Wick from {safe_name}_basis.",
            f"{safe_name}_origen_wick = {{}}",
            *expressions,
            "",
            "# 映射键是从 0 开始的输出槽位；wick_id 保留原始的 1 起编号。",
            f"{safe_name}_origen_wick_to_independent = {relation_literal}",
        ))
    return "\n".join(lines) + "\n"


def write_output(
    correlator: Sequence[Sequence[model.core.Operator]],
    output_path: str | Path | None = None,
    *,
    name: str = "correlator",
    mode: str = "independent",
    isospin_degenerate: bool = True,
    gamma_matrices: Optional[Mapping[str, model.core.Matrix]] = None,
    block_options: Optional[Mapping[str, object]] = None,
):
    """Reduce ``[O1_sink, O2_source]`` and write a Python output document.

    ``mode='block'`` is a normal-mode-only second pass.  It first writes the
    ordinary independent three-table output into a temporary file, parses
    those final tables as symbolic data, and sends only them (plus explicit
    ``TensorSpec`` declarations) to :func:`block_reduce`.  The block writer
    then emits a ``normal_wick_block`` document.  The ordinary
    ``independent`` and ``check`` paths are unchanged.
    """
    operators = normalize_correlator(correlator)
    if mode == "block":
        if model.core.validate_operator_modes(operators) != "normal":
            raise ValueError("mode='block' is available only for normal operators")
        options = dict(block_options or {})
        allowed = {
            "max_w_types", "search_budget", "search_seconds",
            "max_bridge_steps", "pairing_family",
        }
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise TypeError(f"unknown block_options: {unknown}")
        from kernels.block_reduction import block_reduce
        from kernels.block_reduction.output_adapter import (
            load_normal_output, pairing_family_from_operators,
            project_gamma_arrays, write_block_output,
        )

        if output_path is None:
            output_path = default_block_output_path(name)

        with tempfile.TemporaryDirectory(prefix="wick-block-input-") as directory:
            source_path = Path(directory) / f"{_identifier(name)}_independent.py"
            independent = write_output(
                correlator, source_path, name=name, mode="independent",
                isospin_degenerate=isospin_degenerate,
                gamma_matrices=gamma_matrices,
            )
            # The adapter verifies registered Gamma symmetries from concrete
            # arrays.  Convert custom project names to the same symbols used
            # by the generated contract operands before loading the tables.
            gamma_arrays = project_gamma_arrays()
            if gamma_matrices is not None:
                import numpy as np
                gamma_arrays.update({
                    _gamma_symbol(symbol): np.asarray(array, dtype=np.complex128)
                    for symbol, array in gamma_matrices.items()
                })
            adapted = load_normal_output(source_path, gamma_arrays=gamma_arrays)
            # A normal block result is planned from the actual Wick pairing
            # family, not from the reduced three tables alone.  The compact
            # family is built here from the same operators that produced the
            # independent tables, then certified by block_reduce against the
            # tables before it can affect the output.  This makes the public
            # normal-wick-block entry point discover the 48-network ppn plan
            # automatically; callers do not need to provide hidden metadata.
            options.setdefault(
                "pairing_family",
                pairing_family_from_operators(
                    operators, isospin_degenerate=isospin_degenerate,
                    tensor_specs=adapted.tensor_specs, compact=True,
                ),
            )
            plan = block_reduce(
                adapted.basis_contract, adapted.basis, adapted.coefficients,
                tensor_specs=adapted.tensor_specs,
                **options,
            )
            written = write_block_output(
                plan, output_path, name=name, source=source_path,
            )
        result = dict(written)
        result.update({
            "name": name,
            "mode": "block",
            "wick_count": independent["wick_count"],
            "independent_count": independent["independent_count"],
            "dependent_count": independent["dependent_count"],
            "reduced_term_count": independent["reduced_term_count"],
            "block_network_count": len(plan.networks),
            "block_w_type_count": len(plan.w_definitions),
            "block_plan": plan,
            "report": independent["report"],
        })
        return result
    if output_path is None:
        raise ValueError(f"mode={mode!r} requires output_path")
    if mode == "check":
        return _write_check_output_streamed(
            operators, output_path, name=name,
            isospin_degenerate=isospin_degenerate,
            gamma_matrices=gamma_matrices,
        )
    reducer_mode = "independent" if mode == "independent" else "detailed" if mode == "check" else mode
    relation_coefficients = defaultdict(complex)
    callback = None
    if mode == "independent":
        axis_order = _bare_axes(operators)

        def collect_relation(relation, candidate_state, basis):
            if [entry.wick_index for entry in basis] != list(relation["basis_wick_indices"]):
                raise AssertionError("relation basis order does not match the stored source projection")
            for entry, coefficient in zip(basis, relation["coefficients"]):
                permutation = _transpose_for_relation(
                    candidate_state.bare_axis_map,
                    entry.state.bare_axis_map,
                    axis_order,
                )
                relation_coefficients[(entry.wick_index, permutation)] += complex(
                    float(coefficient[0]), float(coefficient[1])
                )

        callback = collect_relation
    report = reduce_operators(
        operators,
        label=name,
        mode=reducer_mode,
        options=ReductionOptions(isospin_degenerate=isospin_degenerate),
        gamma_matrices=gamma_matrices,
        _relation_callback=callback,
    )
    output_terms = None
    if mode == "independent":
        output_terms = _build_reduced_terms(
            report["independent_wicks"], relation_coefficients,
            _bare_axes(operators),
        )
    elif mode == "check":
        relation_coefficients = _collect_report_relations(report, _bare_axes(operators))
        output_terms = _build_reduced_terms(
            report["independent_wicks"], relation_coefficients,
            _bare_axes(operators),
        )
    text = render_python_output(
        report, operators, name=name, mode=mode, reduced_terms=output_terms,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "name": name,
        "mode": mode,
        "output_path": str(path),
        "wick_count": report["wick_count"],
        "independent_count": report["independent_count"],
        "dependent_count": report["dependent_count"],
        "reduced_term_count": len(output_terms or ()),
        "report": report,
    }


def _view_key(permutation: Sequence[int]) -> bytes:
    """Return a lexicographically sortable encoding of a small axis permutation."""
    if any(index < 0 or index > 255 for index in permutation):
        raise ValueError("at most 256 bare axes can be represented in a check output")
    return bytes(permutation)


def _write_check_output_streamed(
    operators: Sequence[model.core.Operator],
    output_path: str | Path,
    *,
    name: str,
    isospin_degenerate: bool,
    gamma_matrices: Optional[Mapping[str, model.core.Matrix]],
):
    """Write check output without materializing per-raw-Wick graph records.

    Independent basis data stays in the reducer report.  Dependent relations
    are spooled to SQLite, which also provides the deterministic basis-view
    ordering and the final Wick-ID ordering required by the public format.
    Raw contractions are regenerated from the exact Wick iterator while the
    output is written.
    """
    # Use the package-qualified module in both invocation styles.  Importing
    # ``reducer`` as a top-level script would create a second module object and
    # make its ``ReductionOptions`` class differ from the one imported above.
    from kernels.wick_reducer import reducer as reducer_module

    safe_name = _identifier(name)
    fields, quarks, antiquarks = model.fields_for_operators(operators)
    axis_order = _bare_axes(operators)
    scalar_weight = _operator_scalar_weight(operators)
    independent_position = {}

    with tempfile.TemporaryDirectory(prefix="wick-check-") as temporary_directory:
        database_path = Path(temporary_directory) / "relations.sqlite3"
        connection = sqlite3.connect(str(database_path))
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE relations (wick_id INTEGER PRIMARY KEY, terms TEXT NOT NULL)"
        )
        connection.execute(
            """CREATE TABLE views (
                wick_id INTEGER NOT NULL,
                transpose_key BLOB NOT NULL,
                transpose TEXT NOT NULL,
                independent_position INTEGER NOT NULL,
                coefficient_real REAL NOT NULL DEFAULT 0,
                coefficient_imag REAL NOT NULL DEFAULT 0,
                output_position INTEGER,
                PRIMARY KEY (wick_id, transpose_key)
            )"""
        )
        connection.execute(
            """CREATE TABLE ordered_views (
                output_position INTEGER PRIMARY KEY,
                wick_id INTEGER NOT NULL,
                transpose_key BLOB NOT NULL,
                UNIQUE (wick_id, transpose_key)
            )"""
        )

        def collect_relation(relation, candidate_state, basis, _term):
            basis_ids = [int(entry.wick_index) for entry in basis]
            if basis_ids != [int(value) for value in relation["basis_wick_indices"]]:
                raise AssertionError("relation basis order does not match the stored source projection")
            terms = []
            for entry, coefficient in zip(basis, relation["coefficients"]):
                permutation = _transpose_for_relation(
                    candidate_state.bare_axis_map,
                    entry.state.bare_axis_map,
                    axis_order,
                )
                value = complex(float(coefficient[0]), float(coefficient[1]))
                terms.append([
                    int(entry.wick_index), value.real, value.imag,
                    list(permutation),
                ])
            connection.execute(
                "INSERT INTO relations(wick_id, terms) VALUES (?, ?)",
                (int(relation["wick_index"]), json.dumps(terms, separators=(",", ":"))),
            )

        report = reducer_module.reduce_operators(
            operators,
            label=name,
            mode="independent",
            options=ReductionOptions(isospin_degenerate=isospin_degenerate),
            gamma_matrices=gamma_matrices,
            _relation_term_callback=collect_relation,
        )
        independent_items = report["independent_wicks"]
        independent_by_id = {
            int(item["wick_index"]): item for item in independent_items
        }
        independent_position = {
            int(item["wick_index"]): position
            for position, item in enumerate(independent_items)
        }
        identity = tuple(range(len(axis_order)))

        for item in independent_items:
            wick_id = int(item["wick_index"])
            terms = [[wick_id, 1.0, 0.0, list(identity)]]
            connection.execute(
                "INSERT INTO relations(wick_id, terms) VALUES (?, ?)",
                (wick_id, json.dumps(terms, separators=(",", ":"))),
            )
            connection.execute(
                """INSERT INTO views
                   (wick_id, transpose_key, transpose, independent_position,
                    coefficient_real, coefficient_imag)
                   VALUES (?, ?, ?, ?, 1.0, 0.0)""",
                (wick_id, sqlite3.Binary(_view_key(identity)), json.dumps(identity),
                 independent_position[wick_id]),
            )

        relation_count = connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        if relation_count != int(report["wick_count"]):
            raise RuntimeError(
                f"streamed check captured {relation_count} of {report['wick_count']} Wick relations"
            )

        for relation_wick_id, terms_text in connection.execute(
            "SELECT wick_id, terms FROM relations"
        ):
            # Independent representatives contribute their identity view once,
            # which was inserted above.  Their stored identity relation is
            # still needed for the final per-Wick mapping but must not be
            # counted a second time in the global coefficient vector.
            if int(relation_wick_id) in independent_position:
                continue
            for wick_id, real, imag, permutation_values in json.loads(terms_text):
                wick_id = int(wick_id)
                if wick_id not in independent_position:
                    raise ValueError(f"relation references non-independent Wick {wick_id}")
                permutation = tuple(int(value) for value in permutation_values)
                transpose_key = sqlite3.Binary(_view_key(permutation))
                transpose_text = json.dumps(permutation, separators=(",", ":"))
                position = independent_position[wick_id]
                connection.execute(
                    """INSERT INTO views
                       (wick_id, transpose_key, transpose, independent_position,
                        coefficient_real, coefficient_imag)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(wick_id, transpose_key) DO UPDATE SET
                         coefficient_real = coefficient_real + excluded.coefficient_real,
                         coefficient_imag = coefficient_imag + excluded.coefficient_imag""",
                    (wick_id, transpose_key, transpose_text, position, float(real), float(imag)),
                )

        connection.execute(
            """INSERT INTO ordered_views (output_position, wick_id, transpose_key)
               SELECT ROW_NUMBER() OVER (
                   ORDER BY independent_position, transpose_key
               ) - 1, wick_id, transpose_key
               FROM views"""
        )
        connection.execute(
            """UPDATE views SET output_position = (
                   SELECT output_position FROM ordered_views
                   WHERE ordered_views.wick_id = views.wick_id
                     AND ordered_views.transpose_key = views.transpose_key
               )"""
        )
        connection.commit()

        @lru_cache(maxsize=8192)
        def view_index(wick_id: int, permutation: Tuple[int, ...]) -> int:
            row = connection.execute(
                "SELECT output_position FROM ordered_views WHERE wick_id=? AND transpose_key=?",
                (wick_id, sqlite3.Binary(_view_key(permutation))),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"missing basis view for Wick {wick_id}, transpose {tuple(permutation)}"
                )
            return int(row[0])

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", delete=False,
                dir=str(path.parent), prefix=path.name + ".", suffix=".tmp",
            ) as output:
                temporary_output = Path(output.name)
                for line in _output_header(
                    operators, axis_order, isospin_degenerate=isospin_degenerate,
                ):
                    output.write(line + "\n")
                output.write("\n# 最终系数数组只保存一次；coefficients[i] 对应 basis[i]。\n")
                output.write(f"{safe_name}_coefficients = [")
                nonzero_count = 0
                first_coefficient = True
                for real, imag in connection.execute(
                    """SELECT views.coefficient_real, views.coefficient_imag
                       FROM ordered_views
                       JOIN views USING (wick_id, transpose_key)
                       ORDER BY ordered_views.output_position"""
                ):
                    value = complex(real, imag)
                    if abs(value) > 1.0e-12:
                        nonzero_count += 1
                    if not first_coefficient:
                        output.write(", ")
                    output.write(_python_complex(value))
                    first_coefficient = False
                output.write("]\n")

                output.write("# 原始逐图收缩；输出槽位从 0 起，注释中的 Wick ID 从 1 起。\n")
                output.write(f"{safe_name}_origen_wick_raw = {{}}\n")
                output.write(f"{safe_name}_origen_wick_sign = {{}}\n")
                raw_count = 0
                for raw_count, term in enumerate(model.core.enumerate_wicks(quarks, antiquarks), 1):
                    equation, operands, raw_axes = _contraction_expression_from_fields(
                        operators, fields, quarks, antiquarks, term.pairing,
                        isospin_degenerate=isospin_degenerate,
                    )
                    if raw_axes != axis_order:
                        raise ValueError("raw Wick expressions disagree on output-axis order")
                    call = _contract_call(
                        equation, operands, scalar_weight=scalar_weight,
                    ).strip()
                    if call.endswith(","):
                        call = call[:-1]
                    slot = raw_count - 1
                    output.write(f"{safe_name}_origen_wick_raw[{slot}] = {call}\n")
                    output.write(f"{safe_name}_origen_wick_sign[{slot}] = {int(term.sign)}\n")
                if raw_count != int(report["wick_count"]):
                    raise RuntimeError(
                        f"raw Wick enumeration produced {raw_count}; reducer reported {report['wick_count']}"
                    )

                output.write("\n\n# 第一步：每个独立 Wick 代表只做一次原始 contract 收缩；字典键是 1 起始 Wick 编号。\n")
                output.write(f"{safe_name}_basis_contract = {{}}\n")
                for item in independent_items:
                    wick_id = int(item["wick_index"])
                    equation, operands, _ = _contraction_expression_from_fields(
                        operators, fields, quarks, antiquarks, item["pairing"],
                        isospin_degenerate=isospin_degenerate,
                    )
                    call = _contract_call(
                        equation, operands, scalar_weight=scalar_weight,
                    ).strip()
                    if call.endswith(","):
                        call = call[:-1]
                    output.write(f"{safe_name}_basis_contract[{wick_id}] = {call}\n")

                output.write("\n# 第二步：构造真正参与线性组合的基底；已包含代表 Wick 符号和裸轴转置。\n")
                output.write(f"{safe_name}_basis = {{}}\n")
                for row in connection.execute(
                    """SELECT ordered_views.output_position, views.wick_id, views.transpose
                       FROM ordered_views
                       JOIN views USING (wick_id, transpose_key)
                       ORDER BY ordered_views.output_position"""
                ):
                    basis_index, wick_id, transpose_text = row
                    permutation = tuple(json.loads(transpose_text))
                    sign = int(independent_by_id[int(wick_id)]["wick_sign"])
                    expression = f"{safe_name}_basis_contract[{wick_id}]"
                    if sign == -1:
                        expression = f"(-{expression})"
                    elif sign != 1:
                        expression = f"({sign} * {expression})"
                    if permutation != identity:
                        expression += f".transpose({', '.join(map(str, permutation))})"
                    output.write(
                        f"# basis[{basis_index}] derives from Wick {wick_id}; output-axis permutation {permutation}.\n"
                    )
                    output.write(f"{safe_name}_basis[{basis_index}] = {expression}\n")

                output.write("\n# 用顶部的一维系数数组与同下标 basis 相乘并求和。\n")
                output.write(
                    f"{safe_name}_result = sum(coefficient * {safe_name}_basis[basis_index] "
                    f"for basis_index, coefficient in enumerate({safe_name}_coefficients) if coefficient)\n"
                )
                output.write("\n")
                output.write(
                    f"# Definition: {safe_name}_origen_wick[slot] = "
                    f"{safe_name}_origen_wick_sign[slot] * "
                    f"{safe_name}_origen_wick_raw[slot].\n"
                )
                output.write(
                    f"# The following right-hand sides reconstruct the signed raw Wick "
                    f"from {safe_name}_basis.\n"
                )
                output.write(f"{safe_name}_origen_wick = {{}}\n")

                for wick_id, terms_text in connection.execute(
                    "SELECT wick_id, terms FROM relations ORDER BY wick_id"
                ):
                    slot = int(wick_id) - 1
                    terms = json.loads(terms_text)
                    expression_terms = []
                    for basis_wick_id, real, imag, permutation_values in terms:
                        permutation = tuple(int(value) for value in permutation_values)
                        basis_index = view_index(int(basis_wick_id), permutation)
                        value = complex(float(real), float(imag))
                        expression_terms.append(
                            f"({_python_complex(value)} * {safe_name}_basis[{basis_index}])"
                        )
                    expression = " + ".join(expression_terms) if expression_terms else "0"
                    output.write(f"{safe_name}_origen_wick[{slot}] = {expression}\n")

                output.write("\n# 映射键是从 0 开始的输出槽位；wick_id 保留原始的 1 起编号。\n")
                output.write(f"{safe_name}_origen_wick_to_independent = {{\n")
                for wick_id, terms_text in connection.execute(
                    "SELECT wick_id, terms FROM relations ORDER BY wick_id"
                ):
                    terms_source = []
                    for basis_wick_id, real, imag, permutation_values in json.loads(terms_text):
                        permutation = tuple(int(value) for value in permutation_values)
                        basis_index = view_index(int(basis_wick_id), permutation)
                        terms_source.append(
                            "{" +
                            f"'basis_index': {int(basis_index)}, "
                            f"'coefficient': {_python_complex(complex(float(real), float(imag)))}, "
                            f"'basis_wick_id': {int(basis_wick_id)}, "
                            f"'transpose': {permutation!r}" +
                            "}"
                        )
                    slot = int(wick_id) - 1
                    output.write(
                        f"    {slot}: {{'wick_id': {int(wick_id)}, 'terms': [{', '.join(terms_source)}]}},\n"
                    )
                output.write("}\n")
            os.replace(str(temporary_output), str(path))
            temporary_output = None
        finally:
            if temporary_output is not None:
                try:
                    temporary_output.unlink()
                except FileNotFoundError:
                    pass
        connection.close()

    report["check_output_streamed"] = True
    return {
        "name": name,
        "mode": "check",
        "output_path": str(path),
        "wick_count": report["wick_count"],
        "independent_count": report["independent_count"],
        "dependent_count": report["dependent_count"],
        "reduced_term_count": nonzero_count,
        "report": report,
    }


def write_case_output(
    example: str,
    output_path: str | Path | None = None,
    *,
    mode: str = "independent",
    isospin_degenerate: bool = True,
    block_options: Optional[Mapping[str, object]] = None,
):
    try:
        operators = model.operator_library()[example]
    except KeyError as exc:
        raise ValueError(f"unknown correlator example: {example}") from exc
    return write_output(
        split_operators(operators), output_path, name=example, mode=mode,
        isospin_degenerate=isospin_degenerate, block_options=block_options,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", required=True, help="built-in correlator name, for example ppn")
    parser.add_argument("--mode", choices=("independent", "check", "block"), default="independent")
    parser.add_argument(
        "--output", type=Path,
        help="Python output document path; optional for --mode block (defaults to result/normal_wick_block/)",
    )
    parser.add_argument("--no-isospin-symmetry", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None and args.mode != "block":
        parser.error("--output is required unless --mode block is selected")
    result = write_case_output(
        args.example, args.output, mode=args.mode,
        isospin_degenerate=not args.no_isospin_symmetry,
    )
    print(
        f"{result['name']}: {result['wick_count']} Wick, "
        f"{result['independent_count']} independent, "
        f"{result['dependent_count']} dependent -> {result['output_path']}"
    )


if __name__ == "__main__":
    main()
