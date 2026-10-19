"""Versioned JSON input for general ordinary-colour Wick reductions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .user_api import HadronOperator, make_operator


if TYPE_CHECKING:
    from . import exact_operator_backend as exact


INPUT_FORMAT = "wick-colour-orbit-input-v1"


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a JSON array")
    return value


def _complex_value(value: Any, where: str) -> complex:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a number or [real, imaginary]")
    if isinstance(value, (int, float)):
        return complex(value)
    if isinstance(value, list) and len(value) == 2:
        real, imaginary = value
        if any(isinstance(part, bool) or not isinstance(part, (int, float))
               for part in (real, imaginary)):
            raise ValueError(f"{where} complex pair must contain two numbers")
        return complex(real, imaginary)
    if isinstance(value, dict) and set(value) == {"real", "imag"}:
        real, imaginary = value["real"], value["imag"]
        if any(isinstance(part, bool) or not isinstance(part, (int, float))
               for part in (real, imaginary)):
            raise ValueError(f"{where} complex object must contain two numbers")
        return complex(real, imaginary)
    raise ValueError(f"{where} must be a number or [real, imaginary]")


def _matrix(value: Any, where: str) -> tuple[tuple[complex, ...], ...]:
    rows = _sequence(value, where)
    if len(rows) != 4:
        raise ValueError(f"{where} must have 4 rows")
    result = []
    for row_index, row in enumerate(rows):
        entries = _sequence(row, f"{where}[{row_index}]")
        if len(entries) != 4:
            raise ValueError(f"{where}[{row_index}] must have 4 entries")
        result.append(tuple(
            _complex_value(entry, f"{where}[{row_index}][{column_index}]")
            for column_index, entry in enumerate(entries)
        ))
    return tuple(result)


def _coordinate(value: Any, where: str) -> tuple[int, int]:
    pair = _sequence(value, where)
    if len(pair) != 2 or any(type(item) is not int for item in pair):
        raise ValueError(f"{where} must be [colour, spin] integers")
    return pair[0], pair[1]


def _hadron_operator(
    record: Mapping[str, Any], side: str, index: int, *, where: str | None = None
) -> HadronOperator:
    where = where or f"operators[{side}][{index}]"
    declared_side = record.get("side", side)
    if declared_side != side:
        raise ValueError(f"{where}.side must match its [sink, source] position")
    required = ("kind", "species", "label", "flavors")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"{where} is missing required fields: {', '.join(missing)}")
    kind = record["kind"]
    if not isinstance(kind, str) or kind not in {"meson", "baryon"}:
        raise ValueError(f"{where}.kind must be 'meson' or 'baryon'")
    flavors = _sequence(record["flavors"], f"{where}.flavors")
    if any(not isinstance(flavor, str) or not flavor for flavor in flavors):
        raise ValueError(f"{where}.flavors entries must be nonempty strings")
    gamma = record.get("gamma", "")
    if gamma is not None and not isinstance(gamma, str):
        raise ValueError(f"{where}.gamma must be a string or null")
    gamma_slots = record.get("gamma_slots")
    if gamma_slots is not None:
        gamma_slots = _sequence(gamma_slots, f"{where}.gamma_slots")
    external_tag = record.get("external_tag", "")
    point = record.get("point", side)
    if not isinstance(external_tag, str) or not isinstance(point, str) or not point:
        raise ValueError(f"{where}.external_tag and point must be strings")
    if "species" in record and not isinstance(record["species"], str):
        raise ValueError(f"{where}.species must be a string")
    if not isinstance(record["label"], str) or not record["label"]:
        raise ValueError(f"{where}.label must be a nonempty string")
    kwargs = {
        "external_tag": external_tag,
        "external_spin": record.get("external_spin"),
        "point": point,
    }
    if "open_slot" in record:
        kwargs["open_slot"] = record["open_slot"]
    return make_operator(
        side,
        kind,
        record["species"],
        record["label"],
        flavors,
        gamma or "",
        gamma_slots,
        **kwargs,
    )


def _explicit_operator(
    record: Mapping[str, Any], side: str, index: int, *, where: str | None = None
) -> exact.Operator:
    from . import exact_operator_backend as exact

    where = where or f"operators[{side}][{index}]"
    name = record.get("name", record.get("label"))
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}.name must be a nonempty string")
    if "factors" in record:
        raw_factors = _sequence(record["factors"], f"{where}.factors")
        factors = []
        for factor_index, raw_factor in enumerate(raw_factors):
            factor_where = f"{where}.factors[{factor_index}]"
            factor = _mapping(raw_factor, factor_where)
            factors.append(_explicit_operator(
                factor, side, index, where=f"{factor_where}",
            ))
        return exact.product_operator(name, factors)

    raw_fields = _sequence(record.get("fields"), f"{where}.fields")
    fields = []
    for field_index, raw_field in enumerate(raw_fields):
        field_where = f"{where}.fields[{field_index}]"
        field = _mapping(raw_field, field_where)
        flavor = field.get("flavor")
        barred = field.get("barred")
        point = field.get("point", side)
        if not isinstance(flavor, str) or not flavor:
            raise ValueError(f"{field_where}.flavor must be a nonempty string")
        if type(barred) is not bool:
            raise ValueError(f"{field_where}.barred must be true or false")
        if not isinstance(point, str) or not point:
            raise ValueError(f"{field_where}.point must be a nonempty string")
        fields.append(exact.Field(flavor, barred, point))

    open_dimensions = _sequence(record.get("open_dimensions", []), f"{where}.open_dimensions")
    if any(type(dimension) is not int for dimension in open_dimensions):
        raise ValueError(f"{where}.open_dimensions entries must be integers")
    raw_terms = _sequence(record.get("terms"), f"{where}.terms")
    terms = []
    for term_index, raw_term in enumerate(raw_terms):
        term_where = f"{where}.terms[{term_index}]"
        term = _mapping(raw_term, term_where)
        raw_coordinates = _sequence(term.get("coordinates"), f"{term_where}.coordinates")
        coordinates = tuple(
            _coordinate(coordinate, f"{term_where}.coordinates[{coordinate_index}]")
            for coordinate_index, coordinate in enumerate(raw_coordinates)
        )
        coefficient = _complex_value(term.get("coefficient", 1), f"{term_where}.coefficient")
        raw_open = _sequence(term.get("open_indices", []), f"{term_where}.open_indices")
        if any(type(value) is not int for value in raw_open):
            raise ValueError(f"{term_where}.open_indices entries must be integers")
        terms.append(exact.TensorTerm(coordinates, coefficient, tuple(raw_open)))
    return exact.explicit_operator(name, fields, terms, open_dimensions=open_dimensions)


def parse_input_document(document: Any) -> dict[str, Any]:
    """Parse the public JSON schema into ``write_output`` arguments.

    ``operators`` may be a flat sequence of records with explicit ``side``
    fields or the legacy ``[sink_operators, source_operators]`` pair. ``type``
    may be ``hadron`` or ``explicit_sparse_tensor``; omitting it defaults
    ``kind=meson|baryon`` records to hadrons.
    """
    document = _mapping(document, "input")
    supported_formats = {
        None, INPUT_FORMAT,
        "wick-colour-orbit-json-v2", "wick-colour-orbit-json-v3",
        "wick-colour-orbit-count-v2", "wick-colour-orbit-count-v3",
    }
    if document.get("format") not in supported_formats:
        raise ValueError(
            "input.format must be an input schema or a supported result JSON version"
        )
    name = document.get("name", "correlator")
    if not isinstance(name, str) or not name:
        raise ValueError("input.name must be a nonempty string")
    mode = document.get("mode", "independent")
    if not isinstance(mode, str) or mode not in {"independent", "check", "count"}:
        raise ValueError("input.mode must be 'independent', 'check', or 'count'")
    normal_ordered = document.get("normal_ordered", False)
    if type(normal_ordered) is not bool:
        raise ValueError("input.normal_ordered must be true or false")
    wick_scope = document.get("wick_scope", "complete")
    if wick_scope not in {"complete", "cross_time"}:
        raise ValueError("input.wick_scope must be 'complete' or 'cross_time'")
    tolerance = document.get("tolerance", 1.0e-12)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("input.tolerance must be a nonnegative number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("input.tolerance must be a finite nonnegative number")
    isospin_degenerate_perams = document.get("isospin_degenerate_perams", True)
    if type(isospin_degenerate_perams) is not bool:
        raise ValueError("input.isospin_degenerate_perams must be true or false")

    gamma_matrices = {}
    raw_gamma = _mapping(document.get("gamma_matrices", {}), "input.gamma_matrices")
    for matrix_name, value in raw_gamma.items():
        if not isinstance(matrix_name, str) or not matrix_name:
            raise ValueError("input.gamma_matrices keys must be nonempty strings")
        gamma_matrices[matrix_name] = _matrix(
            value, f"input.gamma_matrices[{matrix_name!r}]"
        )

    raw_operators = _sequence(document.get("operators"), "input.operators")
    if len(raw_operators) == 2 and all(isinstance(side, list) for side in raw_operators):
        side_records = {
            side: _sequence(records, f"input.operators[{side}]")
            for side, records in zip(("sink", "source"), raw_operators)
        }
        side_where = {"sink": "operators[sink]", "source": "operators[source]"}
    else:
        side_records = {"sink": [], "source": []}
        for index, raw_record in enumerate(raw_operators):
            record = _mapping(raw_record, f"input.operators[{index}]")
            side = record.get("side")
            if side not in side_records:
                raise ValueError(
                    f"input.operators[{index}].side must be 'sink' or 'source'"
                )
            side_records[side].append(record)
        side_where = {"sink": "operators", "source": "operators"}

    operators = []
    for side in ("sink", "source"):
        records = side_records[side]
        if not records:
            raise ValueError(f"input.operators[{side}] must not be empty")
        parsed = []
        for index, raw_record in enumerate(records):
            where = (
                f"input.operators[{side}][{index}]"
                if side_where[side] != "operators"
                else f"input.operators[{index}]"
            )
            record = _mapping(raw_record, where)
            operator_type = record.get("type")
            if operator_type is None:
                kind = record.get("kind")
                operator_type = (
                    "hadron" if isinstance(kind, str) and kind in {"meson", "baryon"}
                    else "product" if "factors" in record
                    else "explicit_sparse_tensor"
                )
            if operator_type == "hadron":
                parsed.append(_hadron_operator(record, side, index, where=where))
            elif operator_type == "explicit_sparse_tensor":
                parsed.append(_explicit_operator(record, side, index, where=where))
            elif operator_type == "product":
                parsed.append(_explicit_operator(record, side, index, where=where))
            else:
                raise ValueError(
                    f"{where}.type must be 'hadron', 'explicit_sparse_tensor', or 'product'"
                )
        operators.append(parsed)

    return {
        "operators": operators,
        "name": name,
        "mode": mode,
        "normal_ordered": normal_ordered,
        "wick_scope": wick_scope,
        "tolerance": tolerance,
        "gamma_matrices": gamma_matrices,
        "isospin_degenerate_perams": isospin_degenerate_perams,
    }


def load_input(path: str | Path) -> dict[str, Any]:
    """Read and validate a JSON input document."""
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in {input_path} at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error
    return parse_input_document(document)


__all__ = ["INPUT_FORMAT", "load_input", "parse_input_document"]
