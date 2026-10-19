"""Audit-only input normalization for Wick orbit experiments.

The production command line and reducer are intentionally untouched.  This
module accepts the operator forms already used by the project and exposes a
single raw-Wick enumeration helper for verify_code scripts:

* a flat sequence of ``Operator`` objects;
* ``[sink_operators, source_operators]``;
* a JSON specification mapping or path with the same fields as the existing
  correlator normalizer.

It preserves ordered flavor slots, Gamma slots, open slots, and
``external_tag``.  No orbit action is inferred here: arbitrary operators need
an explicitly supplied action and contract builder before they can be reduced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from kernels import correlator_wick_normalizer as core
from kernels.wick_reducer.operator_model import make_operator, operator_library


Operator = core.Operator


def _operator_from_mapping(item: Mapping[str, Any]) -> Operator:
    required = ("side", "kind", "species", "label", "flavors")
    missing = [name for name in required if name not in item]
    if missing:
        raise ValueError("operator specification is missing " + ", ".join(missing))
    # operator_model.make_operator handles both the legacy one-flavor meson
    # spelling and the explicit (antiquark, quark) pair used by verify_code docs.
    return make_operator(
        item["side"], item["kind"], item["species"], item["label"],
        item["flavors"], item.get("gamma", ""), item.get("gamma_slots"),
        item.get("open_slot"), item.get("external_tag", ""),
        item.get("contraction_metadata", item.get("metadata")),
        mode=item.get("mode", "distillation"),
        momentum=item.get("momentum"), projector=item.get("projector"),
        clebsch=item.get("clebsch"), phase=item.get("phase"),
    )


def _flatten_operator_value(value: Any) -> Tuple[Operator, ...]:
    if isinstance(value, core.Operator):
        return (value,)
    if isinstance(value, Mapping):
        if "operators" in value:
            return _flatten_operator_value(value["operators"])
        if "sink_operators" in value or "source_operators" in value:
            return (
                _flatten_operator_value(value.get("sink_operators", ()))
                + _flatten_operator_value(value.get("source_operators", ()))
            )
        return (_operator_from_mapping(value),)
    if isinstance(value, (str, bytes, Path)):
        raise TypeError("a string/path is a JSON source; call load_operators first")
    if not isinstance(value, Sequence):
        raise TypeError("operators must be an Operator, mapping, or sequence")
    values = tuple(value)
    # A two-list value is the project-friendly [sink_operators, source_operators]
    # form.  A flat list remains in the caller's declared order.
    if len(values) == 2 and all(isinstance(item, (list, tuple)) for item in values):
        return _flatten_operator_value(values[0]) + _flatten_operator_value(values[1])
    result = []
    for item in values:
        result.extend(_flatten_operator_value(item))
    return tuple(result)


def load_operators(value: Any) -> Tuple[Operator, ...]:
    """Normalize an operator object, nested list, JSON mapping, or JSON path."""
    if isinstance(value, Path) or isinstance(value, str):
        path = Path(value)
        document = json.loads(path.read_text(encoding="utf-8"))
        operators = _flatten_operator_value(document)
    elif isinstance(value, Mapping) and "name" in value and "operators" not in value:
        name = value["name"]
        try:
            operators = tuple(operator_library()[name])
        except KeyError as exc:
            raise ValueError("unknown operator-library name " + repr(name)) from exc
    else:
        operators = _flatten_operator_value(value)
    core.validate_operator_modes(operators)
    return operators


def enumerate_raw_wicks(value: Any):
    """Return ``(operators, fields, quarks, antiquarks, wick_terms)``."""
    operators = load_operators(value)
    fields, quarks, antiquarks = core._fields(operators)
    terms = tuple(core.enumerate_wicks(quarks, antiquarks))
    return operators, fields, quarks, antiquarks, terms


__all__ = ["Operator", "enumerate_raw_wicks", "load_operators"]
