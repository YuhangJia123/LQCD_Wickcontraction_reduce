"""Cheap, labeled perambulator incidence keys used before Dirac checks."""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from kernels import correlator_wick_normalizer as core


def _effective_species(operator: core.Operator) -> str:
    """Return the species component used by the generic candidate key.

    Species is an interpolator label, not a contraction factor.  The exact
    quark/antiquark flavors are carried by the directed peram edges, while
    Gamma and contraction factors are carried by the operator metadata;
    normal-mode momentum is carried by the field endpoints.  Keeping this
    field empty therefore lets a label-only rename
    reuse a contraction without merging physically different flavor content.
    """
    del operator
    return ""


def _operator_color(
    operator: core.Operator,
) -> str:
    color = (
        f"operator|{operator.side}|{operator.kind}|"
        f"species:{_effective_species(operator)}|"
        f"gamma:{operator.gamma}|slots:{tuple(operator.gamma_slots)}|"
        f"metadata:{core.operator_factor_metadata(operator)!r}|"
        f"open:{operator.open_slot}|{core.operator_factor_tag(operator)}"
    )
    return color if operator.mode == "distillation" else f"{color}|mode:{operator.mode}"


def _edge_color(q: core.Field, anti: core.Field, isospin_degenerate: bool) -> Tuple:
    def flavor_class(flavor: str) -> str:
        normalized = flavor.lower()
        if isospin_degenerate and normalized in ("u", "d"):
            return "ud"
        return normalized

    peram_type = core.peram_direction(q, anti)
    role = core._line_role(q, anti)
    q_axis = f"bare|{q.side}" if q.bare else "dummy"
    anti_axis = f"bare|{anti.side}" if anti.bare else "dummy"
    q_flavor = flavor_class(q.flavor)
    anti_flavor = flavor_class(anti.flavor)
    endpoint_tags = f"{q.external_tag}>{anti.external_tag}" if (q.external_tag or anti.external_tag) else ""
    color = (
        peram_type, role, endpoint_tags, q_flavor, anti_flavor,
        q_axis, anti_axis, q.kind, anti.kind,
    )
    if q.mode == "normal" or anti.mode == "normal":
        # Preserve the ordered quark/antiquark endpoint labels, including
        # momentum, even when both operator vertices are exchangeable.
        # The shared class label serializes frozen metadata deterministically
        # and keeps descriptive graph colors sortable for heterogeneous p.
        return color + (q.mode, anti.mode, core.peram_class_label(q, anti))
    return color


def incidence_key(
    operators: Sequence[core.Operator],
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    term: core.WickTerm,
    *,
    isospin_degenerate: bool = True,
    operator_maps: Optional[Sequence[Mapping[int, int]]] = None,
) -> Tuple:
    """Build an operator-level key without concrete bare-axis names."""
    raw_edges = []
    for q_index, anti_index in enumerate(term.pairing):
        q, anti = quarks[q_index], antiquarks[anti_index]
        raw_edges.append((
            q.operator,
            anti.operator,
            _edge_color(q, anti, isospin_degenerate),
        ))

    colors = tuple(
        _operator_color(
            operator,
        )
        for operator in operators
    )
    if operator_maps is None:
        return (colors, tuple(sorted(raw_edges)))

    best = None
    for operator_map in operator_maps:
        candidate = (
            colors,
            tuple(sorted(
                (operator_map[source], operator_map[target], color)
                for source, target, color in raw_edges
            )),
        )
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best
