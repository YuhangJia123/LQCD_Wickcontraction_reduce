#!/usr/bin/env python3
"""Generic symbolic Wick and incidence normalizer for distillation correlators.

This program enumerates flavor-compatible Gaussian contractions and builds
two colored incidence graphs per contraction:

* an elemental graph with VVV legs unordered and VDV q/bar legs directed;
* a decorated graph that also includes ordered Gamma ports and open axes.

It evaluates factorized Dirac Gram products for registered 4x4 Gamma
matrices, but does not evaluate numerical VVV or perambulator arrays. The
reported rank is therefore a coefficient-space rank, not the numerical rank
of a physical correlator.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


ROLE_ORDER = ("bb", "b0", "0b", "00")
PERAM_ORDER = ("sin_sin", "sin_sou", "sou_sin", "sou_sou")
OPERATOR_MODES = ("distillation", "normal")
AXIS_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
Matrix = Tuple[Tuple[complex, ...], ...]
SpinAxis = Tuple[int, str]


def validate_operator_mode(mode: str) -> str:
    if mode not in OPERATOR_MODES:
        raise ValueError("operator mode must be 'distillation' or 'normal'")
    return mode


@dataclass(frozen=True)
class Operator:
    side: str
    kind: str
    species: str
    label: str
    flavors: Tuple[str, ...]
    gamma: str
    gamma_slots: Tuple[int, ...]
    open_slot: Optional[int]
    external_tag: str
    # Additional factors that affect the contraction but are not part of the
    # hadron name.  Values are recursively frozen by ``normalize_metadata``
    # so the field remains hashable and deterministic in canonical keys.
    contraction_metadata: Tuple[Tuple[str, object], ...] = ()
    mode: str = "distillation"

    def __post_init__(self) -> None:
        validate_operator_mode(self.mode)


def _freeze_metadata(value: Any) -> object:
    """Return a deterministic, hashable representation of metadata values."""
    if isinstance(value, Mapping):
        if set(value) == {"$complex"}:
            parts = value["$complex"]
            if isinstance(parts, (list, tuple)) and len(parts) == 2:
                return ("complex", float(parts[0]), float(parts[1]))
        return tuple(
            (str(key), _freeze_metadata(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_metadata(item) for item in value), key=repr))
    if isinstance(value, complex):
        return ("complex", value.real, value.imag)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def normalize_contraction_metadata(
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    momentum: Any = None,
    projector: Any = None,
    clebsch: Any = None,
    phase: Any = None,
) -> Tuple[Tuple[str, object], ...]:
    """Normalize optional contraction factors into a stable tuple.

    ``momentum``, ``projector``, ``clebsch`` and ``phase`` are convenience
    names for common factors.  A mapping can carry additional application
    specific factors.  ``None`` means that a factor was not supplied, while
    values such as ``0`` and ``1`` remain significant.
    """
    values: Dict[str, Any] = {}
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("contraction_metadata must be a mapping")
        values.update(metadata)
    for name, value in (
        ("momentum", momentum),
        ("projector", projector),
        ("clebsch", clebsch),
        ("phase", phase),
    ):
        if value is not None:
            values[name] = value
    return tuple(
        (str(key), _freeze_metadata(value))
        for key, value in sorted(values.items(), key=lambda pair: str(pair[0]))
    )


def _metadata_json_value(value: object) -> object:
    """Convert frozen metadata back to JSON-compatible values for reports."""
    if isinstance(value, tuple):
        if len(value) == 3 and value[0] == "complex":
            return {"$complex": [value[1], value[2]]}
        return [_metadata_json_value(item) for item in value]
    return value


def contraction_metadata_dict(metadata: Sequence[Tuple[str, object]]) -> Dict[str, object]:
    """Return a readable JSON-compatible mapping for a normalized tuple."""
    return {key: _metadata_json_value(value) for key, value in metadata}


@dataclass(frozen=True)
class Field:
    index: int
    operator: int
    side: str
    species: str
    label: str
    external_tag: str
    kind: str
    flavor: str
    slot: int
    bare: bool
    axis: Optional[str]
    mode: str = "distillation"
    endpoint_descriptor: Tuple[object, ...] = ("v",)


def validate_operator_modes(operators: Sequence[Operator]) -> str:
    """Validate the construction mode shared by a complete correlator."""
    modes = {validate_operator_mode(operator.mode) for operator in operators}
    if len(modes) > 1:
        raise ValueError("all operators in a correlator must use the same operator mode; mixed normal/distillation input is unsupported")
    return next(iter(modes), "distillation")


def operator_factor_tag(operator: Operator) -> str:
    """Normal momentum labels belong to peram endpoints, not elementals."""
    return "" if operator.mode == "normal" else operator.external_tag


def operator_factor_metadata(operator: Operator) -> Tuple[Tuple[str, object], ...]:
    if operator.mode == "normal":
        return tuple(item for item in operator.contraction_metadata if item[0] != "momentum")
    return operator.contraction_metadata


def operator_endpoint_descriptor(operator: Operator) -> Tuple[object, ...]:
    """Hashable endpoint space and p label, shared by both field builders."""
    if operator.mode == "normal":
        momentum = tuple(
            (name, _freeze_metadata(value))
            for name, value in operator.contraction_metadata if name == "momentum"
        )
        return ("x", operator.external_tag, momentum)
    return ("v",)


def field_endpoint_metadata(field: Field) -> Dict[str, object]:
    result: Dict[str, object] = {
        "mode": field.mode,
        "representation": field.endpoint_descriptor[0],
        "external_tag": field.external_tag,
    }
    if field.mode == "normal":
        result.update(contraction_metadata_dict(field.endpoint_descriptor[2]))
    return result


def peram_direction(q: Field, anti: Field) -> str:
    return f"{q.side}_{anti.side}".replace("sink", "sin").replace("source", "sou")


def peram_class_label(q: Field, anti: Field, flavor_tag: str = "") -> str:
    """Describe an ordered peram class without changing legacy spellings."""
    direction = peram_direction(q, anti)
    if q.mode == "normal" or anti.mode == "normal":
        spaces = f"{q.endpoint_descriptor[0]}_{anti.endpoint_descriptor[0]}"
        direction += f"_{spaces}"
        endpoint_tags = repr((q.endpoint_descriptor, anti.endpoint_descriptor))
    else:
        endpoint_tags = f"{q.external_tag}>{anti.external_tag}" if q.external_tag or anti.external_tag else ""
    suffix = "|".join(value for value in (endpoint_tags, flavor_tag) if value)
    return f"{direction}|{suffix}" if suffix else direction


@dataclass(frozen=True)
class WickTerm:
    # For each quark field in field-list order, the matched antiquark index.
    pairing: Tuple[int, ...]
    sign: int


@dataclass(frozen=True)
class TensorNetwork:
    node_colors: Tuple[str, ...]
    node_meta: Tuple[Dict[str, object], ...]
    edges: Tuple[Tuple[int, int, str], ...]
    peram_nodes: Tuple[int, ...]
    peram_lines: Tuple[Dict[str, object], ...]
    sink_rows: Tuple[Dict[str, object], ...]
    vvv_rows: Tuple[Dict[str, object], ...]


def _matrix_product(left: Matrix, right: Matrix) -> Matrix:
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(len(right))) for j in range(len(right[0])))
        for i in range(len(left))
    )


def _matrix_dagger(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[j][i].conjugate() for j in range(len(matrix))) for i in range(len(matrix[0])))


def _matrix_transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(matrix[j][i] for j in range(len(matrix))) for i in range(len(matrix[0])))


def _euclidean_gamma_matrices() -> Dict[str, Matrix]:
    identity2: Matrix = ((1 + 0j, 0j), (0j, 1 + 0j))
    zero2: Matrix = ((0j, 0j), (0j, 0j))
    sigma1: Matrix = ((0j, 1 + 0j), (1 + 0j, 0j))
    sigma2: Matrix = ((0j, -1j), (1j, 0j))
    sigma3: Matrix = ((1 + 0j, 0j), (0j, -1 + 0j))

    def block(a: Matrix, b: Matrix, c: Matrix, d: Matrix) -> Matrix:
        return tuple(
            tuple(a[i][j] if i < 2 and j < 2 else
                  b[i][j - 2] if i < 2 else
                  c[i - 2][j] if j < 2 else
                  d[i - 2][j - 2] for j in range(4))
            for i in range(4)
        )

    def spatial(pauli: Matrix) -> Matrix:
        minus_i_pauli = tuple(tuple(-1j * value for value in row) for row in pauli)
        plus_i_pauli = tuple(tuple(1j * value for value in row) for row in pauli)
        return block(zero2, minus_i_pauli, plus_i_pauli, zero2)

    g1, g2, g3 = (spatial(pauli) for pauli in (sigma1, sigma2, sigma3))
    g4 = block(identity2, zero2, zero2, tuple(tuple(-value for value in row) for row in identity2))
    g5 = _matrix_product(_matrix_product(_matrix_product(g1, g2), g3), g4)
    charge = _matrix_product(g4, g2)
    cg5 = _matrix_product(charge, g5)
    cg3 = _matrix_product(charge, g3)
    return {
        "gamma1": g1,
        "gamma2": g2,
        "gamma3": g3,
        "gamma4": g4,
        "gamma5": g5,
        "Cgamma5": cg5,
        "Cgamma5_dagger": _matrix_dagger(cg5),
        "Cgamma3": cg3,
        "Cgamma3_dagger": _matrix_dagger(cg3),
    }


def _parse_matrix(value: Sequence[Sequence[object]]) -> Matrix:
    rows = []
    for row in value:
        parsed = []
        for entry in row:
            parsed.append(complex(entry[0], entry[1]) if isinstance(entry, list) else complex(entry))
        rows.append(tuple(parsed))
    matrix = tuple(rows)
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Dirac matrices must be 4x4")
    return matrix


def op(
    side: str,
    kind: str,
    species: str,
    label: str,
    flavors: Sequence[str],
    gamma: str,
    gamma_slots: Optional[Sequence[int]] = None,
    open_slot: Optional[int] = None,
    external_tag: str = "",
    contraction_metadata: Optional[Mapping[str, Any]] = None,
    *,
    mode: str = "distillation",
    momentum: Any = None,
    projector: Any = None,
    clebsch: Any = None,
    phase: Any = None,
) -> Operator:
    validate_operator_mode(mode)
    if side not in ("sink", "source"):
        raise ValueError("side must be 'sink' or 'source'")
    if kind not in ("baryon", "meson"):
        raise ValueError("kind must be 'baryon' or 'meson'")
    flavors = tuple(flavors)
    if kind == "baryon" and len(flavors) != 3:
        raise ValueError("baryon operators require exactly three flavor slots")
    if kind == "meson":
        if len(flavors) not in (1, 2):
            raise ValueError("meson flavors must be (antiquark_flavor, quark_flavor)")
    slots = tuple(gamma_slots if gamma_slots is not None else ((0, 1) if gamma else ()))
    if any(slot < 0 or slot >= (3 if kind == "baryon" else 2) for slot in slots):
        raise ValueError("gamma_slots contains an invalid field slot")
    if open_slot is None and kind == "baryon":
        open_slot = 2
    if open_slot is not None and (kind != "baryon" or not 0 <= open_slot < 3):
        raise ValueError("open_slot must identify a baryon field slot")
    if open_slot is not None and open_slot in slots:
        raise ValueError("an open slot cannot also be a Gamma slot")
    normalized_metadata = normalize_contraction_metadata(
        contraction_metadata,
        momentum=momentum,
        projector=projector,
        clebsch=clebsch,
        phase=phase,
    )
    return Operator(
        side, kind, species, label, flavors, gamma, slots, open_slot,
        external_tag, normalized_metadata, mode,
    )


def _example_specs() -> Dict[str, Tuple[Operator, ...]]:
    def baryon(side: str, species: str, label: str, flavors: Sequence[str], gamma: str) -> Operator:
        return op(side, "baryon", species, label, flavors, gamma, (0, 1), 2)

    def pion(side: str, label: str) -> Operator:
        # pi0 is deliberately the single u-bar-u component requested here.
        return op(side, "meson", "pi0_uubar", label, ("u",), "gamma5", (0, 1), None)

    p = ("u", "d", "u")
    n = ("d", "u", "d")
    delta_pp = ("u", "u", "u")
    delta_mm = ("d", "d", "d")

    return {
        "pipi": (
            pion("sink", "sink_pi_1"), pion("sink", "sink_pi_2"),
            pion("source", "source_pi_1"), pion("source", "source_pi_2"),
        ),
        "delta": (
            baryon("sink", "Delta++", "sink_Delta++", delta_pp, "Cgamma3"),
            baryon("source", "Delta++", "source_Delta++", delta_pp, "Cgamma3_dagger"),
        ),
        "pn": (
            baryon("sink", "p", "sink_p", p, "Cgamma5"),
            baryon("sink", "n", "sink_n", n, "Cgamma5"),
            baryon("source", "p", "source_p", p, "Cgamma5_dagger"),
            baryon("source", "n", "source_n", n, "Cgamma5_dagger"),
        ),
        "ppn": (
            baryon("sink", "p", "sink_p_1", p, "Cgamma5"),
            baryon("sink", "p", "sink_p_2", p, "Cgamma5"),
            baryon("sink", "n", "sink_n", n, "Cgamma5"),
            baryon("source", "p", "source_p_1", p, "Cgamma5_dagger"),
            baryon("source", "p", "source_p_2", p, "Cgamma5_dagger"),
            baryon("source", "n", "source_n", n, "Cgamma5_dagger"),
        ),
        "pip": (
            pion("sink", "sink_pi"),
            baryon("sink", "p", "sink_p", p, "Cgamma5"),
            pion("source", "source_pi"),
            baryon("source", "p", "source_p", p, "Cgamma5_dagger"),
        ),
        # Off-diagonal matrix element: sink p, source pi0+p.
        "pip-p": (
            baryon("sink", "p", "sink_p", p, "Cgamma5"),
            pion("source", "source_pi",),
            baryon("source", "p", "source_p", p, "Cgamma5_dagger"),
        ),
        "delta++delta-": (
            baryon("sink", "Delta++", "sink_Delta++", delta_pp, "Cgamma3"),
            baryon("sink", "Delta-", "sink_Delta-", delta_mm, "Cgamma3"),
            baryon("source", "Delta++", "source_Delta++", delta_pp, "Cgamma3_dagger"),
            baryon("source", "Delta-", "source_Delta-", delta_mm, "Cgamma3_dagger"),
        ),
    }


def _fields(operators: Sequence[Operator]) -> Tuple[List[Field], List[Field], List[Field]]:
    validate_operator_modes(operators)
    sink_axes = iter(AXIS_NAMES)
    sink_open_count = sum(
        operator.open_slot is not None for operator in operators if operator.side == "sink"
    )
    source_axes = iter(AXIS_NAMES[sink_open_count:])
    fields: List[Field] = []
    for op_index, operator in enumerate(operators):
        if operator.kind == "baryon":
            kinds = ("q", "q", "q") if operator.side == "sink" else ("bar", "bar", "bar")
            flavors = operator.flavors
        else:
            # The VDV ports are directed and ordered (antiquark, quark).
            kinds = ("bar", "q")
            if len(operator.flavors) == 1:
                flavors = (operator.flavors[0], operator.flavors[0])
            elif len(operator.flavors) == 2:
                flavors = operator.flavors
            else:
                raise ValueError("meson operator needs one legacy flavor or an (anti, quark) pair")
        for slot, (kind, flavor) in enumerate(zip(kinds, flavors)):
            bare = operator.open_slot == slot
            axis = None
            if bare:
                axis = next(sink_axes if operator.side == "sink" else source_axes)
            fields.append(Field(
                index=len(fields), operator=op_index, side=operator.side,
                species=operator.species, label=operator.label, external_tag=operator.external_tag, kind=kind,
                flavor=flavor, slot=slot, bare=bare, axis=axis,
                mode=operator.mode,
                endpoint_descriptor=operator_endpoint_descriptor(operator),
            ))
    quarks = [field for field in fields if field.kind == "q"]
    antiquarks = [field for field in fields if field.kind == "bar"]
    return fields, quarks, antiquarks


def _permutation_sign(values: Sequence[int]) -> int:
    inversions = sum(
        values[i] > values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def enumerate_wicks(quarks: Sequence[Field], antiquarks: Sequence[Field]) -> Iterator[WickTerm]:
    if len(quarks) != len(antiquarks):
        raise ValueError(f"unbalanced quark/antiquark count: {len(quarks)} != {len(antiquarks)}")
    q_by_flavor: Dict[str, List[int]] = defaultdict(list)
    bar_by_flavor: Dict[str, List[int]] = defaultdict(list)
    for i, field in enumerate(quarks):
        q_by_flavor[field.flavor].append(i)
    for i, field in enumerate(antiquarks):
        bar_by_flavor[field.flavor].append(i)
    if set(q_by_flavor) != set(bar_by_flavor):
        raise ValueError("quark and antiquark flavor sets differ")
    for flavor in q_by_flavor:
        if len(q_by_flavor[flavor]) != len(bar_by_flavor[flavor]):
            raise ValueError(f"unbalanced flavor {flavor}")

    flavors = sorted(q_by_flavor)
    choices = [tuple(itertools.permutations(bar_by_flavor[flavor])) for flavor in flavors]
    for selected in itertools.product(*choices):
        pairing = [-1] * len(quarks)
        for flavor, anti_permutation in zip(flavors, selected):
            for q_index, anti_index in zip(q_by_flavor[flavor], anti_permutation):
                pairing[q_index] = anti_index
        # The sign is the parity of the full fermionic reordering from the
        # original operator field order to ``q_0 ... q_n bar_{pi(0)} ...``.
        # For baryon-only inputs this differs from the pairing-only parity by
        # at most a correlator-wide convention sign; it also accounts for the
        # intrinsic ``bar,q`` order of a meson and interleaved operators.
        # Any additional (-1)^(N(N-1)/2) Berezin convention is global and is
        # intentionally left to the caller.
        ordered_original_positions = tuple(field.index for field in quarks) + tuple(
            antiquarks[anti_index].index for anti_index in pairing
        )
        yield WickTerm(tuple(pairing), _permutation_sign(ordered_original_positions))


def _profile(
    term: WickTerm,
    quarks: Sequence[Field],
    antiquarks: Sequence[Field],
    isospin_degenerate: bool,
):
    roles: Dict[str, Counter] = defaultdict(Counter)
    line_types: List[str] = []
    open_routing: List[Tuple[str, str]] = []
    for q_index, anti_index in enumerate(term.pairing):
        q, anti = quarks[q_index], antiquarks[anti_index]
        peram_type = peram_direction(q, anti)
        line_types.append(peram_type)
        flavor_tag = "" if isospin_degenerate else q.flavor
        peram_class = peram_class_label(q, anti, flavor_tag)
        sink_bare = (q.bare and q.side == "sink") or (anti.bare and anti.side == "sink")
        source_bare = (q.bare and q.side == "source") or (anti.bare and anti.side == "source")
        role = "bb" if sink_bare and source_bare else "b0" if sink_bare else "0b" if source_bare else "00"
        roles[peram_class][role] += 1
        if q.axis or anti.axis:
            open_routing.append((q.axis or "-", anti.axis or "-"))
    profile = tuple(
        (peram_class, tuple(roles[peram_class].get(role, 0) for role in ROLE_ORDER))
        for peram_class in sorted(roles)
    )
    type_counts = tuple((name, line_types.count(name)) for name in PERAM_ORDER)
    return profile, type_counts, tuple(open_routing)


def _line_role(q: Field, anti: Field) -> str:
    sink_bare = (q.bare and q.side == "sink") or (anti.bare and anti.side == "sink")
    source_bare = (q.bare and q.side == "source") or (anti.bare and anti.side == "source")
    return "bb" if sink_bare and source_bare else "b0" if sink_bare else "0b" if source_bare else "00"


def _canonical_graph(
    node_colors: Sequence[str], edges: Sequence[Tuple[int, int, str]]
) -> Tuple[Tuple, Tuple[int, ...]]:
    """Canonicalize a directed colored multigraph and return its node order."""
    n = len(node_colors)
    adjacency: List[List[Tuple[int, str]]] = [[] for _ in range(n)]
    for source, target, edge_color in edges:
        adjacency[source].append((target, f"out|{edge_color}"))
        adjacency[target].append((source, f"in|{edge_color}"))

    def refine(labels: List[int]) -> List[int]:
        while True:
            signatures = [
                (labels[node], tuple(sorted((edge, labels[other]) for other, edge in adjacency[node])))
                for node in range(n)
            ]
            unique = {signature for signature in signatures}
            ranks = {signature: rank for rank, signature in enumerate(sorted(unique))}
            updated = [ranks[signature] for signature in signatures]
            same_partition = all(
                (labels[i] == labels[j]) == (updated[i] == updated[j])
                for i in range(n) for j in range(i)
            )
            labels = updated
            if same_partition:
                return labels

    initial_values = {value: index for index, value in enumerate(sorted(set(node_colors)))}
    initial = [initial_values[value] for value in node_colors]

    def twin(left: int, right: int) -> bool:
        left_neighbours = Counter((edge, node) for node, edge in adjacency[left] if node != right)
        right_neighbours = Counter((edge, node) for node, edge in adjacency[right] if node != left)
        left_to_right = Counter(edge for node, edge in adjacency[left] if node == right)
        right_to_left = Counter(edge for node, edge in adjacency[right] if node == left)
        left_loops = Counter(edge for node, edge in adjacency[left] if node == left)
        right_loops = Counter(edge for node, edge in adjacency[right] if node == right)
        return (
            left_neighbours == right_neighbours
            and left_to_right == right_to_left
            and left_loops == right_loops
        )

    def visit(labels: List[int]) -> Tuple[Tuple, Tuple[int, ...]]:
        labels = refine(labels)
        cells: Dict[int, List[int]] = defaultdict(list)
        for node, color in enumerate(labels):
            cells[color].append(node)
        nonsingletons = [cell for cell in cells.values() if len(cell) > 1]
        if not nonsingletons:
            order = [node for node, _ in sorted(enumerate(labels), key=lambda item: item[1])]
            position = {node: rank for rank, node in enumerate(order)}
            canonical_edges = tuple(sorted(
                (position[source], position[target], color)
                for source, target, color in edges
            ))
            return (tuple(node_colors[node] for node in order), canonical_edges), tuple(order)

        cell = min(nonsingletons, key=lambda group: (len(group), labels[group[0]]))
        best: Optional[Tuple[Tuple, Tuple[int, ...]]] = None
        representatives: List[int] = []
        for candidate in cell:
            if any(twin(candidate, previous) for previous in representatives):
                continue
            representatives.append(candidate)
            individualized = list(labels)
            individualized[candidate] = max(labels, default=-1) + 1
            value = visit(individualized)
            if best is None or repr(value[0]) < repr(best[0]):
                best = value
        assert best is not None
        return best

    return visit(initial)


def _build_tensor_network(
    operators: Sequence[Operator],
    fields: Sequence[Field],
    quarks: Sequence[Field],
    antiquarks: Sequence[Field],
    term: WickTerm,
    isospin_degenerate: bool,
) -> TensorNetwork:
    """Build a full factor graph with independent V and Dirac incidences."""
    colors: List[str] = []
    meta: List[Dict[str, object]] = []
    edges: List[Tuple[int, int, str]] = []
    op_nodes: Dict[int, int] = {}
    elemental_nodes: Dict[int, int] = {}
    gamma_nodes: Dict[int, int] = {}
    open_nodes: Dict[int, int] = {}

    def add_node(color: str, **attributes: object) -> int:
        index = len(colors)
        colors.append(color)
        meta.append({"id": index, **attributes})
        return index

    for op_index, operator in enumerate(operators):
        mode_suffix = "|mode:normal" if operator.mode == "normal" else ""
        op_nodes[op_index] = add_node(
            f"operator|{operator.side}|{operator.kind}|{operator.species}|{operator_factor_tag(operator)}|"
            f"metadata:{operator_factor_metadata(operator)!r}{mode_suffix}",
            kind="operator", operator=op_index, label=operator.label,
            species=operator.species, side=operator.side, mode=operator.mode,
            contraction_metadata=contraction_metadata_dict(operator.contraction_metadata),
        )
        tensor = "VVV" if operator.kind == "baryon" else "VDV"
        elemental_nodes[op_index] = add_node(
            f"elemental|{tensor}|{operator.side}{mode_suffix}", kind="elemental",
            operator=op_index, tensor=tensor, side=operator.side, mode=operator.mode,
        )
        edges.append((op_nodes[op_index], elemental_nodes[op_index], "operator:elemental"))
        if operator.gamma_slots:
            gamma_nodes[op_index] = add_node(
                f"gamma|{operator.side}|{operator.gamma}|arity:{len(operator.gamma_slots)}",
                kind="gamma", operator=op_index, gamma=operator.gamma,
                ordered_ports=len(operator.gamma_slots), side=operator.side,
            )
            edges.append((op_nodes[op_index], gamma_nodes[op_index], "operator:gamma"))

    for field in fields:
        if field.bare:
            operator = operators[field.operator]
            open_nodes[field.index] = add_node(
                f"open|{operator.side}", kind="open", operator=field.operator,
                axis=field.axis, slot=field.slot, side=operator.side,
            )
            edges.append((op_nodes[field.operator], open_nodes[field.index], "operator:open"))

    def endpoint(field: Field) -> Dict[str, object]:
        operator = operators[field.operator]
        return {
            "field": field.index, "operator": field.operator,
            "operator_label": operator.label, "species": operator.species,
            "side": field.side, "field_kind": field.kind,
            "slot": field.slot, "flavor": field.flavor,
            "bare": field.bare, "axis": field.axis,
            **field_endpoint_metadata(field),
        }

    peram_nodes: List[int] = []
    peram_lines: List[Dict[str, object]] = []
    sink_rows: Dict[int, List[Tuple[str, int, str]]] = defaultdict(list)
    elemental_legs: Dict[int, List[Tuple[int, int, str]]] = defaultdict(list)
    for line_id, anti_index in enumerate(term.pairing):
        q, anti = quarks[line_id], antiquarks[anti_index]
        q_operator, anti_operator = operators[q.operator], operators[anti.operator]
        peram_type = peram_direction(q, anti)
        role = _line_role(q, anti)
        flavor_tag = "" if isospin_degenerate else q.flavor
        color = "peram|" + peram_class_label(q, anti, flavor_tag)
        line_node = add_node(
            color, kind="peram", line_id=line_id, peram_type=peram_type,
            role=role, flavor=q.flavor,
        )
        peram_nodes.append(line_node)
        peram_lines.append({
            "id": line_id, "node": line_node, "peram_type": peram_type,
            "role": role,
            "numeric_flavor_class": "ud-degenerate" if isospin_degenerate else q.flavor,
            "q": endpoint(q), "bar": endpoint(anti),
        })

        for field, operator, endpoint_kind in ((q, q_operator, "q"), (anti, anti_operator, "bar")):
            tensor = "VVV" if operator.kind == "baryon" else "VDV"
            v_port = "V:VVV" if tensor == "VVV" else f"V:VDV:{endpoint_kind}"
            edges.append((elemental_nodes[field.operator], line_node, v_port))
            sink_rows[field.operator].append((f"{peram_type}:{role}", field.slot, endpoint_kind))
            elemental_legs[field.operator].append((line_id, field.slot, endpoint_kind))
            if field.slot in operator.gamma_slots:
                gamma_port = operator.gamma_slots.index(field.slot)
                edges.append((gamma_nodes[field.operator], line_node,
                              f"D:Gamma:{endpoint_kind}:port:{gamma_port}"))
            elif field.bare:
                edges.append((open_nodes[field.index], line_node, f"D:open:{endpoint_kind}"))
            else:
                raise ValueError(
                    f"field {operator.label}[{field.slot}] needs a Gamma port or an open axis"
                )

    rows = []
    vvv_rows = []
    for op_index, operator in enumerate(operators):
        if operator.kind == "baryon":
            vvv_rows.append({
                "operator": op_index,
                "label": operator.label,
                "species": operator.species,
                "side": operator.side,
                "legs": tuple(sorted(elemental_legs.get(op_index, ()), key=lambda item: item[1])),
            })
        if operator.side != "sink":
            continue
        tensor = "VVV" if operator.kind == "baryon" else "VDV"
        entries = sink_rows.get(op_index, [])
        signature = (
            tuple(sorted(color for color, _slot, _end in entries))
            if tensor == "VVV"
            else tuple(sorted((end, color) for color, _slot, end in entries))
        )
        rows.append({
            "operator": op_index, "label": operator.label,
            "species": operator.species, "tensor": tensor,
            "signature": signature,
            "legs": tuple(sorted(entries, key=lambda item: (item[1], item[2]))),
        })
    return TensorNetwork(tuple(colors), tuple(meta), tuple(edges), tuple(peram_nodes),
                         tuple(peram_lines), tuple(rows), tuple(vvv_rows))


def _project_network(
    graph: TensorNetwork,
    sides: Sequence[str],
    include_gamma: bool,
    include_open: bool,
    label_open_axes: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[Tuple[int, int, str], ...], Tuple[int, ...]]:
    allowed_sides = set(sides)
    remap: Dict[int, int] = {}
    colors: List[str] = []
    for old, node in enumerate(graph.node_meta):
        kind, side = node["kind"], node.get("side")
        keep = (
            kind == "peram"
            or (kind in ("operator", "elemental") and side in allowed_sides)
            or (kind == "gamma" and include_gamma and side in allowed_sides)
            or (kind == "open" and include_open and side in allowed_sides)
        )
        if not keep:
            continue
        color = graph.node_colors[old]
        if kind == "open":
            color = f"open|{side}"
            if label_open_axes:
                color += f"|axis:{node['axis']}"
        remap[old] = len(colors)
        colors.append(color)
    edges = tuple(
        (remap[source], remap[target], color)
        for source, target, color in graph.edges
        if source in remap and target in remap
    )
    line_nodes = tuple(remap[node] for node in graph.peram_nodes)
    return tuple(colors), edges, line_nodes


def _canonical_projection(
    graph: TensorNetwork,
    sides: Sequence[str],
    include_gamma: bool,
    include_open: bool,
    label_open_axes: bool = False,
) -> Tuple[Tuple, Tuple[int, ...], Tuple[int, ...]]:
    colors, edges, line_nodes = _project_network(
        graph, sides, include_gamma, include_open, label_open_axes
    )
    key, order = _canonical_graph(colors, edges)
    ranks = {node: rank for rank, node in enumerate(order)}
    line_order = tuple(sorted(range(len(line_nodes)), key=lambda line: ranks[line_nodes[line]]))
    return key, order, line_order


def _vvv_characters(
    rows: Sequence[Dict[str, object]], line_order: Sequence[int]
) -> Tuple[Dict[str, int], List[Dict[str, object]]]:
    line_rank = {line_id: rank for rank, line_id in enumerate(line_order)}
    side_signs = {"sink": 1, "source": 1}
    details = []
    for row in rows:
        ordered_legs = sorted(row["legs"], key=lambda item: item[1])
        canonical_positions = [line_rank[leg[0]] for leg in ordered_legs]
        sign = _permutation_sign(canonical_positions)
        side_signs[row["side"]] *= sign
        details.append({
            "operator": row["operator"],
            "label": row["label"],
            "side": row["side"],
            "slot_to_canonical_line_rank": canonical_positions,
            "character": sign,
        })
    return side_signs, details


def _peram_template_order(graph: TensorNetwork) -> Tuple[int, ...]:
    type_rank = {name: rank for rank, name in enumerate(PERAM_ORDER)}
    ordered = sorted(
        graph.peram_lines,
        key=lambda line: (
            type_rank[line["peram_type"]],
            line["numeric_flavor_class"],
            line["q"]["field"],
        ),
    )
    return tuple(line["id"] for line in ordered)


def _dirac_payload(
    operators: Sequence[Operator],
    fields: Sequence[Field],
    quarks: Sequence[Field],
    antiquarks: Sequence[Field],
    term: WickTerm,
    side: str,
    line_order: Sequence[int],
    matrix_registry: Mapping[str, Matrix],
) -> Optional[Dict[str, object]]:
    line_rank = {line_id: rank for rank, line_id in enumerate(line_order)}
    field_endpoint: Dict[int, SpinAxis] = {}
    for line_id, anti_index in enumerate(term.pairing):
        field_endpoint[quarks[line_id].index] = (line_rank[line_id], "q")
        field_endpoint[antiquarks[anti_index].index] = (line_rank[line_id], "bar")

    fields_by_operator: Dict[int, Dict[int, Field]] = defaultdict(dict)
    for field in fields:
        fields_by_operator[field.operator][field.slot] = field

    axes = tuple(sorted(
        field_endpoint[field.index]
        for field in fields
        if field.side == side
    ))
    factors = []
    signature = []
    for op_index, operator in enumerate(operators):
        if operator.side != side or not operator.gamma_slots:
            continue
        if len(operator.gamma_slots) != 2:
            return None
        matrix = matrix_registry.get(operator.gamma)
        if matrix is None:
            return None
        first, second = (fields_by_operator[op_index][slot] for slot in operator.gamma_slots)
        first_axis, second_axis = field_endpoint[first.index], field_endpoint[second.index]
        factors.append((first_axis, second_axis, matrix))
        signature.append((operator.gamma, first_axis, second_axis))
    return {"axes": axes, "factors": tuple(factors), "signature": tuple(signature)}


def _dirac_tensor_inner(left: Dict[str, object], right: Dict[str, object]) -> complex:
    if left["axes"] != right["axes"]:
        raise ValueError("Dirac payloads use different peram boundary axes")
    axes = left["axes"]
    edges = []
    for conjugate, payload in ((True, left), (False, right)):
        for first, second, matrix in payload["factors"]:
            values = tuple(
                tuple(value.conjugate() if conjugate else value for value in row)
                for row in matrix
            )
            edges.append((first, second, values))

    adjacency: Dict[SpinAxis, List[int]] = {axis: [] for axis in axes}
    for edge_id, (first, second, _matrix) in enumerate(edges):
        adjacency[first].append(edge_id)
        adjacency[second].append(edge_id)

    def other(edge_id: int, vertex: SpinAxis) -> SpinAxis:
        first, second, _matrix = edges[edge_id]
        return second if vertex == first else first

    def oriented_matrix(edge_id: int, current: SpinAxis, following: SpinAxis) -> Matrix:
        first, second, matrix = edges[edge_id]
        return matrix if current == first and following == second else _matrix_transpose(matrix)

    result = 1 + 0j
    visited_vertices = set()
    visited_edges = set()
    for seed in axes:
        if seed in visited_vertices:
            continue
        component = []
        stack = [seed]
        visited_vertices.add(seed)
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for edge_id in adjacency[vertex]:
                neighbour = other(edge_id, vertex)
                if neighbour not in visited_vertices:
                    visited_vertices.add(neighbour)
                    stack.append(neighbour)

        degrees = [len(adjacency[vertex]) for vertex in component]
        if max(degrees, default=0) > 2:
            raise ValueError("a Dirac index occurs in more than one Gamma factor per tensor")
        component_edges = {edge for vertex in component for edge in adjacency[vertex]}
        if not component_edges:
            result *= 4
            continue
        is_cycle = all(degree == 2 for degree in degrees)
        start = min(component, key=repr) if is_cycle else min(
            (vertex for vertex in component if len(adjacency[vertex]) == 1), key=repr
        )
        current = start
        product_matrix: Optional[Matrix] = None
        while True:
            candidates = [edge for edge in adjacency[current] if edge not in visited_edges]
            if not candidates:
                break
            edge_id = candidates[0]
            following = other(edge_id, current)
            factor = oriented_matrix(edge_id, current, following)
            product_matrix = factor if product_matrix is None else _matrix_product(product_matrix, factor)
            visited_edges.add(edge_id)
            current = following
            if is_cycle and current == start:
                break
        assert product_matrix is not None
        if is_cycle:
            result *= sum(product_matrix[i][i] for i in range(4))
        else:
            result *= sum(value for row in product_matrix for value in row)
    return result


def _dirac_payload_cache_key(payload: Mapping[str, object]) -> Tuple:
    """Return the symbolic identity used by repeated Dirac inner products."""
    return (tuple(payload.get("axes", ())), tuple(payload.get("signature", ())))


def _solve_linear(matrix: Sequence[Sequence[complex]], rhs: Sequence[complex], tolerance: float) -> List[complex]:
    size = len(rhs)
    augmented = [list(matrix[row]) + [rhs[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValueError("singular Gram basis during Dirac reduction")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            scale = augmented[row][column]
            if scale:
                augmented[row] = [
                    value - scale * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [augmented[row][-1] for row in range(size)]


def _dirac_gram_reduce(items: Sequence[Dict[str, object]], tolerance: float = 1.0e-10) -> Dict[str, object]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for item in items:
        groups[(item["profile"], item["fixed_elemental_key"])].append(item)

    reports = []
    total_rank = 0
    total_candidates = 0
    inner_cache: Dict[Tuple[Tuple, Tuple], complex] = {}

    def inner(left: Dict[str, object], right: Dict[str, object]) -> complex:
        left_key = _dirac_payload_cache_key(left)
        right_key = _dirac_payload_cache_key(right)
        key = (left_key, right_key)
        if key in inner_cache:
            return inner_cache[key]
        reverse = (right_key, left_key)
        if reverse in inner_cache:
            return inner_cache[reverse].conjugate()
        value = _dirac_tensor_inner(left, right)
        inner_cache[key] = value
        return value

    for (profile, fixed_key), terms in sorted(groups.items(), key=lambda item: repr(item[0])):
        count = len(terms)
        gram = [[0j for _ in terms] for _ in terms]
        for row in range(count):
            for column in range(row, count):
                sink_inner = inner(terms[row]["sink_dirac"], terms[column]["sink_dirac"])
                source_inner = inner(terms[row]["source_dirac"], terms[column]["source_dirac"])
                value = sink_inner * source_inner
                gram[row][column] = value
                gram[column][row] = value.conjugate()

        diagonal_scale = max((abs(gram[i][i].real) for i in range(count)), default=0.0)
        threshold = tolerance * max(diagonal_scale, 1.0)
        basis = []
        dependent = []
        weights = [complex(item["canonical_coefficient"]) for item in terms]
        aggregate = []
        for index in range(count):
            if not basis:
                residual = gram[index][index].real
                coefficients: List[complex] = []
            else:
                basis_gram = [[gram[left][right] for right in basis] for left in basis]
                basis_column = [gram[left][index] for left in basis]
                coefficients = _solve_linear(basis_gram, basis_column, threshold * 1.0e-4)
                residual = (gram[index][index] - sum(
                    coefficient.conjugate() * value
                    for coefficient, value in zip(coefficients, basis_column)
                )).real
            if residual > threshold:
                basis.append(index)
                aggregate.append(weights[index])
            else:
                dependent.append({
                    "wick_index": terms[index]["wick_index"],
                    "basis_wick_indices": [terms[basis[i]]["wick_index"] for i in range(len(basis))],
                    "coefficients": [[value.real, value.imag] for value in coefficients],
                    "residual_norm2": max(0.0, residual),
                })
                for basis_index, coefficient in enumerate(coefficients):
                    aggregate[basis_index] += weights[index] * coefficient

        total_rank += len(basis)
        total_candidates += count
        reports.append({
            "elemental_class": _key_digest(fixed_key),
            "peram_profile": {name: dict(zip(ROLE_ORDER, counts)) for name, counts in profile if any(counts)},
            "candidate_count": count,
            "exact_dirac_signature_count": len({
                (term["sink_dirac"]["signature"], term["source_dirac"]["signature"])
                for term in terms
            }),
            "dirac_rank": len(basis),
            "independent_wick_indices": [terms[index]["wick_index"] for index in basis],
            "dependent_relations": dependent,
            "aggregated_canonical_coefficients": [
                [value.real, value.imag] for value in aggregate
            ],
            "vvv_characters": [terms[index]["vvv_character"] for index in range(count)],
        })
    return {
        "supported": True,
        "candidate_count": total_candidates,
        "elemental_buckets": len(reports),
        "dirac_rank_sum": total_rank,
        "buckets": reports,
    }


class _StreamingDiracReduction:
    """Incremental bucketed Gram reduction.

    The older list-based reducer is useful for audit exports, but retaining a
    Dirac payload for every Wick term is unnecessary in production.  This
    accumulator stores only the current basis and the small relation record
    for each bucket.  The numerical rule is identical to
    ``_dirac_gram_reduce``; only the lifetime of non-basis candidates differs.
    """

    def __init__(self, tolerance: float = 1.0e-10, keep_relations: bool = False) -> None:
        self.tolerance = tolerance
        self.keep_relations = keep_relations
        self.buckets: Dict[Tuple[object, ...], Dict[str, object]] = {}

    def add(self, item: Dict[str, object]) -> None:
        key = (item["profile"], item["fixed_elemental_key"])
        state = self.buckets.setdefault(key, {
            "basis": [], "aggregate": [], "dependent": [], "dependent_count": 0,
            "candidate_count": 0, "signatures": set(), "vvv_characters": [],
            "vvv_histogram": Counter(),
        })
        state["candidate_count"] += 1
        if self.keep_relations:
            state["signatures"].add((item["sink_dirac"]["signature"], item["source_dirac"]["signature"]))
            state["vvv_characters"].append(item["vvv_character"])
        state["vvv_histogram"][str(item["vvv_character"])] += 1
        basis = state["basis"]
        weights = state["aggregate"]
        if basis:
            gram = [[0j for _ in basis] for _ in basis]
            rhs = []
            for left, left_item in enumerate(basis):
                rhs.append(
                    _dirac_tensor_inner(left_item["sink_dirac"], item["sink_dirac"])
                    * _dirac_tensor_inner(left_item["source_dirac"], item["source_dirac"])
                )
                for right in range(left, len(basis)):
                    value = (
                        _dirac_tensor_inner(left_item["sink_dirac"], basis[right]["sink_dirac"])
                        * _dirac_tensor_inner(left_item["source_dirac"], basis[right]["source_dirac"])
                    )
                    gram[left][right] = value
                    gram[right][left] = value.conjugate()
            diagonal_scale = max((abs(gram[index][index].real) for index in range(len(basis))), default=0.0)
            threshold = self.tolerance * max(diagonal_scale, 1.0)
            coefficients = _solve_linear(gram, rhs, threshold * 1.0e-4)
            candidate_norm = (
                _dirac_tensor_inner(item["sink_dirac"], item["sink_dirac"])
                * _dirac_tensor_inner(item["source_dirac"], item["source_dirac"])
            )
            residual = (candidate_norm - sum(
                coefficient.conjugate() * value
                for coefficient, value in zip(coefficients, rhs)
            )).real
        else:
            coefficients = []
            residual = (
                _dirac_tensor_inner(item["sink_dirac"], item["sink_dirac"])
                * _dirac_tensor_inner(item["source_dirac"], item["source_dirac"])
            ).real
            threshold = self.tolerance
        if residual > threshold:
            basis.append(item)
            weights.append(complex(item["canonical_coefficient"]))
            return
        state["dependent_count"] += 1
        if self.keep_relations:
            state["dependent"].append({
                "wick_index": item["wick_index"],
                "basis_wick_indices": [candidate["wick_index"] for candidate in basis],
                "coefficients": [[value.real, value.imag] for value in coefficients],
                "residual_norm2": max(0.0, residual),
            })
        for index, coefficient in enumerate(coefficients):
            weights[index] += complex(item["canonical_coefficient"]) * coefficient

    def finish(self) -> Dict[str, object]:
        reports = []
        total_rank = 0
        total_candidates = 0
        for (profile, fixed_key), state in sorted(self.buckets.items(), key=lambda item: repr(item[0])):
            basis = state["basis"]
            total_rank += len(basis)
            total_candidates += int(state["candidate_count"])
            reports.append({
                "elemental_class": _key_digest(fixed_key),
                "peram_profile": {name: dict(zip(ROLE_ORDER, counts)) for name, counts in profile if any(counts)},
                "candidate_count": int(state["candidate_count"]),
                "exact_dirac_signature_count": len(state["signatures"]) if self.keep_relations else None,
                "dirac_rank": len(basis),
                "independent_wick_indices": [item["wick_index"] for item in basis],
                "dependent_relations": state["dependent"] if self.keep_relations else [],
                "dependent_count": int(state["dependent_count"]),
                "aggregated_canonical_coefficients": [[value.real, value.imag] for value in state["aggregate"]],
                "vvv_characters": state["vvv_characters"] if self.keep_relations else [],
                "vvv_character_histogram": dict(state["vvv_histogram"]),
            })
        return {
            "supported": True,
            "candidate_count": total_candidates,
            "elemental_buckets": len(reports),
            "dirac_rank_sum": total_rank,
            "buckets": reports,
        }


def _key_digest(key: Tuple) -> str:
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:16]


def process(
    operators: Sequence[Operator],
    isospin_degenerate: bool = True,
    retain_graphs: bool = False,
    gamma_matrices: Optional[Mapping[str, Matrix]] = None,
    stream_dirac: bool = False,
    max_wicks: Optional[int] = None,
) -> Dict[str, object]:
    start = time.perf_counter()
    matrix_registry = _euclidean_gamma_matrices()
    matrix_registry.update(gamma_matrices or {})
    fields, quarks, antiquarks = _fields(operators)
    wick_count = 0
    wick_elapsed = 0.0
    graph_elapsed = 0.0
    profile_counts: Counter = Counter()
    n_bb_histogram: Counter = Counter()
    sin_sin_histogram: Counter = Counter()
    base_buckets: Counter = Counter()
    decorated_buckets: Counter = Counter()
    open_axis_buckets: Counter = Counter()
    sink_elemental_buckets: Counter = Counter()
    sink_decorated_buckets: Counter = Counter()
    source_after_sink_buckets: Counter = Counter()
    sign_histogram: Counter = Counter()
    vvv_character_histogram: Counter = Counter()
    canonical_sign_histogram: Counter = Counter()
    graph_records: List[Dict[str, object]] = []
    dirac_items: List[Dict[str, object]] = []
    streaming_dirac = _StreamingDiracReduction(keep_relations=False) if stream_dirac else None
    dirac_unsupported = set()
    dirac_payload_elapsed = 0.0

    wick_iterator = iter(enumerate_wicks(quarks, antiquarks))
    while True:
        wick_start = time.perf_counter()
        try:
            term = next(wick_iterator)
        except StopIteration:
            wick_elapsed += time.perf_counter() - wick_start
            break
        if max_wicks is not None and wick_count >= max_wicks:
            break
        wick_elapsed += time.perf_counter() - wick_start
        graph_start = time.perf_counter()
        wick_count += 1
        profile, type_counts, _open_routing = _profile(term, quarks, antiquarks, isospin_degenerate)
        graph = _build_tensor_network(operators, fields, quarks, antiquarks, term, isospin_degenerate)
        peram_template_lines = _peram_template_order(graph)
        sink_elemental_key, sink_elemental_order, sink_elemental_lines = _canonical_projection(
            graph, ("sink",), False, True
        )
        sink_decorated_key, sink_decorated_order, sink_decorated_lines = _canonical_projection(
            graph, ("sink",), True, True
        )
        base_key, full_elemental_order, full_elemental_lines = _canonical_projection(
            graph, ("sink", "source"), False, True
        )
        decorated_key, full_decorated_order, full_decorated_lines = _canonical_projection(
            graph, ("sink", "source"), True, True
        )
        fixed_elemental_key, fixed_elemental_order, fixed_elemental_lines = _canonical_projection(
            graph, ("sink", "source"), False, True, True
        )
        fixed_axis_key, fixed_open_order, fixed_open_lines = _canonical_projection(
            graph, ("sink", "source"), True, True, True
        )
        side_characters, vvv_details = _vvv_characters(graph.vvv_rows, fixed_elemental_lines)
        vvv_character = side_characters["sink"] * side_characters["source"]
        canonical_sign = term.sign * vvv_character
        bucket = profile
        profile_counts[bucket] += 1
        n_bb_histogram[sum(counts[0] for _name, counts in profile)] += 1
        sin_sin_histogram[dict(type_counts)["sin_sin"]] += 1
        base_buckets[(bucket, base_key)] += 1
        decorated_buckets[(bucket, decorated_key)] += 1
        open_axis_buckets[(bucket, fixed_axis_key)] += 1
        sink_elemental_buckets[(bucket, sink_elemental_key)] += 1
        sink_decorated_buckets[(bucket, sink_decorated_key)] += 1
        source_after_sink_buckets[(bucket, sink_decorated_key, decorated_key)] += 1
        sign_histogram["+" if term.sign > 0 else "-"] += 1
        vvv_character_histogram["+" if vvv_character > 0 else "-"] += 1
        canonical_sign_histogram["+" if canonical_sign > 0 else "-"] += 1
        payload_start = time.perf_counter()
        sink_payload = _dirac_payload(
            operators, fields, quarks, antiquarks, term, "sink", fixed_elemental_lines, matrix_registry
        )
        source_payload = _dirac_payload(
            operators, fields, quarks, antiquarks, term, "source", fixed_elemental_lines, matrix_registry
        )
        payload_duration = time.perf_counter() - payload_start
        dirac_payload_elapsed += payload_duration
        if sink_payload is None or source_payload is None:
            dirac_unsupported.update(
                operator.gamma for operator in operators if operator.gamma_slots
                and operator.gamma not in matrix_registry
            )
        else:
            dirac_item = {
                "profile": bucket,
                "fixed_elemental_key": fixed_elemental_key,
                "wick_index": wick_count,
                "canonical_coefficient": canonical_sign,
                "vvv_character": vvv_character,
                "sink_dirac": sink_payload,
                "source_dirac": source_payload,
            }
            if streaming_dirac is not None:
                streaming_dirac.add(dirac_item)
            else:
                dirac_items.append(dirac_item)
        if retain_graphs:
            graph_records.append({
                "wick_sign": term.sign,
                "pairing": list(term.pairing),
                "sink_rows": [
                    {"operator": row["operator"], "label": row["label"],
                     "species": row["species"], "tensor": row["tensor"],
                     "signature": list(row["signature"]),
                     "legs": [list(leg) for leg in row["legs"]]}
                    for row in graph.sink_rows
                ],
                "peram_lines": [
                    {key: value for key, value in line.items() if key != "node"}
                    for line in graph.peram_lines
                ],
                "vvv_character": {
                    "sink": side_characters["sink"],
                    "source": side_characters["source"],
                    "combined": vvv_character,
                    "per_operator": vvv_details,
                },
                "canonical_coefficient_sign": canonical_sign,
                "nodes": [dict(meta, color=graph.node_colors[index])
                          for index, meta in enumerate(graph.node_meta)],
                "edges": [
                    {"from": source, "to": target, "port": color}
                    for source, target, color in graph.edges
                ],
                "canonical_keys": {
                    "sink_elemental": _key_digest(sink_elemental_key),
                    "sink_gamma": _key_digest(sink_decorated_key),
                    "full_elemental": _key_digest(base_key),
                    "fixed_elemental": _key_digest(fixed_elemental_key),
                    "full_gamma": _key_digest(decorated_key),
                    "fixed_open_axes": _key_digest(fixed_axis_key),
                },
                "canonical_peram_line_orders": {
                    "sink_elemental": list(sink_elemental_lines),
                    "sink_gamma": list(sink_decorated_lines),
                    "full_elemental": list(full_elemental_lines),
                    "fixed_elemental": list(fixed_elemental_lines),
                    "peram_template": list(peram_template_lines),
                    "full_gamma": list(full_decorated_lines),
                    "fixed_open_axes": list(fixed_open_lines),
                },
            })
        graph_elapsed += time.perf_counter() - graph_start - payload_duration

    dirac_start = time.perf_counter()
    dirac_reduction = (
        (streaming_dirac.finish() if streaming_dirac is not None else _dirac_gram_reduce(dirac_items))
        if not dirac_unsupported else {
        "supported": False,
        "reason": "No numeric matrix is registered for every Gamma label",
        "unregistered_gamma_matrices": sorted(dirac_unsupported),
        "candidate_count": 0,
        "elemental_buckets": 0,
        "dirac_rank_sum": None,
        "buckets": [],
        }
    )
    dirac_elapsed = time.perf_counter() - dirac_start
    end = time.perf_counter()
    return {
        "operator_mode": validate_operator_modes(operators),
        "operators": [
            {"side": x.side, "kind": x.kind, "species": x.species, "label": x.label,
             "flavors": list(x.flavors), "gamma": x.gamma, "gamma_slots": list(x.gamma_slots),
             "open_slot": x.open_slot, "external_tag": x.external_tag, "mode": x.mode,
             "contraction_metadata": contraction_metadata_dict(x.contraction_metadata)}
            for x in operators
        ],
        "isospin_degenerate_perams": isospin_degenerate,
        "field_counts": {"q": len(quarks), "bar": len(antiquarks)},
        "flavor_counts": {
            "q": dict(Counter(x.flavor for x in quarks)),
            "bar": dict(Counter(x.flavor for x in antiquarks)),
        },
        "wick_count": wick_count,
        "wick_sign_histogram": dict(sign_histogram),
        "VVV_character_histogram": dict(vvv_character_histogram),
        "wick_times_VVV_character_histogram": dict(canonical_sign_histogram),
        "peram_profile_histogram": [
            {"profile": {name: dict(zip(ROLE_ORDER, counts)) for name, counts in profile if any(counts)},
             "wick_count": count}
            for profile, count in sorted(profile_counts.items())
        ],
        "N_bb_histogram": dict(sorted(n_bb_histogram.items())),
        "N_sin_sin_histogram": dict(sorted(sin_sin_histogram.items())),
        "sink_elemental_graph_classes": len(sink_elemental_buckets),
        "sink_gamma_decorated_graph_classes": len(sink_decorated_buckets),
        "elemental_graph_classes": len(base_buckets),
        "gamma_decorated_graph_classes": len(decorated_buckets),
        "fixed_open_axis_graph_classes": len(open_axis_buckets),
        "source_graph_classes_after_sink_buckets": len(source_after_sink_buckets),
        "dirac_reduction": dirac_reduction,
        "sink_elemental_class_size_histogram": dict(sorted(Counter(sink_elemental_buckets.values()).items())),
        "sink_gamma_class_size_histogram": dict(sorted(Counter(sink_decorated_buckets.values()).items())),
        "elemental_class_size_histogram": dict(sorted(Counter(base_buckets.values()).items())),
        "gamma_class_size_histogram": dict(sorted(Counter(decorated_buckets.values()).items())),
        "open_axis_class_size_histogram": dict(sorted(Counter(open_axis_buckets.values()).items())),
        "timing_seconds": {
            "wick_enumeration": wick_elapsed,
            "graph_normalization": graph_elapsed,
            "dirac_payload": dirac_payload_elapsed,
            "dirac_gram": dirac_elapsed,
            "total": end - start,
        },
        "full_graphs": graph_records if retain_graphs else None,
    }


def _from_json(path: Path) -> Tuple[str, Tuple[Operator, ...], Dict[str, Matrix]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    operators = tuple(
        op(
            item["side"], item["kind"], item["species"], item["label"], item["flavors"],
            item.get("gamma", ""), item.get("gamma_slots"), item.get("open_slot"),
            item.get("external_tag", ""),
            item.get("contraction_metadata", item.get("metadata")),
            mode=item.get("mode", "distillation"),
            momentum=item.get("momentum"), projector=item.get("projector"),
            clebsch=item.get("clebsch"), phase=item.get("phase"),
        )
        for item in document["operators"]
    )
    validate_operator_modes(operators)
    matrices = {
        name: _parse_matrix(value)
        for name, value in document.get("gamma_matrices", {}).items()
    }
    return document.get("name", path.stem), operators, matrices


def main() -> None:
    examples = _example_specs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=("all", *examples.keys()), default="all")
    parser.add_argument("--input", type=Path, help="JSON correlator specification")
    parser.add_argument(
        "--export-graphs", type=Path,
        help="write one compressed full factor-graph JSON and one readable summary per correlator",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    selected: List[Tuple[str, Tuple[Operator, ...]]]
    isospin_degenerate = True
    gamma_matrices: Dict[str, Matrix] = {}
    if args.input:
        name, operators, gamma_matrices = _from_json(args.input)
        input_data = json.loads(args.input.read_text(encoding="utf-8"))
        isospin_degenerate = input_data.get("isospin_degenerate_perams", True)
        selected = [(name, operators)]
    elif args.example == "all":
        selected = list(examples.items())
    else:
        selected = [(args.example, examples[args.example])]

    results = []
    for name, operators in selected:
        result = process(
            operators,
            isospin_degenerate,
            retain_graphs=args.export_graphs is not None,
            gamma_matrices=gamma_matrices,
        )
        result["name"] = name
        if args.export_graphs is not None:
            args.export_graphs.mkdir(parents=True, exist_ok=True)
            graph_records = result.pop("full_graphs")
            safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
            graph_path = args.export_graphs / f"{safe_name}_wick_factor_graphs.json.gz"
            result["graph_export"] = str(graph_path)
            with gzip.open(graph_path, "wt", encoding="utf-8", compresslevel=6) as stream:
                json.dump(
                    {"name": name, "summary": result, "wick_graphs": graph_records},
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
            summary_path = graph_path.with_suffix("")
            summary_path.write_text(
                json.dumps(
                    {"name": name, "summary": result, "full_graph_file": graph_path.name},
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            result["summary_export"] = str(summary_path)
        results.append(result)

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    for result in results:
        print(f"[{result['name']}]")
        print(f"Wick={result['wick_count']} signs={result['wick_sign_histogram']}")
        print(
            f"VVV character={result['VVV_character_histogram']} "
            f"Wick x VVV={result['wick_times_VVV_character_histogram']}"
        )
        print(f"N_bb={result['N_bb_histogram']} N_sin_sin={result['N_sin_sin_histogram']}")
        print(f"peram role profiles={len(result['peram_profile_histogram'])}")
        print(
            "sink candidates: VVV/VDV={sink_elemental_graph_classes}, "
            "with sink Gamma/open ports={sink_gamma_decorated_graph_classes}; "
            "full graph: VVV/VDV={elemental_graph_classes}, "
            "with both-side Gamma={gamma_decorated_graph_classes}, "
            "fixed open axes={fixed_open_axis_graph_classes}".format(**result)
        )
        print(
            "full graph class sizes: VVV/VDV={elemental_class_size_histogram}, "
            "Gamma={gamma_class_size_histogram}, fixed-open={open_axis_class_size_histogram}".format(**result)
        )
        dirac = result["dirac_reduction"]
        if dirac["supported"]:
            rank_distribution = dict(sorted(Counter(
                bucket["dirac_rank"] for bucket in dirac["buckets"]
            ).items()))
            print(
                f"Dirac Gram: {dirac['candidate_count']} candidates in {dirac['elemental_buckets']} "
                f"fixed elemental buckets; rank sum={dirac['dirac_rank_sum']}, "
                f"bucket rank distribution={rank_distribution}"
            )
        else:
            print(f"Dirac Gram unavailable: {dirac['unregistered_gamma_matrices']}")
        if result.get("graph_export"):
            print(f"full graphs: {result['graph_export']}")
        timing = result["timing_seconds"]
        print("time: wick={wick_enumeration:.6f}s graph={graph_normalization:.6f}s total={total:.6f}s".format(**timing))


if __name__ == "__main__":
    main()
