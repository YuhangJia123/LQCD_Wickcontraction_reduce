"""User-facing [sink operators, source operators] input adapter.

The convenience constructors mirror wick_diss_reduce.  Their exact sparse
tensor representation is also available through ``explicit_operator`` for
operators that are not products of meson and three-quark baryon vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import TYPE_CHECKING, Mapping, Sequence

from importlib import import_module

from .symmetry import identical_vertices, vertex_slot_permutation_phase
from .model import (
    Correlator as FastCorrelator,
    Field as FastField,
    LocalVertex,
    Operator as FastOperator,
    parity,
)


if TYPE_CHECKING:
    from . import exact_operator_backend as exact


Matrix = Sequence[Sequence[complex]]
_DEFAULT_OPEN = object()


def _exact_backend():
    """Load sparse reports only when a caller requests those modes."""
    return import_module(".exact_operator_backend", __package__)


def _matmul(left: Matrix, right: Matrix):
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(4))
                       for j in range(4)) for i in range(4))


def _dagger(matrix: Matrix):
    return tuple(tuple(complex(matrix[j][i]).conjugate() for j in range(4))
                 for i in range(4))


def standard_gamma_matrices() -> dict[str, tuple[tuple[complex, ...], ...]]:
    """Return the Euclidean gamma matrices used by the sibling input API."""
    zero = ((0j, 0j), (0j, 0j))
    identity = ((1 + 0j, 0j), (0j, 1 + 0j))
    paulis = (
        ((0j, 1 + 0j), (1 + 0j, 0j)),
        ((0j, -1j), (1j, 0j)),
        ((1 + 0j, 0j), (0j, -1 + 0j)),
    )

    def block(a, b, c, d):
        return tuple(tuple(
            a[i][j] if i < 2 and j < 2 else
            b[i][j - 2] if i < 2 else
            c[i - 2][j] if j < 2 else
            d[i - 2][j - 2]
            for j in range(4)) for i in range(4))

    gammas = []
    for sigma in paulis:
        upper = tuple(tuple(-1j * value for value in row) for row in sigma)
        lower = tuple(tuple(1j * value for value in row) for row in sigma)
        gammas.append(block(zero, upper, lower, zero))
    g1, g2, g3 = gammas
    g4 = block(identity, zero, zero,
               tuple(tuple(-value for value in row) for row in identity))
    g5 = _matmul(_matmul(_matmul(g1, g2), g3), g4)
    charge = _matmul(g4, g2)
    cg5 = _matmul(charge, g5)
    cg3 = _matmul(charge, g3)
    values = {
        "gamma1": g1, "gamma2": g2, "gamma3": g3,
        "gamma4": g4, "gamma5": g5,
        "Cgamma5": cg5, "Cgamma5_dagger": _dagger(cg5),
        "Cgamma3": cg3, "Cgamma3_dagger": _dagger(cg3),
    }
    return {name: tuple(tuple(complex(x) for x in row) for row in matrix)
            for name, matrix in values.items()}


@dataclass(frozen=True)
class HadronOperator:
    side: str
    kind: str
    species: str
    label: str
    flavors: tuple[str, ...]
    gamma: str | None
    gamma_slots: tuple[int, ...]
    open_slot: int | None
    external_spin: int | None
    external_tag: str
    point: str


def make_operator(
    side: str,
    kind: str,
    species: str,
    label: str,
    flavors: Sequence[str],
    gamma: str | None = "",
    gamma_slots: Sequence[int] | None = None,
    open_slot: object = _DEFAULT_OPEN,
    external_tag: str = "",
    *,
    external_spin: int | None = None,
    point: str | None = None,
) -> HadronOperator:
    """Create an ordered meson/baryon operator like wick_diss_reduce.

    Baryon sink fields are quarks and baryon source fields are antiquarks.
    Meson slots are always ``(antiquark, quark)``.  Matrix daggering and slot
    reordering are never implicit. ``open_slot`` is the baryon field slot not
    covered by Gamma; its quark spin remains a propagator coordinate.
    ``external_spin`` optionally fixes that spin coordinate. ``point``
    distinguishes propagator endpoints and defaults to ``side``.
    """
    if side not in {"sink", "source"}:
        raise ValueError("side must be 'sink' or 'source'")
    if kind not in {"meson", "baryon"}:
        raise ValueError("kind must be 'meson' or 'baryon'")
    ordered_flavors = tuple(str(flavor) for flavor in flavors)
    if kind == "baryon" and len(ordered_flavors) != 3:
        raise ValueError("baryon operators require three ordered quark flavors")
    if kind == "meson":
        if len(ordered_flavors) == 1:
            ordered_flavors *= 2
        if len(ordered_flavors) != 2:
            raise ValueError("meson flavors must be (antiquark_flavor, quark_flavor)")

    arity = 3 if kind == "baryon" else 2
    if open_slot is _DEFAULT_OPEN:
        selected_open = 0 if kind == "baryon" else None
    else:
        selected_open = open_slot
    if selected_open is not None and (
        kind != "baryon" or not isinstance(selected_open, int)
        or not 0 <= selected_open < 3
    ):
        raise ValueError("open_slot must be a baryon slot in {0, 1, 2}")

    matrix_name = str(gamma) if gamma else None
    if gamma_slots is None:
        if matrix_name:
            if kind == "meson":
                selected_slots = (0, 1)
            else:
                selected_slots = tuple(slot for slot in range(3)
                                       if slot != selected_open)
        else:
            selected_slots = ()
    else:
        selected_slots = tuple(gamma_slots)
    if any(not isinstance(slot, int) or not 0 <= slot < arity
           for slot in selected_slots):
        raise ValueError("gamma_slots contains an invalid field slot")
    if len(set(selected_slots)) != len(selected_slots):
        raise ValueError("gamma_slots must contain distinct field slots")
    if matrix_name and len(selected_slots) != 2:
        raise ValueError("a Gamma matrix connects exactly two slots")
    if not matrix_name and selected_slots:
        raise ValueError("gamma_slots require a Gamma matrix")
    if selected_open is not None and selected_open in selected_slots:
        raise ValueError("an open slot cannot also be a Gamma slot")
    if external_spin is not None and (
        kind != "baryon" or selected_open is None
        or not isinstance(external_spin, int) or not 0 <= external_spin < 4
    ):
        raise ValueError("external_spin must fix an open baryon slot in [0, 3]")
    return HadronOperator(
        side, kind, species, label, ordered_flavors, matrix_name,
        selected_slots, selected_open, external_spin, external_tag, point or side,
    )


def _matrix_for(operator: HadronOperator, registry: Mapping[str, Matrix]):
    if operator.gamma is None:
        return None
    try:
        matrix = tuple(tuple(complex(value) for value in row)
                       for row in registry[operator.gamma])
    except KeyError as error:
        raise ValueError(f"Gamma matrix {operator.gamma!r} is not registered") from error
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError(f"Gamma matrix {operator.gamma!r} must be 4 x 4")
    return matrix


def adapt_operator(operator):
    """Adapt a sibling wick_diss_reduce operator without importing that project."""
    if isinstance(operator, HadronOperator):
        return operator
    required = (
        "side", "kind", "species", "label", "flavors", "gamma",
        "gamma_slots", "open_slot", "external_tag",
    )
    if not all(hasattr(operator, attribute) for attribute in required):
        if isinstance(operator, _exact_backend().Operator):
            return operator
        raise TypeError(
            "operators must be HadronOperator, sparse Operator, or expose the "
            "wick_diss_reduce operator fields"
        )
    return make_operator(
        operator.side,
        operator.kind,
        operator.species,
        operator.label,
        operator.flavors,
        operator.gamma,
        operator.gamma_slots,
        operator.open_slot,
        operator.external_tag,
        external_spin=getattr(operator, "external_spin", None),
        point=getattr(operator, "point", None),
    )


def _adapt_operator_pair(operators):
    if len(operators) != 2:
        raise ValueError("correlator input must be [sink_operators, source_operators]")
    return tuple(tuple(adapt_operator(operator) for operator in side)
                 for side in operators)


def _build_exact(operator: HadronOperator, registry: Mapping[str, Matrix]):
    exact = _exact_backend()
    arity = len(operator.flavors)
    if operator.kind == "meson":
        fields = tuple(exact.Field(flavor, slot == 0, operator.point)
                       for slot, flavor in enumerate(operator.flavors))
    else:
        barred = operator.side == "source"
        fields = tuple(exact.Field(flavor, barred, operator.point)
                       for flavor in operator.flavors)
    matrix = _matrix_for(operator, registry)
    open_dimensions = ()
    terms = []
    if operator.kind == "baryon":
        color_assignments = tuple((values, parity(values))
                                  for values in permutations(range(3)))
    else:
        color_assignments = tuple(((color, color), 1) for color in range(3))

    if matrix is not None:
        gamma_entries = tuple(
            (row, column, complex(matrix[row][column]))
            for row in range(4) for column in range(4)
            if matrix[row][column] != 0
        )
    else:
        gamma_entries = ((None, None, 1 + 0j),)

    free_slots = tuple(
        slot for slot in range(arity)
        if slot not in operator.gamma_slots
        and not (slot == operator.open_slot and operator.external_spin is not None)
    )
    for colors, color_sign in color_assignments:
        for row, column, gamma_value in gamma_entries:
            gamma_spins = {}
            if row is not None:
                gamma_spins[operator.gamma_slots[0]] = row
                gamma_spins[operator.gamma_slots[1]] = column
            if operator.external_spin is not None:
                gamma_spins[operator.open_slot] = operator.external_spin
            for free_spins in product(range(4), repeat=len(free_slots)):
                spins = dict(gamma_spins)
                spins.update(zip(free_slots, free_spins))
                terms.append(exact.TensorTerm(
                    tuple((colors[slot], spins[slot]) for slot in range(arity)),
                    color_sign * gamma_value,
                ))

    return exact.explicit_operator(
        operator.label, fields, terms
    )


def _build_fast_vertex(operator: HadronOperator, registry: Mapping[str, Matrix]):
    matrix = _matrix_for(operator, registry)
    if matrix is None:
        spin_terms = tuple(
            (tuple(
                operator.external_spin if slot == operator.open_slot
                and operator.external_spin is not None else spin
                for slot, spin in enumerate(spins)
            ), 1 + 0j)
            for spins in product(*(
                (operator.external_spin,)
                if slot == operator.open_slot and operator.external_spin is not None
                else range(4)
                for slot in range(len(operator.flavors))
            ))
        )
    else:
        free_slots = tuple(
            slot for slot in range(len(operator.flavors))
            if slot not in operator.gamma_slots
            and not (slot == operator.open_slot and operator.external_spin is not None)
        )
        spin_terms = []
        for row in range(4):
            for column in range(4):
                value = matrix[row][column]
                if value == 0:
                    continue
                gamma_spins = {
                    operator.gamma_slots[0]: row,
                    operator.gamma_slots[1]: column,
                }
                if operator.external_spin is not None:
                    gamma_spins[operator.open_slot] = operator.external_spin
                for free_spins in product(range(4), repeat=len(free_slots)):
                    spins = dict(gamma_spins)
                    spins.update(zip(free_slots, free_spins))
                    spin_terms.append((tuple(spins[slot] for slot in range(len(operator.flavors))),
                                       complex(value)))
        spin_terms = tuple(spin_terms)

    if operator.kind == "meson":
        fields = tuple(FastField(flavor, slot == 0, operator.point)
                       for slot, flavor in enumerate(operator.flavors))
    else:
        barred = operator.side == "source"
        fields = tuple(FastField(flavor, barred, operator.point)
                       for flavor in operator.flavors)
    return LocalVertex(
        operator.label, fields,
        "epsilon" if operator.kind == "baryon" else "delta",
        spin_terms,
    )


def normalize_correlator(
    operators: Sequence[Sequence[HadronOperator | exact.Operator]],
    *,
    gamma_matrices: Mapping[str, Matrix] | None = None,
):
    """Normalize ``[sink_operators, source_operators]`` to exact backend form."""
    operators = _adapt_operator_pair(operators)
    exact = _exact_backend()
    registry = dict(standard_gamma_matrices())
    registry.update(gamma_matrices or {})
    sink_specs = operators[0]
    source_specs = operators[1]
    if not sink_specs or not source_specs:
        raise ValueError("sink and source operator lists must be nonempty")
    if any(isinstance(operator, HadronOperator) and operator.side != "sink"
           for operator in sink_specs):
        raise ValueError("the first list must contain only sink operators")
    if any(isinstance(operator, HadronOperator) and operator.side != "source"
           for operator in source_specs):
        raise ValueError("the second list must contain only source operators")
    def exact_operator(operator):
        if isinstance(operator, exact.Operator):
            operator.validate()
            return operator
        if isinstance(operator, HadronOperator):
            return _build_exact(operator, registry)
        raise TypeError("unsupported operator type after input adaptation")

    sink = exact.product_operator("sink", tuple(map(exact_operator, sink_specs)))
    source = exact.product_operator("source", tuple(map(exact_operator, source_specs)))
    return exact.Correlator(sink, source), registry


def normalize_fast_correlator(
    operators: Sequence[Sequence[HadronOperator]],
    *,
    gamma_matrices: Mapping[str, Matrix] | None = None,
) -> FastCorrelator:
    """Convert a meson/baryon pair to the epsilon/delta S3 coefficient path.

    The fast result is a source coefficient tensor over ordered sink legs.
    Multiple source points are retained in endpoint-resolved output keys.
    """
    operators = _adapt_operator_pair(operators)
    registry = dict(standard_gamma_matrices())
    registry.update(gamma_matrices or {})
    sink_specs = tuple(operators[0])
    source_specs = tuple(operators[1])
    if not sink_specs or not source_specs:
        raise ValueError("sink and source operator lists must be nonempty")
    if any(not isinstance(operator, HadronOperator)
           for operator in sink_specs + source_specs):
        raise TypeError("the S3 fast path accepts meson/baryon HadronOperator inputs")
    if any(operator.side != "sink" for operator in sink_specs):
        raise ValueError("the first list must contain only sink operators")
    if any(operator.side != "source" for operator in source_specs):
        raise ValueError("the second list must contain only source operators")
    sink_fields = tuple(
        FastField(flavor, operator.kind == "meson" and slot == 0, operator.point)
        for operator in sink_specs
        for slot, flavor in enumerate(operator.flavors)
    )
    sink_routes = tuple(
        (operator_index, slot)
        for operator_index, operator in enumerate(sink_specs)
        for slot, _flavor in enumerate(operator.flavors)
    )
    sink_axis_transports = []
    axis_offset = 0
    sink_tensors = tuple(_build_fast_vertex(operator, registry)
                         for operator in sink_specs)
    for operator, tensor in zip(sink_specs, sink_tensors):
        identity = tuple(range(len(operator.flavors)))
        for local_map in permutations(range(len(operator.flavors))):
            if local_map == identity:
                continue
            phase = vertex_slot_permutation_phase(tensor, tuple(local_map))
            if phase is None:
                continue
            axis_map = list(range(len(sink_fields)))
            for old_slot, new_slot in enumerate(local_map):
                axis_map[axis_offset + old_slot] = axis_offset + new_slot
            sink_axis_transports.append((tuple(axis_map), phase))
        axis_offset += len(operator.flavors)
    axis_offsets = []
    axis_offset = 0
    for operator in sink_specs:
        axis_offsets.append(axis_offset)
        axis_offset += len(operator.flavors)
    for left_index in range(len(sink_specs)):
        for right_index in range(left_index + 1, len(sink_specs)):
            left = sink_tensors[left_index]
            right = sink_tensors[right_index]
            if not identical_vertices(left, right):
                continue
            left_offset = axis_offsets[left_index]
            right_offset = axis_offsets[right_index]
            axis_map = list(range(len(sink_fields)))
            for slot in range(len(left.fields)):
                axis_map[left_offset + slot] = right_offset + slot
                axis_map[right_offset + slot] = left_offset + slot
            sink_axis_transports.append((tuple(axis_map), 1 + 0j))
    source_vertices = tuple(_build_fast_vertex(operator, registry)
                            for operator in source_specs)
    sink = FastOperator("sink", sink_fields)
    source = FastOperator.from_vertices("source", source_vertices)
    correlator = FastCorrelator(
        sink, source, sink_routes, tuple(sink_axis_transports),
    )
    correlator.validate()
    return correlator


def normalize_general_fast_correlator(
    operators: Sequence[Sequence[HadronOperator]],
    *,
    gamma_matrices: Mapping[str, Matrix] | None = None,
) -> FastCorrelator:
    """Convert factorized operators on both sides to the full graph model.

    Unlike :func:`normalize_fast_correlator`, sink vertices remain present so
    same-side and disconnected Wick edges can be enumerated by the generalized
    fused reducer.
    """
    operators = _adapt_operator_pair(operators)
    registry = dict(standard_gamma_matrices())
    registry.update(gamma_matrices or {})
    if any(not isinstance(operator, HadronOperator)
           for side in operators for operator in side):
        raise TypeError("the generalized factorized path accepts HadronOperator inputs")
    if any(operator.side != side_name
           for side_name, side in zip(("sink", "source"), operators)
           for operator in side):
        raise ValueError("operator sides do not match the [sink, source] lists")
    sink_vertices = tuple(
        _build_fast_vertex(operator, registry) for operator in operators[0]
    )
    source_vertices = tuple(
        _build_fast_vertex(operator, registry) for operator in operators[1]
    )
    correlator = FastCorrelator(
        FastOperator.from_vertices("sink", sink_vertices),
        FastOperator.from_vertices("source", source_vertices),
    )
    correlator.validate_general()
    return correlator


__all__ = [
    "HadronOperator", "adapt_operator", "make_operator", "normalize_correlator",
    "normalize_fast_correlator", "normalize_general_fast_correlator",
    "standard_gamma_matrices",
]
