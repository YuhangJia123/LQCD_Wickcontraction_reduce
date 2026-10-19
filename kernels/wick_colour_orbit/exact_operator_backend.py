"""General exact fallback for full ordinary-colour Wick contractions.

Operators are supplied as ordered fields plus sparse local colour-spin tensor
entries. This path supports same-operator pairings and tensors that do not
factor into the epsilon/delta vertices accepted by the fused fast path. It
returns coefficients over ordinary propagator matrix-element monomials.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from itertools import permutations, permutations as colour_permutations
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

from .grouping import classify_output_pairing, classify_pairing


Complex = complex
Matrix = Sequence[Sequence[Complex]]
Coordinate = Tuple[int, int]  # (colour, Dirac)
Monomial = Tuple[Tuple[object, ...], ...]
TensorKey = Tuple[Tuple[int, ...], Monomial]


def parity(values: Sequence[int]) -> int:
    inversions = sum(
        values[i] > values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    return -1 if inversions % 2 else 1


@dataclass(frozen=True)
class Field:
    flavor: str
    barred: bool
    point: str


@dataclass(frozen=True)
class TensorTerm:
    """One sparse entry of an operator tensor, in field-slot order."""

    coordinates: Tuple[Coordinate, ...]
    coefficient: Complex
    open_indices: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Operator:
    name: str
    fields: Tuple[Field, ...]
    terms: Tuple[TensorTerm, ...]
    open_dimensions: Tuple[int, ...] = ()
    factors: Tuple["Operator", ...] = ()

    def validate(self) -> None:
        if not self.fields or (not self.terms and not self.factors):
            raise ValueError(f"operator {self.name!r} needs fields and tensor terms")
        if any(dimension < 1 for dimension in self.open_dimensions):
            raise ValueError("open-index dimensions must be positive")
        if self.factors:
            for factor in self.factors:
                factor.validate()
            expected_fields = tuple(field for factor in self.factors for field in factor.fields)
            expected_dimensions = tuple(
                dimension for factor in self.factors for dimension in factor.open_dimensions
            )
            if self.fields != expected_fields or self.open_dimensions != expected_dimensions:
                raise ValueError(f"operator {self.name!r} has inconsistent product factors")
            if self.terms:
                raise ValueError("a factored operator cannot also store expanded terms")
            return
        for term in self.terms:
            if len(term.coordinates) != len(self.fields):
                raise ValueError(f"operator {self.name!r} has a malformed tensor term")
            if len(term.open_indices) != len(self.open_dimensions):
                raise ValueError(f"operator {self.name!r} has malformed open indices")
            if any(not 0 <= index < dimension
                   for index, dimension in zip(term.open_indices, self.open_dimensions)):
                raise ValueError(f"operator {self.name!r} has an out-of-range open index")
            for colour, spin in term.coordinates:
                if not 0 <= colour < 3 or not 0 <= spin < 4:
                    raise ValueError("ordinary colour-spin coordinates must be in 3 x 4")


@dataclass(frozen=True)
class Correlator:
    """An explicitly ordered [sink, source] operator pair."""

    sink: Operator
    source: Operator

    def validate(self) -> None:
        self.sink.validate()
        self.source.validate()


@dataclass(frozen=True)
class WickPairing:
    # Each pair is (quark field index, antiquark field index) in the flattened
    # sink-then-source field order.
    pairs: Tuple[Tuple[int, int], ...]
    sign: int


@dataclass(frozen=True)
class TypedProfile:
    """Literal edge typing plus the ordered open-output route."""

    edges: Tuple[Tuple[object, ...], ...]
    output_route: Tuple[Tuple[int, str, int, int, int], ...]
    connectivity_class: Tuple[int, int]
    topology_signature: Tuple[int, int]


@dataclass(frozen=True)
class ProfileTransporter:
    representative_wick_index: int
    field_map: Tuple[int, ...]
    open_axis_map: Tuple[int, ...]
    tensor_phase: Complex
    wick_sign_ratio: Complex
    phase: Complex


@dataclass(frozen=True)
class IndependentWick:
    index: int
    pairing: WickPairing
    propagator_profile: Tuple[Tuple[Tuple[str, str, str], int], ...]
    vector: Dict[TensorKey, Complex]
    typed_profile: TypedProfile


@dataclass(frozen=True)
class WickBasisRepresentation:
    wick_index: int
    pairing: WickPairing
    propagator_profile: Tuple[Tuple[Tuple[str, str, str], int], ...]
    basis_coefficients: Tuple[Tuple[int, Complex], ...]
    typed_profile: Optional[TypedProfile] = None
    transporter: Optional[ProfileTransporter] = None
    raw_vector: Optional[Dict[TensorKey, Complex]] = None


def explicit_operator(
    name: str,
    fields: Sequence[Field],
    terms: Iterable[TensorTerm],
    *,
    open_dimensions: Sequence[int] = (),
) -> Operator:
    """Build a general operator without imposing a Gamma or particle type."""
    operator = Operator(name, tuple(fields), tuple(terms), tuple(open_dimensions))
    operator.validate()
    return operator


def meson_operator(
    name: str,
    *,
    antiquark_flavor: str,
    quark_flavor: str,
    gamma: Matrix,
    point: str,
) -> Operator:
    """Build ``bar(qbar) Gamma q`` with a colour delta tensor.

    The two field slots are kept exactly in the stated order.  The caller
    supplies sink and source operators independently; no conjugate-flavour or
    slot-order rewriting is performed.
    """
    if len(gamma) != 4 or any(len(row) != 4 for row in gamma):
        raise ValueError("gamma must be a 4 x 4 matrix")
    fields = (
        Field(antiquark_flavor, True, point),
        Field(quark_flavor, False, point),
    )
    terms = []
    for colour in range(3):
        for anti_spin in range(4):
            for quark_spin in range(4):
                value = complex(gamma[anti_spin][quark_spin])
                if value:
                    terms.append(TensorTerm(
                        ((colour, anti_spin), (colour, quark_spin)), value
                    ))
    return explicit_operator(name, fields, terms)


def epsilon_three_quark_operator(
    name: str,
    *,
    flavors: Sequence[str],
    point: str,
    diquark_matrix: Matrix,
    third_slot_matrix: Matrix,
    external_spin: int | None,
    barred: bool = False,
) -> Operator:
    """Build a three-slot epsilon tensor with caller-defined Dirac matrices.

    ``barred=True`` changes only the field orientation.  It deliberately does
    not conjugate matrices or reverse slot/flavour order; the caller supplies
    the desired source tensor convention explicitly.  Use ``external_spin=None``
    to retain the third Dirac index as an open output axis.
    """
    if len(flavors) != 3:
        raise ValueError("a three-quark epsilon operator needs three ordered flavors")
    if external_spin is not None and not 0 <= external_spin < 4:
        raise ValueError("external_spin must be in [0, 3]")
    if any(len(matrix) != 4 or any(len(row) != 4 for row in matrix)
           for matrix in (diquark_matrix, third_slot_matrix)):
        raise ValueError("Dirac matrices must be 4 x 4")
    fields = tuple(Field(flavor, barred, point) for flavor in flavors)
    terms = []
    for colours in colour_permutations(range(3)):
        colour_sign = parity(colours)
        for first_spin in range(4):
            for second_spin in range(4):
                first_value = complex(diquark_matrix[first_spin][second_spin])
                if not first_value:
                    continue
                for third_spin in range(4):
                    external_values = (external_spin,) if external_spin is not None else range(4)
                    for selected_external_spin in external_values:
                        third_value = complex(
                            third_slot_matrix[third_spin][selected_external_spin]
                        )
                        if third_value:
                            terms.append(TensorTerm(
                                tuple(zip(colours, (first_spin, second_spin, third_spin))),
                                colour_sign * first_value * third_value,
                                () if external_spin is not None else (selected_external_spin,),
                            ))
    return explicit_operator(
        name, fields, terms,
        open_dimensions=() if external_spin is not None else (4,),
    )


def product_operator(name: str, factors: Sequence[Operator]) -> Operator:
    """Compose factors without materialising their Cartesian-product tensor."""
    if not factors:
        raise ValueError("at least one operator factor is required")
    for factor in factors:
        factor.validate()
    flat_factors = tuple(
        nested
        for factor in factors
        for nested in (factor.factors or (factor,))
    )
    fields = tuple(field for factor in flat_factors for field in factor.fields)
    open_dimensions = tuple(
        dimension for factor in flat_factors for dimension in factor.open_dimensions
    )
    operator = Operator(name, fields, (), open_dimensions, flat_factors)
    operator.validate()
    return operator


def _flatten(correlator: Correlator):
    correlator.validate()
    fields = []
    operator_index = 0

    def append_factors(side: str, operator: Operator) -> None:
        nonlocal operator_index
        if operator.factors:
            for factor in operator.factors:
                append_factors(side, factor)
            return
        current_index = operator_index
        operator_index += 1
        fields.extend((side, current_index, field) for field in operator.fields)

    append_factors("sink", correlator.sink)
    append_factors("source", correlator.source)
    return tuple(fields)


def _atom_layout(correlator: Correlator):
    """Return atomic factors with their global field and output-axis offsets."""
    result = []
    field_offset = 0
    axis_offset = 0
    owner = 0

    def visit(side: str, operator: Operator) -> None:
        nonlocal field_offset, axis_offset, owner
        if operator.factors:
            for factor in operator.factors:
                visit(side, factor)
            return
        result.append((side, owner, operator, field_offset, axis_offset))
        field_offset += len(operator.fields)
        axis_offset += len(operator.open_dimensions)
        owner += 1

    visit("sink", correlator.sink)
    visit("source", correlator.source)
    return tuple(result)


def _tensor_entries(operator: Operator):
    entries = defaultdict(complex)
    for term in operator.terms:
        entries[(term.coordinates, term.open_indices)] += complex(term.coefficient)
    return {key: value for key, value in entries.items() if value != 0}


def _slot_permutation_phase(operator: Operator, slot_map: Tuple[int, ...], tolerance=1e-12):
    """Return the phase mapping representative terms to their routed slots."""
    original = _tensor_entries(operator)
    if len(slot_map) != len(operator.fields):
        return None
    if any(operator.fields[index] != operator.fields[slot_map[index]]
           for index in range(len(slot_map))):
        return None
    transported = defaultdict(complex)
    for (coordinates, open_indices), coefficient in original.items():
        mapped = [None] * len(coordinates)
        for old_slot, new_slot in enumerate(slot_map):
            mapped[new_slot] = coordinates[old_slot]
        transported[(tuple(mapped), open_indices)] += coefficient
    keys = set(original) | set(transported)
    if not keys:
        return None
    pivot = next((key for key in keys
                  if abs(original.get(key, 0j)) > tolerance
                  and abs(transported.get(key, 0j)) > tolerance), None)
    if pivot is None:
        return None
    phase = original[pivot] / transported[pivot]
    if abs(abs(phase) - 1.0) > tolerance:
        return None
    phase /= abs(phase)
    scale = max((abs(value) for value in original.values()), default=1.0)
    if any(abs(original.get(key, 0j) - phase * transported.get(key, 0j))
           > tolerance * scale for key in keys):
        return None
    return phase


def _same_atom(left: Operator, right: Operator) -> bool:
    return (
        left.fields == right.fields
        and left.open_dimensions == right.open_dimensions
        and _tensor_entries(left) == _tensor_entries(right)
    )


def _transporter_generators(correlator: Correlator):
    """Build only slot/factor permutations verified against complete tensors."""
    layouts = _atom_layout(correlator)
    fields = _flatten(correlator)
    field_count = len(fields)
    axis_count = sum(len(operator.open_dimensions)
                     for _side, _owner, operator, _field, _axis in layouts)
    identity_fields = tuple(range(field_count))
    identity_axes = tuple(range(axis_count))
    generators = []

    def add(field_map, axis_map, phase):
        value = (tuple(field_map), tuple(axis_map), complex(phase))
        if value[:2] == (identity_fields, identity_axes):
            return
        if not any(existing[:2] == value[:2]
                   and abs(existing[2] - value[2]) <= 1e-12
                   for existing in generators):
            generators.append(value)

    for _side, _owner, operator, field_offset, _axis_offset in layouts:
        rank = len(operator.fields)
        if rank > 3:
            continue
        for local_map in permutations(range(rank)):
            if local_map == tuple(range(rank)):
                continue
            # Transporters require tensor self-equivalence up to floating-point
            # roundoff. Approximate covariance is not safe because small terms
            # can combine later.
            phase = _slot_permutation_phase(
                operator, tuple(local_map), tolerance=1.0e-15,
            )
            if phase is None:
                continue
            field_map = list(identity_fields)
            for old_slot, new_slot in enumerate(local_map):
                field_map[field_offset + old_slot] = field_offset + new_slot
            add(field_map, identity_axes, phase)

    # Swapping identical factors also permutes their open output routes.
    for side, root in (("sink", correlator.sink), ("source", correlator.source)):
        factors = root.factors
        if not factors or any(factor.factors for factor in factors):
            continue
        side_layouts = [layout for layout in layouts if layout[0] == side]
        for left_index in range(len(factors)):
            for right_index in range(left_index + 1, len(factors)):
                left = factors[left_index]
                right = factors[right_index]
                if not _same_atom(left, right):
                    continue
                left_layout = side_layouts[left_index]
                right_layout = side_layouts[right_index]
                left_field, right_field = left_layout[3], right_layout[3]
                left_axis, right_axis = left_layout[4], right_layout[4]
                field_map = list(identity_fields)
                for slot in range(len(left.fields)):
                    field_map[left_field + slot] = right_field + slot
                    field_map[right_field + slot] = left_field + slot
                axis_map = list(identity_axes)
                for axis in range(len(left.open_dimensions)):
                    axis_map[left_axis + axis] = right_axis + axis
                    axis_map[right_axis + axis] = left_axis + axis
                add(field_map, axis_map, 1 + 0j)

    return tuple(generators), layouts


def _slot_roles(layouts, fields):
    roles = {}
    axis_route = {}
    for side, owner, operator, field_offset, axis_offset in layouts:
        entries = _tensor_entries(operator)
        for slot in range(len(operator.fields)):
            support = tuple(sorted(
                (coordinates[slot], open_indices, value.real, value.imag)
                for (coordinates, open_indices), value in entries.items()
                if value
            ))
            role = sha256(repr(support).encode("utf-8")).hexdigest()[:16]
            roles[field_offset + slot] = role
        for local_axis, dimension in enumerate(operator.open_dimensions):
            axis_route[axis_offset + local_axis] = (
                side, owner, local_axis, dimension,
            )
    slot_numbers = defaultdict(int)
    descriptors = {}
    for index, (side, owner, field) in enumerate(fields):
        slot = slot_numbers[owner]
        slot_numbers[owner] += 1
        descriptors[index] = (
            side, owner, slot, field, roles.get(index, ""),
        )
    return descriptors, axis_route


def _typed_profile(fields, pairing, layouts, descriptors, axis_route):
    edges = []
    for quark_index, anti_index in pairing.pairs:
        q_index, a_index = ((quark_index, anti_index)
                            if not fields[quark_index][2].barred
                            else (anti_index, quark_index))
        q_side, q_owner, q_slot, quark, q_role = descriptors[q_index]
        a_side, a_owner, a_slot, antiquark, a_role = descriptors[a_index]
        edges.append((
            quark.flavor,
            (q_index, q_side, q_owner, q_slot, quark.point, q_role),
            (a_index, a_side, a_owner, a_slot, antiquark.point, a_role),
        ))
    route = tuple(
        (axis, *axis_route[axis]) for axis in sorted(axis_route)
    )
    connectivity_class, topology_signature = classify_pairing(
        pairing.pairs,
        tuple((side, owner) for side, owner, _field in fields),
    )
    return TypedProfile(
        tuple(sorted(edges)), route, connectivity_class, topology_signature,
    )


def _pairing_from_pairs(fields, pairs):
    ordered_pairs = tuple(sorted(pairs))
    quarks: Dict[str, list[int]] = defaultdict(list)
    antiquarks: Dict[str, list[int]] = defaultdict(list)
    for index, (_side, _owner, field) in enumerate(fields):
        (antiquarks if field.barred else quarks)[field.flavor].append(index)
    if set(quarks) != set(antiquarks) or any(
        len(quarks[flavor]) != len(antiquarks[flavor]) for flavor in quarks
    ):
        raise ValueError("transporter changed the literal flavor profile")
    if len(ordered_pairs) != sum(map(len, quarks.values())):
        raise ValueError("transporter produced an incomplete Wick pairing")
    pairing_by_quark = {}
    for left, right in ordered_pairs:
        if fields[left][2].barred:
            left, right = right, left
        if fields[left][2].barred or not fields[right][2].barred:
            raise ValueError("transporter did not preserve field orientation")
        if fields[left][2].flavor != fields[right][2].flavor:
            raise ValueError("transporter did not preserve literal flavor")
        pairing_by_quark[left] = right
    if set(pairing_by_quark) != {index for group in quarks.values() for index in group}:
        raise ValueError("transporter changed the quark incidence")
    flavors = sorted(quarks)
    ordered_quarks = tuple(index for flavor in flavors for index in quarks[flavor])
    ordered_antiquarks = tuple(index for flavor in flavors for index in antiquarks[flavor])
    anti_position = {field_index: i for i, field_index in enumerate(ordered_antiquarks)}
    permutation = tuple(anti_position[pairing_by_quark[index]]
                        for index in ordered_quarks)
    order_sign = parity(ordered_quarks + ordered_antiquarks)
    return WickPairing(ordered_pairs, order_sign * parity(permutation))


def _compose_transporter(first, second):
    first_fields, first_axes, first_phase = first
    second_fields, second_axes, second_phase = second
    return (
        tuple(second_fields[index] for index in first_fields),
        tuple(second_axes[index] for index in first_axes),
        first_phase * second_phase,
    )


def _map_typed_profile(profile, transform, fields, descriptors, axis_route):
    field_map, axis_map, _phase = transform
    mapped_edges = []
    mapped_pairs = []
    for flavor, q_edge, a_edge in profile.edges:
        q_index = field_map[q_edge[0]]
        a_index = field_map[a_edge[0]]
        mapped_pairs.append(tuple(sorted((q_index, a_index))))
        q_side, q_owner, q_slot, quark, q_role = descriptors[q_index]
        a_side, a_owner, a_slot, antiquark, a_role = descriptors[a_index]
        mapped_edges.append((
            flavor,
            (q_index, q_side, q_owner, q_slot, quark.point, q_role),
            (a_index, a_side, a_owner, a_slot, antiquark.point, a_role),
        ))
    mapped_route = tuple(sorted(
        (axis_map[axis], *axis_route[axis_map[axis]])
        for axis, *_route in profile.output_route
    ))
    connectivity_class, topology_signature = classify_pairing(
        mapped_pairs,
        tuple((side, owner) for side, owner, _field in fields),
    )
    return TypedProfile(
        tuple(sorted(mapped_edges)), mapped_route,
        connectivity_class, topology_signature,
    )


def _pairing_orbit(pairing, wick_index, fields, generators, profile,
                   descriptors, axis_route):
    identity = (tuple(range(len(fields))),
                tuple(range(len(axis_route))), 1 + 0j)
    by_pairs = {pairing.pairs: (pairing, identity)}
    queue = [pairing.pairs]
    consistent = True
    while queue:
        current_pairs = queue.pop()
        current_pairing, current_transform = by_pairs[current_pairs]
        for generator in generators:
            transform = _compose_transporter(current_transform, generator)
            mapped_pairs = tuple(sorted(
                (generator[0][left], generator[0][right])
                for left, right in current_pairs
            ))
            try:
                mapped_pairing = _pairing_from_pairs(fields, mapped_pairs)
            except ValueError:
                consistent = False
                continue
            current_profile = _typed_profile(
                fields, current_pairing, (), descriptors, axis_route
            )
            mapped_profile = _typed_profile(
                fields, mapped_pairing, (), descriptors, axis_route
            )
            if _map_typed_profile(
                current_profile, generator, fields, descriptors, axis_route
            ) != mapped_profile:
                consistent = False
                continue
            if profile is not None and (
                mapped_profile.connectivity_class != profile.connectivity_class
                or mapped_profile.topology_signature != profile.topology_signature
            ):
                consistent = False
                continue
            previous = by_pairs.get(mapped_pairs)
            if previous is None:
                by_pairs[mapped_pairs] = (mapped_pairing, transform)
                queue.append(mapped_pairs)
            elif (previous[1][1] != transform[1]
                  or abs(previous[1][2] - transform[2]) > 1e-12):
                consistent = False
    ordered = []
    for pairs, (mapped_pairing, transform) in by_pairs.items():
        sign_ratio = complex(mapped_pairing.sign / pairing.sign)
        transporter = ProfileTransporter(
            wick_index, transform[0], transform[1], transform[2],
            sign_ratio, transform[2] * sign_ratio,
        )
        ordered.append((pairs, mapped_pairing, transporter))
    return ordered, consistent


def _transport_vector(vector, transporter, *, tolerance=1e-12):
    result = defaultdict(complex)
    axis_map = transporter.open_axis_map
    for (open_indices, monomial), coefficient in vector.items():
        if axis_map:
            mapped_indices = [0] * len(open_indices)
            for old_axis, new_axis in enumerate(axis_map):
                mapped_indices[new_axis] = open_indices[old_axis]
            open_indices = tuple(mapped_indices)
        result[(open_indices, monomial)] += transporter.phase * coefficient
    return {key: value for key, value in result.items() if abs(value) > tolerance}


def _transport_exact_key(key: TensorKey, transporter: ProfileTransporter) -> TensorKey:
    open_indices, monomial = key
    axis_map = transporter.open_axis_map
    if not axis_map:
        return key
    if len(axis_map) != len(open_indices):
        raise ValueError("transporter output-axis map does not match the exact key")
    mapped_indices = [0] * len(open_indices)
    for old_axis, new_axis in enumerate(axis_map):
        mapped_indices[new_axis] = open_indices[old_axis]
    return tuple(mapped_indices), monomial


def _pairing_vector(correlator, fields, pairing, *, tolerance,
                    isospin_degenerate_perams):
    vector: Dict[TensorKey, Complex] = defaultdict(complex)
    for key, value in _iter_pairing_terms(
        correlator, fields, pairing,
        isospin_degenerate_perams=isospin_degenerate_perams,
    ):
        vector[key] += value
    scale = max((abs(value) for value in vector.values()), default=0.0)
    threshold = tolerance * scale
    return {key: value for key, value in vector.items() if abs(value) > threshold}


def _reduce_one_vector(wick_index, pairing, profile, typed_profile, original,
                       echelon_by_profile, basis_count, tolerance,
                       echelon_partition=(0, 0)):
    scale = max((abs(value) for value in original.values()), default=0.0)
    if not scale:
        return None, WickBasisRepresentation(
            wick_index, pairing, profile, (), typed_profile,
        ), basis_count

    threshold = tolerance * scale
    work = dict(original)
    row_coefficients = []
    echelon = echelon_by_profile[(echelon_partition, profile)]
    for pivot, row, row_representation in echelon:
        factor = work.get(pivot, 0j)
        if abs(factor) <= threshold:
            work.pop(pivot, None)
            continue
        _sparse_add_scaled(work, row, -factor, threshold)
        row_coefficients.append((factor, row_representation))

    if not work:
        basis_coefficients: Dict[int, Complex] = defaultdict(complex)
        for factor, row_representation in row_coefficients:
            for basis_index, coefficient in row_representation.items():
                basis_coefficients[basis_index] += factor * coefficient
        relation = WickBasisRepresentation(
            wick_index, pairing, profile,
            tuple(sorted((index, value)
                         for index, value in basis_coefficients.items()
                         if value != 0)),
            typed_profile,
        )
        return None, relation, basis_count

    pivot = min(work, key=lambda key: (-abs(work[key]), key))
    pivot_value = work[pivot]
    basis_index = basis_count
    basis_count += 1
    normalized = {key: value / pivot_value for key, value in work.items()}
    row_representation: Dict[int, Complex] = defaultdict(complex)
    row_representation[basis_index] = 1 / pivot_value
    for factor, previous_representation in row_coefficients:
        for previous_index, coefficient in previous_representation.items():
            row_representation[previous_index] -= factor * coefficient / pivot_value
    echelon.append((
        pivot,
        normalized,
        {index: value for index, value in row_representation.items() if value != 0},
    ))
    basis_entry = IndependentWick(
        basis_index, pairing, profile, original, typed_profile,
    )
    relation = WickBasisRepresentation(
        wick_index, pairing, profile, ((basis_index, 1 + 0j),), typed_profile,
    )
    return basis_entry, relation, basis_count


def enumerate_wick_pairings(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
) -> Iterator[WickPairing]:
    """Enumerate all flavor-preserving quark-antiquark Wick pairings.

    By default, contractions inside a single operator are included, which is
    required for flavor-neutral mesons and disconnected loops.  Set
    ``normal_ordered`` to exclude contractions between fields in the same
    operator.  The Wick sign follows the declared field order: sink slots,
    then source slots, each in their supplied order.
    """
    if wick_scope not in {"complete", "cross_time"}:
        raise ValueError("wick_scope must be 'complete' or 'cross_time'")
    fields = _flatten(correlator)
    quarks: Dict[str, list[int]] = defaultdict(list)
    antiquarks: Dict[str, list[int]] = defaultdict(list)
    for index, (_side, _operator_name, field) in enumerate(fields):
        (antiquarks if field.barred else quarks)[field.flavor].append(index)
    if set(quarks) != set(antiquarks) or any(
        len(quarks[flavor]) != len(antiquarks[flavor]) for flavor in quarks
    ):
        return

    flavors = sorted(quarks)
    ordered_quarks = tuple(index for flavor in flavors for index in quarks[flavor])
    ordered_antiquarks = tuple(index for flavor in flavors for index in antiquarks[flavor])
    canonical_field_order = ordered_quarks + ordered_antiquarks
    order_sign = parity(canonical_field_order)
    anti_position = {field_index: i for i, field_index in enumerate(ordered_antiquarks)}

    def visit(flavor_index: int, pairs: list[Tuple[int, int]]):
        if flavor_index == len(flavors):
            ordered_pairs = tuple(sorted(pairs))
            pairing_by_quark = dict(ordered_pairs)
            permutation = tuple(
                anti_position[pairing_by_quark[quark_index]]
                for quark_index in ordered_quarks
            )
            yield WickPairing(ordered_pairs, order_sign * parity(permutation))
            return

        flavor = flavors[flavor_index]
        flavor_quarks = quarks[flavor]
        for anti_targets in permutations(antiquarks[flavor]):
            if normal_ordered and any(
                fields[quark_index][1] == fields[anti_index][1]
                for quark_index, anti_index in zip(flavor_quarks, anti_targets)
            ):
                continue
            if wick_scope == "cross_time" and any(
                fields[quark_index][0] == fields[anti_index][0]
                for quark_index, anti_index in zip(flavor_quarks, anti_targets)
            ):
                continue
            start = len(pairs)
            pairs.extend(zip(flavor_quarks, anti_targets))
            yield from visit(flavor_index + 1, pairs)
            del pairs[start:]

    yield from visit(0, [])


def _operator_assignments(operator: Operator) -> Iterator[TensorTerm]:
    if not operator.factors:
        yield from operator.terms
        return

    def combine(position: int, coordinates: list[Coordinate], open_indices: list[int], coefficient: complex):
        if position == len(operator.factors):
            yield TensorTerm(tuple(coordinates), coefficient, tuple(open_indices))
            return
        factor = operator.factors[position]
        coordinate_start = len(coordinates)
        open_start = len(open_indices)
        for term in _operator_assignments(factor):
            coordinates.extend(term.coordinates)
            open_indices.extend(term.open_indices)
            yield from combine(position + 1, coordinates, open_indices, coefficient * term.coefficient)
            del coordinates[coordinate_start:]
            del open_indices[open_start:]

    yield from combine(0, [], [], 1 + 0j)


def _peram_flavor(flavor: str, isospin_degenerate_perams: bool) -> str:
    """Canonical propagator label; Wick pairing continues to use literal flavor."""
    if isospin_degenerate_perams and flavor in {"u", "d"}:
        return "ud"
    return flavor


def iter_exact_terms(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
) -> Iterator[Tuple[TensorKey, Complex]]:
    """Stream raw exact monomial contributions without retaining the result."""
    fields = _flatten(correlator)
    for pairing in enumerate_wick_pairings(
        correlator, normal_ordered=normal_ordered, wick_scope=wick_scope,
    ):
        yield from _iter_pairing_terms(
            correlator, fields, pairing,
            isospin_degenerate_perams=isospin_degenerate_perams,
        )


def _iter_pairing_terms(
    correlator: Correlator,
    fields,
    pairing: WickPairing,
    *,
    isospin_degenerate_perams: bool = True,
):
    for sink_term in _operator_assignments(correlator.sink):
        for source_term in _operator_assignments(correlator.source):
            coordinates = sink_term.coordinates + source_term.coordinates
            factors = []
            for quark_index, anti_index in pairing.pairs:
                _q_side, _q_operator, quark = fields[quark_index]
                _a_side, _a_operator, antiquark = fields[anti_index]
                q_colour, q_spin = coordinates[quark_index]
                anti_colour, anti_spin = coordinates[anti_index]
                factors.append((
                    quark.flavor,
                    quark.point,
                    q_colour,
                    q_spin,
                    antiquark.point,
                    anti_colour,
                    anti_spin,
                ))
            monomial = tuple(sorted(
                (_peram_flavor(factor[0], isospin_degenerate_perams), *factor[1:])
                for factor in factors
            ))
            open_indices = sink_term.open_indices + source_term.open_indices
            coefficient = pairing.sign * sink_term.coefficient * source_term.coefficient
            if coefficient:
                yield (open_indices, monomial), coefficient


def _propagator_profile(
    fields,
    pairing: WickPairing,
    *,
    isospin_degenerate_perams: bool = True,
):
    counts = Counter()
    for quark_index, anti_index in pairing.pairs:
        _q_side, _q_operator, quark = fields[quark_index]
        _a_side, _a_operator, antiquark = fields[anti_index]
        counts[(_peram_flavor(quark.flavor, isospin_degenerate_perams),
                quark.point, antiquark.point)] += 1
    return tuple(sorted(counts.items()))


def _typed_peram_profile(
    fields,
    pairing: WickPairing,
    *,
    isospin_degenerate_perams: bool = True,
):
    counts = Counter()
    for quark_index, anti_index in pairing.pairs:
        q_side, _q_operator, quark = fields[quark_index]
        anti_side, _anti_operator, antiquark = fields[anti_index]
        counts[(
            _peram_flavor(quark.flavor, isospin_degenerate_perams),
            f"{q_side}_to_{anti_side}",
            "quark_to_antiquark",
            quark.point,
            antiquark.point,
        )] += 1
    return tuple(sorted(counts.items()))


def _sparse_add_scaled(target: Dict[TensorKey, Complex], source, scale: Complex, tolerance: float):
    for key, value in source.items():
        updated = target.get(key, 0j) + scale * value
        if abs(updated) <= tolerance:
            target.pop(key, None)
        else:
            target[key] = updated


def iter_wick_vectors(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    tolerance: float = 1.0e-12,
    isospin_degenerate_perams: bool = True,
):
    """Yield each raw Wick as one sparse vector, releasing it before the next."""
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    fields = _flatten(correlator)
    for wick_index, pairing in enumerate(
        enumerate_wick_pairings(
            correlator, normal_ordered=normal_ordered, wick_scope=wick_scope,
        )
    ):
        profile = _propagator_profile(
            fields, pairing,
            isospin_degenerate_perams=isospin_degenerate_perams,
        )
        vector: Dict[TensorKey, Complex] = defaultdict(complex)
        for key, value in _iter_pairing_terms(
            correlator, fields, pairing,
            isospin_degenerate_perams=isospin_degenerate_perams,
        ):
            vector[key] += value
        scale = max((abs(value) for value in vector.values()), default=0.0)
        vector_threshold = tolerance * scale
        sparse_vector = {
            key: value for key, value in vector.items()
            if abs(value) > vector_threshold
        }
        yield wick_index, pairing, profile, sparse_vector


def iter_reduced_wicks(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    tolerance: float = 1.0e-12,
    isospin_degenerate_perams: bool = True,
) -> Iterator[Tuple[Optional[IndependentWick], WickBasisRepresentation]]:
    """Yield a transporter-reduced basis and reconstructible raw-Wick relations.

    Only complete local tensor automorphisms and swaps of identical atomic
    factors are candidates. Candidate orbits with an inconsistent stabilizer
    are expanded through the exact per-Wick fallback.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    correlator.validate()
    echelon_by_profile: Dict[Tuple[object, object], list] = defaultdict(list)
    basis_count = 0
    fields = _flatten(correlator)
    generators, layouts = _transporter_generators(correlator)
    descriptors, axis_route = _slot_roles(layouts, fields)

    def profile_for(pairing):
        return _typed_profile(fields, pairing, layouts, descriptors, axis_route)

    if not generators:
        for wick_index, pairing in enumerate(
            enumerate_wick_pairings(
                correlator, normal_ordered=normal_ordered,
                wick_scope=wick_scope,
            )
        ):
            profile = _propagator_profile(
                fields, pairing,
                isospin_degenerate_perams=isospin_degenerate_perams,
            )
            typed_profile = profile_for(pairing)
            original = _pairing_vector(
                correlator, fields, pairing, tolerance=tolerance,
                isospin_degenerate_perams=isospin_degenerate_perams,
            )
            basis_entry, relation, basis_count = _reduce_one_vector(
                wick_index, pairing, profile, typed_profile, original,
                echelon_by_profile, basis_count, tolerance,
                (typed_profile.connectivity_class, typed_profile.topology_signature),
            )
            yield basis_entry, relation
        return

    all_pairings = tuple(enumerate_wick_pairings(
        correlator, normal_ordered=normal_ordered, wick_scope=wick_scope,
    ))
    index_by_pairs = {pairing.pairs: (index, pairing)
                      for index, pairing in enumerate(all_pairings)}
    visited = set()
    for wick_index, pairing in enumerate(all_pairings):
        if pairing.pairs in visited:
            continue
        typed_profile = profile_for(pairing)
        orbit, consistent = _pairing_orbit(
            pairing, wick_index, fields, generators, typed_profile,
            descriptors, axis_route,
        )
        if any(pairs not in index_by_pairs for pairs, _mapped, _transporter in orbit):
            consistent = False
        orbit = [(index_by_pairs[pairs][0], index_by_pairs[pairs][1], transporter)
                 for pairs, _mapped, transporter in orbit
                 if pairs in index_by_pairs]
        if any(member[1].pairs in visited for member in orbit):
            consistent = False
        orbit = [member for member in orbit if member[1].pairs not in visited]
        visited.update(member[1].pairs for member in orbit)
        orbit.sort(key=lambda member: member[0])

        if not consistent:
            for member_index, member_pairing, _transporter in orbit:
                profile = _propagator_profile(
                    fields, member_pairing,
                    isospin_degenerate_perams=isospin_degenerate_perams,
                )
                member_profile = profile_for(member_pairing)
                original = _pairing_vector(
                    correlator, fields, member_pairing, tolerance=tolerance,
                    isospin_degenerate_perams=isospin_degenerate_perams,
                )
                basis_entry, relation, basis_count = _reduce_one_vector(
                    member_index, member_pairing, profile, member_profile,
                    original, echelon_by_profile, basis_count, tolerance,
                    (member_profile.connectivity_class,
                     member_profile.topology_signature),
                )
                yield basis_entry, relation
            continue

        representative_index, representative_pairing, _identity_transporter = orbit[0]
        representative_profile = _propagator_profile(
            fields, representative_pairing,
            isospin_degenerate_perams=isospin_degenerate_perams,
        )
        representative_typed_profile = profile_for(representative_pairing)
        representative_vector = _pairing_vector(
            correlator, fields, representative_pairing, tolerance=tolerance,
            isospin_degenerate_perams=isospin_degenerate_perams,
        )
        basis_entry, representative_relation, basis_count = _reduce_one_vector(
            representative_index, representative_pairing,
            representative_profile, representative_typed_profile,
            representative_vector, echelon_by_profile, basis_count, tolerance,
            (representative_typed_profile.connectivity_class,
             representative_typed_profile.topology_signature),
        )
        yield basis_entry, representative_relation

        for member_index, member_pairing, transporter in orbit[1:]:
            member_profile = _propagator_profile(
                fields, member_pairing,
                isospin_degenerate_perams=isospin_degenerate_perams,
            )
            member_typed_profile = profile_for(member_pairing)
            full_transporter = ProfileTransporter(
                representative_index,
                transporter.field_map,
                transporter.open_axis_map,
                transporter.tensor_phase,
                complex(member_pairing.sign / representative_pairing.sign),
                transporter.tensor_phase
                * complex(member_pairing.sign / representative_pairing.sign),
            )
            raw_vector = _transport_vector(
                representative_vector, full_transporter, tolerance=tolerance,
            )
            relation = WickBasisRepresentation(
                member_index, member_pairing, member_profile,
                representative_relation.basis_coefficients,
                member_typed_profile, full_transporter, raw_vector,
            )
            yield None, relation


def reduce_wick_basis(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    tolerance: float = 1.0e-12,
    isospin_degenerate_perams: bool = True,
) -> Tuple[Tuple[IndependentWick, ...], Tuple[WickBasisRepresentation, ...]]:
    """Materialize the exact sparse basis and all raw-Wick relations."""
    independent = []
    relations = []
    for basis_entry, relation in iter_reduced_wicks(
        correlator,
        normal_ordered=normal_ordered,
        tolerance=tolerance,
        isospin_degenerate_perams=isospin_degenerate_perams,
    ):
        if basis_entry is not None:
            independent.append(basis_entry)
        relations.append(relation)
    return tuple(independent), tuple(relations)


def reconstruct_wick_vector(
    basis: Sequence[IndependentWick],
    representation: WickBasisRepresentation,
    *,
    tolerance: float = 1.0e-12,
) -> Dict[TensorKey, Complex]:
    """Reconstruct one exact sparse Wick vector from its stored basis relation."""
    result: Dict[TensorKey, Complex] = defaultdict(complex)
    by_index = {entry.index: entry.vector for entry in basis}
    for basis_index, coefficient in representation.basis_coefficients:
        if basis_index not in by_index:
            raise ValueError(f"unknown basis index {basis_index}")
        _sparse_add_scaled(result, by_index[basis_index], coefficient, tolerance)
    if representation.transporter is not None:
        result = defaultdict(complex, _transport_vector(
            result, representation.transporter, tolerance=tolerance,
        ))
    return result


def accumulate_exact(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
    tolerance: float = 1.0e-12,
    max_entries: int | None = None,
    group_by_connectivity: bool = False,
) -> Dict[TensorKey, Complex] | "GroupedWickResult":
    """Expand an exact coefficient dictionary over propagator matrix elements.

    A key is ``(open_indices, monomial)``.  Each line monomial is
    ``(flavour, q_point, q_colour, q_spin, bar_point, bar_colour, bar_spin)``.
    Factors are sorted because ordinary propagator matrix elements commute as
    scalar variables. Thus equal explicit propagator products are combined,
    while distinct endpoint coordinates and open output axes remain distinct.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if type(group_by_connectivity) is not bool:
        raise ValueError("group_by_connectivity must be true or false")
    if not group_by_connectivity:
        output: Dict[TensorKey, Complex] = defaultdict(complex)
        for key, coefficient in iter_exact_terms(
            correlator,
            normal_ordered=normal_ordered,
            wick_scope=wick_scope,
            isospin_degenerate_perams=isospin_degenerate_perams,
        ):
            output[key] += coefficient
            if max_entries is not None and len(output) > max_entries:
                raise MemoryError("coefficient dictionary exceeded max_entries")
        return {key: value for key, value in output.items()
                if abs(value) > tolerance}

    from .grouping import GroupedWickResult

    fields = _flatten(correlator)
    field_owners = tuple((side, owner) for side, owner, _field in fields)
    grouped: Dict[Tuple[int, int], Dict[TensorKey, Complex]] = defaultdict(
        lambda: defaultdict(complex)
    )
    for pairing in enumerate_wick_pairings(
        correlator, normal_ordered=normal_ordered, wick_scope=wick_scope,
    ):
        signature = classify_output_pairing(pairing.pairs, field_owners)
        output = grouped[signature]
        for key, coefficient in _iter_pairing_terms(
            correlator, fields, pairing,
            isospin_degenerate_perams=isospin_degenerate_perams,
        ):
            output[key] += coefficient
        if max_entries is not None and sum(map(len, grouped.values())) > max_entries:
            raise MemoryError("coefficient dictionary exceeded max_entries")
    cleaned = {
        signature: {
            key: value for key, value in values.items()
            if abs(value) > tolerance
        }
        for signature, values in grouped.items()
    }
    return GroupedWickResult(
        cleaned, tolerance=tolerance,
        total_groups={signature: dict(values)
                      for signature, values in grouped.items()},
    )


_COLOUR_GROUP = tuple(colour_permutations(range(3)))

def _operator_colour_character(
    operator: Operator,
) -> Optional[dict[Tuple[int, int, int], Complex]]:
    """Return a validated scalar S3 character, or None when unsupported."""
    if operator.open_dimensions:
        return None
    if operator.factors:
        character = {permutation: 1 + 0j for permutation in _COLOUR_GROUP}
        for factor in operator.factors:
            factor_character = _operator_colour_character(
                factor,
            )
            if factor_character is None:
                return None
            for permutation in _COLOUR_GROUP:
                character[permutation] *= factor_character[permutation]
        return character if _is_s3_character(character) else None

    tensor = defaultdict(complex)
    for term in operator.terms:
        tensor[(term.coordinates, term.open_indices)] += term.coefficient
    tensor = {key: value for key, value in tensor.items() if value != 0}
    if not tensor:
        return None

    pivot = max(tensor, key=lambda key: abs(tensor[key]))
    character = {}
    for permutation in _COLOUR_GROUP:
        mapped_keys = {
            key: (
                tuple((permutation[colour], spin) for colour, spin in key[0]),
                key[1],
            )
            for key in tensor
        }
        if set(mapped_keys.values()) != set(tensor):
            return None
        mapped_pivot = tensor[mapped_keys[pivot]]
        original_pivot = tensor[pivot]
        if mapped_pivot == original_pivot:
            phase = 1 + 0j
        elif mapped_pivot == -original_pivot:
            phase = -1 + 0j
        else:
            return None
        if any(
            tensor[mapped_keys[key]] != phase * coefficient
            for key, coefficient in tensor.items()
        ):
            return None
        character[permutation] = phase
    return character if _is_s3_character(character) else None


def _compose_colours(left, right):
    return tuple(left[right[colour]] for colour in range(3))


def _is_s3_character(character) -> bool:
    identity = (0, 1, 2)
    if character.get(identity) != 1:
        return False
    return all(
        character[_compose_colours(left, right)]
        == character[left] * character[right]
        for left in _COLOUR_GROUP
        for right in _COLOUR_GROUP
    )


def _correlator_colour_character(
    correlator: Correlator,
) -> Optional[dict[Tuple[int, int, int], Complex]]:
    sink_character = _operator_colour_character(correlator.sink)
    source_character = _operator_colour_character(correlator.source)
    if sink_character is None or source_character is None:
        return None
    return {
        permutation: sink_character[permutation] * source_character[permutation]
        for permutation in _COLOUR_GROUP
    }


def _character_name(character) -> str:
    transposition = (1, 0, 2)
    return "sign" if character[transposition].real < 0 else "trivial"


def _colour_transform_key(key: TensorKey, permutation) -> TensorKey:
    open_indices, monomial = key
    mapped = []
    for factor in monomial:
        values = list(factor)
        values[2] = permutation[values[2]]
        values[5] = permutation[values[5]]
        mapped.append(tuple(values))
    return open_indices, tuple(sorted(mapped))


def _colour_canonical_key(key: TensorKey) -> Tuple[TensorKey, int]:
    orbit = {_colour_transform_key(key, permutation) for permutation in _COLOUR_GROUP}
    return min(orbit), len(orbit)


def _is_colour_representative(key: TensorKey) -> Tuple[bool, int]:
    """Check canonicality cheaply when the peram sort order cannot change."""
    open_indices, monomial = key
    prefixes = tuple((factor[0], factor[1]) for factor in monomial)
    colours = tuple(
        colour
        for factor in monomial
        for colour in (factor[2], factor[5])
    )
    if len(set(prefixes)) == len(prefixes):
        stabilizer_size = 0
        is_representative = True
        for permutation in _COLOUR_GROUP:
            image = tuple(permutation[colour] for colour in colours)
            if image == colours:
                stabilizer_size += 1
            elif image < colours:
                is_representative = False
        return is_representative, len(_COLOUR_GROUP) // stabilizer_size

    stabilizer_size = 1
    is_representative = True
    identity = (0, 1, 2)
    for permutation in _COLOUR_GROUP:
        if permutation == identity:
            continue
        image = _colour_transform_key((open_indices, monomial), permutation)
        if image == key:
            stabilizer_size += 1
        elif image < key:
            is_representative = False
    return is_representative, len(_COLOUR_GROUP) // stabilizer_size


def _count_exact_sqlite(
    correlator: Correlator,
    database_path: str | Path,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
    batch_size: int = 4096,
    zero_tolerance: float = 1.0e-12,
    use_colour_orbits: bool = False,
) -> dict[str, int | str | None]:
    """Count exact nonzero coordinates using a bounded-memory SQLite merge.

    The destination must not already exist. Returns the number of nonzero rows
    after cancellation cleanup, plus exact-path Wick and peram-profile statistics.
    Keys use the same ``(open_indices, monomial)`` structure as
    :func:`accumulate_exact` and are JSON encoded in SQLite.
    """
    path = Path(database_path)
    if path.exists():
        raise FileExistsError(path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and nonnegative")
    correlator.validate()
    if wick_scope not in {"complete", "cross_time"}:
        raise ValueError("wick_scope must be 'complete' or 'cross_time'")
    colour_character = (
        _correlator_colour_character(correlator) if use_colour_orbits else None
    )
    colour_orbit_order = len(_COLOUR_GROUP) if colour_character is not None else 1
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA cache_size = -8192")
        connection.create_function(
            "complex_within_tolerance", 3,
            lambda real, imag, tolerance: int(
                abs(complex(real, imag)) <= tolerance
            ),
            deterministic=True,
        )
        connection.execute(
            "CREATE TABLE exact_terms (key TEXT PRIMARY KEY, real REAL NOT NULL, imag REAL NOT NULL, orbit_size INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE grouped_terms (class_sin INTEGER NOT NULL, class_sou INTEGER NOT NULL, key TEXT NOT NULL, real REAL NOT NULL, imag REAL NOT NULL, orbit_size INTEGER NOT NULL, PRIMARY KEY(class_sin, class_sou, key)) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE profiles (key TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE incidence_classes (key TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE seen_wicks (key TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        statement = (
            "INSERT INTO exact_terms(key, real, imag, orbit_size) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "real = real + excluded.real, imag = imag + excluded.imag, "
            "orbit_size = excluded.orbit_size"
        )
        grouped_statement = (
            "INSERT INTO grouped_terms(class_sin, class_sou, key, real, imag, orbit_size) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(class_sin, class_sou, key) "
            "DO UPDATE SET real = real + excluded.real, imag = imag + excluded.imag, "
            "orbit_size = excluded.orbit_size"
        )
        profile_statement = "INSERT OR IGNORE INTO profiles(key) VALUES (?)"
        incidence_statement = (
            "INSERT OR IGNORE INTO incidence_classes(key) VALUES (?)"
        )
        fields = _flatten(correlator)
        batch = []
        grouped_batch = []
        profile_batch = []
        incidence_batch = []
        wick_counts = Counter()
        wick_count = 0
        transporter_orbits = 0
        validated_transporters = 0
        transporter_wicks_saved = 0
        transporter_fallback_orbits = 0
        generators, layouts = _transporter_generators(correlator)
        descriptors, axis_route = _slot_roles(layouts, fields)
        def incidence_key(pairing):
            typed = _typed_profile(
                fields, pairing, layouts, descriptors, axis_route,
            )
            return json.dumps(
                [typed.connectivity_class, typed.topology_signature,
                 typed.edges, typed.output_route], separators=(",", ":"),
            )

        def flush_batches():
            if batch:
                connection.executemany(statement, batch)
                batch.clear()
            if grouped_batch:
                connection.executemany(grouped_statement, grouped_batch)
                grouped_batch.clear()
            if profile_batch:
                connection.executemany(profile_statement, profile_batch)
                profile_batch.clear()
            if incidence_batch:
                connection.executemany(incidence_statement, incidence_batch)
                incidence_batch.clear()

        def add_term(key, coefficient, signature):
            orbit_size = 1
            if colour_character is not None:
                is_representative, orbit_size = _is_colour_representative(key)
                if not is_representative:
                    return
            encoded_key = json.dumps(
                [list(key[0]), [list(factor) for factor in key[1]]],
                separators=(",", ":"),
            )
            batch.append((encoded_key, coefficient.real, coefficient.imag, orbit_size))
            grouped_batch.append((
                signature[0], signature[1], encoded_key,
                coefficient.real, coefficient.imag, orbit_size,
            ))
            if len(batch) >= batch_size:
                flush_batches()
                connection.commit()

        def add_pairing(pairing, signature):
            for key, coefficient in _iter_pairing_terms(
                correlator, fields, pairing,
                isospin_degenerate_perams=isospin_degenerate_perams,
            ):
                add_term(key, coefficient, signature)

        def is_allowed_pairing(pairing):
            normal_ok = not normal_ordered or all(
                fields[quark_index][1] != fields[anti_index][1]
                for quark_index, anti_index in pairing.pairs
            )
            scope_ok = wick_scope != "cross_time" or all(
                fields[quark_index][0] != fields[anti_index][0]
                for quark_index, anti_index in pairing.pairs
            )
            return normal_ok and scope_ok

        for pairing in enumerate_wick_pairings(
            correlator, normal_ordered=normal_ordered, wick_scope=wick_scope,
        ):
            wick_count += 1
            typed = _typed_profile(
                fields, pairing, layouts, descriptors, axis_route,
            )
            signature = typed.connectivity_class
            topology_signature = typed.topology_signature
            wick_counts[signature] += 1
            profile = _typed_peram_profile(
                fields, pairing,
                isospin_degenerate_perams=isospin_degenerate_perams,
            )
            profile_batch.append((json.dumps(
                [signature, topology_signature, profile], separators=(",", ":"),
            ),))
            if not generators:
                incidence_batch.append((incidence_key(pairing),))
                add_pairing(pairing, signature)
            else:
                pairing_key = json.dumps(pairing.pairs, separators=(",", ":"))
                if connection.execute(
                    "SELECT 1 FROM seen_wicks WHERE key = ?", (pairing_key,),
                ).fetchone() is not None:
                    if len(profile_batch) >= batch_size:
                        flush_batches()
                        connection.commit()
                    continue

                orbit, consistent = _pairing_orbit(
                    pairing, wick_count - 1, fields, generators,
                    typed,
                    descriptors, axis_route,
                )
                allowed_orbit = [
                    member for member in orbit
                    if is_allowed_pairing(member[1])
                ]
                closed_under_filters = len(allowed_orbit) == len(orbit)
                if consistent and closed_under_filters:
                    incidence_batch.append((min(
                        incidence_key(member_pairing)
                        for _member_pairs, member_pairing, _transporter in orbit
                    ),))
                else:
                    fallback_members = {
                        member_pairs: member_pairing
                        for member_pairs, member_pairing, _transporter
                        in allowed_orbit
                    }
                    fallback_members.setdefault(pairing.pairs, pairing)
                    incidence_batch.extend(
                        (incidence_key(member_pairing),)
                        for member_pairing in fallback_members.values()
                    )
                if consistent and closed_under_filters:
                    for member_pairs, _member_pairing, _transporter in orbit:
                        connection.execute(
                            "INSERT OR IGNORE INTO seen_wicks(key) VALUES (?)",
                            (json.dumps(member_pairs, separators=(",", ":")),),
                        )
                    transporter_orbits += 1
                    validated_transporters += max(0, len(orbit) - 1)
                    transporter_wicks_saved += max(0, len(orbit) - 1)

                    # Expand the representative once, then account for every
                    # member's transported coordinate and phase in SQLite.
                    for key, coefficient in _iter_pairing_terms(
                        correlator, fields, pairing,
                        isospin_degenerate_perams=isospin_degenerate_perams,
                    ):
                        orbit_terms = defaultdict(complex)
                        for _member_pairs, _member_pairing, transporter in orbit:
                            mapped_key = _transport_exact_key(key, transporter)
                            orbit_terms[mapped_key] += transporter.phase * coefficient
                        for mapped_key, mapped_coefficient in orbit_terms.items():
                            if mapped_coefficient:
                                add_term(mapped_key, mapped_coefficient, signature)
                else:
                    transporter_fallback_orbits += 1
                    for member_pairs, member_pairing in fallback_members.items():
                        connection.execute(
                            "INSERT OR IGNORE INTO seen_wicks(key) VALUES (?)",
                            (json.dumps(member_pairs, separators=(",", ":")),),
                        )
                        add_pairing(
                            member_pairing,
                            classify_output_pairing(
                                member_pairing.pairs,
                                tuple((side, owner) for side, owner, _field in fields),
                            ),
                        )

            if len(profile_batch) >= batch_size:
                flush_batches()
                connection.commit()
        flush_batches()
        connection.commit()
        connection.execute(
            "DELETE FROM exact_terms "
            "WHERE complex_within_tolerance(real, imag, ?) = 1",
            (zero_tolerance,),
        )
        connection.execute(
            "DELETE FROM grouped_terms "
            "WHERE complex_within_tolerance(real, imag, ?) = 1",
            (zero_tolerance,),
        )
        connection.execute("DROP TABLE seen_wicks")
        connection.commit()
        representative_rows = int(connection.execute(
            "SELECT COUNT(*) FROM exact_terms"
        ).fetchone()[0])
        nlist = int(connection.execute(
            "SELECT COALESCE(SUM(orbit_size), 0) FROM exact_terms"
        ).fetchone()[0])
        grouped_rows = {
            (int(left), int(right)): (int(rows), int(size))
            for left, right, rows, size in connection.execute(
                "SELECT class_sin, class_sou, COUNT(*), COALESCE(SUM(orbit_size), 0) "
                "FROM grouped_terms GROUP BY class_sin, class_sou"
            )
        }
        group_summaries = [
            {
                "signature": list(signature),
                "raw_wicks": int(wick_counts[signature]),
                "representative_rows": grouped_rows.get(signature, (0, 0))[0],
                "nlist": grouped_rows.get(signature, (0, 0))[1],
            }
            for signature in sorted(
                set(wick_counts) | set(grouped_rows),
                key=lambda value: (value != (0, 0), value),
            )
        ]
        return {
            "algorithm": (
                "exact-colour-orbit-sqlite-transporter"
                if colour_character is not None and validated_transporters
                else "exact-colour-orbit-sqlite"
                if colour_character is not None
                else "exact-coordinate-sqlite-transporter"
                if validated_transporters
                else "exact-coordinate-sqlite-fallback"
            ),
            "raw_wicks": wick_count,
            "peram_profiles": int(connection.execute(
                "SELECT COUNT(*) FROM profiles"
            ).fetchone()[0]),
            "incidence_classes": int(connection.execute(
                "SELECT COUNT(*) FROM incidence_classes"
            ).fetchone()[0]),
            "validated_transporters": validated_transporters,
            "transporter_orbits": transporter_orbits,
            "transporter_wicks_saved": transporter_wicks_saved,
            "transporter_fallback_orbits": transporter_fallback_orbits,
            "colour_symmetry_order": colour_orbit_order,
            "colour_character": (
                _character_name(colour_character)
                if colour_character is not None else None
            ),
            "colour_orbits": representative_rows if colour_character is not None else 0,
            "representative_rows": representative_rows,
            "nlist": nlist,
            "wick_scope": wick_scope,
            "groups": group_summaries,
        }
    finally:
        connection.close()


def count_exact_sqlite(
    correlator: Correlator,
    database_path: str | Path,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
    batch_size: int = 4096,
    zero_tolerance: float = 1.0e-12,
) -> dict[str, int | str | None]:
    """Count exact nonzero coordinates using a bounded-memory SQLite merge."""
    return _count_exact_sqlite(
        correlator, database_path,
        normal_ordered=normal_ordered,
        wick_scope=wick_scope,
        isospin_degenerate_perams=isospin_degenerate_perams,
        batch_size=batch_size,
        zero_tolerance=zero_tolerance,
    )


def count_colour_orbit_sqlite(
    correlator: Correlator,
    database_path: str | Path,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
    batch_size: int = 4096,
    zero_tolerance: float = 1.0e-12,
) -> dict[str, int | str | None]:
    """Count exact coordinates, storing one validated global-colour orbit rep."""
    return _count_exact_sqlite(
        correlator, database_path,
        normal_ordered=normal_ordered,
        wick_scope=wick_scope,
        isospin_degenerate_perams=isospin_degenerate_perams,
        batch_size=batch_size,
        zero_tolerance=zero_tolerance,
        use_colour_orbits=True,
    )


def accumulate_exact_sqlite(
    correlator: Correlator,
    database_path: str | Path,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
    batch_size: int = 4096,
    zero_tolerance: float = 1.0e-12,
) -> int:
    """Accumulate the exact dictionary on disk with bounded Python memory."""
    return int(count_exact_sqlite(
        correlator,
        database_path,
        normal_ordered=normal_ordered,
        wick_scope=wick_scope,
        isospin_degenerate_perams=isospin_degenerate_perams,
        batch_size=batch_size,
        zero_tolerance=zero_tolerance,
    )["nlist"])


def count_exact(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
    isospin_degenerate_perams: bool = True,
    batch_size: int = 4096,
    zero_tolerance: float = 1.0e-12,
) -> dict[str, int | str | None]:
    """Count complete exact coordinates without retaining a Python dictionary."""
    with tempfile.TemporaryDirectory(prefix="wick-colour-count-") as directory:
        return count_exact_sqlite(
            correlator,
            Path(directory) / "exact_terms.sqlite3",
            normal_ordered=normal_ordered,
            wick_scope=wick_scope,
            isospin_degenerate_perams=isospin_degenerate_perams,
            batch_size=batch_size,
            zero_tolerance=zero_tolerance,
        )


def iter_exact_sqlite(database_path: str | Path) -> Iterator[Tuple[TensorKey, Complex]]:
    """Read a disk-accumulated exact dictionary one row at a time."""
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.execute("SELECT key, real, imag FROM exact_terms ORDER BY key")
        for encoded_key, real, imag in cursor:
            open_indices, monomial = json.loads(encoded_key)
            yield (
                (tuple(open_indices), tuple(tuple(factor) for factor in monomial)),
                complex(real, imag),
            )
    finally:
        connection.close()


def iter_count_representatives_sqlite(
    database_path: str | Path,
) -> Iterator[Tuple[TensorKey, Complex, int]]:
    """Read count representatives with their exact global-colour orbit sizes."""
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.execute(
            "SELECT key, real, imag, orbit_size FROM exact_terms ORDER BY key"
        )
        for encoded_key, real, imag, orbit_size in cursor:
            open_indices, monomial = json.loads(encoded_key)
            yield (
                (tuple(open_indices), tuple(tuple(factor) for factor in monomial)),
                complex(real, imag),
                int(orbit_size),
            )
    finally:
        connection.close()


def iter_grouped_count_representatives_sqlite(
    database_path: str | Path,
) -> Iterator[Tuple[Tuple[int, int], TensorKey, Complex, int]]:
    """Read representative rows grouped by same-side Wick edge counts."""
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.execute(
            "SELECT class_sin, class_sou, key, real, imag, orbit_size "
            "FROM grouped_terms ORDER BY class_sin, class_sou, key"
        )
        for class_sin, class_sou, encoded_key, real, imag, orbit_size in cursor:
            open_indices, monomial = json.loads(encoded_key)
            yield (
                (int(class_sin), int(class_sou)),
                (tuple(open_indices), tuple(tuple(factor) for factor in monomial)),
                complex(real, imag),
                int(orbit_size),
            )
    finally:
        connection.close()
