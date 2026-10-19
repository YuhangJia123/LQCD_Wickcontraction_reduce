"""Input model for ordinary-colour source coefficient reduction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
from math import isfinite
from typing import Iterator, Literal, Sequence, Tuple


ColourKind = Literal["epsilon", "delta"]
SpinTerm = Tuple[Tuple[int, ...], complex]


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
    barred: bool = False
    point: str = ""

    def __post_init__(self) -> None:
        if not self.flavor:
            raise ValueError("field flavor must be a nonempty string")
        if not isinstance(self.point, str):
            raise TypeError("field point must be a string")


@dataclass(frozen=True)
class LocalVertex:
    """One factorized local colour-singlet vertex.

    ``spin_terms`` stores sparse Dirac tensor entries in the same ordered slot
    convention as ``fields``. The colour part is either epsilon or delta.
    """

    name: str
    fields: Tuple[Field, ...]
    colour_kind: ColourKind
    spin_terms: Tuple[SpinTerm, ...]
    phase: complex = 1 + 0j

    def validate(self) -> None:
        expected_rank = 3 if self.colour_kind == "epsilon" else 2
        if self.colour_kind not in ("epsilon", "delta"):
            raise ValueError(f"unsupported colour tensor: {self.colour_kind}")
        if len(self.fields) != expected_rank:
            raise ValueError(
                f"{self.colour_kind} vertex {self.name!r} needs {expected_rank} fields"
            )
        if not self.spin_terms:
            raise ValueError(f"vertex {self.name!r} needs at least one spin term")
        for spins, coefficient in self.spin_terms:
            if len(spins) != expected_rank:
                raise ValueError(f"vertex {self.name!r} has a malformed spin term")
            if any(not 0 <= spin < 4 for spin in spins):
                raise ValueError("Dirac indices must lie in [0, 3]")
            if not isinstance(coefficient, (int, float, complex)):
                raise TypeError("spin coefficients must be numeric")


@dataclass(frozen=True)
class Operator:
    """An ordered operator field list, optionally factored into local vertices."""

    name: str
    fields: Tuple[Field, ...]
    vertices: Tuple[LocalVertex, ...] = ()

    @classmethod
    def from_vertices(cls, name: str, vertices: Sequence[LocalVertex]) -> "Operator":
        factors = tuple(vertices)
        return cls(name, tuple(field for vertex in factors for field in vertex.fields), factors)

    def validate(self, *, require_vertices: bool = False) -> None:
        if not self.fields:
            raise ValueError(f"operator {self.name!r} has no fields")
        if require_vertices and not self.vertices:
            raise ValueError(f"source operator {self.name!r} needs factorized vertices")
        if self.vertices:
            for vertex in self.vertices:
                vertex.validate()
            flattened = tuple(field for vertex in self.vertices for field in vertex.fields)
            if flattened != self.fields:
                raise ValueError(f"operator {self.name!r} fields differ from its vertex slots")


@dataclass(frozen=True)
class Correlator:
    """An ordered ``[sink, source]`` pair for source-coefficient reduction.

    Sink fields are the open output legs. Their local operator tensor is not
    part of the coefficient produced by this model.
    """

    sink: Operator
    source: Operator
    sink_routes: Tuple[Tuple[int, int], ...] = ()
    sink_axis_transports: Tuple[Tuple[Tuple[int, ...], complex], ...] | None = None

    @property
    def line_count(self) -> int:
        return len(self.sink.fields)

    def validate(self) -> None:
        self.sink.validate()
        self.source.validate(require_vertices=True)
        if self.sink.vertices:
            raise ValueError(
                "the fast coefficient path keeps sink legs open; omit sink vertices "
                "and contract the sink tensor separately"
            )
        if len(self.sink.fields) != len(self.source.fields):
            raise ValueError("sink and source must contain the same number of fields")
        if self.sink_routes and (
            len(self.sink_routes) != len(self.sink.fields)
            or any(len(route) != 2 or any(index < 0 for index in route)
                   for route in self.sink_routes)
        ):
            raise ValueError("sink output routes must identify each operator slot")
        if self.sink_axis_transports is not None:
            identity = tuple(range(len(self.sink.fields)))
            for axis_map, phase in self.sink_axis_transports:
                if (len(axis_map) != len(identity)
                        or any(type(axis) is not int for axis in axis_map)
                        or tuple(sorted(axis_map)) != identity):
                    raise ValueError("sink-axis transports must be complete permutations")
                if any(self.sink.fields[old] != self.sink.fields[new]
                       for old, new in enumerate(axis_map)):
                    raise ValueError("sink-axis transports must preserve typed fields")
                try:
                    phase_magnitude = abs(complex(phase))
                except (TypeError, ValueError) as error:
                    raise ValueError("sink-axis transport phase must be numeric") from error
                if (not isfinite(phase_magnitude)
                        or abs(phase_magnitude - 1) > 1.0e-10):
                    raise ValueError("sink-axis transport phase must be unit magnitude")

        sink_counts = Counter((field.flavor, field.barred) for field in self.sink.fields)
        source_counts = Counter((field.flavor, not field.barred) for field in self.source.fields)
        if sink_counts != source_counts:
            raise ValueError(
                "cross-time Wick pairing requires matching flavor and opposite field orientation"
            )

    def validate_general(self) -> None:
        """Validate a fully factorized graph with tensors on both sides."""
        self.sink.validate(require_vertices=True)
        self.source.validate(require_vertices=True)


@dataclass(frozen=True)
class WickMap:
    """A cross-time map from each ordered source slot to a sink line."""

    source_to_sink: Tuple[int, ...]
    sign: int


@dataclass(frozen=True)
class FullWick:
    """A flavor-valid pairing for a fully factorized sink/source graph."""

    pairs: Tuple[Tuple[int, int], ...]
    sign: int


@dataclass(frozen=True, order=True)
class EndpointRoute:
    """Propagator endpoints and open sink slot for one output line."""

    sink_route: Tuple[int, int]
    flavor: str
    quark_point: str
    antiquark_point: str


@dataclass(frozen=True, order=True)
class EndpointResolvedKey:
    """A packed source coefficient paired with its ordered peram endpoints."""

    endpoints: Tuple[EndpointRoute, ...]
    coordinates: int


def general_field_records(correlator: Correlator):
    """Return ``(side, vertex_index, local_slot, field)`` in field order."""
    records = []
    for side, operator in (("sink", correlator.sink), ("source", correlator.source)):
        for vertex_index, vertex in enumerate(operator.vertices):
            for local_slot, field in enumerate(vertex.fields):
                records.append((side, vertex_index, local_slot, field))
    return tuple(records)


def enumerate_general_wicks(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    wick_scope: str = "complete",
) -> Iterator[FullWick]:
    """Enumerate flavor-valid pairings for a graph with both sides factored."""
    if wick_scope not in {"complete", "cross_time"}:
        raise ValueError("wick_scope must be 'complete' or 'cross_time'")
    correlator.validate_general()
    records = general_field_records(correlator)
    quarks: dict[str, list[int]] = {}
    antiquarks: dict[str, list[int]] = {}
    owners = []
    for index, (_side, vertex_index, _slot, field) in enumerate(records):
        owners.append((_side, vertex_index))
        (antiquarks if field.barred else quarks).setdefault(field.flavor, []).append(index)
    if set(quarks) != set(antiquarks) or any(
        len(quarks[flavor]) != len(antiquarks[flavor]) for flavor in quarks
    ):
        return

    flavors = tuple(sorted(quarks))
    ordered_quarks = tuple(index for flavor in flavors for index in quarks[flavor])
    ordered_antiquarks = tuple(index for flavor in flavors for index in antiquarks[flavor])
    canonical_order_sign = parity(ordered_quarks + ordered_antiquarks)
    anti_position = {field_index: i for i, field_index in enumerate(ordered_antiquarks)}

    def visit(flavor_index: int, pairs: list[Tuple[int, int]]):
        if flavor_index == len(flavors):
            ordered_pairs = tuple(sorted(pairs))
            by_quark = dict(ordered_pairs)
            permutation = tuple(
                anti_position[by_quark[quark_index]] for quark_index in ordered_quarks
            )
            yield FullWick(ordered_pairs, canonical_order_sign * parity(permutation))
            return
        flavor = flavors[flavor_index]
        for anti_targets in permutations(antiquarks[flavor]):
            if normal_ordered and any(
                owners[quark_index] == owners[anti_index]
                for quark_index, anti_index in zip(quarks[flavor], anti_targets)
            ):
                continue
            if wick_scope == "cross_time" and any(
                owners[quark_index][0] == owners[anti_index][0]
                for quark_index, anti_index in zip(quarks[flavor], anti_targets)
            ):
                continue
            start = len(pairs)
            pairs.extend(zip(quarks[flavor], anti_targets))
            yield from visit(flavor_index + 1, pairs)
            del pairs[start:]

    yield from visit(0, [])


def enumerate_cross_time_wicks(correlator: Correlator) -> Iterator[WickMap]:
    """Enumerate flavor-preserving Wick pairings with every line crossing time.

    The fermion sign follows the declared sink-then-source field ordering.
    Same-operator contractions, including neutral-meson disconnected loops,
    are outside this source-coefficient fast path.
    """

    correlator.validate()
    fields = correlator.sink.fields + correlator.source.fields
    sink_count = len(correlator.sink.fields)
    quarks: dict[str, list[int]] = {}
    antiquarks: dict[str, list[int]] = {}
    for index, field in enumerate(fields):
        destination = antiquarks if field.barred else quarks
        destination.setdefault(field.flavor, []).append(index)

    if set(quarks) != set(antiquarks) or any(
        len(quarks[flavor]) != len(antiquarks[flavor]) for flavor in quarks
    ):
        return

    flavors = tuple(sorted(quarks))
    ordered_quarks = tuple(index for flavor in flavors for index in quarks[flavor])
    ordered_antiquarks = tuple(index for flavor in flavors for index in antiquarks[flavor])
    canonical_order_sign = parity(ordered_quarks + ordered_antiquarks)
    anti_position = {field_index: index for index, field_index in enumerate(ordered_antiquarks)}
    choices = tuple(tuple(permutations(antiquarks[flavor])) for flavor in flavors)

    for selected in product(*choices):
        targets = tuple(target for group in selected for target in group)
        pairs = tuple(zip(ordered_quarks, targets))
        if any((quark < sink_count) == (antiquark < sink_count)
               for quark, antiquark in pairs):
            continue

        source_to_sink = [-1] * sink_count
        for quark, antiquark in pairs:
            if quark < sink_count:
                source_slot, sink_line = antiquark - sink_count, quark
            else:
                source_slot, sink_line = quark - sink_count, antiquark
            source_to_sink[source_slot] = sink_line

        permutation = tuple(anti_position[target] for target in targets)
        yield WickMap(tuple(source_to_sink), canonical_order_sign * parity(permutation))
