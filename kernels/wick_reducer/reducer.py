#!/usr/bin/env python3
"""Sink-first Wick reduction with explicit multiple transporter solutions.

This standalone project bundles qct's operator/Wick definitions and its small
factorized Dirac contraction primitive. The reduction itself is local here:
peram incidence is matched first, every allowed transporter is retained,
sink Dirac tensors are screened and rank-reduced next, and source tensors are
examined only for sink-compatible transporters.
"""

from __future__ import annotations

import itertools
import json
import math
import struct
import sys
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernels import correlator_wick_normalizer as core  # noqa: E402

from kernels.wick_reducer import operator_model as model
from kernels.wick_reducer import peram_incidence


TOLERANCE = 1.0e-10
MAX_TRANSPORTERS = 16384
EXAMPLE_ORDER = (
    "pipi", "pn", "ppn", "delta", "delta++delta-", "pip", "pip-p",
    "pn_omega_pi+",
)


@dataclass(frozen=True)
class ReductionOptions:
    """Policies that apply after exact raw Wick generation.

    ``isospin_degenerate`` only changes the numerical flavor class used for
    peram matching (``u`` and ``d`` become ``ud``).  It never changes the raw
    flavor-valid Wick enumeration.  The reducer always uses the generic
    factor-graph canonicalization path after that enumeration.  The graph
    retains kind, Gamma/slots, external/momentum tags and directed peram
    edges, but does not use a hadron name as a color.

    The policy is intentionally independent of a correlator name such as
    ``ppn`` or ``ppnn``.  Those names are only labels for built-in fixtures.
    """

    isospin_degenerate: bool = True


@dataclass
class TransportState:
    operator_map: Tuple[int, ...]
    line_map: Tuple[int, ...]
    line_order: Tuple[int, ...]
    sink: Dict[str, object]
    source: Optional[Dict[str, object]]
    coefficient: complex
    sink_vvv_sign: int
    source_vvv_sign: int
    bare_axis_map: Dict[str, str]


@dataclass
class BasisState:
    wick_index: int
    state: TransportState
    aggregate: complex = 1.0 + 0.0j


@dataclass
class SinkClass:
    class_id: int
    reference_sink: Dict[str, object]
    basis: List[BasisState] = field(default_factory=list)
    orthonormal_rows: List[List[complex]] = field(default_factory=list)


@dataclass
class PeramFamily:
    root_index: int
    root_term: core.WickTerm
    canonical_operator_map: Optional[Tuple[int, ...]]
    sink_classes: List[SinkClass] = field(default_factory=list)


class CompactWickRecord:
    """Small raw-Wick record kept in a factor bucket.

    A pairing is stored as little-endian unsigned integers rather than a
    Python tuple of boxed integers.  The width is shared by the reduction and
    is therefore not repeated for every record. The first-pass canonical
    operator witness is stored beside the pairing to avoid a second graph
    canonicalization.
    """

    __slots__ = (
        "wick_index", "pairing_data", "pairing_offset", "pairing_size",
        "pairing_width", "operator_map_data", "operator_map_offset",
        "operator_map_size", "operator_map_width", "sign",
    )

    def __init__(
        self,
        wick_index: int,
        pairing_data: bytearray,
        pairing_offset: int,
        pairing_size: int,
        pairing_width: int,
        operator_map_data: bytearray,
        operator_map_offset: int,
        operator_map_size: int,
        operator_map_width: int,
        sign: int,
    ) -> None:
        self.wick_index = int(wick_index)
        self.pairing_data = pairing_data
        self.pairing_offset = pairing_offset
        self.pairing_size = pairing_size
        self.pairing_width = pairing_width
        self.operator_map_data = operator_map_data
        self.operator_map_offset = operator_map_offset
        self.operator_map_size = operator_map_size
        self.operator_map_width = operator_map_width
        self.sign = int(sign)

    def pairing(self) -> Tuple[int, ...]:
        return _unpack_pairing_from(
            self.pairing_data, self.pairing_offset,
            self.pairing_size, self.pairing_width,
        )

    def operator_map(self) -> Tuple[int, ...]:
        return _unpack_pairing_from(
            self.operator_map_data, self.operator_map_offset,
            self.operator_map_size, self.operator_map_width,
        )


class CompactWickBucket:
    """Structure-of-arrays storage for the raw terms sharing one graph key."""

    __slots__ = (
        "wick_indices", "pairing_data", "signs", "pairing_width",
        "pairing_size", "operator_map_data", "operator_map_width",
        "operator_map_size",
    )

    def __init__(
        self,
        pairing_width: int,
        pairing_size: int,
        operator_map_width: int = 1,
        operator_map_size: int = 0,
    ) -> None:
        self.wick_indices = array("I")
        self.pairing_data = bytearray()
        self.signs = bytearray()
        self.pairing_width = pairing_width
        self.pairing_size = pairing_size
        self.operator_map_data = bytearray()
        self.operator_map_width = operator_map_width
        self.operator_map_size = operator_map_size

    def append(
        self,
        wick_index: int,
        term: core.WickTerm,
        operator_map: Sequence[int] = (),
    ) -> None:
        if len(term.pairing) != self.pairing_size:
            raise ValueError("all Wick pairings in a bucket must have the same size")
        if len(operator_map) != self.operator_map_size:
            raise ValueError("all canonical operator maps must have the same size")
        if wick_index < 0:
            raise ValueError("Wick indices must be non-negative")
        try:
            self.wick_indices.append(wick_index)
        except OverflowError:
            if self.wick_indices.typecode != "I":
                raise
            self.wick_indices = array("Q", self.wick_indices)
            self.wick_indices.append(wick_index)
        self.pairing_data.extend(_pack_pairing(term.pairing, self.pairing_width))
        self.operator_map_data.extend(
            _pack_pairing(operator_map, self.operator_map_width)
        )
        self.signs.append(1 if term.sign > 0 else 0)

    def __len__(self) -> int:
        return len(self.wick_indices)

    def __getitem__(self, index: int) -> CompactWickRecord:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        pairing_offset = index * self.pairing_size * self.pairing_width
        operator_map_offset = index * self.operator_map_size * self.operator_map_width
        return CompactWickRecord(
            self.wick_indices[index],
            self.pairing_data,
            pairing_offset,
            self.pairing_size,
            self.pairing_width,
            self.operator_map_data,
            operator_map_offset,
            self.operator_map_size,
            self.operator_map_width,
            1 if self.signs[index] else -1,
        )

    def __iter__(self) -> Iterator[CompactWickRecord]:
        for index in range(len(self)):
            yield self[index]


def _pairing_width(size: int) -> int:
    if size <= 1 << 8:
        return 1
    if size <= 1 << 16:
        return 2
    if size <= 1 << 32:
        return 4
    if size <= 1 << 64:
        return 8
    raise OverflowError("pairing indices must fit in unsigned 64-bit values")


def _pack_pairing(pairing: Sequence[int], width: int) -> bytes:
    if not pairing:
        return b""
    return _pairing_struct(len(pairing), width).pack(*pairing)


@lru_cache(maxsize=256)
def _pairing_struct(size: int, width: int) -> struct.Struct:
    format_codes = {1: "B", 2: "H", 4: "I", 8: "Q"}
    code = format_codes.get(width)
    if code is None:
        raise ValueError(f"unsupported pairing width {width}")
    return struct.Struct("<" + code * size)


def _unpack_pairing(blob: bytes, width: int) -> Tuple[int, ...]:
    if not blob:
        return ()
    size = len(blob) // width
    return _pairing_struct(size, width).unpack(blob)


def _unpack_pairing_from(
    data: bytearray, offset: int, size: int, width: int,
) -> Tuple[int, ...]:
    if not size:
        return ()
    return _pairing_struct(size, width).unpack_from(data, offset)


def _pack_compact_graph_signature(
    ordered_groups: Sequence[Tuple[Tuple, Sequence[int]]],
    vocabulary: _FactorGraphVocabulary,
    graph: Tuple,
    *,
    bipartite: bool,
) -> bytes:
    """Encode a canonical graph with explicit lengths and minimal-width IDs."""
    values = [len(ordered_groups)]
    for color, indices in ordered_groups:
        values.extend((vocabulary.vertex_ids[color], len(indices)))
    values.append(len(graph))
    if bipartite:
        for row in graph:
            values.append(len(row))
            for cell in row:
                values.append(len(cell))
                values.extend(cell)
        marker = b"B"
    else:
        for source, target, edge_id in graph:
            values.extend((source, target, edge_id))
        marker = b"G"
    maximum = max(values, default=0)
    if maximum <= 0xFF:
        width, code = 1, "B"
    elif maximum <= 0xFFFF:
        width, code = 2, "H"
    elif maximum <= 0xFFFFFFFF:
        width, code = 4, "I"
    elif maximum <= 0xFFFFFFFFFFFFFFFF:
        width, code = 8, "Q"
    else:
        raise OverflowError("factor graph IDs must fit in unsigned 64-bit values")
    try:
        payload = struct.pack("<" + code * len(values), *values)
    except struct.error as exc:
        raise OverflowError("factor graph IDs must fit in unsigned 64-bit values") from exc
    return marker + bytes((width,)) + payload


def _term_from_compact(
    record: CompactWickRecord, width: Optional[int] = None,
) -> core.WickTerm:
    del width  # Retain the historical helper signature for callers in verify_code.
    return core.WickTerm(record.pairing(), record.sign)


class _FactorGraphVocabulary:
    """Deterministic integer IDs for vertex and edge colors.

    The public canonical-key helper continues to expose descriptive colors.
    The reducer uses this vocabulary to make bucket keys compact while keeping
    the complete descriptor tables available for diagnostics and future
    collision audits.
    """

    def __init__(
        self,
        operators: Sequence[core.Operator],
        classified_quarks: Sequence[core.Field],
        classified_antiquarks: Sequence[core.Field],
        *,
        isospin_degenerate: bool,
    ) -> None:
        vertex_colors = sorted(
            {_operator_color(operator, isospin_degenerate) for operator in operators},
            key=repr,
        )
        edge_colors = sorted(
            {
                peram_incidence._edge_color(q, anti, isospin_degenerate=isospin_degenerate)
                for q in classified_quarks
                for anti in classified_antiquarks
            },
            key=repr,
        )
        self.vertex_ids = {color: index for index, color in enumerate(vertex_colors)}
        self.edge_ids = {color: index for index, color in enumerate(edge_colors)}
        self.vertex_colors = tuple(vertex_colors)
        self.edge_colors = tuple(edge_colors)


@lru_cache(maxsize=None)
def _permutation_table(size: int) -> Tuple[Tuple[int, ...], ...]:
    return tuple(itertools.permutations(range(size)))


class _FactorGraphCanonicalizationContext:
    """Precomputed operator groups and endpoint colors for one reduction."""

    def __init__(
        self,
        operators: Sequence[core.Operator],
        quarks: Sequence[core.Field],
        antiquarks: Sequence[core.Field],
        classified_quarks: Sequence[core.Field],
        classified_antiquarks: Sequence[core.Field],
        *,
        isospin_degenerate: bool,
        vocabulary: _FactorGraphVocabulary,
    ) -> None:
        groups: Dict[Tuple, List[int]] = defaultdict(list)
        for index, operator in enumerate(operators):
            groups[_operator_color(operator, isospin_degenerate)].append(index)
        self.ordered_groups = tuple(sorted(groups.items(), key=lambda item: repr(item[0])))
        self.offsets: Dict[int, int] = {}
        self.group_id: Dict[int, Tuple[int, int]] = {}
        offset = 0
        for gid, (_color, indices) in enumerate(self.ordered_groups):
            self.offsets[gid] = offset
            for local, index in enumerate(indices):
                self.group_id[index] = (gid, local)
            offset += len(indices)

        routes = []
        for q_index, q in enumerate(classified_quarks):
            q_gid, q_local = self.group_id[q.operator]
            anti_routes = []
            for anti in classified_antiquarks:
                anti_gid, anti_local = self.group_id[anti.operator]
                edge_color = peram_incidence._edge_color(
                    q, anti, isospin_degenerate=isospin_degenerate
                )
                anti_routes.append((
                    q_gid, q_local, anti_gid, anti_local,
                    edge_color, vocabulary.edge_ids[edge_color],
                ))
            routes.append(tuple(anti_routes))
        self.edge_routes = tuple(routes)
        self.refinement_fast_paths = 0
        self.refinement_fallbacks = 0


def _rank_signatures(signatures: Sequence[Tuple]) -> Tuple[int, ...]:
    ranks = {signature: rank for rank, signature in enumerate(sorted(set(signatures)))}
    return tuple(ranks[signature] for signature in signatures)


def _refined_bipartite_orders(
    cells: Sequence[Sequence[Sequence[int]]],
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Return canonical vertex orders when color refinement distinguishes all vertices."""
    sink_count = len(cells)
    source_count = len(cells[0]) if sink_count else 0
    sink_colors = (0,) * sink_count
    source_colors = (0,) * source_count

    for _iteration in range(max(sink_count, source_count) + 1):
        sink_signatures = tuple(
            (
                sink_colors[sink],
                tuple(sorted(
                    (source_colors[source], tuple(sorted(cells[sink][source])))
                    for source in range(source_count)
                )),
            )
            for sink in range(sink_count)
        )
        source_signatures = tuple(
            (
                source_colors[source],
                tuple(sorted(
                    (sink_colors[sink], tuple(sorted(cells[sink][source])))
                    for sink in range(sink_count)
                )),
            )
            for source in range(source_count)
        )
        new_sink_colors = _rank_signatures(sink_signatures)
        new_source_colors = _rank_signatures(source_signatures)
        if (
            len(set(new_sink_colors)) == sink_count
            and len(set(new_source_colors)) == source_count
        ):
            return (
                tuple(sorted(range(sink_count), key=new_sink_colors.__getitem__)),
                tuple(sorted(range(source_count), key=new_source_colors.__getitem__)),
            )
        if new_sink_colors == sink_colors and new_source_colors == source_colors:
            break
        sink_colors, source_colors = new_sink_colors, new_source_colors
    return None


def _operator_color(
    operator: core.Operator,
    isospin_degenerate: bool,
) -> Tuple:
    # Species and exact ordered flavors belong to input interpretation and raw
    # Wick generation.  Candidate colors retain actual contraction factors;
    # flavor classes live on the directed peram edges.
    color = (
        operator.side,
        operator.kind,
        "",
        core.operator_factor_tag(operator),
        operator.gamma,
        tuple(operator.gamma_slots),
        operator.open_slot,
        core.operator_factor_metadata(operator),
    )
    return color if operator.mode == "distillation" else color + (operator.mode,)


def _operator_maps(
    operators: Sequence[core.Operator], isospin_degenerate: bool,
) -> Tuple[Tuple[int, ...], ...]:
    cells: Dict[Tuple, List[int]] = defaultdict(list)
    for index, operator in enumerate(operators):
        cells[_operator_color(operator, isospin_degenerate)].append(index)
    count = math.prod(math.factorial(len(indices)) for indices in cells.values())
    if count > MAX_TRANSPORTERS:
        raise RuntimeError(f"operator exchange orbit ({count}) exceeds the configured limit")
    choices = [tuple(itertools.permutations(indices)) for indices in cells.values()]
    result = []
    for selected in itertools.product(*choices):
        mapping = list(range(len(operators)))
        for indices, permutation in zip(cells.values(), selected):
            for source, target in zip(indices, permutation):
                mapping[source] = target
        result.append(tuple(mapping))
    return tuple(result)


def canonical_factor_graph_key(
    operators: Sequence[core.Operator],
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    term: core.WickTerm,
    classified_quarks: Sequence[core.Field],
    classified_antiquarks: Sequence[core.Field],
    *,
    isospin_degenerate: bool = True,
    max_permutations: int = MAX_TRANSPORTERS,
    vocabulary: Optional[_FactorGraphVocabulary] = None,
    compact: bool = False,
    include_operator_map: bool = True,
    include_key: bool = True,
    context: Optional[_FactorGraphCanonicalizationContext] = None,
) -> Tuple[Optional[object], Tuple[int, ...]]:
    """Return a canonical key for a colored contraction factor graph.

    Raw Wick generation has already enforced the exact u/d flavor pairing.
    Vertices are grouped by the factors visible to the contraction: side,
    operator kind, Gamma/slots, open slot and contraction factors.  Normal
    mode carries its external/momentum labels on the peram endpoints.  Hadron
    species and exact ordered flavors are deliberately absent.  Vertices in
    one group may be permuted.  The edge color retains peram direction, line
    role, endpoint descriptors, flavor class, bare/dummy roles, and field kinds.

    A two-color factor graph first uses exact color refinement to identify a
    unique vertex order; unresolved symmetries fall back to the exact minimum
    over ``S_n x S_m``. General colored graphs use bounded exhaustive group
    permutations. Exceeding the bound is an explicit error rather than an
    incomplete quotient.

    The returned operator map is a canonicalization witness.  It is not a
    separate p/n transporter and does not by itself prove tensor equality.
    ``include_key=False`` skips serialization when a caller needs only that
    witness for the current raw term.
    """
    if compact and vocabulary is None:
        vocabulary = _FactorGraphVocabulary(
            operators, classified_quarks, classified_antiquarks,
            isospin_degenerate=isospin_degenerate,
        )
    if context is None:
        context = _FactorGraphCanonicalizationContext(
            operators, quarks, antiquarks,
            classified_quarks, classified_antiquarks,
            isospin_degenerate=isospin_degenerate,
            vocabulary=vocabulary or _FactorGraphVocabulary(
                operators, classified_quarks, classified_antiquarks,
                isospin_degenerate=isospin_degenerate,
            ),
        )
    ordered_groups = context.ordered_groups
    group_id = context.group_id
    offsets = context.offsets

    edges = []
    for q_index, anti_index in enumerate(term.pairing):
        route = context.edge_routes[q_index][anti_index]
        edges.append(route[:4] + (route[5] if compact else route[4],))

    group_colors = None
    if include_key and not compact:
        group_colors = tuple(color for color, _indices in ordered_groups)

    def make_map(permutations: Sequence[Tuple[int, ...]]) -> Tuple[int, ...]:
        operator_map = list(range(len(operators)))
        for gid, permutation in enumerate(permutations):
            indices = ordered_groups[gid][1]
            for target_local, original_local in enumerate(permutation):
                operator_map[indices[original_local]] = offsets[gid] + target_local
        return tuple(operator_map)

    def mapped_edges(permutations: Sequence[Tuple[int, ...]]) -> Tuple:
        local_targets = []
        for permutation in permutations:
            inverse = [0] * len(permutation)
            for target_local, original_local in enumerate(permutation):
                inverse[original_local] = target_local
            local_targets.append(inverse)
        mapped = []
        for q_gid, q_local, anti_gid, anti_local, color in edges:
            mapped.append((
                offsets[q_gid] + local_targets[q_gid][q_local],
                offsets[anti_gid] + local_targets[anti_gid][anti_local],
                color,
            ))
        return tuple(sorted(mapped))

    # A two-color directed factor graph admits an exact factorial reduction:
    # enumerate one side and sort rows on the other side.  The vertex counts,
    # colors, and field content are entirely determined by the input graph.
    if len(ordered_groups) == 2:
        first_indices = ordered_groups[0][1]
        second_indices = ordered_groups[1][1]
        first_side = operators[first_indices[0]].side
        second_side = operators[second_indices[0]].side
        bipartite = (
            {first_side, second_side} == {"sink", "source"}
            and all(q_gid != anti_gid for q_gid, _ql, anti_gid, _al, _c in edges)
            and all(
                {q_gid, anti_gid} == {0, 1}
                for q_gid, _ql, anti_gid, _al, _c in edges
            )
        )
        if bipartite:
            sink_gid = 0 if first_side == "sink" else 1
            source_gid = 1 - sink_gid
            sink_count = len(ordered_groups[sink_gid][1])
            source_count = len(ordered_groups[source_gid][1])
            total = math.factorial(source_count)
            if total > max_permutations:
                raise RuntimeError(
                    f"factor-graph source automorphism ({total}) exceeds the configured limit"
                )
            cells = [[[] for _ in range(source_count)] for _ in range(sink_count)]
            for q_gid, q_local, anti_gid, anti_local, color in edges:
                if q_gid == sink_gid:
                    sink_local, source_local = q_local, anti_local
                else:
                    sink_local, source_local = anti_local, q_local
                cells[sink_local][source_local].append(color)

            best = None
            best_permutations = None
            refined_orders = _refined_bipartite_orders(cells)
            if refined_orders is not None:
                context.refinement_fast_paths += 1
                sink_order, source_order = refined_orders
                best = tuple(
                    tuple(tuple(sorted(cells[row][column])) for column in source_order)
                    for row in sink_order
                )
                if include_operator_map:
                    permutations = [
                        tuple(range(len(indices)))
                        for _color, indices in ordered_groups
                    ]
                    permutations[sink_gid] = sink_order
                    permutations[source_gid] = source_order
                    best_permutations = tuple(permutations)
            else:
                context.refinement_fallbacks += 1
                for source_order in _permutation_table(source_count):
                    row_values = [
                        tuple(tuple(sorted(cells[row][column])) for column in source_order)
                        for row in range(sink_count)
                    ]
                    sink_order = tuple(sorted(
                        range(sink_count), key=lambda row: (row_values[row], row)
                    ))
                    candidate = tuple(row_values[row] for row in sink_order)
                    if best is None or candidate < best:
                        best = candidate
                        if include_operator_map:
                            permutations = [
                                tuple(range(len(indices)))
                                for _color, indices in ordered_groups
                            ]
                            permutations[sink_gid] = sink_order
                            permutations[source_gid] = source_order
                            best_permutations = tuple(permutations)
            assert best is not None
            operator_map = (
                make_map(best_permutations)
                if best_permutations is not None
                else ()
            )
            if not include_key:
                key = None
            elif compact and vocabulary is not None:
                key = _pack_compact_graph_signature(
                    ordered_groups, vocabulary, best, bipartite=True
                )
            else:
                key = (group_colors, best)
            return key, operator_map

    # Generic colored graph path.  It is exact for the retained colors and
    # refuses to run when the finite automorphism search would be too large.
    permutation_counts = [
        math.factorial(len(indices)) for _color, indices in ordered_groups
    ]
    total = math.prod(permutation_counts)
    if total > max_permutations:
        raise RuntimeError(
            f"factor-graph automorphism ({total}) exceeds the configured limit"
        )
    choices = [
        _permutation_table(len(indices)) for _color, indices in ordered_groups
    ]
    best = None
    best_map = None
    for selected in itertools.product(*choices):
        candidate = mapped_edges(selected)
        if best is None or candidate < best:
            best = candidate
            if include_operator_map:
                best_map = make_map(selected)
    assert best is not None
    if best_map is None:
        best_map = ()
    if not include_key:
        key = None
    elif compact and vocabulary is not None:
        key = _pack_compact_graph_signature(
            ordered_groups, vocabulary, best, bipartite=False
        )
    else:
        key = (group_colors, best)
    return key, best_map


# Backward-compatible private name used by earlier verify_code scripts.
_blind_factor_key = canonical_factor_graph_key


def _compose_operator_maps(
    candidate_to_canonical: Sequence[int],
    reference_to_canonical: Sequence[int],
) -> Tuple[int, ...]:
    """Return the candidate-to-reference map from two canonical maps."""
    inverse = {canonical: reference for reference, canonical in enumerate(reference_to_canonical)}
    return tuple(inverse[candidate_to_canonical[index]] for index in range(len(candidate_to_canonical)))


def _peram_type(q: core.Field, anti: core.Field) -> str:
    return core.peram_direction(q, anti)


def _field_line_maps(
    quarks: Sequence[core.Field], antiquarks: Sequence[core.Field], term: core.WickTerm
) -> Tuple[Dict[int, int], Dict[int, int]]:
    q_lines = {field.index: index for index, field in enumerate(quarks)}
    bar_lines = {
        antiquarks[anti_index].index: q_index
        for q_index, anti_index in enumerate(term.pairing)
    }
    return q_lines, bar_lines


def _line_candidates(
    operators: Sequence[core.Operator],
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    term: core.WickTerm,
    reference: core.WickTerm,
    ref_quarks: Sequence[core.Field],
    ref_antiquarks: Sequence[core.Field],
    operator_map: Sequence[int],
    isospin_degenerate: bool,
    field_flavor_classes: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None,
) -> List[Tuple[int, ...]]:
    candidates: List[Tuple[int, ...]] = []
    if field_flavor_classes is None:
        q_flavors = tuple(model.flavor_class(field.flavor, isospin_degenerate) for field in quarks)
        anti_flavors = tuple(model.flavor_class(field.flavor, isospin_degenerate) for field in antiquarks)
        ref_q_flavors = tuple(model.flavor_class(field.flavor, isospin_degenerate) for field in ref_quarks)
        ref_anti_flavors = tuple(model.flavor_class(field.flavor, isospin_degenerate) for field in ref_antiquarks)
    else:
        q_flavors, anti_flavors = field_flavor_classes
        ref_q_flavors, ref_anti_flavors = q_flavors, anti_flavors
    for q_index, anti_index in enumerate(term.pairing):
        q, anti = quarks[q_index], antiquarks[anti_index]
        options = []
        for ref_q_index, ref_anti_index in enumerate(reference.pairing):
            ref_q, ref_anti = ref_quarks[ref_q_index], ref_antiquarks[ref_anti_index]
            if operator_map[q.operator] != ref_q.operator or operator_map[anti.operator] != ref_anti.operator:
                continue
            if _peram_type(q, anti) != _peram_type(ref_q, ref_anti):
                continue
            if (q.mode, anti.mode, q.endpoint_descriptor, anti.endpoint_descriptor) != (
                ref_q.mode, ref_anti.mode, ref_q.endpoint_descriptor, ref_anti.endpoint_descriptor
            ):
                continue
            if core._line_role(q, anti) != core._line_role(ref_q, ref_anti):
                continue
            if q.bare != ref_q.bare or anti.bare != ref_anti.bare:
                continue
            if (q.side, anti.side, q.kind, anti.kind) != (
                ref_q.side, ref_anti.side, ref_q.kind, ref_anti.kind
            ):
                continue
            if (q_flavors[q_index], anti_flavors[anti_index]) != (
                ref_q_flavors[ref_q_index], ref_anti_flavors[ref_anti_index]
            ):
                continue
            q_operator = operators[q.operator]
            anti_operator = operators[anti.operator]
            if q_operator.kind == "meson" and q.slot != ref_q.slot:
                continue
            if anti_operator.kind == "meson" and anti.slot != ref_anti.slot:
                continue
            options.append(ref_q_index)
        candidates.append(tuple(options))
    return candidates


def _iter_line_maps(candidates: Sequence[Sequence[int]], limit: int) -> Iterator[Tuple[int, ...]]:
    """Enumerate perfect line matchings using most-constrained-first DFS."""
    size = len(candidates)
    if any(not options for options in candidates):
        return
    assignment = [-1] * size
    order = sorted(range(size), key=lambda index: (len(candidates[index]), index))
    emitted = 0

    def visit(depth: int, used: set[int]) -> Iterator[Tuple[int, ...]]:
        nonlocal emitted
        if depth == size:
            emitted += 1
            if emitted > limit:
                raise OverflowError
            yield tuple(assignment)
            return
        line = order[depth]
        for target in candidates[line]:
            if target in used:
                continue
            assignment[line] = target
            yield from visit(depth + 1, used | {target})
            assignment[line] = -1

    yield from visit(0, set())


def _iter_transport_maps(
    operators: Sequence[core.Operator],
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    term: core.WickTerm,
    reference: core.WickTerm,
    *,
    isospin_degenerate: bool,
    operator_maps: Optional[Sequence[Tuple[int, ...]]] = None,
    field_flavor_classes: Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]] = None,
    max_solutions: int = MAX_TRANSPORTERS,
) -> Tuple[List[Tuple[Tuple[int, ...], Tuple[int, ...]]], bool]:
    ref_quarks, ref_antiquarks = quarks, antiquarks
    solutions: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    seen = set()
    truncated = False
    total_operator_maps = operator_maps or _operator_maps(operators, isospin_degenerate)
    for operator_map in total_operator_maps:
        candidates = _line_candidates(
            operators, quarks, antiquarks, term, reference, ref_quarks,
            ref_antiquarks, operator_map, isospin_degenerate,
            field_flavor_classes,
        )
        try:
            for line_map in _iter_line_maps(candidates, max_solutions - len(solutions)):
                key = (operator_map, line_map)
                if key not in seen:
                    seen.add(key)
                    solutions.append(key)
        except OverflowError:
            truncated = True
            break
        if len(solutions) >= max_solutions:
            truncated = True
            break
    return solutions, truncated


def _vvv_signs(
    operators: Sequence[core.Operator],
    fields: Sequence[core.Field],
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    term: core.WickTerm,
    reference: core.WickTerm,
    operator_map: Sequence[int],
    line_map: Sequence[int],
) -> Tuple[int, int]:
    q_lines, bar_lines = _field_line_maps(quarks, antiquarks, term)
    ref_q_lines, ref_bar_lines = _field_line_maps(quarks, antiquarks, reference)
    ref_q_slot = {ref_q_lines[field.index]: field.slot for field in quarks}
    ref_bar_slot = {ref_bar_lines[field.index]: field.slot for field in antiquarks}
    by_operator: Dict[int, List[core.Field]] = defaultdict(list)
    for field in fields:
        by_operator[field.operator].append(field)
    signs = {"sink": 1, "source": 1}
    for op_index, operator in enumerate(operators):
        if operator.kind != "baryon":
            continue
        ordered_fields = sorted(by_operator[op_index], key=lambda item: item.slot)
        if operator.side == "sink":
            positions = [ref_q_slot[line_map[q_lines[field.index]]] for field in ordered_fields]
        else:
            positions = [ref_bar_slot[line_map[bar_lines[field.index]]] for field in ordered_fields]
        signs[operator.side] *= core._permutation_sign(positions)
    return signs["sink"], signs["source"]


def _bare_axis_map(
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    term: core.WickTerm,
    reference: core.WickTerm,
    line_map: Sequence[int],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for q_index, anti_index in enumerate(term.pairing):
        target_line = line_map[q_index]
        target_anti_index = reference.pairing[target_line]
        for field, target in (
            (quarks[q_index], quarks[target_line]),
            (antiquarks[anti_index], antiquarks[target_anti_index]),
        ):
            if field.bare and field.axis and target.axis:
                mapping[field.axis] = target.axis
    return mapping


def _transport_state(
    operators: Sequence[core.Operator],
    fields: Sequence[core.Field],
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    term: core.WickTerm,
    reference: core.WickTerm,
    operator_map: Tuple[int, ...],
    line_map: Tuple[int, ...],
    registry: Mapping[str, core.Matrix],
) -> TransportState:
    inverse = [0] * len(line_map)
    for current, target in enumerate(line_map):
        inverse[target] = current
    line_order = tuple(inverse)
    sink = core._dirac_payload(
        operators, fields, quarks, antiquarks, term, "sink", line_order, registry
    )
    if sink is None:
        raise ValueError("every two-port Gamma must have a registered 4x4 matrix")
    sink_sign, source_sign = _vvv_signs(
        operators, fields, quarks, antiquarks, term, reference, operator_map, line_map
    )
    return TransportState(
        operator_map=operator_map,
        line_map=line_map,
        line_order=line_order,
        sink=sink,
        source=None,
        coefficient=complex(term.sign * sink_sign * source_sign),
        sink_vvv_sign=sink_sign,
        source_vvv_sign=source_sign,
        bare_axis_map=_bare_axis_map(quarks, antiquarks, term, reference, line_map),
    )


def _ensure_source_payload(
    state: TransportState,
    operators: Sequence[core.Operator],
    fields: Sequence[core.Field],
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    term: core.WickTerm,
    registry: Mapping[str, core.Matrix],
) -> Dict[str, object]:
    if state.source is None:
        source = core._dirac_payload(
            operators, fields, quarks, antiquarks, term,
            "source", state.line_order, registry,
        )
        if source is None:
            raise ValueError("every two-port Gamma must have a registered 4x4 matrix")
        state.source = source
    return state.source


def _payload_key(payload: Mapping[str, object]) -> Tuple:
    key = (tuple(payload["axes"]), tuple(payload["signature"]))
    factors = tuple(payload["factors"])
    previous = _PAYLOADS.get(key)
    if previous is not None and tuple(previous["factors"]) != factors:
        raise ValueError("one Gamma signature resolved to different matrices in the same reduction")
    _PAYLOADS[key] = {"axes": key[0], "factors": factors}
    return key


def _concrete_gamma_registry(
    gamma_matrices: Optional[Mapping[str, core.Matrix]],
) -> Dict[str, core.Matrix]:
    registry = core._euclidean_gamma_matrices()
    for name, matrix in (gamma_matrices or {}).items():
        try:
            rows = tuple(tuple(complex(value) for value in row) for row in matrix)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Gamma matrix {name!r} must contain numeric values") from exc
        if not isinstance(name, str) or len(rows) != 4 or any(len(row) != 4 for row in rows):
            raise ValueError(f"Gamma matrix {name!r} must be a concrete 4x4 matrix")
        registry[name] = rows
    return registry


@lru_cache(maxsize=65536)
def _cached_dirac_inner(left: Tuple, right: Tuple) -> complex:
    left_payload = _PAYLOADS[left]
    right_payload = _PAYLOADS[right]
    return core._dirac_tensor_inner(left_payload, right_payload)


_PAYLOADS: Dict[Tuple, Dict[str, object]] = {}


def _inner_payload(left: Mapping[str, object], right: Mapping[str, object]) -> complex:
    left_key = _payload_key(left)
    right_key = _payload_key(right)
    if left_key <= right_key:
        return _cached_dirac_inner(left_key, right_key)
    return _cached_dirac_inner(right_key, left_key).conjugate()


def _state_inner(left: TransportState, right: TransportState) -> complex:
    if left.source is None or right.source is None:
        raise ValueError("source Dirac payloads must be built after sink matching")
    return (
        left.coefficient.conjugate() * right.coefficient
        * _inner_payload(left.sink, right.sink)
        * _inner_payload(left.source, right.source)
    )


def _projective_sink_match(
    reference: Mapping[str, object], candidate: Mapping[str, object], tolerance: float
) -> Tuple[bool, complex, float, complex]:
    norm_ref = _inner_payload(reference, reference).real
    norm_candidate = _inner_payload(candidate, candidate).real
    overlap = _inner_payload(reference, candidate)
    if norm_ref <= tolerance or norm_candidate <= tolerance:
        return False, 0j, max(0.0, norm_candidate), overlap
    coefficient = overlap / norm_ref
    residual = max(0.0, norm_candidate - (coefficient.conjugate() * overlap).real)
    matched = residual <= tolerance * max(norm_candidate, 1.0)
    return matched, coefficient, residual, overlap


def _gram_projection(
    candidate: TransportState,
    basis: Sequence[BasisState],
    *,
    tolerance: float,
    use_sink_only: bool = False,
) -> Tuple[bool, List[complex], float, List[complex]]:
    """Project incrementally onto an existing positive-definite Gram basis."""
    if not basis:
        norm = _inner_payload(candidate.sink, candidate.sink).real
        if not use_sink_only:
            norm *= _inner_payload(candidate.source, candidate.source).real
        norm *= abs(candidate.coefficient) ** 2
        return False, [], max(0.0, norm), []

    def inner_states(left: TransportState, right: TransportState) -> complex:
        value = _inner_payload(left.sink, right.sink)
        if not use_sink_only:
            value *= _inner_payload(left.source, right.source)
        return left.coefficient.conjugate() * right.coefficient * value

    vectors = [entry.state for entry in basis]
    rhs = [inner_states(state, candidate) for state in vectors]
    gram = [
        [inner_states(vectors[row], vectors[column]) for column in range(len(vectors))]
        for row in range(len(vectors))
    ]
    scale = max((abs(gram[i][i].real) for i in range(len(gram))), default=1.0)
    threshold = tolerance * max(scale, 1.0)
    try:
        coefficients = core._solve_linear(gram, rhs, threshold * 1.0e-5)
    except ValueError:
        # The basis is built incrementally, but tiny round-off can still make
        # a large Gram system numerically singular.  Keep the candidate as a
        # new direction rather than claiming a dependency.
        norm = inner_states(candidate, candidate).real
        return False, [], max(0.0, norm), rhs
    norm = inner_states(candidate, candidate).real
    residual = (norm - sum(c.conjugate() * value for c, value in zip(coefficients, rhs))).real
    residual = max(0.0, residual)
    return residual <= threshold, coefficients, residual, rhs


def _streaming_gram_projection(
    candidate: TransportState,
    basis: Sequence[BasisState],
    orthonormal_rows: Sequence[Sequence[complex]],
    *,
    tolerance: float,
) -> Tuple[bool, List[complex], float, Optional[List[complex]]]:
    """Project on a stored orthonormal Gram basis without resolving its Gram matrix."""
    norm = max(0.0, _state_inner(candidate, candidate).real)
    if not basis:
        threshold = tolerance
        if norm <= threshold:
            return True, [], norm, None
        return False, [], norm, [1.0 / math.sqrt(norm)]

    cross = [_state_inner(entry.state, candidate) for entry in basis]
    scale = max((max(0.0, _state_inner(entry.state, entry.state).real) for entry in basis), default=1.0)
    threshold = tolerance * max(scale, 1.0)
    projections = [
        sum(coefficient.conjugate() * value for coefficient, value in zip(row, cross))
        for row in orthonormal_rows
    ]
    residual = max(0.0, norm - sum(abs(value) ** 2 for value in projections))
    if residual <= threshold:
        relation = [
            sum(projections[row_index] * row[column] for row_index, row in enumerate(orthonormal_rows))
            for column in range(len(basis))
        ]
        return True, relation, residual, None

    row = [
        -sum(projections[row_index] * old_row[column]
             for row_index, old_row in enumerate(orthonormal_rows))
        for column in range(len(basis))
    ] + [1.0 + 0.0j]
    scale_row = math.sqrt(residual)
    row = [value / scale_row for value in row]
    return False, [], residual, row


def _append_basis(sink_class: SinkClass, wick_index: int, state: TransportState) -> None:
    dependent, _coefficients, _residual, row = _streaming_gram_projection(
        state, sink_class.basis, sink_class.orthonormal_rows, tolerance=TOLERANCE
    )
    if dependent or row is None:
        raise AssertionError("attempted to append a dependent Dirac state to the basis")
    for old_row in sink_class.orthonormal_rows:
        old_row.append(0j)
    sink_class.basis.append(BasisState(wick_index, state))
    sink_class.orthonormal_rows.append(row)


def _sink_orbit_diagnostics(
    reference_sink: Mapping[str, object], states: Sequence[TransportState], tolerance: float
) -> Dict[str, object]:
    """Apply the requested zero-overlap short circuit, then rank the sink orbit."""
    unique: List[Tuple[int, Mapping[str, object]]] = []
    seen = set()
    for index, state in enumerate(states):
        key = _payload_key(state.sink)
        if key not in seen:
            seen.add(key)
            unique.append((index, state.sink))
    baseline_overlaps = [
        _inner_payload(reference_sink, payload) for _index, payload in unique
    ]
    nonzero_positions = [
        position for position, value in enumerate(baseline_overlaps)
        if abs(value) > tolerance
    ]
    report: Dict[str, object] = {
        "transporter_solution_count": len(states),
        "unique_sink_payload_count": len(unique),
        "baseline_sink_overlaps": [
            {"solution": unique[position][0], "value": [value.real, value.imag]}
            for position, value in enumerate(baseline_overlaps)
        ],
        "sink_rank_checked": False,
        "sink_solution_rank": None,
        "sink_solution_basis": [],
        "sink_baseline_tests": [],
        "zero_overlap_short_circuit": not nonzero_positions,
    }
    if not nonzero_positions:
        return report

    report["sink_rank_checked"] = True
    basis: List[Tuple[int, Mapping[str, object]]] = []
    for unique_position, (solution_index, payload) in enumerate(unique):
        if not basis:
            norm = _inner_payload(payload, payload).real
            if norm > tolerance:
                basis.append((solution_index, payload))
            continue
        proxy = TransportState((), (), (), dict(payload), {"axes": (), "factors": (), "signature": ()}, 1, 1, 1, {})
        basis_states = [
            BasisState(wick_index=old_solution, state=TransportState(
                (), (), (), dict(old_payload), {"axes": (), "factors": (), "signature": ()},
                1, 1, 1, {}
            ))
            for old_solution, old_payload in basis
        ]
        dependent, _coefficients, _residual, _rhs = _gram_projection(
            proxy, basis_states, tolerance=tolerance, use_sink_only=True
        )
        if not dependent:
            basis.append((solution_index, payload))

    basis_tests = []
    for solution_index, payload in basis:
        matched, coefficient, residual, overlap = _projective_sink_match(
            reference_sink, payload, tolerance
        )
        norm_ref = max(_inner_payload(reference_sink, reference_sink).real, tolerance)
        norm_payload = max(_inner_payload(payload, payload).real, tolerance)
        normalized = overlap / math.sqrt(norm_ref * norm_payload)
        basis_tests.append({
            "solution": solution_index,
            "proportional_to_baseline": matched,
            "baseline_scale": [coefficient.real, coefficient.imag],
            "normalized_overlap": [normalized.real, normalized.imag],
            "sink_residual_norm2": residual,
        })
    report["sink_solution_rank"] = len(basis)
    report["sink_solution_basis"] = [index for index, _payload in basis]
    report["sink_baseline_tests"] = basis_tests
    return report


def _state_record(state: TransportState, solution_index: int) -> Dict[str, object]:
    bare_map = state.bare_axis_map
    return {
        "solution": solution_index,
        "operator_map": list(state.operator_map),
        "line_map_candidate_to_reference": list(state.line_map),
        "line_order_candidate_for_reference": list(state.line_order),
        "sink_vvv_sign": state.sink_vvv_sign,
        "source_vvv_sign": state.source_vvv_sign,
        "coefficient": [state.coefficient.real, state.coefficient.imag],
        "bare_axis_map": bare_map,
        "transpose_required": any(source != target for source, target in bare_map.items()),
    }


def _record_for_term(
    wick_index: int, term: core.WickTerm, fields: Sequence[core.Field],
    quarks: Sequence[core.Field], antiquarks: Sequence[core.Field],
) -> Dict[str, object]:
    lines = []
    for q_index, anti_index in enumerate(term.pairing):
        q, anti = quarks[q_index], antiquarks[anti_index]
        lines.append({
            "id": q_index,
            "peram_type": _peram_type(q, anti),
            "peram_class": core.peram_class_label(q, anti),
            "role": core._line_role(q, anti),
            "q": {"operator": q.operator, "slot": q.slot, "flavor": q.flavor,
                  "side": q.side, "bare": q.bare, "axis": q.axis,
                  **core.field_endpoint_metadata(q)},
            "bar": {"operator": anti.operator, "slot": anti.slot, "flavor": anti.flavor,
                   "side": anti.side, "bare": anti.bare, "axis": anti.axis,
                   **core.field_endpoint_metadata(anti)},
        })
    return {"wick_index": wick_index, "wick_sign": term.sign, "pairing": list(term.pairing), "peram_lines": lines}


def _reduce_bucket(
    operators: Sequence[core.Operator],
    fields: Sequence[core.Field],
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
    records: Sequence[CompactWickRecord],
    *,
    isospin_degenerate: bool,
    operator_maps: Sequence[Tuple[int, ...]],
    field_flavor_classes: Tuple[Tuple[str, ...], Tuple[str, ...]],
    classified_quarks: Sequence[core.Field],
    classified_antiquarks: Sequence[core.Field],
    vocabulary: _FactorGraphVocabulary,
    pairing_width: int,
    registry: Mapping[str, core.Matrix],
    detailed: bool,
    counters: Counter,
    relation_callback: Optional[Callable[[Mapping[str, object], TransportState, Sequence[BasisState]], None]] = None,
    relation_term_callback: Optional[
        Callable[[Mapping[str, object], TransportState, Sequence[BasisState], core.WickTerm], None]
    ] = None,
    basis_callback: Optional[
        Callable[[int, core.WickTerm, TransportState, complex, Sequence[core.Operator],
                 Sequence[core.Field], Sequence[core.Field], Sequence[core.Field]], None]
    ] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    families: List[PeramFamily] = []
    independent: List[Dict[str, object]] = []
    dependent: List[Dict[str, object]] = []
    all_records: List[Dict[str, object]] = []
    basis_terms: Dict[int, core.WickTerm] = {}
    for record in records:
        wick_index = record.wick_index
        term = _term_from_compact(record)
        canonical_operator_map = record.operator_map()
        if detailed:
            all_records.append(_record_for_term(wick_index, term, fields, quarks, antiquarks))
        matched_relation = None
        selected_state = None
        selected_class = None
        selected_family = None
        selected_new_row: Optional[List[complex]] = None
        selected_solution_index: Optional[int] = None
        selected_solution_reports = []
        selected_diagnostics = None
        had_mapping = False
        root_transport_states: Optional[List[TransportState]] = None
        root_transport_maps: Optional[List[Tuple[Tuple[int, ...], Tuple[int, ...]]]] = None

        for family in families:
            if canonical_operator_map is None or family.canonical_operator_map is None:
                candidate_operator_maps = operator_maps
            else:
                candidate_operator_maps = (_compose_operator_maps(
                    canonical_operator_map, family.canonical_operator_map,
                ),)
            transport_maps, truncated = _iter_transport_maps(
                operators, quarks, antiquarks, term, family.root_term,
                isospin_degenerate=isospin_degenerate,
                operator_maps=candidate_operator_maps,
                field_flavor_classes=field_flavor_classes,
            )
            counters["transporter_pairs_tested"] += 1
            if truncated:
                raise RuntimeError(
                    f"Wick {wick_index} has more than {MAX_TRANSPORTERS} transporter solutions; "
                    "refusing to silently classify a truncated orbit"
                )
            if not transport_maps:
                continue
            had_mapping = True
            states = [
                _transport_state(
                    operators, fields, quarks, antiquarks, term, family.root_term,
                    operator_map, line_map, registry,
                )
                for operator_map, line_map in transport_maps
            ]
            if family.root_index == records[0].wick_index:
                root_transport_states = states
                root_transport_maps = transport_maps
            counters["transporter_solutions"] += len(states)
            counters["max_transporter_solutions"] = max(
                counters["max_transporter_solutions"], len(states)
            )

            baseline_sink = family.sink_classes[0].reference_sink
            diagnostics = _sink_orbit_diagnostics(baseline_sink, states, TOLERANCE)
            if diagnostics["zero_overlap_short_circuit"]:
                counters["sink_zero_overlap_short_circuits"] += 1
            else:
                counters["sink_rank_checks"] += 1
                counters["sink_rank_directions"] += int(diagnostics["sink_solution_rank"] or 0)

            family_matches: List[Tuple[SinkClass, TransportState, complex, int]] = []
            for solution_index, state in enumerate(states):
                for sink_class in family.sink_classes:
                    matched, alpha, _residual, overlap = _projective_sink_match(
                        sink_class.reference_sink, state.sink, TOLERANCE
                    )
                    if matched:
                        family_matches.append((sink_class, state, alpha, solution_index))
                        break
            if not family_matches:
                continue

            tested_keys = set()
            for sink_class, state, alpha, solution_index in family_matches:
                source = _ensure_source_payload(
                    state, operators, fields, quarks, antiquarks, term, registry
                )
                normalized = TransportState(
                    state.operator_map, state.line_map, state.line_order,
                    sink_class.reference_sink, source,
                    state.coefficient * alpha,
                    state.sink_vvv_sign, state.source_vvv_sign, state.bare_axis_map,
                )
                key = (_payload_key(normalized.source), normalized.coefficient, sink_class.class_id)
                if key in tested_keys:
                    continue
                tested_keys.add(key)
                relation, coefficients, residual, new_row = _streaming_gram_projection(
                    normalized, sink_class.basis, sink_class.orthonormal_rows,
                    tolerance=TOLERANCE,
                )
                counters["source_gram_candidate_sets"] += 1
                if relation:
                    matched_relation = {
                        "wick_index": wick_index,
                        "sink_class": sink_class.class_id,
                        "basis_wick_indices": [entry.wick_index for entry in sink_class.basis],
                        "coefficients": [[value.real, value.imag] for value in coefficients],
                        "residual_norm2": residual,
                        "transporter_solution": solution_index,
                        "transporter_tie_count": sum(
                            1 for other_class, _other_state, _alpha, _other_index in family_matches
                            if other_class.class_id == sink_class.class_id
                        ),
                    }
                    selected_state = normalized
                    selected_class = sink_class
                    selected_family = family
                    selected_new_row = None
                    selected_solution_index = solution_index
                    break
                if selected_state is None:
                    selected_state = normalized
                    selected_class = sink_class
                    selected_family = family
                    selected_new_row = new_row
                    selected_solution_index = solution_index
            if matched_relation is not None:
                selected_diagnostics = diagnostics
                if detailed:
                    selected_solution_reports = [
                        _state_record(state, index) for index, state in enumerate(states)
                    ]
                break

            if family_matches and selected_state is not None and selected_family is family:
                selected_diagnostics = diagnostics
                if detailed:
                    selected_solution_reports = [
                        _state_record(item, index) for index, item in enumerate(states)
                    ]

        if matched_relation is not None:
            if relation_callback is not None:
                if selected_state is None or selected_class is None:
                    raise AssertionError("dependent Wick relation lost its selected transport state")
                relation_callback(matched_relation, selected_state, tuple(selected_class.basis))
            if relation_term_callback is not None:
                if selected_state is None or selected_class is None:
                    raise AssertionError("dependent Wick relation lost its selected transport state")
                relation_term_callback(
                    matched_relation, selected_state, tuple(selected_class.basis), term,
                )
            if detailed:
                dependent.append(matched_relation)
            counters["dependent_wicks"] += 1
            if detailed:
                matched_relation["transporter_solutions"] = selected_solution_reports
                matched_relation["sink_orbit"] = selected_diagnostics
                matched_relation["basis_transports"] = [
                    {
                        "wick_index": entry.wick_index,
                        "operator_map": list(entry.state.operator_map),
                        "line_order_candidate_for_reference": list(entry.state.line_order),
                        "bare_axis_map": dict(entry.state.bare_axis_map),
                        "coefficient": [
                            entry.state.coefficient.real,
                            entry.state.coefficient.imag,
                        ],
                    }
                    for entry in selected_class.basis
                ]
            # Aggregation is attached to the canonical dependency witness only;
            # alternate valid transporters are recorded as one orbit, not
            # counted as extra Wick terms.
            for entry, coefficient in zip(selected_class.basis, coefficients):
                entry.aggregate += coefficient
            continue

        if selected_state is not None and selected_class is not None:
            # An independent source direction in an existing sink class.
            if selected_new_row is None:
                raise AssertionError("selected source state is not independent")
            for old_row in selected_class.orthonormal_rows:
                old_row.append(0j)
            selected_class.basis.append(BasisState(wick_index, selected_state))
            selected_class.orthonormal_rows.append(selected_new_row)
            item = {
                "wick_index": wick_index,
                "sink_class": selected_class.class_id,
                "coefficient": [selected_state.coefficient.real, selected_state.coefficient.imag],
            }
            if detailed:
                item["transporter_solutions"] = selected_solution_reports
                item["sink_orbit"] = selected_diagnostics
                item["selected_transporter_solution"] = selected_solution_index
                if selected_solution_index is not None:
                    item["transport_witness"] = selected_solution_reports[selected_solution_index]
            independent.append(item)
            basis_terms[wick_index] = term
            counters["independent_wicks"] += 1
            continue

        if had_mapping:
            # No transporter lands in an existing sink Dirac class.  Keep one
            # canonical orbit representative and start a new sink class.
            family = next(
                f for f in families
                if f.root_term is not None and f.root_index == records[0].wick_index
            )
            transport_maps = root_transport_maps
            states = root_transport_states
            if transport_maps is None or states is None:
                if canonical_operator_map is None or family.canonical_operator_map is None:
                    fallback_operator_maps = operator_maps
                else:
                    fallback_operator_maps = (_compose_operator_maps(
                        canonical_operator_map, family.canonical_operator_map,
                    ),)
                transport_maps, truncated = _iter_transport_maps(
                    operators, quarks, antiquarks, term, family.root_term,
                    isospin_degenerate=isospin_degenerate,
                    operator_maps=fallback_operator_maps,
                    field_flavor_classes=field_flavor_classes,
                )
                if truncated:
                    raise RuntimeError("transporter orbit exceeded the configured limit")
                states = [
                    _transport_state(
                        operators, fields, quarks, antiquarks, term, family.root_term,
                        operator_map, line_map, registry,
                    )
                    for operator_map, line_map in transport_maps
                ]
            if transport_maps:
                state = states[0]
                _ensure_source_payload(
                    state, operators, fields, quarks, antiquarks, term, registry
                )
                new_class = SinkClass(len(family.sink_classes), state.sink)
                _append_basis(new_class, wick_index, state)
                family.sink_classes.append(new_class)
                item = {"wick_index": wick_index, "sink_class": new_class.class_id,
                        "coefficient": [state.coefficient.real, state.coefficient.imag]}
                if detailed:
                    item["transporter_solutions"] = [
                        _state_record(candidate, index)
                        for index, candidate in enumerate(states)
                    ]
                    item["selected_transporter_solution"] = 0
                    item["transport_witness"] = _state_record(state, 0)
                    item["sink_orbit"] = _sink_orbit_diagnostics(
                        family.sink_classes[0].reference_sink,
                        states, TOLERANCE,
                    )
                independent.append(item)
                basis_terms[wick_index] = term
                counters["independent_wicks"] += 1
                continue

        # The peram incidence key may be a coarse collision (for example a
        # directed VDV slot mismatch). Start a separately normalized family.
        family = PeramFamily(wick_index, term, canonical_operator_map)
        identity_map = tuple(range(len(operators)))
        identity_lines = tuple(range(len(quarks)))
        root_state = _transport_state(
            operators, fields, quarks, antiquarks, term, term,
            identity_map, identity_lines, registry,
        )
        _ensure_source_payload(
            root_state, operators, fields, quarks, antiquarks, term, registry
        )
        sink_class = SinkClass(0, root_state.sink)
        _append_basis(sink_class, wick_index, root_state)
        family.sink_classes.append(sink_class)
        families.append(family)
        item = {"wick_index": wick_index, "sink_class": 0,
                "coefficient": [root_state.coefficient.real, root_state.coefficient.imag]}
        if detailed:
            item["selected_transporter_solution"] = 0
            item["transport_witness"] = _state_record(root_state, 0)
        independent.append(item)
        basis_terms[wick_index] = term
        counters["independent_wicks"] += 1

    aggregate_by_wick: Dict[int, complex] = {}
    for family in families:
        for sink_class in family.sink_classes:
            for entry in sink_class.basis:
                entry.aggregate = complex(entry.aggregate)
                aggregate_by_wick[entry.wick_index] = entry.state.coefficient * entry.aggregate
                if basis_callback is not None:
                    basis_callback(
                        entry.wick_index, basis_terms[entry.wick_index], entry.state,
                        entry.aggregate, operators, fields, quarks, antiquarks,
                    )
    for item in independent:
        wick_index = int(item["wick_index"])
        coefficient = aggregate_by_wick.get(wick_index, complex(item["coefficient"][0], item["coefficient"][1]))
        item["coefficient"] = [coefficient.real, coefficient.imag]
        item.update(_record_for_term(
            wick_index, basis_terms[wick_index], fields, quarks, antiquarks
        ))
    return independent, dependent, all_records


def reduce_case(
    name: str,
    operators: Optional[Sequence[core.Operator]] = None,
    *,
    mode: str = "independent",
    options: Optional[ReductionOptions] = None,
    isospin_degenerate: bool = True,
    gamma_matrices: Optional[Mapping[str, core.Matrix]] = None,
    max_wicks: Optional[int] = None,
    block: bool = False,
    output_path: Optional[str | Path] = None,
    block_options: Optional[Mapping[str, object]] = None,
    _precomputed_wicks: Optional[Sequence[core.WickTerm]] = None,
    _relation_callback: Optional[Callable[[Mapping[str, object], TransportState, Sequence[BasisState]], None]] = None,
    _relation_term_callback: Optional[
        Callable[[Mapping[str, object], TransportState, Sequence[BasisState], core.WickTerm], None]
    ] = None,
    _basis_callback: Optional[
        Callable[[int, core.WickTerm, TransportState, complex, Sequence[core.Operator],
                 Sequence[core.Field], Sequence[core.Field], Sequence[core.Field]], None]
    ] = None,
) -> Dict[str, object]:
    if options is not None:
        if not isinstance(options, ReductionOptions):
            raise TypeError("options must be a ReductionOptions instance")
        isospin_degenerate = options.isospin_degenerate
    if mode not in ("coarse", "independent", "detailed", "block"):
        raise ValueError("mode must be 'coarse', 'independent', 'detailed', or 'block'")
    if operators is None:
        try:
            operators = model.operator_library()[name]
        except KeyError as exc:
            raise ValueError(f"unknown correlator example: {name}") from exc
    operators = tuple(operators)
    operator_mode = core.validate_operator_modes(operators)
    if block or mode == "block":
        if operator_mode != "normal":
            raise ValueError("block reduction is available only for normal operators")
        if output_path is None:
            from .expression_output import default_block_output_path
            output_path = default_block_output_path(name)
        # Keep the reducer's graph/Wick implementation independent from the
        # second-pass block layer.  The output module first materializes the
        # normal independent three tables, then invokes block_reduce on those
        # tables and writes the normal-wick-block document.
        from . import expression_output
        return expression_output.write_output(
            expression_output.split_operators(operators), output_path,
            name=name, mode="block", isospin_degenerate=isospin_degenerate,
            gamma_matrices=gamma_matrices, block_options=block_options,
        )
    _cached_dirac_inner.cache_clear()
    _PAYLOADS.clear()
    started = time.perf_counter()
    fields, quarks, antiquarks = model.fields_for_operators(operators)
    q_flavors = tuple(
        model.flavor_class(field.flavor, isospin_degenerate) for field in quarks
    )
    precomputed_wicks = None if _precomputed_wicks is None else tuple(_precomputed_wicks)
    if mode == "coarse":
        profiles = Counter()
        signs = Counter()
        count = 0
        wick_iter = precomputed_wicks
        if wick_iter is None:
            wick_iter = core.enumerate_wicks(quarks, antiquarks)
        for index, term in enumerate(wick_iter, start=1):
            if max_wicks is not None and index > max_wicks:
                break
            profile, _peram_counts, _routing = model.profile(
                term, quarks, antiquarks, isospin_degenerate,
                quark_flavor_classes=q_flavors,
            )
            profiles[profile] += 1
            signs["+" if term.sign > 0 else "-"] += 1
            count += 1
        elapsed = time.perf_counter() - started
        return {
            "name": name, "mode": mode, "wick_count": count,
            "operator_mode": operator_mode,
            "isospin_symmetric": isospin_degenerate,
            "factor_graph": True,
            "independent_count": None, "dependent_count": None,
            "peram_profile_count": len(profiles),
            "wick_sign_histogram": dict(signs),
            "timing_seconds": {"coarse_total": elapsed, "total": elapsed},
        }

    bucket_started = time.perf_counter()
    identity_operator_map = tuple(range(len(operators)))
    # The generic factor bucket canonicalizes operator vertices itself.  The
    # bucket reducer therefore starts from the identity map and performs its
    # usual line/transporter and Dirac checks afterwards.
    factor_graph_mode = True
    operator_maps = (identity_operator_map,)
    compact_operator_maps = ({index: index for index in range(len(operators))},)
    anti_flavors = tuple(
        model.flavor_class(field.flavor, isospin_degenerate) for field in antiquarks
    )
    field_flavor_classes = (q_flavors, anti_flavors)
    classified_quarks = tuple(
        replace(field, flavor=flavor) for field, flavor in zip(quarks, q_flavors)
    )
    classified_antiquarks = tuple(
        replace(field, flavor=flavor) for field, flavor in zip(antiquarks, anti_flavors)
    )
    vocabulary = _FactorGraphVocabulary(
        operators, classified_quarks, classified_antiquarks,
        isospin_degenerate=isospin_degenerate,
    )
    canonicalization_context = _FactorGraphCanonicalizationContext(
        operators, quarks, antiquarks,
        classified_quarks, classified_antiquarks,
        isospin_degenerate=isospin_degenerate,
        vocabulary=vocabulary,
    )
    pairing_width = _pairing_width(len(antiquarks))
    operator_map_width = _pairing_width(len(operators))
    buckets: Dict[Tuple, CompactWickBucket] = {}
    wick_iter = precomputed_wicks
    if wick_iter is None:
        wick_iter = core.enumerate_wicks(quarks, antiquarks)
    for wick_index, term in enumerate(wick_iter, start=1):
        if max_wicks is not None and wick_index > max_wicks:
            break
        profile, _counts, _routing = model.profile(
            term, quarks, antiquarks, isospin_degenerate,
            quark_flavor_classes=q_flavors,
        )
        if factor_graph_mode:
            peram_key, canonical_operator_map = canonical_factor_graph_key(
                operators, quarks, antiquarks, term,
                classified_quarks, classified_antiquarks,
                isospin_degenerate=isospin_degenerate,
                vocabulary=vocabulary,
                compact=True,
                include_operator_map=True,
                context=canonicalization_context,
            )
        else:
            peram_key = peram_incidence.incidence_key(
                operators, classified_quarks, classified_antiquarks, term,
                isospin_degenerate=False,
                operator_maps=compact_operator_maps,
            )
            canonical_operator_map = None
        if canonical_operator_map is None:
            canonical_operator_map = identity_operator_map
        bucket_key = (profile, peram_key)
        bucket = buckets.get(bucket_key)
        if bucket is None:
            bucket = CompactWickBucket(
                pairing_width, len(term.pairing),
                operator_map_width, len(operators),
            )
            buckets[bucket_key] = bucket
        bucket.append(wick_index, term, canonical_operator_map)
    bucket_elapsed = time.perf_counter() - bucket_started

    registry = _concrete_gamma_registry(gamma_matrices)
    counters = Counter()
    independent_wicks: List[Dict[str, object]] = []
    dependent_relations: List[Dict[str, object]] = []
    graph_records: List[Dict[str, object]] = []
    transporter_started = time.perf_counter()
    for _key, records in buckets.items():
        independent, dependent, all_records = _reduce_bucket(
            operators, fields, quarks, antiquarks, records,
            isospin_degenerate=isospin_degenerate,
            operator_maps=operator_maps,
            field_flavor_classes=field_flavor_classes,
            classified_quarks=classified_quarks,
            classified_antiquarks=classified_antiquarks,
            vocabulary=vocabulary,
            pairing_width=pairing_width,
            registry=registry,
            detailed=(mode == "detailed"),
            counters=counters,
            relation_callback=_relation_callback,
            relation_term_callback=_relation_term_callback,
            basis_callback=_basis_callback,
        )
        independent_wicks.extend(independent)
        if mode == "detailed":
            dependent_relations.extend(dependent)
        if mode == "detailed":
            graph_records.extend(all_records)
    transporter_elapsed = time.perf_counter() - transporter_started
    wick_count = sum(len(values) for values in buckets.values())
    result = {
        "name": name,
        "mode": mode,
        "operator_mode": operator_mode,
        "isospin_symmetric": isospin_degenerate,
        "factor_graph": factor_graph_mode,
        "canonicalization": {
            "refinement_fast_paths": canonicalization_context.refinement_fast_paths,
            "exhaustive_fallbacks": canonicalization_context.refinement_fallbacks,
        },
        "wick_count": wick_count,
        "independent_count": len(independent_wicks),
        "dependent_count": counters["dependent_wicks"],
        "peram_incidence_buckets": len(buckets),
        "independent_wicks": independent_wicks,
        "dependent_relations": dependent_relations if mode == "detailed" else [],
        "graph_records": graph_records if mode == "detailed" else None,
        "multi_solution": {
            "transporter_pairs_tested": counters["transporter_pairs_tested"],
            "transporter_solutions": counters["transporter_solutions"],
            "max_transporter_solutions": counters["max_transporter_solutions"],
            "sink_zero_overlap_short_circuits": counters["sink_zero_overlap_short_circuits"],
            "sink_rank_checks": counters["sink_rank_checks"],
            "sink_rank_directions": counters["sink_rank_directions"],
            "source_gram_candidate_sets": counters["source_gram_candidate_sets"],
        },
        "timing_seconds": {
            "peram_bucket": bucket_elapsed,
            "transporter_sink_source": transporter_elapsed,
            "total": time.perf_counter() - started,
        },
    }
    return result


def reduce_operators(
    operators: Sequence[core.Operator],
    *,
    label: str = "correlator",
    mode: str = "independent",
    options: Optional[ReductionOptions] = None,
    gamma_matrices: Optional[Mapping[str, core.Matrix]] = None,
    max_wicks: Optional[int] = None,
    block: bool = False,
    output_path: Optional[str | Path] = None,
    block_options: Optional[Mapping[str, object]] = None,
    _precomputed_wicks: Optional[Sequence[core.WickTerm]] = None,
    _relation_callback: Optional[Callable[[Mapping[str, object], TransportState, Sequence[BasisState]], None]] = None,
    _relation_term_callback: Optional[
        Callable[[Mapping[str, object], TransportState, Sequence[BasisState], core.WickTerm], None]
    ] = None,
    _basis_callback: Optional[
        Callable[[int, core.WickTerm, TransportState, complex, Sequence[core.Operator],
                 Sequence[core.Field], Sequence[core.Field], Sequence[core.Field]], None]
    ] = None,
) -> Dict[str, object]:
    """Reduce arbitrary operators without a built-in correlator name.

    ``label`` is output metadata only.  The reduction policy is supplied by
    ``options`` and therefore does not branch on ``ppn``, ``ppnn`` or any
    other hadron-system label.  If no policy is supplied, the generic
    species-blind factor-graph policy is used.  Exact operator flavors are
    still consumed by raw Wick generation before the policy is applied.
    """
    if options is None:
        options = ReductionOptions()
    return reduce_case(
        label,
        tuple(operators),
        mode=mode,
        options=options,
        gamma_matrices=gamma_matrices,
        max_wicks=max_wicks,
        block=block,
        output_path=output_path,
        block_options=block_options,
        _precomputed_wicks=_precomputed_wicks,
        _relation_callback=_relation_callback,
        _relation_term_callback=_relation_term_callback,
        _basis_callback=_basis_callback,
    )


def benchmark_modes(
    name: str,
    operators: Optional[Sequence[core.Operator]] = None,
    *,
    isospin_degenerate: bool = True,
) -> Dict[str, object]:
    reports = {}
    wall = {}
    for mode in ("coarse", "independent", "detailed"):
        start = time.perf_counter()
        report = reduce_case(
            name, operators, mode=mode, isospin_degenerate=isospin_degenerate
        )
        wall[mode] = time.perf_counter() - start
        reports[mode] = {
            "operator_mode": report["operator_mode"],
            "wick_count": report["wick_count"],
            "independent_count": report["independent_count"],
            "dependent_count": report["dependent_count"],
            "timing_seconds": report["timing_seconds"],
            "multi_solution": report.get("multi_solution"),
        }
    return {
        "name": name,
        "operator_mode": reports["coarse"]["operator_mode"],
        "isospin_symmetric": isospin_degenerate,
        "mode_reports": reports,
        "mode_wall_seconds": wall,
    }


def run_examples(
    example: str = "all", *, benchmark: bool = True,
    isospin_degenerate: bool = True,
) -> List[Dict[str, object]]:
    library = model.operator_library()
    names = EXAMPLE_ORDER if example == "all" else (example,)
    unknown = [name for name in names if name not in library]
    if unknown:
        raise ValueError(f"unknown examples: {unknown}")
    if benchmark:
        return [
            benchmark_modes(name, library[name], isospin_degenerate=isospin_degenerate)
            for name in names
        ]
    return [
        reduce_case(name, library[name], mode="independent", isospin_degenerate=isospin_degenerate)
        for name in names
    ]


def format_benchmark_table(reports: Sequence[Mapping[str, object]]) -> str:
    rows = [
        "example | Wick | independent (new) | coarse s | independent s | detailed s | max transporters | sink rank checks",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for item in reports:
        modes = item["mode_reports"]
        independent = modes["independent"]
        counts = independent["multi_solution"]
        rows.append(
            f"{item['name']} | {independent['wick_count']} | {independent['independent_count']} | "
            f"{item['mode_wall_seconds']['coarse']:.6f} | "
            f"{item['mode_wall_seconds']['independent']:.6f} | "
            f"{item['mode_wall_seconds']['detailed']:.6f} | "
            f"{counts['max_transporter_solutions']} | {counts['sink_rank_checks']}"
        )
    return "\n".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="all", help="one built-in example or 'all'")
    parser.add_argument("--mode", choices=("coarse", "independent", "detailed", "benchmark"), default="benchmark")
    parser.add_argument("--json", action="store_true", help="print full JSON reports")
    parser.add_argument("--output", type=Path, help="write full report JSON to a file")
    parser.add_argument(
        "--no-isospin-symmetry", action="store_true",
        help="keep u and d peram flavor classes separate",
    )
    args = parser.parse_args(argv)
    isospin_degenerate = not args.no_isospin_symmetry
    if args.mode == "benchmark":
        reports = run_examples(
            args.example, benchmark=True, isospin_degenerate=isospin_degenerate
        )
        print(format_benchmark_table(reports))
        payload = reports
    else:
        library = model.operator_library()
        names = EXAMPLE_ORDER if args.example == "all" else (args.example,)
        unknown = [name for name in names if name not in library]
        if unknown:
            parser.error(f"unknown examples: {unknown}")
        reports = [
            reduce_case(
                name, library[name], mode=args.mode,
                isospin_degenerate=isospin_degenerate,
            )
            for name in names
        ]
        payload = reports
        if args.json:
            print(json.dumps(reports, ensure_ascii=False, indent=2))
        else:
            print("\n".join(format_single_report(report) for report in reports))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_single_report(report: Mapping[str, object]) -> str:
    if report["mode"] == "coarse":
        return f"{report['name']}: {report['wick_count']} Wick, coarse elapsed {report['timing_seconds']['total']:.6f} s"
    return (
        f"{report['name']}: {report['wick_count']} Wick, "
        f"{report['independent_count']} independent, {report['dependent_count']} dependent; "
        f"elapsed {report['timing_seconds']['total']:.6f} s"
    )


if __name__ == "__main__":
    main()
