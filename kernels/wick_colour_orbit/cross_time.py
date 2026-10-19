"""Cross-time source-coefficient fusion for the normal-special path.

This module keeps the compact representation used when every Wick line crosses
between sink and source.  The complete graph path lives in ``general.py``;
both paths share the incidence/spin/S3 primitives in ``fusion.py``.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import permutations, product
from typing import Dict, Mapping, Tuple, Union

from .model import (
    Correlator,
    EndpointResolvedKey,
    EndpointRoute,
    LocalVertex,
    WickMap,
    enumerate_cross_time_wicks,
    parity,
)
from .symmetry import _spin_permutation_phase
from .fusion import (
    COLOUR_BASE, SPIN_BASE, Incidence, _EPSILON_TERMS,
    _fuse_incidence_spin, encode_digits,
)
PackedKey = Union[int, EndpointResolvedKey]
PackedDictionary = Dict[PackedKey, complex]


def _needs_endpoint_resolved_keys(correlator: Correlator) -> bool:
    return len({field.point for field in correlator.source.fields}) > 1


def _endpoint_route(correlator: Correlator, wick: WickMap):
    source_by_sink = [-1] * correlator.line_count
    for source_slot, sink_line in enumerate(wick.source_to_sink):
        source_by_sink[sink_line] = source_slot
    sink_routes = correlator.sink_routes or tuple(
        (0, line) for line in range(correlator.line_count)
    )
    routes = []
    for sink_line, source_slot in enumerate(source_by_sink):
        if source_slot < 0:
            raise ValueError("Wick map does not route every sink field")
        sink_field = correlator.sink.fields[sink_line]
        source_field = correlator.source.fields[source_slot]
        quark, antiquark = (
            (source_field, sink_field) if sink_field.barred
            else (sink_field, source_field)
        )
        routes.append(EndpointRoute(
            sink_routes[sink_line], quark.flavor,
            quark.point, antiquark.point,
        ))
    return tuple(routes)


def _output_key(correlator: Correlator, coordinates: int,
                endpoint_route=None) -> PackedKey:
    if endpoint_route is None:
        return coordinates
    return EndpointResolvedKey(endpoint_route, coordinates)


def _source_incidence(correlator: Correlator, wick: WickMap) -> Tuple[Incidence, int]:
    factors = []
    orientation = 1
    source_slot = 0
    for vertex in correlator.source.vertices:
        mapped = tuple(wick.source_to_sink[source_slot + offset]
                       for offset in range(len(vertex.fields)))
        source_slot += len(vertex.fields)
        if vertex.colour_kind == "epsilon":
            orientation *= parity(mapped)
        factors.append((vertex.colour_kind, tuple(sorted(mapped))))
    return tuple(sorted(factors)), orientation


def _spin_terms_by_vertex(vertices: Tuple[LocalVertex, ...]):
    return tuple(
        tuple((tuple(spins), complex(coefficient) * complex(vertex.phase))
              for spins, coefficient in vertex.spin_terms)
        for vertex in vertices
    )


def _source_transporter_generators(correlator: Correlator):
    vertices = correlator.source.vertices
    sink_fields = correlator.sink.fields
    slot_offsets = []
    offset = 0
    for vertex in vertices:
        slot_offsets.append(offset)
        offset += len(vertex.fields)
    identity = tuple(range(offset))
    identity_sink = tuple(range(len(sink_fields)))
    generators = []

    def add(source_map, sink_map, tensor_phase, spin_phase):
        value = (tuple(source_map), tuple(sink_map),
                 complex(tensor_phase), complex(spin_phase))
        if value[:2] == (identity, identity_sink):
            return
        if not any(previous[:2] == value[:2]
                   and abs(previous[2] - value[2]) <= 1e-12
                   for previous in generators):
            generators.append(value)

    for vertex, slot_offset in zip(vertices, slot_offsets):
        rank = len(vertex.fields)
        if rank > 3:
            continue
        for local_map in permutations(range(rank)):
            if local_map == tuple(range(rank)):
                continue
            if any(vertex.fields[slot] != vertex.fields[local_map[slot]]
                   for slot in range(rank)):
                continue
            spin_phase = _spin_permutation_phase(vertex, local_map)
            if spin_phase is None:
                continue
            colour_phase = (
                parity(local_map) if vertex.colour_kind == "epsilon" else 1
            )
            mapped = list(identity)
            for old_slot, new_slot in enumerate(local_map):
                mapped[slot_offset + old_slot] = slot_offset + new_slot
            add(mapped, identity_sink, colour_phase * spin_phase, spin_phase)

    # Identical factors can exchange as a complete typed local tensor.
    for left_index in range(len(vertices)):
        for right_index in range(left_index + 1, len(vertices)):
            left, right = vertices[left_index], vertices[right_index]
            if (left.fields != right.fields
                    or left.colour_kind != right.colour_kind
                    or left.spin_terms != right.spin_terms
                    or complex(left.phase) != complex(right.phase)):
                continue
            if len(left.fields) != len(right.fields):
                continue
            mapped = list(identity)
            left_offset, right_offset = slot_offsets[left_index], slot_offsets[right_index]
            for slot in range(len(left.fields)):
                mapped[left_offset + slot] = right_offset + slot
                mapped[right_offset + slot] = left_offset + slot
            add(mapped, identity_sink, 1 + 0j, 1 + 0j)

    if correlator.sink_axis_transports is not None:
        for sink_map, sink_tensor_phase in correlator.sink_axis_transports:
            if (len(sink_map) != len(identity_sink)
                    or tuple(sorted(sink_map)) != identity_sink):
                continue
            if any(sink_fields[old_axis] != sink_fields[new_axis]
                   for old_axis, new_axis in enumerate(sink_map)):
                continue
            add(identity, sink_map, sink_tensor_phase, 1 + 0j)
    return tuple(generators), tuple(slot_offsets)


def _source_slot_roles(correlator: Correlator, slot_offsets):
    slot_owner = {}
    for owner, (vertex, offset) in enumerate(zip(
        correlator.source.vertices, slot_offsets
    )):
        for local_slot, field in enumerate(vertex.fields):
            support = tuple(sorted(
                (tuple(spins), complex(coefficient).real, complex(coefficient).imag)
                for spins, coefficient in vertex.spin_terms
            ))
            role = sha256(repr(support).encode("utf-8")).hexdigest()[:16]
            slot_owner[offset + local_slot] = (owner, local_slot, role)
    return slot_owner


def _fast_profile(correlator: Correlator, wick: WickMap, slot_offsets,
                  slot_owner=None):
    source_fields = correlator.source.fields
    sink_fields = correlator.sink.fields
    sink_routes = correlator.sink_routes or tuple(
        (0, line) for line in range(len(sink_fields))
    )
    if slot_owner is None:
        slot_owner = _source_slot_roles(correlator, slot_offsets)
    edges = []
    for source_slot, sink_line in enumerate(wick.source_to_sink):
        source_field = source_fields[source_slot]
        sink_field = sink_fields[sink_line]
        owner, local_slot, role = slot_owner[source_slot]
        sink_owner, sink_slot = sink_routes[sink_line]
        q_endpoint, anti_endpoint = (
            (("source", source_slot, owner, local_slot, source_field.point, role),
             ("sink", sink_line, sink_owner, sink_slot, sink_field.point, "open"))
            if not source_field.barred else
            (("sink", sink_line, sink_owner, sink_slot, sink_field.point, "open"),
             ("source", source_slot, owner, local_slot, source_field.point, role))
        )
        edges.append((source_field.flavor, q_endpoint, anti_endpoint, sink_line))
    route = tuple(
        (line, field.flavor, field.barred, field.point, *sink_routes[line])
        for line, field in enumerate(sink_fields)
    )
    return tuple(sorted(edges)), route


def _coarse_profile(correlator: Correlator, wick: WickMap):
    counts = defaultdict(int)
    for source_slot, sink_line in enumerate(wick.source_to_sink):
        source_field = correlator.source.fields[source_slot]
        sink_field = correlator.sink.fields[sink_line]
        quark, antiquark = ((source_field, sink_field)
                            if not source_field.barred else (sink_field, source_field))
        counts[(quark.flavor, quark.point, antiquark.point)] += 1
    return tuple(sorted(counts.items()))


def _map_fast_profile(profile, source_map, sink_map, correlator, slot_offsets,
                      slot_owner=None):
    source_fields = correlator.source.fields
    sink_fields = correlator.sink.fields
    sink_routes = correlator.sink_routes or tuple(
        (0, line) for line in range(len(sink_fields))
    )
    if slot_owner is None:
        slot_owner = _source_slot_roles(correlator, slot_offsets)
    edges = []
    for _flavor, q_endpoint, anti_endpoint, sink_line in profile[0]:
        source_slot = q_endpoint[1] if q_endpoint[0] == "source" else anti_endpoint[1]
        target_slot = source_map[source_slot]
        source_sink_line = (
            q_endpoint[1] if q_endpoint[0] == "sink" else anti_endpoint[1]
        )
        target_sink_line = sink_map[source_sink_line]
        target_source = source_fields[target_slot]
        sink_field = sink_fields[target_sink_line]
        sink_owner, sink_slot = sink_routes[target_sink_line]
        owner, local_slot, role = slot_owner[target_slot]
        q_endpoint, anti_endpoint = (
            (("source", target_slot, owner, local_slot, target_source.point, role),
             ("sink", target_sink_line, sink_owner, sink_slot, sink_field.point, "open"))
            if not target_source.barred else
            (("sink", target_sink_line, sink_owner, sink_slot, sink_field.point, "open"),
             ("source", target_slot, owner, local_slot, target_source.point, role))
        )
        edges.append((target_source.flavor, q_endpoint, anti_endpoint,
                      target_sink_line))
    route = tuple(sorted(
        (sink_map[line], sink_fields[sink_map[line]].flavor,
         sink_fields[sink_map[line]].barred, sink_fields[sink_map[line]].point,
         *sink_routes[sink_map[line]])
        for line, *_route in profile[1]
    ))
    return tuple(sorted(edges)), route


def _map_fast_incidence(incidence, sink_map):
    return tuple(sorted(
        (kind, tuple(sorted(sink_map[axis] for axis in axes)))
        for kind, axes in incidence
    ))


def _wick_sign_for_map(correlator: Correlator, source_to_sink):
    sink_count = len(correlator.sink.fields)
    fields = correlator.sink.fields + correlator.source.fields
    quarks = defaultdict(list)
    antiquarks = defaultdict(list)
    for index, field in enumerate(fields):
        (antiquarks if field.barred else quarks)[field.flavor].append(index)
    flavors = sorted(quarks)
    ordered_quarks = tuple(index for flavor in flavors for index in quarks[flavor])
    ordered_antiquarks = tuple(index for flavor in flavors for index in antiquarks[flavor])
    pairing_by_quark = {}
    for source_slot, sink_line in enumerate(source_to_sink):
        source_index = sink_count + source_slot
        sink_index = sink_line
        if fields[source_index].barred:
            quark_index, anti_index = sink_index, source_index
        else:
            quark_index, anti_index = source_index, sink_index
        pairing_by_quark[quark_index] = anti_index
    anti_position = {field_index: index
                     for index, field_index in enumerate(ordered_antiquarks)}
    permutation = tuple(anti_position[pairing_by_quark[index]]
                        for index in ordered_quarks)
    return parity(ordered_quarks + ordered_antiquarks) * parity(permutation)


def _transporter_target_map(source_to_sink, generator):
    """Apply a transporter to one Wick map without profile validation."""
    source_map, sink_map = generator[:2]
    target_map = [None] * len(source_to_sink)
    for old_slot, new_slot in enumerate(source_map):
        target_map[new_slot] = sink_map[source_to_sink[old_slot]]
    return tuple(target_map)


def _estimate_transporter_savings(wicks, generators):
    """Estimate reducible Wick maps before constructing validated orbits.

    The estimate follows only the discrete map action.  It deliberately omits
    profile, incidence, and phase checks, which are the expensive part of the
    verified transporter path.  The result is therefore a conservative
    candidate count, not evidence that the transporter is valid.
    """
    all_maps = {wick.source_to_sink for wick in wicks}
    seen = set()
    saved = 0
    for start in all_maps:
        if start in seen:
            continue
        orbit = {start}
        queue = [start]
        while queue:
            current = queue.pop()
            for generator in generators:
                target = _transporter_target_map(current, generator)
                if target in all_maps and target not in orbit:
                    orbit.add(target)
                    queue.append(target)
        seen.update(orbit)
        saved += max(0, len(orbit) - 1)
    return saved


def _auto_use_transporters(wicks, generators):
    """Choose the verified path from a cheap discrete-orbit estimate.

    Small inputs with many generators spend more time validating candidate
    relations than reducing repeated Wick maps.  Requiring at least half of
    the maps to be reducible keeps this policy conservative while preserving
    the useful low-generator cases (including the one-baryon Delta fixture).
    """
    if not generators or not wicks:
        return False
    estimated_saved = _estimate_transporter_savings(wicks, generators)
    if estimated_saved == 0:
        return False
    # A small map set with four or more independent candidates is dominated
    # by transporter validation in practice.  The map-only estimate can
    # overstate savings when profile/incidence checks reject candidates, so
    # keep this conservative cap in addition to the reduction threshold.
    if (len(generators) >= 4
            and (len(wicks) <= 64 or estimated_saved * 2 < len(wicks))):
        return False
    return True


def _source_vertices_are_distinct(correlator: Correlator) -> bool:
    """Cheaply reject inter-vertex transporters with distinct local tensors."""
    vertices = correlator.source.vertices
    if len(vertices) < 2:
        return False
    signatures = {
        (
            vertex.colour_kind,
            tuple((field.flavor, field.barred, field.point)
                  for field in vertex.fields),
            tuple(vertex.spin_terms),
        )
        for vertex in vertices
    }
    return len(signatures) == len(vertices)


def _wick_transporter_orbit(correlator, wick, generators, slot_offsets,
                            wick_by_map, slot_owner=None):
    identity = (
        tuple(range(len(wick.source_to_sink))),
        tuple(range(correlator.line_count)), 1 + 0j, 1 + 0j,
    )
    by_map = {wick.source_to_sink: identity}
    queue = [wick.source_to_sink]
    consistent = True
    while queue:
        current_map = queue.pop()
        current_transform = by_map[current_map]
        for (generator_source_map, generator_sink_map,
             generator_tensor_phase, generator_spin_phase) in generators:
            composed_source_map = tuple(
                generator_source_map[index] for index in current_transform[0]
            )
            composed_sink_map = tuple(
                generator_sink_map[index] for index in current_transform[1]
            )
            target_map = [None] * len(composed_source_map)
            for old_slot, new_slot in enumerate(generator_source_map):
                target_map[new_slot] = generator_sink_map[current_map[old_slot]]
            target_map = tuple(target_map)
            target_transform = (
                composed_source_map, composed_sink_map,
                current_transform[2] * generator_tensor_phase,
                current_transform[3] * generator_spin_phase,
            )
            previous = by_map.get(target_map)
            if previous is None:
                by_map[target_map] = target_transform
                queue.append(target_map)
            elif (previous[0] != target_transform[0]
                  or previous[1] != target_transform[1]
                  or abs(previous[2] - target_transform[2]) > 1e-12
                  or abs(previous[3] - target_transform[3]) > 1e-12):
                consistent = False
    endpoint_resolved = _needs_endpoint_resolved_keys(correlator)
    members = []
    for target_map, (source_map, sink_map, tensor_phase, spin_phase) in by_map.items():
        target_wick = wick_by_map.get(target_map)
        if target_wick is None:
            target_wick = WickMap(
                target_map, _wick_sign_for_map(correlator, target_map),
            )
        phase = tensor_phase * complex(target_wick.sign / wick.sign)
        # The profile/incidence validation below is also all the metadata the
        # accumulator needs for an accepted orbit.  Keep it on the member so a
        # successful transporter does not repeat these calculations while
        # building incidence/spin coefficients.
        members.append((target_wick, source_map, sink_map,
                        tensor_phase, spin_phase, phase, None, None, None, None))
    if slot_owner is None:
        slot_owner = _source_slot_roles(correlator, slot_offsets)
    representative_profile = _fast_profile(
        correlator, wick, slot_offsets, slot_owner,
    )
    representative_incidence, _orientation = _source_incidence(correlator, wick)
    for member_index, (target_wick, source_map, sink_map, _tensor_phase,
                       _spin_phase, _phase, _profile, _incidence,
                       _member_orientation, _cached_route) in enumerate(members):
        if target_wick.source_to_sink == wick.source_to_sink:
            target_profile = representative_profile
            target_incidence = representative_incidence
            target_orientation = _orientation
        else:
            target_profile = _fast_profile(
                correlator, target_wick, slot_offsets, slot_owner,
            )
            target_incidence, target_orientation = _source_incidence(
                correlator, target_wick,
            )
        target_route = (
            _endpoint_route(correlator, target_wick)
            if endpoint_resolved else None
        )
        members[member_index] = (
            target_wick, source_map, sink_map, _tensor_phase,
            _spin_phase, _phase, True, target_incidence,
            target_orientation, target_route,
        )
        if (target_wick.source_to_sink == wick.source_to_sink
                and source_map == identity[0] and sink_map == identity[1]):
            continue
        if _map_fast_profile(
            representative_profile, source_map, sink_map,
            correlator, slot_offsets, slot_owner,
        ) != target_profile:
            consistent = False
            break
        if _map_fast_incidence(representative_incidence, sink_map) != target_incidence:
            consistent = False
            break
    return members, consistent


def _wick_groups(correlator: Correlator, use_transporters: bool | str):
    if use_transporters not in (True, False, "auto"):
        raise ValueError("use_transporters must be true, false, or 'auto'")
    wicks = tuple(enumerate_cross_time_wicks(correlator))
    source_count = len(correlator.source.fields)
    sink_count = correlator.line_count

    def singleton(wick):
        return (wick, [(wick, tuple(range(source_count)), tuple(range(sink_count)),
                        1 + 0j, 1 + 0j, 1 + 0j)])

    if use_transporters == "auto" and len(wicks) <= 36:
        # The full candidate/profile validation has a fixed cost that cannot
        # pay back on a few dozen Wick maps.  Forced profile mode remains the
        # diagnostic and regression path for these same inputs.
        return [singleton(wick) for wick in wicks], {
            "profile_keys": len({_coarse_profile(correlator, wick) for wick in wicks}),
            "transporter_generators": 0,
            "transporter_orbits": 0,
            "transporter_wicks_saved": 0,
            "transporter_stabilizer_fallbacks": 0,
            "transporter_auto_fallbacks": 1,
        }, len(wicks)

    if not use_transporters:
        slot_offsets = []
        offset = 0
        for vertex in correlator.source.vertices:
            slot_offsets.append(offset)
            offset += len(vertex.fields)
        return [singleton(wick) for wick in wicks], {
            "profile_keys": len({_coarse_profile(correlator, wick) for wick in wicks}),
            "transporter_generators": 0,
            "transporter_orbits": 0,
            "transporter_wicks_saved": 0,
            "transporter_stabilizer_fallbacks": 0,
            "transporter_auto_fallbacks": 0,
        }, len(wicks)
    if (use_transporters == "auto"
            and not correlator.sink_axis_transports
            and _source_vertices_are_distinct(correlator)):
        return [singleton(wick) for wick in wicks], {
            "profile_keys": len({_coarse_profile(correlator, wick) for wick in wicks}),
            "transporter_generators": 0,
            "transporter_orbits": 0,
            "transporter_wicks_saved": 0,
            "transporter_stabilizer_fallbacks": 0,
            "transporter_auto_fallbacks": 1,
        }, len(wicks)
    generators, slot_offsets = _source_transporter_generators(correlator)
    if not generators:
        return [singleton(wick) for wick in wicks], {
            "profile_keys": len({_coarse_profile(correlator, wick) for wick in wicks}),
            "transporter_generators": 0,
            "transporter_orbits": 0,
            "transporter_wicks_saved": 0,
            "transporter_stabilizer_fallbacks": 0,
            "transporter_auto_fallbacks": 0,
        }, len(wicks)
    auto_fallback = 0
    if use_transporters == "auto" and not _auto_use_transporters(
        wicks, generators,
    ):
        auto_fallback = 1
        return [singleton(wick) for wick in wicks], {
            "profile_keys": len({_coarse_profile(correlator, wick) for wick in wicks}),
            "transporter_generators": 0,
            "transporter_orbits": 0,
            "transporter_wicks_saved": 0,
            "transporter_stabilizer_fallbacks": 0,
            "transporter_auto_fallbacks": auto_fallback,
        }, len(wicks)
    seen = set()
    groups = []
    slot_owner = _source_slot_roles(correlator, slot_offsets)
    wick_indices = {wick.source_to_sink: index for index, wick in enumerate(wicks)}
    wick_by_map = {wick.source_to_sink: wick for wick in wicks}
    profiles = {_coarse_profile(correlator, wick) for wick in wicks}
    transported_orbits = 0
    saved = 0
    fallbacks = 0
    for wick in wicks:
        if wick.source_to_sink in seen:
            continue
        orbit, consistent = _wick_transporter_orbit(
            correlator, wick, generators, slot_offsets, wick_by_map,
            slot_owner,
        )
        if any(member[0].source_to_sink in seen for member in orbit):
            consistent = False
        orbit = [member for member in orbit
                 if member[0].source_to_sink not in seen]
        seen.update(member[0].source_to_sink for member in orbit)
        if not consistent:
            fallbacks += 1
            groups.extend(singleton(member[0]) for member in orbit)
            continue
        orbit.sort(key=lambda member: wick_indices[member[0].source_to_sink])
        groups.append((orbit[0][0], orbit))
        if len(orbit) > 1:
            transported_orbits += 1
            saved += len(orbit) - 1
    return groups, {
        "profile_keys": len(profiles),
        "transporter_generators": len(generators),
        "transporter_orbits": transported_orbits,
        "transporter_wicks_saved": saved,
        "transporter_stabilizer_fallbacks": fallbacks,
        "transporter_auto_fallbacks": auto_fallback,
    }, len(wicks)


def _relative_prune(values: Mapping[PackedKey, complex],
                    tolerance: float) -> PackedDictionary:
    # Standard single-point output is homogeneous integer packed keys.  Avoid
    # copying and then rescanning the multi-million-entry dictionary; the
    # endpoint-resolved path below still needs independent route scales.
    if not values:
        return {}
    first_key = next(iter(values))
    if not isinstance(first_key, EndpointResolvedKey):
        scale = max((abs(value) for value in values.values()), default=0.0)
        threshold = tolerance * scale
        return {
            key: value for key, value in values.items()
            if abs(value) > threshold
        }
    plain_values = {
        key: value for key, value in values.items()
        if not isinstance(key, EndpointResolvedKey)
    }
    scale = max((abs(value) for value in plain_values.values()), default=0.0)
    threshold = tolerance * scale
    plain = {key: value for key, value in plain_values.items()
             if abs(value) > threshold}
    by_endpoint: Dict[Tuple[EndpointRoute, ...], Dict[EndpointResolvedKey, complex]] = defaultdict(dict)
    for key, value in values.items():
        if isinstance(key, EndpointResolvedKey):
            by_endpoint[key.endpoints][key] = value
    for profile_values in by_endpoint.values():
        profile_scale = max((abs(value) for value in profile_values.values()), default=0.0)
        profile_threshold = tolerance * profile_scale
        plain.update({
            key: value for key, value in profile_values.items()
            if abs(value) > profile_threshold
        })
    return plain


def accumulate_cross_time_direct(
    correlator: Correlator,
    *,
    tolerance: float = 1.0e-12,
    max_entries: int | None = None,
) -> PackedDictionary:
    """Expand all epsilon/delta colour and sparse source-spin terms directly.

    This intentionally simple implementation is useful as a small-fixture
    oracle for the fused path. Integer keys pack one colour and one Dirac
    coordinate for each sink line; multi-point sources use endpoint-resolved
    keys to keep distinct propagator routes separate. Relative pruning uses one
    scale per endpoint route for resolved keys and one global scale otherwise.
    """

    correlator.validate()
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    result: Dict[PackedKey, complex] = defaultdict(complex)
    vertices = correlator.source.vertices
    line_count = correlator.line_count
    spin_multiplier = SPIN_BASE ** line_count
    spin_support = _spin_terms_by_vertex(vertices)
    endpoint_resolved = _needs_endpoint_resolved_keys(correlator)

    for wick in enumerate_cross_time_wicks(correlator):
        endpoint_route = (
            _endpoint_route(correlator, wick) if endpoint_resolved else None
        )
        local_colour_terms = tuple(_colour_terms(vertex) for vertex in vertices)
        for colour_assignment in product(*local_colour_terms):
            colour_digits = [-1] * line_count
            colour_coefficient = complex(wick.sign)
            valid = True
            source_slot = 0
            for vertex, (colours, coefficient) in zip(vertices, colour_assignment):
                colour_coefficient *= coefficient
                for colour, _field in zip(colours, vertex.fields):
                    line = wick.source_to_sink[source_slot]
                    source_slot += 1
                    previous = colour_digits[line]
                    if previous >= 0 and previous != colour:
                        valid = False
                        break
                    colour_digits[line] = colour
                if not valid:
                    break
            if not valid:
                continue
            if any(colour < 0 for colour in colour_digits):
                raise ValueError("source vertices must cover every propagator colour axis")
            colour_code = encode_digits(colour_digits, COLOUR_BASE)

            for spin_assignment in product(*spin_support):
                spin_digits = [-1] * line_count
                coefficient = colour_coefficient
                source_slot = 0
                for vertex, (spins, spin_coefficient) in zip(vertices, spin_assignment):
                    coefficient *= spin_coefficient
                    for spin, _field in zip(spins, vertex.fields):
                        line = wick.source_to_sink[source_slot]
                        source_slot += 1
                        spin_digits[line] = spin
                spin_code = encode_digits(spin_digits, SPIN_BASE)
                key = colour_code * spin_multiplier + spin_code
                result[_output_key(correlator, key, endpoint_route)] += coefficient
                if max_entries is not None and len(result) > max_entries:
                    raise MemoryError("coefficient dictionary exceeded max_entries")

    return _relative_prune(result, tolerance)


def _colour_terms(vertex: LocalVertex):
    if vertex.colour_kind == "epsilon":
        return _EPSILON_TERMS
    return tuple(((colour, colour), 1) for colour in range(COLOUR_BASE))


def accumulate_cross_time_fused(
    correlator: Correlator,
    *,
    tolerance: float = 1.0e-12,
    max_entries: int | None = None,
    use_transporters: bool | str = "auto",
) -> Tuple[PackedDictionary, Dict[str, int]]:
    """Reduce source tensors while retaining endpoint profiles when needed."""

    correlator.validate()
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    vertices = correlator.source.vertices
    line_count = correlator.line_count
    spin_support = _spin_terms_by_vertex(vertices)
    endpoint_resolved = _needs_endpoint_resolved_keys(correlator)
    incidence_spin: Dict[object, Dict[int, complex]] = defaultdict(
        lambda: defaultdict(complex)
    )

    wick_groups, transporter_stats, wick_count = _wick_groups(
        correlator, use_transporters,
    )
    identity_sink_map = tuple(range(line_count))
    endpoint_profiles = set()
    for wick, orbit in wick_groups:
        # Sink tensor phases validate the open-axis route; this reducer emits
        # only source coefficients, so only source spin phases enter here.
        routed = []
        for member in orbit:
            (member_wick, _source_map, sink_map, _tensor_phase,
             spin_phase, _phase) = member[:6]
            if len(member) >= 10 and member[6] is not None:
                member_incidence = member[7]
                member_orientation = member[8]
                endpoint_route = member[9]
            else:
                # Rejected transporter candidates deliberately use
                # the original singleton representation.  Keep their
                # established path and only consume cached metadata for a
                # fully verified orbit.
                endpoint_route = (
                    _endpoint_route(correlator, member_wick)
                    if endpoint_resolved else None
                )
                member_incidence, member_orientation = _source_incidence(
                    correlator, member_wick,
                )
            routed.append((member_wick, sink_map, spin_phase,
                           endpoint_route, member_incidence,
                           member_orientation))
        routed_members = tuple(routed)
        endpoint_profiles.update(member[3] for member in routed_members)
        for spin_assignment in product(*spin_support):
            spin_digits = [-1] * line_count
            coefficient = 1 + 0j
            source_slot = 0
            for vertex, (spins, spin_coefficient) in zip(vertices, spin_assignment):
                coefficient *= spin_coefficient
                for spin, _field in zip(spins, vertex.fields):
                    line = wick.source_to_sink[source_slot]
                    source_slot += 1
                    spin_digits[line] = spin
            spin_code = encode_digits(spin_digits, SPIN_BASE)
            for (member_wick, sink_map, spin_phase, endpoint_route,
                 member_incidence, member_orientation) in routed_members:
                if sink_map == identity_sink_map:
                    mapped_spin_code = spin_code
                else:
                    mapped_spin_digits = [0] * line_count
                    for old_axis, new_axis in enumerate(sink_map):
                        mapped_spin_digits[new_axis] = spin_digits[old_axis]
                    mapped_spin_code = encode_digits(mapped_spin_digits, SPIN_BASE)
                member_coefficient = (
                    member_wick.sign * member_orientation * spin_phase * coefficient
                )
                incidence_key = (
                    (endpoint_route, member_incidence)
                    if endpoint_resolved else member_incidence
                )
                incidence_spin[incidence_key][mapped_spin_code] += member_coefficient

    clean_incidence_spin = {
        profile: {
            spin_code: coefficient
            for spin_code, coefficient in coefficients.items()
            if coefficient != 0
        }
        for profile, coefficients in incidence_spin.items()
    }
    clean_incidence_spin = {
        profile: coefficients
        for profile, coefficients in clean_incidence_spin.items()
        if coefficients
    }

    incidence_spin_by_endpoint: Dict[
        Tuple[EndpointRoute, ...] | None, Dict[Incidence, Dict[int, complex]]
    ]
    if endpoint_resolved:
        incidence_spin_by_endpoint = defaultdict(dict)
        for (endpoint_route, incidence), coefficients in clean_incidence_spin.items():
            incidence_spin_by_endpoint[endpoint_route][incidence] = coefficients
    else:
        incidence_spin_by_endpoint = {None: clean_incidence_spin}

    epsilon_count = sum(vertex.colour_kind == "epsilon" for vertex in vertices)
    result: PackedDictionary = {}
    spin_multiplier = SPIN_BASE ** line_count
    fusion_stats = defaultdict(int)
    for endpoint_route, profile_incidence_spin in incidence_spin_by_endpoint.items():
        for members, spin_coefficients in _fuse_incidence_spin(
            profile_incidence_spin, line_count, epsilon_count, fusion_stats,
        ):
            for colour_code, character in members.items():
                for spin_code, coefficient in spin_coefficients.items():
                    coordinates = colour_code * spin_multiplier + spin_code
                    key = _output_key(correlator, coordinates, endpoint_route)
                    result[key] = character * coefficient
                    if max_entries is not None and len(result) > max_entries:
                        raise MemoryError("coefficient dictionary exceeded max_entries")

    result = _relative_prune(result, tolerance)
    auto_fallbacks = transporter_stats["transporter_auto_fallbacks"]
    stats = {
        "wick_maps": wick_count,
        "profile_keys": transporter_stats["profile_keys"],
        "endpoint_profiles": len(endpoint_profiles),
        "transporter_generators": transporter_stats["transporter_generators"],
        "transporter_orbits": transporter_stats["transporter_orbits"],
        "transporter_wicks_saved": transporter_stats["transporter_wicks_saved"],
        "transporter_stabilizer_fallbacks": transporter_stats[
            "transporter_stabilizer_fallbacks"
        ],
        "transporter_auto_fallbacks": transporter_stats[
            "transporter_auto_fallbacks"
        ],
        "route": "cross-time-fused",
        "key_kind": "packed",
        "fallback_reason": (
            "transporter estimate below validation cost"
            if auto_fallbacks else None
        ),
        "orbit_cache_hits": 0,
        "colour_orbit_enabled": True,
        "incidence_keys": len(clean_incidence_spin),
        "fusion_kernel": "incidence-spin-s3",
        **fusion_stats,
        "dictionary_entries": len(result),
    }
    return result, stats


__all__ = [
    "accumulate_cross_time_direct",
    "accumulate_cross_time_fused",
]
