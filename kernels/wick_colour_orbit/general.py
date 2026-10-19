"""Full factorized Wick graphs with compact peram routes and grouped output."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import isfinite

from .fusion import COLOUR_BASE, _COLOUR_PERMUTATIONS, _fuse_incidence_spin, decode_digits
from .grouping import GroupedWickResult, classify_output_pairing
from .model import Correlator, enumerate_general_wicks, general_field_records, parity


def _general_compact_key_from_encoded_edges(
    edge_route_ids, edge_anti_point_ids, colours, spins,
):
    """Build a compact monomial key from already encoded edge metadata.

    The direct general-fused loop visits the same incidence and route layout
    many times. Keeping the route and point ids numeric lets the hot loop
    combine colour and spin coordinates without rebuilding string-bearing edge
    records or performing dictionary lookups for every term.
    """
    compact = tuple(sorted(
        (
            edge_route_ids[edge],
            colours[2 * edge],
            spins[2 * edge],
            edge_anti_point_ids[edge],
            colours[2 * edge + 1],
            spins[2 * edge + 1],
        )
        for edge in range(len(edge_route_ids))
    ))
    return (), compact


def _general_expand_compact_key(key, prefixes, points):
    """Expand compact integer coordinates into an exact monomial key."""
    open_indices, compact = key
    factors = tuple(
        (
            prefixes[route_id][0],
            prefixes[route_id][1],
            q_colour,
            q_spin,
            points[anti_point_id],
            anti_colour,
            anti_spin,
        )
        for route_id, q_colour, q_spin, anti_point_id,
        anti_colour, anti_spin in compact
    )
    return tuple(open_indices), factors


def _general_compact_colour_transform(key, permutation):
    """Apply global colour relabeling to integer compact coordinates."""
    open_indices, compact = key
    return (
        open_indices,
        tuple(sorted(
            (route_id, permutation[q_colour], q_spin, anti_point_id,
             permutation[anti_colour], anti_spin)
            for route_id, q_colour, q_spin, anti_point_id,
            anti_colour, anti_spin in compact
        )),
    )


@lru_cache(maxsize=8192)
def _general_compact_colour_orbit(key, epsilon_count):
    """Return one compact colour orbit without constructing full monomials."""
    members = {}
    for permutation in _COLOUR_PERMUTATIONS:
        mapped = _general_compact_colour_transform(key, permutation)
        members.setdefault(mapped, set()).add(
            parity(permutation) ** epsilon_count
        )
    representative = min(members)
    return representative, members, len(members)


def _expand_general_compact_values(
    compact_values, prefixes, points, epsilon_count, *, tolerance,
    max_entries, use_colour_orbits,
):
    colour_orbits = 0
    colour_order = 1
    orbit_validated = False
    result_compact = compact_values
    validate_colour_orbits = use_colour_orbits is True
    if validate_colour_orbits and compact_values:
        orbit_data = {}
        valid = True
        checked = set()
        for key, value in compact_values.items():
            if key in checked:
                continue
            representative, members, orbit_size = _general_compact_colour_orbit(
                key, epsilon_count,
            )
            orbit_data.setdefault(representative, (members, orbit_size))
            checked.update(members)
            for member, phases in members.items():
                expected = next(iter(phases)) * value
                observed = compact_values.get(member, 0j)
                if len(phases) != 1 or abs(observed - expected) > tolerance:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            result_compact = {}
            for representative, (members, _orbit_size) in orbit_data.items():
                value = compact_values.get(representative, 0j)
                if not value:
                    valid = False
                    break
                for member, phases in members.items():
                    result_compact[member] = next(iter(phases)) * value
                colour_orbits += 1
            if valid:
                colour_order = len(_COLOUR_PERMUTATIONS)
                orbit_validated = True

    if max_entries is not None and len(result_compact) > max_entries:
        raise MemoryError("coefficient dictionary exceeded max_entries")
    result = defaultdict(complex)
    for compact_key, value in result_compact.items():
        result[_general_expand_compact_key(compact_key, prefixes, points)] += value
    full_result = dict(result)
    if max_entries is not None and len(full_result) > max_entries:
        raise MemoryError("coefficient dictionary exceeded max_entries")
    nonzero_entries = sum(abs(value) > tolerance for value in full_result.values())
    return full_result, {
        "colour_symmetry_order": colour_order,
        "colour_orbits": colour_orbits,
        "colour_orbit_validated": int(orbit_validated),
        "dictionary_entries": nonzero_entries,
        "fallback_reason": (
            "colour orbit validation failed"
            if validate_colour_orbits and compact_values and not orbit_validated
            else None
        ),
    }


def _general_field_profile(records, pairing, isospin_degenerate_perams):
    profile = []
    for quark_index, anti_index in pairing.pairs:
        q_side, q_vertex, q_slot, quark = records[quark_index]
        anti_side, anti_vertex, anti_slot, antiquark = records[anti_index]
        q_endpoint = "sin" if q_side == "sink" else "sou"
        anti_endpoint = "sin" if anti_side == "sink" else "sou"
        flavor = ("ud" if isospin_degenerate_perams
                  and quark.flavor in {"u", "d"} else quark.flavor)
        profile.append((
            flavor, f"{q_endpoint}_{anti_endpoint}",
            quark.point, antiquark.point,
            q_vertex, q_slot, anti_vertex, anti_slot,
        ))
    return tuple(sorted(profile))


def accumulate_general_fused(
    correlator: Correlator,
    *,
    normal_ordered: bool = False,
    isospin_degenerate_perams: bool = True,
    tolerance: float = 1.0e-12,
    max_entries: int | None = None,
    use_colour_orbits: bool | str = "auto",
    use_transporters: bool | str = "auto",
    wick_scope: str = "complete",
    group_by_connectivity: bool = False,
):
    """Fuse complete factorized graphs through the shared incidence/S3 kernel.

    The full-monomial key and optional GroupedWickResult are unchanged.
    ``use_transporters`` is accepted for call compatibility; all values use
    this kernel. No sparse backend or per-Wick vector conversion is invoked.
    ``use_colour_orbits=True`` additionally checks the final colour symmetry.
    """
    correlator.validate_general()
    if tolerance < 0 or not isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    if type(isospin_degenerate_perams) is not bool:
        raise ValueError("isospin_degenerate_perams must be true or false")
    if type(normal_ordered) is not bool:
        raise ValueError("normal_ordered must be true or false")
    if use_colour_orbits not in (True, False, "auto"):
        raise ValueError("use_colour_orbits must be true, false, or 'auto'")
    if use_transporters not in (True, False, "auto"):
        raise ValueError("use_transporters must be true, false, or 'auto'")
    if wick_scope not in {"complete", "cross_time"}:
        raise ValueError("wick_scope must be 'complete' or 'cross_time'")
    if type(group_by_connectivity) is not bool:
        raise ValueError("group_by_connectivity must be true or false")

    records = general_field_records(correlator)
    vertices = correlator.sink.vertices + correlator.source.vertices
    prefixes = tuple(sorted({
        (
            "ud" if isospin_degenerate_perams and field.flavor in {"u", "d"}
            else field.flavor,
            field.point,
        )
        for _side, _vertex, _slot, field in records
        if not field.barred
    }))
    points = tuple(sorted({field.point for *_owner, field in records}))
    route_ids = {prefix: index for index, prefix in enumerate(prefixes)}
    point_ids = {point: index for index, point in enumerate(points)}
    vertex_offsets = []
    offset = 0
    for vertex in vertices:
        vertex_offsets.append(offset)
        offset += len(vertex.fields)
    # The direct path first accumulates Wick/profile/incidence spin records.
    # Colour expansion and final key construction happen only after all Wick
    # contributions sharing the same typed route and incidence are merged.
    compact_values = defaultdict(complex)
    compact_groups = defaultdict(lambda: defaultdict(complex))
    compact_group_total = 0
    route_incidence_spin = defaultdict(
        lambda: defaultdict(lambda: defaultdict(complex))
    )
    incidence_classes = set()
    profiles = set()
    endpoint_types = set()
    field_owners = tuple((side, vertex) for side, vertex, _slot, _field in records)
    wick_counts_by_signature = defaultdict(int)
    wick_count = 0
    epsilon_count = sum(vertex.colour_kind == "epsilon" for vertex in vertices)
    compact_orbit_cache_before = _general_compact_colour_orbit.cache_info()
    spin_support = tuple(
        tuple((tuple(spins), complex(coefficient) * complex(vertex.phase))
              for spins, coefficient in vertex.spin_terms)
        for vertex in vertices
    )

    for pairing in enumerate_general_wicks(
        correlator, normal_ordered=normal_ordered, wick_scope=wick_scope,
    ):
        wick_count += 1
        output_signature = classify_output_pairing(pairing.pairs, field_owners)
        wick_counts_by_signature[output_signature] += 1
        edge_axes = {}
        edge_fields = []
        for edge, (quark_index, anti_index) in enumerate(pairing.pairs):
            edge_axes[quark_index] = 2 * edge
            edge_axes[anti_index] = 2 * edge + 1
            edge_fields.append((quark_index, anti_index))
            q_side = records[quark_index][0]
            anti_side = records[anti_index][0]
            q_endpoint = "sin" if q_side == "sink" else "sou"
            anti_endpoint = "sin" if anti_side == "sink" else "sou"
            endpoint_types.add((
                q_endpoint, anti_endpoint,
            ))

        edge_route_ids = []
        edge_anti_point_ids = []
        for quark_index, anti_index in edge_fields:
            _q_side, _q_vertex, _q_slot, quark = records[quark_index]
            _a_side, _a_vertex, _a_slot, antiquark = records[anti_index]
            flavor = ("ud" if isospin_degenerate_perams
                      and quark.flavor in {"u", "d"} else quark.flavor)
            edge_route_ids.append(route_ids[(flavor, quark.point)])
            edge_anti_point_ids.append(point_ids[antiquark.point])
        typed_profile = _general_field_profile(
            records, pairing, isospin_degenerate_perams,
        )
        profiles.add(typed_profile)

        incidence = []
        orientation = 1
        for vertex, vertex_offset in zip(vertices, vertex_offsets):
            axes = tuple(edge_axes[vertex_offset + slot]
                         for slot in range(len(vertex.fields)))
            if vertex.colour_kind == "epsilon":
                orientation *= parity(axes)
            incidence.append((vertex.colour_kind, tuple(sorted(axes))))
        incidence = tuple(sorted(incidence))
        # Keep the route profile with the incidence class.  It preserves
        # endpoint/point information while allowing equivalent Wick records
        # to share the same spin accumulation.
        route_key = (
            output_signature,
            typed_profile,
            tuple(edge_route_ids),
            tuple(edge_anti_point_ids),
        )
        spin_accumulator = route_incidence_spin[route_key][incidence]
        incidence_classes.add((typed_profile, incidence))
        for spin_assignment in product(*spin_support):
            edge_spins = [None] * (2 * len(edge_fields))
            coefficient = complex(pairing.sign * orientation)
            for vertex, (spins, spin_coefficient), vertex_offset in zip(
                vertices, spin_assignment, vertex_offsets,
            ):
                coefficient *= spin_coefficient
                for local_slot, spin in enumerate(spins):
                    edge_spins[edge_axes[vertex_offset + local_slot]] = spin
            spin_accumulator[tuple(edge_spins)] += coefficient

    # Common incidence + spin + S3 stage for the general graph.  Route and
    # output signatures remain outside the fusion kernel, so only equivalent
    # propagator profiles can share colour representatives.
    fusion_stats = defaultdict(int)
    s3_route_classes = 0
    for route_key, profile_incidence_spin in route_incidence_spin.items():
        output_signature, _typed_profile, edge_route_ids, edge_anti_point_ids = route_key
        use_s3 = (
            use_colour_orbits is True
            or (use_colour_orbits == "auto"
                and any(len(spins) > 1 for spins in profile_incidence_spin.values()))
        )
        s3_route_classes += use_s3
        target_values = (
            compact_groups[output_signature]
            if group_by_connectivity else compact_values
        )
        for members, spin_terms in _fuse_incidence_spin(
            profile_incidence_spin,
            2 * len(edge_route_ids),
            epsilon_count,
            fusion_stats,
            use_colour_orbits=use_s3,
        ):
            for colour_code, character in members.items():
                colours = decode_digits(
                    colour_code, 2 * len(edge_route_ids), COLOUR_BASE,
                )
                for edge_spins, coefficient in spin_terms.items():
                    compact_key = _general_compact_key_from_encoded_edges(
                        edge_route_ids, edge_anti_point_ids, colours, edge_spins,
                    )
                    if (group_by_connectivity and max_entries is not None
                            and compact_key not in target_values):
                        compact_group_total += 1
                        if compact_group_total > max_entries:
                            raise MemoryError("coefficient dictionary exceeded max_entries")
                    target_values[compact_key] += character * coefficient

    if group_by_connectivity:
        to_expand = compact_groups
    else:
        to_expand = {None: {key: value for key, value in compact_values.items()
                           if abs(value) > tolerance}}
    output_groups = {}
    group_stats = {}
    expanded_total = 0
    for signature, compact in to_expand.items():
        expanded, colour_stats = _expand_general_compact_values(
            compact, prefixes, points, epsilon_count,
            tolerance=tolerance, max_entries=max_entries,
            use_colour_orbits=use_colour_orbits,
        )
        output_groups[signature] = expanded
        group_stats[signature] = colour_stats
        expanded_total += colour_stats["dictionary_entries"]
        if max_entries is not None and expanded_total > max_entries:
            raise MemoryError("coefficient dictionary exceeded max_entries")
    if group_by_connectivity:
        result = GroupedWickResult(
            output_groups, tolerance=tolerance, filter_groups=True,
        )
    else:
        result = {key: value for key, value in output_groups[None].items()
                  if abs(value) > tolerance}
    stats = {
        "wick_maps": wick_count,
        "peram_profiles": len(profiles),
        "incidence_keys": len(incidence_classes),
        "endpoint_types": len(endpoint_types),
        "transporter_generators": 0,
        "validated_transporters": 0,
        "transporter_orbits": 0,
        "transporter_fallback_relations": 0,
        "colour_symmetry_order": (
            6 if s3_route_classes else max(
                (item["colour_symmetry_order"] for item in group_stats.values()),
                default=1,
            )
        ),
        "colour_orbit_validated": int(bool(group_stats) and all(
            item["colour_orbit_validated"] for item in group_stats.values()
        )),
        "dictionary_entries": expanded_total,
        "route": "general-fused",
        "key_kind": "grouped-full-monomial" if group_by_connectivity else "full-monomial",
        "fallback_reason": next((item["fallback_reason"]
                                  for item in group_stats.values()
                                  if item["fallback_reason"]), None),
        "orbit_cache_hits": (_general_compact_colour_orbit.cache_info().hits
                             - compact_orbit_cache_before.hits),
        "colour_orbit_enabled": bool(s3_route_classes) or use_colour_orbits is True,
        "colour_orbit_policy": (
            "forced-validation" if use_colour_orbits is True
            else "auto-skip-full-output" if use_colour_orbits == "auto"
            else "disabled"
        ),
        "algorithm": "general-fused-incidence",
        "packing": "generalized-packed-coordinates",
        "fusion_kernel": "incidence-spin-s3",
        "fusion_route_classes": len(route_incidence_spin),
        "fusion_s3_route_classes": s3_route_classes,
        "wick_scope": wick_scope,
        "colour_orbits": 0,
        "incidence_spin_pairs": 0,
        "exact_colour_codes": 0,
        "representative_incidence_checks": 0,
        "reconstructed_colour_codes": 0,
        "zero_stabilizer_orbits": 0,
        **fusion_stats,
    }
    if group_by_connectivity:
        stats["connectivity_groups"] = {
            signature: {"wick_maps": wick_counts_by_signature[signature],
                        "dictionary_entries": group_stats[signature]["dictionary_entries"]}
            for signature in output_groups
        }
        stats["topology_signatures"] = "not computed on direct incidence path"
        stats["topology_signature_counts"] = None
    return result, stats


__all__ = ["accumulate_general_fused"]
