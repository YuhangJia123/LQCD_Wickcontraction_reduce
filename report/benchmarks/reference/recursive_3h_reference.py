"""Sparse antisymmetric-product recursion for the relativistic 3H fixture.

This implements the paper's canonical antisymmetric-tuple recursion on the
fixture's sparse local source tensors, then expands the final tensor into the
ordered coordinate dictionary used by the project's UCA reference benchmark.
It does not implement the paper's gauge-field-dependent, sink-pruned Lambda
operation lists; the reported update count is therefore a local recursion
count, not the published N_Lambda.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations, product

from kernels.wick_colour_orbit.examples import make_pnn
from kernels.wick_colour_orbit.model import Correlator, LocalVertex, parity
from kernels.wick_colour_orbit.fusion import _EPSILON_TERMS, encode_digits


def _local_antisymmetric_terms(vertex: LocalVertex, flavor_order: tuple[str, ...]):
    """Return one baryon block in canonical, flavor-grouped spin-color axes."""
    by_flavor: dict[str, list[int]] = {flavor: [] for flavor in flavor_order}
    slot_flavors = tuple(field.flavor for field in vertex.fields)
    for slot, flavor in enumerate(slot_flavors):
        by_flavor[flavor].append(slot)

    if vertex.colour_kind == "epsilon":
        colour_terms = _EPSILON_TERMS
    elif vertex.colour_kind == "delta":
        colour_terms = tuple(((colour, colour), 1) for colour in range(3))
    else:
        raise ValueError(f"unsupported colour tensor: {vertex.colour_kind}")

    terms: dict[tuple[tuple[int, ...], ...], complex] = defaultdict(complex)
    for colours, colour_sign in colour_terms:
        for spins, spin_coefficient in vertex.spin_terms:
            grouped = []
            coefficient = complex(vertex.phase) * colour_sign * complex(spin_coefficient)
            for flavor in flavor_order:
                slots = by_flavor[flavor]
                values = tuple(colours[slot] * 4 + spins[slot] for slot in slots)
                if len(set(values)) != len(values):
                    coefficient = 0j
                    break
                inversions = sum(
                    values[left] > values[right]
                    for left in range(len(values))
                    for right in range(left + 1, len(values))
                )
                if inversions & 1:
                    coefficient = -coefficient
                grouped.append(tuple(sorted(values)))
            if coefficient:
                terms[tuple(grouped)] += coefficient

    return {key: value for key, value in terms.items() if value}


def _merge_sorted(left: tuple[int, ...], right: tuple[int, ...]):
    """Merge two canonical antisymmetric tuples and return wedge sign."""
    if set(left).intersection(right):
        return None
    sign = -1 if sum(a > b for a in left for b in right) & 1 else 1
    merged = tuple(sorted((*left, *right)))
    return merged, sign


def build_recursive_tensor(correlator: Correlator):
    """Recursively wedge sparse source baryon tensors by quark flavor."""
    correlator.validate()
    if any(vertex.colour_kind != "epsilon" for vertex in correlator.source.vertices):
        raise ValueError("the 3H recursive reference expects baryon epsilon vertices")

    flavor_order = tuple(sorted({field.flavor for field in correlator.sink.fields}))
    source_vertices = correlator.source.vertices
    tensor: dict[tuple[tuple[int, ...], ...], complex] = {
        tuple(() for _ in flavor_order): 1 + 0j
    }
    intermediate_components = []
    candidate_updates = 0
    valid_updates = 0

    for vertex in source_vertices:
        local_terms = _local_antisymmetric_terms(vertex, flavor_order)
        next_tensor: dict[tuple[tuple[int, ...], ...], complex] = defaultdict(complex)
        candidate_updates += len(tensor) * len(local_terms)
        for old_key, old_value in tensor.items():
            for local_key, local_value in local_terms.items():
                merged_groups = []
                wedge_sign = 1
                for old_group, local_group in zip(old_key, local_key):
                    merged = _merge_sorted(old_group, local_group)
                    if merged is None:
                        break
                    group, sign = merged
                    merged_groups.append(group)
                    wedge_sign *= sign
                else:
                    valid_updates += 1
                    next_tensor[tuple(merged_groups)] += old_value * local_value * wedge_sign
        tensor = {key: value for key, value in next_tensor.items() if value}
        intermediate_components.append({
            "vertex": vertex.name,
            "local_components": len(local_terms),
            "recursive_components": len(tensor),
        })

    return tensor, {
        "flavor_order": flavor_order,
        "intermediate_components": intermediate_components,
        "candidate_updates": candidate_updates,
        "valid_updates": valid_updates,
    }


def _ordered_group_permutations(canonical: tuple[int, ...]):
    if not canonical:
        return (((), 1),)
    rows = []
    for values in permutations(canonical):
        positions = tuple(canonical.index(value) for value in values)
        rows.append((values, parity(positions)))
    return tuple(rows)


def expand_recursive_tensor(
    tensor: dict[tuple[tuple[int, ...], ...], complex],
    correlator: Correlator,
) -> dict[int, complex]:
    """Expand canonical source tensor components to ordered sink coordinates."""
    return dict(iter_expanded_recursive_tensor(tensor, correlator))


def iter_expanded_recursive_tensor(
    tensor: dict[tuple[tuple[int, ...], ...], complex],
    correlator: Correlator,
):
    """Yield expanded ordered coordinate rows without retaining a second copy."""
    flavors = tuple(sorted({field.flavor for field in correlator.sink.fields}))
    sink_slots = {
        flavor: tuple(
            index for index, field in enumerate(correlator.sink.fields)
            if field.flavor == flavor
        )
        for flavor in flavors
    }
    grouped_sink_order = tuple(
        index for flavor in flavors for index in sink_slots[flavor]
    )
    grouped_source_order = tuple(
        correlator.line_count + index
        for flavor in flavors
        for index, field in enumerate(correlator.source.fields)
        if field.flavor == flavor
    )
    global_order_sign = parity((*grouped_sink_order, *grouped_source_order))
    spin_multiplier = 4 ** correlator.line_count
    for canonical_groups, base_coefficient in tensor.items():
        ordered_options = tuple(
            _ordered_group_permutations(group) for group in canonical_groups
        )
        for group_assignments in product(*ordered_options):
            states = [-1] * correlator.line_count
            coefficient = base_coefficient * global_order_sign
            for flavor, (values, sign) in zip(flavors, group_assignments):
                coefficient *= sign
                for sink_slot, value in zip(sink_slots[flavor], values):
                    states[sink_slot] = value
            if coefficient == 0:
                continue
            colours = tuple(value // 4 for value in states)
            spins = tuple(value % 4 for value in states)
            key = encode_digits(colours, 3) * spin_multiplier + encode_digits(spins, 4)
            yield key, coefficient


def accumulate_recursive(correlator: Correlator):
    tensor, stats = build_recursive_tensor(correlator)
    output = expand_recursive_tensor(tensor, correlator)
    stats = {
        **stats,
        "canonical_tensor_components": len(tensor),
        "expanded_dictionary_entries": len(output),
        "paper_reference": {
            "N_lambda": 381978,
            "N_L": 1311,
            "N_list": 3775680,
            "N_contr": 471960,
            "scope": "published 3H relativistic (0,0,1) calculation; not local update count",
        },
    }
    return output, stats


def make_fixture() -> Correlator:
    return make_pnn()


__all__ = [
    "accumulate_recursive",
    "build_recursive_tensor",
    "expand_recursive_tensor",
    "iter_expanded_recursive_tensor",
    "make_fixture",
]
