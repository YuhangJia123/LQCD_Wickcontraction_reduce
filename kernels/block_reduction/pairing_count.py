"""Conditional Burnside counts for a complete typed pairing family.

This is a combinatorial layout filter, not a tensor identity.  The caller
must separately certify that its target equals the declared full family and
that every supplied endpoint action is a signed, axis-aware tensor symmetry.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import factorial, prod
from operator import index as integer_index
from typing import Hashable, Iterable, Optional, Sequence, Tuple


Permutation = Tuple[int, ...]  # Images of old slot positions.
EndpointAction = Tuple[Permutation, Permutation]  # Sink, source.


@dataclass(frozen=True)
class PairingOrbitCount:
    orbit_count: int
    pairing_count: int
    exchange_order: int
    layout_group_order: int
    full_family_verified: bool
    fixed_point_sum: int


def _permutation(value: Sequence[int], size: int) -> Permutation:
    try:
        result = tuple(integer_index(item) for item in value)
    except TypeError as error:
        raise ValueError("endpoint action must be a permutation") from error
    if len(result) != size or sorted(result) != list(range(size)):
        raise ValueError("endpoint action must permute every slot exactly once")
    return result


def _compose(left: Permutation, right: Permutation) -> Permutation:
    """Apply right first, then left, with old-to-new permutation notation."""
    return tuple(left[i] for i in right)


def _closure(generators: Sequence[EndpointAction], size: int, limit: int):
    identity = tuple(range(size))
    start = (identity, identity)
    found = {start}
    pending = [start]
    while pending:
        current = pending.pop()
        for generator in generators:
            image = (_compose(generator[0], current[0]),
                     _compose(generator[1], current[1]))
            if image not in found:
                found.add(image)
                if len(found) > limit:
                    raise ValueError("endpoint group exceeds max_group_order")
                pending.append(image)
    return found


def _cycles(positions: Iterable[int], permutation: Permutation) -> Counter:
    positions = set(positions)
    remaining = set(positions)
    lengths = Counter()
    while remaining:
        start = next(iter(remaining))
        cursor = start
        length = 0
        while cursor in remaining:
            remaining.remove(cursor)
            length += 1
            cursor = permutation[cursor]
        if cursor != start:
            raise ValueError("endpoint action does not preserve a slot class")
        lengths[length] += 1
    return lengths


def _centralizer(cycles: Counter) -> int:
    return prod(length ** count * factorial(count)
                for length, count in cycles.items())


def _fixed_sum_for_type(pair_cycles, free_cycles, source_cycles):
    """Sum over local pair swaps via a cycle-type dynamic program."""
    state = {tuple(sorted(free_cycles.items())): 1}
    for size in pair_cycles:
        updated = Counter()
        for key, multiplicity in state.items():
            for lengths in ((size, size), (2 * size,)):
                cycles = Counter(dict(key))
                cycles.update(lengths)
                if any(cycles[n] > source_cycles.get(n, 0) for n in cycles):
                    continue
                updated[tuple(sorted(cycles.items()))] += multiplicity * (1 << (size - 1))
        state = updated
        if not state:
            return 0
    return state.get(tuple(sorted(source_cycles.items())), 0) * _centralizer(source_cycles)


def _fixed_point_sum(action, sink_types, source_types, pairs, classes):
    sink_perm, source_perm = action
    index = {pair: i for i, pair in enumerate(pairs)}
    pair_image = [index[tuple(sorted((sink_perm[a], sink_perm[b])))]
                  for a, b in pairs]
    pair_lengths = {kind: [] for kind in classes}
    remaining = set(range(len(pairs)))
    while remaining:
        start = next(iter(remaining))
        cursor = start
        length = 0
        while cursor in remaining:
            remaining.remove(cursor)
            length += 1
            cursor = pair_image[cursor]
        if cursor != start:
            raise AssertionError("matching automorphism did not form cycles")
        pair_lengths[sink_types[pairs[start][0]]].append(length)

    paired = {slot for pair in pairs for slot in pair}
    answer = 1
    for kind in classes:
        free = (i for i, flavor in enumerate(sink_types)
                if flavor == kind and i not in paired)
        source = (i for i, flavor in enumerate(source_types) if flavor == kind)
        answer *= _fixed_sum_for_type(
            pair_lengths[kind], _cycles(free, sink_perm),
            _cycles(source, source_perm),
        )
        if not answer:
            break
    return answer


def count_pairing_layout(
    sink_types: Sequence[Hashable],
    source_types: Sequence[Hashable],
    pairs: Sequence[Tuple[int, int]],
    endpoint_generators: Sequence[EndpointAction] = (),
    *,
    pairings: Optional[Iterable[Sequence[int]]] = None,
    max_group_order: int = 4096,
) -> PairingOrbitCount:
    """Count full-bijection orbits after W swaps and layout-stable actions.

    Each pairing maps sink slot ``i`` to a source slot.  Slots carry a pairing
    class, usually physical flavor.  A W pair swaps two sink slots of one
    class.  Endpoint generators act on old slots by old-to-new images; the
    resulting group is filtered to elements preserving the chosen matching.

    This count is applicable to the contraction target only when the target
    covers every class-preserving bijection and the endpoint generators were
    independently proven to act on the tensor expression with the proper
    signs and open-axis permutations.  Supplying ``pairings`` verifies the
    combinatorial full-family condition without trusting a declared size.
    """
    sink_types = tuple(sink_types)
    source_types = tuple(source_types)
    size = len(sink_types)
    if len(source_types) != size:
        raise ValueError("sink and source slot counts must match")
    try:
        classes = tuple(dict.fromkeys(sink_types))
        sink_count, source_count = Counter(sink_types), Counter(source_types)
    except TypeError as error:
        raise ValueError("slot classes must be hashable") from error
    if sink_count != source_count:
        raise ValueError("sink and source class multiplicities must match")
    try:
        max_group_order = integer_index(max_group_order)
    except TypeError as error:
        raise ValueError("max_group_order must be a positive integer") from error
    if max_group_order < 1:
        raise ValueError("max_group_order must be positive")

    used = set()
    canonical_pairs = []
    for left, right in pairs:
        try:
            left, right = integer_index(left), integer_index(right)
        except TypeError as error:
            raise ValueError("W pair slots must be integers") from error
        if (left < 0 or right < 0 or left >= size or right >= size
                or left == right or left in used or right in used):
            raise ValueError("W pairs must be distinct disjoint sink slots")
        if sink_types[left] != sink_types[right]:
            raise ValueError("W pair must stay inside one pairing class")
        canonical_pairs.append(tuple(sorted((left, right))))
        used.update((left, right))
    canonical_pairs = tuple(sorted(canonical_pairs))
    pair_set = set(canonical_pairs)

    generators = []
    for sink, source in endpoint_generators:
        sink = _permutation(sink, size)
        source = _permutation(source, size)
        if any(sink_types[i] != sink_types[sink[i]] for i in range(size)):
            raise ValueError("sink generator changes a pairing class")
        if any(source_types[i] != source_types[source[i]] for i in range(size)):
            raise ValueError("source generator changes a pairing class")
        generators.append((sink, source))

    pairing_count = prod(factorial(n) for n in sink_count.values())
    verified = pairings is not None
    if verified:
        seen = set()
        for raw in pairings:
            raw = _permutation(raw, size)
            if any(sink_types[i] != source_types[raw[i]] for i in range(size)):
                raise ValueError("raw pairing violates its slot classes")
            if raw in seen:
                raise ValueError("raw pairing family contains duplicates")
            seen.add(raw)
        if len(seen) != pairing_count:
            raise ValueError("raw pairing family is not complete")

    group = _closure(generators, size, max_group_order)
    stable = [action for action in group if {
        tuple(sorted((action[0][a], action[0][b])))
        for a, b in canonical_pairs
    } == pair_set]
    fixed_sum = sum(_fixed_point_sum(action, sink_types, source_types,
                                     canonical_pairs, classes)
                    for action in stable)
    denominator = (1 << len(canonical_pairs)) * len(stable)
    if fixed_sum % denominator:
        raise AssertionError("Burnside fixed-point sum is not integral")
    return PairingOrbitCount(fixed_sum // denominator, pairing_count,
                             1 << len(canonical_pairs), len(stable),
                             verified, fixed_sum)


__all__ = ["PairingOrbitCount", "count_pairing_layout"]
