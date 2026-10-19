#!/usr/bin/env python3
"""Enumerate a proposed block-permutation action for ppnn Wick classes."""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Callable, Iterable, Optional

Pair = tuple[int, int]


@dataclass(frozen=True, order=True)
class He4State:
    up_protons: tuple[Pair, Pair]
    up_neutrons: tuple[int, int]
    down_neutrons: tuple[Pair, Pair]
    down_protons: tuple[int, int]


@dataclass(frozen=True, order=True)
class He3State:
    up_protons: tuple[Pair, Pair]
    up_neutron: int
    down_protons: tuple[int, int]
    down_neutron: Pair


Action = Callable[[object], object]


def _two_pairs_two_ordered_singles(labels: range) -> Iterable[tuple[Pair, Pair, tuple[int, int]]]:
    for first in combinations(labels, 2):
        remaining = tuple(label for label in labels if label not in first)
        for second in combinations(remaining, 2):
            singles = tuple(label for label in remaining if label not in second)
            for ordered_singles in permutations(singles):
                yield first, second, ordered_singles


def he4_states() -> Iterable[He4State]:
    labels = range(1, 7)
    assignments = tuple(_two_pairs_two_ordered_singles(labels))
    for up, down in product(assignments, repeat=2):
        up_p1, up_p2, up_neutrons = up
        down_n1, down_n2, down_protons = down
        yield He4State(
            up_protons=(up_p1, up_p2),
            up_neutrons=up_neutrons,
            down_neutrons=(down_n1, down_n2),
            down_protons=down_protons,
        )


def _map_pair(pair: Pair, mapping: dict[int, int]) -> Pair:
    return tuple(sorted(mapping[label] for label in pair))


def _relabel_he4(
    state: He4State,
    up_mapping: dict[int, int],
    down_mapping: dict[int, int],
) -> He4State:
    return He4State(
        up_protons=tuple(_map_pair(pair, up_mapping) for pair in state.up_protons),
        up_neutrons=tuple(up_mapping[label] for label in state.up_neutrons),
        down_neutrons=tuple(
            _map_pair(pair, down_mapping) for pair in state.down_neutrons
        ),
        down_protons=tuple(down_mapping[label] for label in state.down_protons),
    )


def swap_sink_protons(state: He4State) -> He4State:
    return He4State(
        up_protons=state.up_protons[::-1],
        up_neutrons=state.up_neutrons,
        down_neutrons=state.down_neutrons,
        down_protons=state.down_protons[::-1],
    )


def swap_sink_neutrons(state: He4State) -> He4State:
    return He4State(
        up_protons=state.up_protons,
        up_neutrons=state.up_neutrons[::-1],
        down_neutrons=state.down_neutrons[::-1],
        down_protons=state.down_protons,
    )


def swap_source_protons(state: He4State) -> He4State:
    return _relabel_he4(
        state,
        up_mapping={1: 3, 2: 4, 3: 1, 4: 2, 5: 5, 6: 6},
        down_mapping={1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 6},
    )


def swap_source_neutrons(state: He4State) -> He4State:
    return _relabel_he4(
        state,
        up_mapping={1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 5},
        down_mapping={1: 1, 2: 2, 3: 5, 4: 6, 5: 3, 6: 4},
    )


def isospin_flip(state: He4State) -> He4State:
    down_to_up = {1: 5, 2: 6, 3: 1, 4: 2, 5: 3, 6: 4}
    up_to_down = {1: 3, 2: 4, 3: 5, 4: 6, 5: 1, 6: 2}
    return He4State(
        up_protons=tuple(
            _map_pair(pair, down_to_up) for pair in state.down_neutrons
        ),
        up_neutrons=tuple(down_to_up[label] for label in state.down_protons),
        down_neutrons=tuple(
            _map_pair(pair, up_to_down) for pair in state.up_protons
        ),
        down_protons=tuple(up_to_down[label] for label in state.up_neutrons),
    )


BLOCK_GENERATORS: tuple[Callable[[He4State], He4State], ...] = (
    swap_sink_protons,
    swap_sink_neutrons,
    swap_source_protons,
    swap_source_neutrons,
)
HE4_GENERATORS = BLOCK_GENERATORS + (isospin_flip,)

HE3_SOURCE_UP_PROTON_SWAP = {1: 3, 2: 4, 3: 1, 4: 2, 5: 5}
HE3_SOURCE_DOWN_PROTON_SWAP = {1: 2, 2: 1, 3: 3, 4: 4}


def _apply_mask(state: object, mask: int, generators: tuple[Action, ...]) -> object:
    for bit, generator in enumerate(generators):
        if mask & (1 << bit):
            state = generator(state)
    return state


def fixed_point_histogram(
    states: tuple[object, ...],
    generators: tuple[Action, ...],
    extra_involution: Optional[Action] = None,
) -> Counter[int]:
    """Count fixed points for an elementary abelian group and an optional coset."""
    fixed_counts: Counter[int] = Counter()
    for mask in range(1 << len(generators)):
        fixed_counts[sum(_apply_mask(state, mask, generators) == state for state in states)] += 1
    if extra_involution is not None:
        for mask in range(1 << len(generators)):
            fixed_counts[
                sum(
                    extra_involution(_apply_mask(state, mask, generators)) == state
                    for state in states
                )
            ] += 1
    return fixed_counts


def orbit_size_histogram(
    states: tuple[object, ...], generators: tuple[Action, ...]
) -> Counter[int]:
    unseen = set(states)
    sizes: Counter[int] = Counter()
    while unseen:
        seed = next(iter(unseen))
        orbit = {seed}
        pending = [seed]
        while pending:
            state = pending.pop()
            for generator in generators:
                image = generator(state)
                if image not in orbit:
                    orbit.add(image)
                    pending.append(image)
        unseen.difference_update(orbit)
        sizes[len(orbit)] += 1
    return sizes


def he3_states() -> Iterable[He3State]:
    labels_u = range(1, 6)
    labels_d = range(1, 5)
    for first in combinations(labels_u, 2):
        after_first = tuple(label for label in labels_u if label not in first)
        for second in combinations(after_first, 2):
            neutron_u = next(label for label in after_first if label not in second)
            for d1 in labels_d:
                for d2 in labels_d:
                    if d1 == d2:
                        continue
                    neutron_d = tuple(label for label in labels_d if label not in (d1, d2))
                    yield He3State(
                        up_protons=(first, second),
                        up_neutron=neutron_u,
                        down_protons=(d1, d2),
                        down_neutron=neutron_d,
                    )


def he3_sink_proton_swap(state: He3State) -> He3State:
    return He3State(
        up_protons=state.up_protons[::-1],
        up_neutron=state.up_neutron,
        down_protons=state.down_protons[::-1],
        down_neutron=state.down_neutron,
    )


def he3_source_proton_swap(state: He3State) -> He3State:
    return He3State(
        up_protons=tuple(
            _map_pair(pair, HE3_SOURCE_UP_PROTON_SWAP) for pair in state.up_protons
        ),
        up_neutron=HE3_SOURCE_UP_PROTON_SWAP[state.up_neutron],
        down_protons=tuple(
            HE3_SOURCE_DOWN_PROTON_SWAP[label] for label in state.down_protons
        ),
        down_neutron=_map_pair(state.down_neutron, HE3_SOURCE_DOWN_PROTON_SWAP),
    )


def analyze_ppnn() -> tuple[int, Counter[int], Counter[int]]:
    states = tuple(he4_states())
    fixed = fixed_point_histogram(states, BLOCK_GENERATORS, isospin_flip)
    orbits = orbit_size_histogram(states, HE4_GENERATORS)
    return len(states), fixed, orbits


def main() -> None:
    state_count, fixed, orbits = analyze_ppnn()
    print(f"ppnn states: {state_count}")
    print(f"Burnside fixed-point histogram: {dict(sorted(fixed.items()))}")
    print(f"orbit-size histogram: {dict(sorted(orbits.items()))}")
    print(f"ppnn orbits: {sum(orbits.values())}")

    states3 = tuple(he3_states())
    generators3 = (he3_sink_proton_swap, he3_source_proton_swap)
    orbits3 = orbit_size_histogram(states3, generators3)
    print(f"ppn states: {len(states3)}")
    print(f"ppn orbit-size histogram: {dict(sorted(orbits3.items()))}")
    print(f"ppn orbits: {sum(orbits3.values())}")


if __name__ == "__main__":
    main()
