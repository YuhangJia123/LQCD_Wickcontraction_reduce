"""Small, independent data structures for verify_code orbit reductions.

The production reducer has no dependency on this module.  The two concrete
verify_code systems expose raw Wick ids, quotient states, and the two associated
group actions through :class:`OrbitSystem`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Hashable, Iterable, Mapping, Sequence, Tuple, TypeVar


T = TypeVar("T", bound=Hashable)


Action = Callable[[T], T]


def _ordered(values: Iterable[T]) -> Tuple[T, ...]:
    values = tuple(values)
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=repr))


@dataclass(frozen=True)
class OrbitRecord(Generic[T]):
    """One orbit, with deterministic member ordering."""

    orbit_id: int
    representative: T
    members: Tuple[T, ...]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class OrbitPartition(Generic[T]):
    """A finite set partitioned by a declared generator action."""

    elements: Tuple[T, ...]
    generators: Tuple[Action[T], ...]
    orbits: Tuple[OrbitRecord[T], ...]
    element_to_orbit: Mapping[T, int]

    @property
    def count(self) -> int:
        return len(self.orbits)

    @property
    def size_histogram(self) -> Counter:
        return Counter(record.size for record in self.orbits)

    def members(self, orbit_id: int) -> Tuple[T, ...]:
        return self.orbits[orbit_id].members


def enumerate_orbits(
    elements: Iterable[T],
    generators: Sequence[Action[T]],
) -> OrbitPartition[T]:
    """Enumerate a finite generator action by breadth-first search.

    Representatives and members are sorted when possible.  All systems in
    this verify_code package use orderable tuple states; the fallback keeps this
    helper useful for other hashable verify_code objects.
    """

    unique = set(elements)
    ordered = _ordered(unique)
    unseen = set(unique)
    records = []
    element_to_orbit: Dict[T, int] = {}
    # Iterate the single sorted sequence instead of repeatedly taking min()
    # from a shrinking set; the latter is quadratic for 32,400 ppnn states.
    for seed in ordered:
        if seed not in unseen:
            continue
        orbit = {seed}
        pending = [seed]
        while pending:
            current = pending.pop()
            for generator in generators:
                image = generator(current)
                if image not in unique:
                    raise ValueError("generator action leaves the declared element set")
                if image not in orbit:
                    orbit.add(image)
                    pending.append(image)
        members = _ordered(orbit)
        orbit_id = len(records)
        record = OrbitRecord(orbit_id, members[0], members)
        records.append(record)
        for member in members:
            element_to_orbit[member] = orbit_id
        unseen.difference_update(orbit)
    return OrbitPartition(
        elements=ordered,
        generators=tuple(generators),
        orbits=tuple(records),
        element_to_orbit=element_to_orbit,
    )


def fixed_point_histogram(
    elements: Iterable[T], group_elements: Sequence[Action[T]]
) -> Counter:
    """Return ``{number_of_fixed_points: multiplicity}`` for a finite group."""

    domain = tuple(elements)
    return Counter(sum(action(element) == element for element in domain)
                   for action in group_elements)


def burnside_count(
    elements: Iterable[T], group_elements: Sequence[Action[T]]
) -> int:
    """Compute ``|X/G|`` from an explicit finite list of group actions."""

    group_elements = tuple(group_elements)
    if not group_elements:
        raise ValueError("Burnside requires at least the identity action")
    total = sum(
        sum(action(element) == element for element in elements)
        for action in group_elements
    )
    if total % len(group_elements):
        raise ValueError("fixed-point sum is not divisible by group order")
    return total // len(group_elements)


def action_from_generators(
    generators: Sequence[Action[T]], word: Sequence[int]
) -> Action[T]:
    """Build a composed action from generator indices.

    This deliberately keeps the word representation visible in verify_code
    output.  Concrete modules use it for their small Burnside action lists.
    """

    selected = tuple(generators[index] for index in word)

    def apply(element: T) -> T:
        for generator in selected:
            element = generator(element)
        return element

    return apply


@dataclass(frozen=True)
class OrbitSystem(Generic[T]):
    """Raw Wick data together with quotient-state and orbit partitions."""

    raw_pairings: Mapping[int, Tuple[int, ...]]
    raw_signs: Mapping[int, int]
    raw_to_state: Mapping[int, T]
    state_to_raw: Mapping[T, Tuple[int, ...]]
    state_partition: OrbitPartition[T]
    raw_partition: OrbitPartition[int]
    state_group_elements: Tuple[Action[T], ...]
    raw_group_elements: Tuple[Action[int], ...]
    quotient_raw_group_elements: Tuple[Action[int], ...] = ()

    @property
    def raw_wick_count(self) -> int:
        return len(self.raw_pairings)

    @property
    def state_count(self) -> int:
        return len(self.state_partition.elements)

    @property
    def orbit_count(self) -> int:
        return self.state_partition.count

    @property
    def state_orbits(self) -> Tuple[OrbitRecord[T], ...]:
        return self.state_partition.orbits

    @property
    def raw_orbits(self) -> Tuple[OrbitRecord[int], ...]:
        return self.raw_partition.orbits

    @property
    def state_to_orbit(self) -> Mapping[T, int]:
        return self.state_partition.element_to_orbit

    @property
    def state_to_representative(self) -> Mapping[T, T]:
        return {
            state: self.state_partition.orbits[orbit_id].representative
            for state, orbit_id in self.state_to_orbit.items()
        }

    @property
    def raw_to_orbit(self) -> Mapping[int, int]:
        return self.raw_partition.element_to_orbit

    @property
    def raw_to_representative(self) -> Mapping[int, int]:
        return {
            raw_id: self.raw_partition.orbits[orbit_id].representative
            for raw_id, orbit_id in self.raw_to_orbit.items()
        }

    @property
    def orbit_size_histogram(self) -> Counter:
        return self.state_partition.size_histogram

    @property
    def raw_orbit_size_histogram(self) -> Counter:
        return self.raw_partition.size_histogram

    def state_fixed_point_histogram(self) -> Counter:
        return fixed_point_histogram(
            self.state_partition.elements, self.state_group_elements
        )

    def raw_fixed_point_histogram(self) -> Counter:
        return fixed_point_histogram(
            self.raw_partition.elements, self.raw_group_elements
        )

    def validate(self) -> None:
        """Check partition coverage and the quotient map invariant."""

        raw_ids = set(self.raw_pairings)
        if raw_ids != set(self.raw_to_state) or raw_ids != set(self.raw_partition.elements):
            raise AssertionError("raw Wick ids are not covered exactly once")
        if set(self.state_to_raw) != set(self.state_partition.elements):
            raise AssertionError("quotient states are not covered exactly once")
        flattened_state_raw = [
            raw_id for raw_ids_for_state in self.state_to_raw.values()
            for raw_id in raw_ids_for_state
        ]
        if len(flattened_state_raw) != len(set(flattened_state_raw)):
            raise AssertionError("a raw Wick id belongs to multiple quotient states")
        if set(flattened_state_raw) != raw_ids:
            raise AssertionError("state fibers do not cover every raw Wick id")
        flattened_state = [
            state for orbit in self.state_partition.orbits for state in orbit.members
        ]
        if len(flattened_state) != len(set(flattened_state)):
            raise AssertionError("a state belongs to multiple state orbits")
        if set(flattened_state) != set(self.state_partition.elements):
            raise AssertionError("state orbits do not cover every state")
        flattened_raw = [
            raw_id for orbit in self.raw_partition.orbits for raw_id in orbit.members
        ]
        if len(flattened_raw) != len(set(flattened_raw)) or set(flattened_raw) != raw_ids:
            raise AssertionError("raw orbits do not cover every raw Wick exactly once")
        for raw_id, state in self.raw_to_state.items():
            if raw_id not in self.state_to_raw[state]:
                raise AssertionError("raw-to-state and state-to-raw maps disagree")
        for raw_orbit in self.raw_partition.orbits:
            state_orbit_ids = {
                self.state_to_orbit[self.raw_to_state[raw_id]]
                for raw_id in raw_orbit.members
            }
            if len(state_orbit_ids) != 1:
                raise AssertionError("a raw orbit projects to multiple state orbits")
        projected_state_orbits = {
            self.state_to_orbit[self.raw_to_state[raw_id]]
            for raw_id in self.raw_partition.elements
        }
        if projected_state_orbits != set(range(self.state_partition.count)):
            raise AssertionError("raw orbits do not project onto every state orbit")

    def check_quotient_equivariance(self) -> None:
        """Verify ``q(g*raw) = g*q(raw)`` for corresponding action lists."""

        raw_actions = self.quotient_raw_group_elements or self.raw_group_elements
        if len(raw_actions) != len(self.state_group_elements):
            raise AssertionError("raw/state action lists have different sizes")
        for raw_action, state_action in zip(
            raw_actions, self.state_group_elements
        ):
            state_images = {
                state: state_action(state) for state in self.state_partition.elements
            }
            for raw_id, state in self.raw_to_state.items():
                image_raw = raw_action(raw_id)
                if self.raw_to_state[image_raw] != state_images[state]:
                    raise AssertionError("quotient equivariance failed")
