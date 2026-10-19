"""Independent ppnn (``1068``) Wick-state orbit reduction verify_code module."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Mapping, Tuple

from kernels import correlator_wick_normalizer as core
from verify_code import ppnn_group_action as action
from verify_code.orbit_common import (
    OrbitSystem,
    burnside_count,
    enumerate_orbits,
)
from verify_code.ppnn_state_quotient_audit import (
    _decode,
    _fixture,
    _induced_field_relabelings,
)


def _swap_quark_slots(pairing: Tuple[int, ...], left: int, right: int) -> Tuple[int, ...]:
    result = list(pairing)
    result[left], result[right] = result[right], result[left]
    return tuple(result)


def _raw_action_from_pairings(pairings, transform):
    pairing_to_id = {pairing: raw_id for raw_id, pairing in pairings.items()}

    def apply(raw_id: int) -> int:
        image = transform(pairings[raw_id])
        return pairing_to_id[image]

    return apply


def _compose(first, second):
    return lambda value: first(second(value))


def _field_action(pair_maps, pairings):
    q_map, anti_map = pair_maps

    def transform(pairing):
        result = [-1] * len(pairing)
        for old_q, old_anti in enumerate(pairing):
            result[q_map[old_q]] = anti_map[old_anti]
        return tuple(result)

    return _raw_action_from_pairings(pairings, transform)


def _check_generator_equivariance(raw_to_state, raw_actions, state_actions):
    """Check the quotient map on all raw items for each group generator.

    Checking generators is sufficient once the action tables have been built
    from the same field relabellings: every group word is a composition of
    these maps.  It avoids repeating the same 32-word certificate for every
    raw Wick term while still checking every one of the 518400 terms.
    """
    if len(raw_actions) != len(state_actions):
        raise AssertionError("raw/state generator lists have different sizes")
    for raw_action, state_action in zip(raw_actions, state_actions):
        state_images = {state: state_action(state) for state in set(raw_to_state.values())}
        for raw_id, state in raw_to_state.items():
            if raw_to_state[raw_action(raw_id)] != state_images[state]:
                raise AssertionError("quotient equivariance failed for a generator")


def _state_group_elements():
    generators = action.HE4_GENERATORS
    elements = []
    for mask in range(32):
        def apply(state, mask=mask):
            for bit, generator in enumerate(action.BLOCK_GENERATORS):
                if mask & (1 << bit):
                    state = generator(state)
            if mask & (1 << 4):
                state = action.isospin_flip(state)
            return state
        elements.append(apply)
    return tuple(elements)


def _build() -> OrbitSystem[action.He4State]:
    mapping = _fixture()
    quarks, antiquarks = mapping[0], mapping[1]
    terms = tuple(core.enumerate_wicks(quarks, antiquarks))
    pairings = {index: tuple(term.pairing) for index, term in enumerate(terms, start=1)}
    signs = {index: int(term.sign) for index, term in enumerate(terms, start=1)}
    raw_to_state = {
        raw_id: _decode(pairing, mapping)[0]
        for raw_id, pairing in pairings.items()
    }
    states = tuple(sorted(set(raw_to_state.values()), key=repr))
    raw_by_state = defaultdict(list)
    for raw_id, state in raw_to_state.items():
        raw_by_state[state].append(raw_id)
    state_to_raw = {
        state: tuple(sorted(raw_by_state[state])) for state in states
    }
    state_generators = action.HE4_GENERATORS
    state_partition = enumerate_orbits(states, state_generators)
    state_group_elements = _state_group_elements()

    q_index = {(field.label, field.slot): index for index, field in enumerate(quarks)}
    internal_transforms = (
        lambda pairing: _swap_quark_slots(
            pairing, q_index[("sink_p_1", 0)], q_index[("sink_p_1", 1)]
        ),
        lambda pairing: _swap_quark_slots(
            pairing, q_index[("sink_p_2", 0)], q_index[("sink_p_2", 1)]
        ),
        lambda pairing: _swap_quark_slots(
            pairing, q_index[("sink_n_1", 0)], q_index[("sink_n_1", 1)]
        ),
        lambda pairing: _swap_quark_slots(
            pairing, q_index[("sink_n_2", 0)], q_index[("sink_n_2", 1)]
        ),
    )
    raw_internal = tuple(_raw_action_from_pairings(pairings, transform)
                         for transform in internal_transforms)
    raw_external = tuple(
        _field_action(field_map, pairings)
        for field_map in _induced_field_relabelings(mapping)
    )
    raw_generators = raw_internal + raw_external
    identity = _raw_action_from_pairings(pairings, lambda pairing: pairing)

    # Explicit external group actions, with the same words as the state group.
    quotient_raw_group_elements = []
    for mask in range(32):
        composed = identity
        for bit in range(4):
            if mask & (1 << bit):
                composed = _compose(raw_external[bit], composed)
        if mask & (1 << 4):
            composed = _compose(raw_external[4], composed)
        quotient_raw_group_elements.append(composed)

    # The 512 element extension has K_ppnn=(C2)^4 as its normal subgroup.
    raw_full_elements = []
    for ext in quotient_raw_group_elements:
        for internal_mask in range(16):
            composed = identity
            for bit, generator in enumerate(raw_internal):
                if internal_mask & (1 << bit):
                    composed = _compose(generator, composed)
            raw_full_elements.append(_compose(ext, composed))

    raw_partition = enumerate_orbits(tuple(pairings), raw_generators)
    system = OrbitSystem(
        raw_pairings=pairings,
        raw_signs=signs,
        raw_to_state=raw_to_state,
        state_to_raw=state_to_raw,
        state_partition=state_partition,
        raw_partition=raw_partition,
        state_group_elements=state_group_elements,
        raw_group_elements=tuple(raw_full_elements),
        quotient_raw_group_elements=tuple(quotient_raw_group_elements),
    )
    system.validate()
    _check_generator_equivariance(raw_to_state, raw_external, state_generators)
    return system


@lru_cache(maxsize=1)
def ppnn_orbit_system() -> OrbitSystem[action.He4State]:
    """Build and cache the complete ppnn raw/state/orbit system."""

    return _build()


build_ppnn_system = ppnn_orbit_system


def burnside_ppnn() -> int:
    system = ppnn_orbit_system()
    return burnside_count(system.state_partition.elements, system.state_group_elements)


def summary() -> Mapping[str, object]:
    system = ppnn_orbit_system()
    return {
        "raw_wick_count": system.raw_wick_count,
        "state_count": system.state_count,
        "orbit_count": system.orbit_count,
        "state_orbit_size_histogram": dict(sorted(system.orbit_size_histogram.items())),
        "raw_orbit_size_histogram": dict(sorted(system.raw_orbit_size_histogram.items())),
        "state_burnside": burnside_ppnn(),
        "raw_group_order": len(system.raw_group_elements),
    }


def main() -> None:
    for key, value in summary().items():
        print(f"{key}: {value}")


__all__ = ["build_ppnn_system", "ppnn_orbit_system", "burnside_ppnn", "summary"]


if __name__ == "__main__":
    main()
