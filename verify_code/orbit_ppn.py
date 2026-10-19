"""Independent ppn (``93``) Wick-state orbit reduction verify_code module."""

from __future__ import annotations

from functools import lru_cache
from collections import defaultdict
from typing import Callable, Dict, Mapping, Sequence, Tuple

from kernels import correlator_wick_normalizer as core
from verify_code import ppnn_group_action as action
from verify_code.orbit_common import (
    OrbitSystem,
    action_from_generators,
    burnside_count,
    enumerate_orbits,
)
from kernels.wick_reducer.operator_model import fields_for_operators, operator_library


def _state_for_pairing(pairing, quarks, antiquarks) -> action.He3State:
    up_labels = {
        field.index: label
        for label, field in enumerate(
            (field for field in antiquarks if field.flavor == "u"), start=1
        )
    }
    down_labels = {
        field.index: label
        for label, field in enumerate(
            (field for field in antiquarks if field.flavor == "d"), start=1
        )
    }
    quark_by_slot = {
        (field.label, field.slot): index for index, field in enumerate(quarks)
    }

    def label(field_label: str, slot: int, labels) -> int:
        anti_index = pairing[quark_by_slot[(field_label, slot)]]
        return labels[antiquarks[anti_index].index]

    return action.He3State(
        up_protons=(
            tuple(sorted(label("sink_p_1", slot, up_labels) for slot in (0, 1))),
            tuple(sorted(label("sink_p_2", slot, up_labels) for slot in (0, 1))),
        ),
        up_neutron=label("sink_n", 2, up_labels),
        down_protons=(
            label("sink_p_1", 2, down_labels),
            label("sink_p_2", 2, down_labels),
        ),
        down_neutron=tuple(
            sorted(label("sink_n", slot, down_labels) for slot in (0, 1))
        ),
    )


def _swap_quark_slots(pairing: Tuple[int, ...], left: int, right: int) -> Tuple[int, ...]:
    result = list(pairing)
    result[left], result[right] = result[right], result[left]
    return tuple(result)


def _raw_action_from_pairings(
    pairings: Mapping[int, Tuple[int, ...]], transform: Callable[[Tuple[int, ...]], Tuple[int, ...]]
) -> Callable[[int], int]:
    pairing_to_id = {pairing: raw_id for raw_id, pairing in pairings.items()}

    def apply(raw_id: int) -> int:
        image = transform(pairings[raw_id])
        try:
            return pairing_to_id[image]
        except KeyError as exc:
            raise ValueError("raw action produced a pairing outside the Wick set") from exc

    return apply


def _field_relabel_action(quarks, antiquarks, label_map):
    q_index = {(field.label, field.slot): index for index, field in enumerate(quarks)}
    anti_index = {(field.label, field.slot): index for index, field in enumerate(antiquarks)}
    q_map = tuple(
        q_index[(label_map.get(field.label, field.label), field.slot)]
        for field in quarks
    )
    anti_map = tuple(
        anti_index[(label_map.get(field.label, field.label), field.slot)]
        for field in antiquarks
    )

    def transform(pairing: Tuple[int, ...]) -> Tuple[int, ...]:
        result = [-1] * len(pairing)
        for old_q, old_anti in enumerate(pairing):
            result[q_map[old_q]] = anti_map[old_anti]
        return tuple(result)

    return transform


def _compose_raw(first, second):
    return lambda raw_id: first(second(raw_id))


def _compose_state(first, second):
    return lambda state: first(second(state))


def _build() -> OrbitSystem[action.He3State]:
    operators = operator_library()["ppn"]
    _fields, quarks, antiquarks = fields_for_operators(operators)
    terms = tuple(core.enumerate_wicks(quarks, antiquarks))
    pairings = {index: tuple(term.pairing) for index, term in enumerate(terms, start=1)}
    signs = {index: int(term.sign) for index, term in enumerate(terms, start=1)}
    raw_to_state = {
        raw_id: _state_for_pairing(pairing, quarks, antiquarks)
        for raw_id, pairing in pairings.items()
    }
    states = tuple(sorted(set(raw_to_state.values()), key=repr))
    raw_by_state = defaultdict(list)
    for raw_id, state in raw_to_state.items():
        raw_by_state[state].append(raw_id)
    state_to_raw: Dict[action.He3State, Tuple[int, ...]] = {
        state: tuple(sorted(raw_by_state[state])) for state in states
    }

    state_generators = (
        action.he3_sink_proton_swap,
        action.he3_source_proton_swap,
    )
    state_partition = enumerate_orbits(states, state_generators)
    state_words = tuple((mask,) for mask in ())  # keep construction below explicit
    state_group_elements = tuple(
        action_from_generators(state_generators, tuple(
            bit for bit in range(2) if mask & (1 << bit)
        ))
        for mask in range(4)
    )

    # Internal same-flavor slot swaps generate K_ppn=(C2)^3.
    q_slot = {(field.label, field.slot): index for index, field in enumerate(quarks)}
    internal_transforms = (
        lambda pairing: _swap_quark_slots(
            pairing, q_slot[("sink_p_1", 0)], q_slot[("sink_p_1", 1)]
        ),
        lambda pairing: _swap_quark_slots(
            pairing, q_slot[("sink_p_2", 0)], q_slot[("sink_p_2", 1)]
        ),
        lambda pairing: _swap_quark_slots(
            pairing, q_slot[("sink_n", 0)], q_slot[("sink_n", 1)]
        ),
    )
    raw_internal = tuple(_raw_action_from_pairings(pairings, transform)
                         for transform in internal_transforms)
    label_maps = (
        {"sink_p_1": "sink_p_2", "sink_p_2": "sink_p_1"},
        {"source_p_1": "source_p_2", "source_p_2": "source_p_1"},
    )
    external_transforms = tuple(
        _field_relabel_action(quarks, antiquarks, label_map)
        for label_map in label_maps
    )
    raw_external = tuple(_raw_action_from_pairings(pairings, transform)
                         for transform in external_transforms)
    raw_generators = raw_internal + raw_external

    # The identity-K external actions are the raw maps corresponding to the
    # state group, used for the quotient equivariance certificate.
    identity_raw = _raw_action_from_pairings(pairings, lambda pairing: pairing)
    raw_group_elements = tuple(
        _compose_raw(
            raw_external[1] if mask & 2 else identity_raw,
            raw_external[0] if mask & 1 else identity_raw,
        )
        for mask in range(4)
    )
    full_raw_partition = enumerate_orbits(tuple(pairings), raw_generators)

    # Generate all 32 actions of K_ppn semidirect G_ppn for the raw Burnside
    # audit.  The orbit partition itself is generated from the six declared
    # generators above.
    raw_full_elements = []
    for ext_mask in range(4):
        external = raw_group_elements[ext_mask]
        for internal_mask in range(8):
            internal = identity_raw
            for bit, generator in enumerate(raw_internal):
                if internal_mask & (1 << bit):
                    internal = _compose_raw(generator, internal)
            raw_full_elements.append(_compose_raw(external, internal))

    system = OrbitSystem(
        raw_pairings=pairings,
        raw_signs=signs,
        raw_to_state=raw_to_state,
        state_to_raw=state_to_raw,
        state_partition=state_partition,
        raw_partition=full_raw_partition,
        state_group_elements=state_group_elements,
        raw_group_elements=tuple(raw_full_elements),
        quotient_raw_group_elements=raw_group_elements,
    )
    system.validate()
    system.check_quotient_equivariance()
    return system


@lru_cache(maxsize=1)
def ppn_orbit_system() -> OrbitSystem[action.He3State]:
    """Build and cache the complete ppn raw/state/orbit system."""

    return _build()


build_ppn_system = ppn_orbit_system


def burnside_ppn() -> int:
    system = ppn_orbit_system()
    return burnside_count(system.state_partition.elements, system.state_group_elements)


def summary() -> Mapping[str, object]:
    system = ppn_orbit_system()
    return {
        "raw_wick_count": system.raw_wick_count,
        "state_count": system.state_count,
        "orbit_count": system.orbit_count,
        "state_orbit_size_histogram": dict(sorted(system.orbit_size_histogram.items())),
        "raw_orbit_size_histogram": dict(sorted(system.raw_orbit_size_histogram.items())),
        "state_burnside": burnside_ppn(),
        "raw_group_order": len(system.raw_group_elements),
    }


def main() -> None:
    for key, value in summary().items():
        print(f"{key}: {value}")


__all__ = ["build_ppn_system", "ppn_orbit_system", "burnside_ppn", "summary"]


if __name__ == "__main__":
    main()
