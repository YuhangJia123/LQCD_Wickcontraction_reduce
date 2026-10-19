#!/usr/bin/env python3
"""Exhaustively audit the ppnn Wick-to-state quotient and its group action.

This is a combinatorial audit only. It does not evaluate spin/color contractions
or claim to reproduce the projected 1,107-contraction calculation in the
literature.

Run from the repository root with::

    python3 verify_code/ppnn_state_quotient_audit.py
"""

from __future__ import annotations

from collections import Counter
from math import factorial
from pathlib import Path
import sys
from typing import Dict, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from verify_code import ppnn_group_action as action
from kernels.wick_reducer import operator_model


def _baryon(species: str, side: str, label: str):
    if side == "sink":
        flavors = ("u", "u", "d") if species == "p" else ("d", "d", "u")
        gamma = "Cgamma5"
    else:
        flavors = ("u", "d", "u") if species == "p" else ("d", "u", "d")
        gamma = "Cgamma5_dagger"
    return operator_model.make_operator(
        side,
        "baryon",
        species,
        label,
        flavors,
        gamma,
        gamma_slots=(1, 2),
        open_slot=0,
    )


def _operators():
    sink = tuple(
        _baryon(species, "sink", label)
        for species, label in (
            ("p", "sink_p_1"),
            ("p", "sink_p_2"),
            ("n", "sink_n_1"),
            ("n", "sink_n_2"),
        )
    )
    source = tuple(
        _baryon(species, "source", label)
        for species, label in (
            ("p", "source_p_1"),
            ("p", "source_p_2"),
            ("n", "source_n_1"),
            ("n", "source_n_2"),
        )
    )
    return sink + source


def _fixture():
    operators = _operators()
    _fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    q_index = {(field.label, field.slot): i for i, field in enumerate(quarks)}
    anti_index = {(field.label, field.slot): i for i, field in enumerate(antiquarks)}

    up_source_slots = (
        ("source_p_1", 0),
        ("source_p_1", 2),
        ("source_p_2", 0),
        ("source_p_2", 2),
        ("source_n_1", 1),
        ("source_n_2", 1),
    )
    down_source_slots = (
        ("source_p_1", 1),
        ("source_p_2", 1),
        ("source_n_1", 0),
        ("source_n_1", 2),
        ("source_n_2", 0),
        ("source_n_2", 2),
    )
    up_sink_slots = (
        ("sink_p_1", 0),
        ("sink_p_1", 1),
        ("sink_p_2", 0),
        ("sink_p_2", 1),
        ("sink_n_1", 2),
        ("sink_n_2", 2),
    )
    down_sink_slots = (
        ("sink_p_1", 2),
        ("sink_p_2", 2),
        ("sink_n_1", 0),
        ("sink_n_1", 1),
        ("sink_n_2", 0),
        ("sink_n_2", 1),
    )

    def source_labels(slots: Sequence[Tuple[str, int]]) -> Dict[int, int]:
        return {label: anti_index[slot] for label, slot in enumerate(slots, 1)}

    up_source = source_labels(up_source_slots)
    down_source = source_labels(down_source_slots)
    up_sink = tuple(q_index[slot] for slot in up_sink_slots)
    down_sink = tuple(q_index[slot] for slot in down_sink_slots)

    assert len(quarks) == len(antiquarks) == 12
    assert all(quarks[q].flavor == "u" for q in up_sink)
    assert all(quarks[q].flavor == "d" for q in down_sink)
    assert all(antiquarks[a].flavor == "u" for a in up_source.values())
    assert all(antiquarks[a].flavor == "d" for a in down_source.values())
    return quarks, antiquarks, up_sink, down_sink, up_source, down_source


def _decode(pairing, mapping):
    """Return the canonical quotient state and its four internal swap bits."""
    _quarks, _antiquarks, up_sink, down_sink, up_source, down_source = mapping
    up_labels = {anti: label for label, anti in up_source.items()}
    down_labels = {anti: label for label, anti in down_source.items()}

    def up(q_index: int) -> int:
        return up_labels[pairing[q_index]]

    def down(q_index: int) -> int:
        return down_labels[pairing[q_index]]

    up_p1 = tuple(sorted((up(up_sink[0]), up(up_sink[1]))))
    up_p2 = tuple(sorted((up(up_sink[2]), up(up_sink[3]))))
    down_n1 = tuple(sorted((down(down_sink[2]), down(down_sink[3]))))
    down_n2 = tuple(sorted((down(down_sink[4]), down(down_sink[5]))))
    state = action.He4State(
        up_protons=(up_p1, up_p2),
        up_neutrons=(up(up_sink[4]), up(up_sink[5])),
        down_neutrons=(down_n1, down_n2),
        down_protons=(down(down_sink[0]), down(down_sink[1])),
    )

    swap_pairs = (
        (up_sink[0], up_p1),
        (up_sink[2], up_p2),
        (down_sink[2], down_n1),
        (down_sink[4], down_n2),
    )
    mask = 0
    for bit, (first_quark, labels) in enumerate(swap_pairs):
        if (up if bit < 2 else down)(first_quark) == labels[1]:
            mask |= 1 << bit
    return state, mask


def _reconstruct(state, mask, mapping):
    """Rebuild the exact pairing represented by one state and four swap bits."""
    quarks, antiquarks, up_sink, down_sink, up_source, down_source = mapping
    assignments = []

    def pair_assignments(sink_indices, pair, bit):
        labels = pair[::-1] if mask & (1 << bit) else pair
        source_map = up_source if bit < 2 else down_source
        assignments.extend(
            ((sink_indices[0], source_map[labels[0]]),
             (sink_indices[1], source_map[labels[1]]))
        )

    pair_assignments(up_sink[0:2], state.up_protons[0], 0)
    pair_assignments(up_sink[2:4], state.up_protons[1], 1)
    pair_assignments(down_sink[2:4], state.down_neutrons[0], 2)
    pair_assignments(down_sink[4:6], state.down_neutrons[1], 3)
    assignments.extend(
        (
            (up_sink[4], up_source[state.up_neutrons[0]]),
            (up_sink[5], up_source[state.up_neutrons[1]]),
            (down_sink[0], down_source[state.down_protons[0]]),
            (down_sink[1], down_source[state.down_protons[1]]),
        )
    )

    pairing = [-1] * len(quarks)
    for q_index, anti_index in assignments:
        pairing[q_index] = anti_index
    assert sorted(pairing) == list(range(len(antiquarks)))
    assert all(
        quarks[q].flavor == antiquarks[a].flavor
        for q, a in enumerate(pairing)
    )
    return tuple(pairing)


GENERATOR_NAMES = (
    "sink_protons",
    "sink_neutrons",
    "source_protons",
    "source_neutrons",
    "isospin_flip",
)


def _induced_field_relabelings(mapping):
    """Build old-index -> new-index maps for quark and antiquark fields."""
    quarks, antiquarks = mapping[0], mapping[1]
    q_index = {(field.label, field.slot): i for i, field in enumerate(quarks)}
    anti_index = {
        (field.label, field.slot): i for i, field in enumerate(antiquarks)
    }
    label_maps = (
        {"sink_p_1": "sink_p_2", "sink_p_2": "sink_p_1"},
        {"sink_n_1": "sink_n_2", "sink_n_2": "sink_n_1"},
        {"source_p_1": "source_p_2", "source_p_2": "source_p_1"},
        {"source_n_1": "source_n_2", "source_n_2": "source_n_1"},
        {
            "sink_p_1": "sink_n_1",
            "sink_n_1": "sink_p_1",
            "sink_p_2": "sink_n_2",
            "sink_n_2": "sink_p_2",
            "source_p_1": "source_n_1",
            "source_n_1": "source_p_1",
            "source_p_2": "source_n_2",
            "source_n_2": "source_p_2",
        },
    )
    result = []
    for label_map in label_maps:
        q_map = tuple(
            q_index[(label_map.get(field.label, field.label), field.slot)]
            for field in quarks
        )
        anti_map = tuple(
            anti_index[(label_map.get(field.label, field.label), field.slot)]
            for field in antiquarks
        )
        assert sorted(q_map) == list(range(len(quarks)))
        assert sorted(anti_map) == list(range(len(antiquarks)))
        result.append((q_map, anti_map))
    return tuple(result)


def _permutation_sign(permutation):
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _pairing_sign(pairing):
    """Parity of the anti-field permutation for this baryon-only fixture."""
    inversions = 0
    for i, anti_index in enumerate(pairing):
        inversions += sum(anti_index > later for later in pairing[i + 1 :])
    return -1 if inversions % 2 else 1


def _transform_pairing(pairing, q_map, anti_map):
    transformed = [-1] * len(pairing)
    for q_index, anti_index in enumerate(pairing):
        transformed[q_map[q_index]] = anti_map[anti_index]
    return tuple(transformed)


def _apply_group_element(state, mask):
    for bit, generator in enumerate(action.BLOCK_GENERATORS):
        if mask & (1 << bit):
            state = generator(state)
    if mask & (1 << 4):
        state = action.isospin_flip(state)
    return state


def _audit_group(states):
    state_set = set(states)
    assert len(state_set) == len(states) == 32400

    for state in states:
        assert all(generator(state) in state_set for generator in action.HE4_GENERATORS)
        assert all(generator(generator(state)) == state for generator in action.HE4_GENERATORS)
        for i, left in enumerate(action.BLOCK_GENERATORS):
            for right in action.BLOCK_GENERATORS[:i]:
                assert left(right(state)) == right(left(state))

        flip = action.isospin_flip
        assert flip(action.swap_sink_protons(flip(state))) == action.swap_sink_neutrons(state)
        assert flip(action.swap_sink_neutrons(flip(state))) == action.swap_sink_protons(state)
        assert flip(action.swap_source_protons(flip(state))) == action.swap_source_neutrons(state)
        assert flip(action.swap_source_neutrons(flip(state))) == action.swap_source_protons(state)

    fixed_histogram = Counter()
    for mask in range(32):
        fixed = sum(_apply_group_element(state, mask) == state for state in states)
        fixed_histogram[fixed] += 1
    expected_fixed = Counter({32400: 1, 576: 1, 180: 4, 96: 4, 24: 4, 0: 18})
    assert fixed_histogram == expected_fixed
    burnside = sum(fixed * count for fixed, count in fixed_histogram.items()) // 32
    assert burnside == 1068

    unseen = set(states)
    orbit_histogram = Counter()
    while unseen:
        seed = next(iter(unseen))
        orbit = {seed}
        pending = [seed]
        while pending:
            current = pending.pop()
            for generator in action.HE4_GENERATORS:
                image = generator(current)
                if image not in orbit:
                    orbit.add(image)
                    pending.append(image)
        unseen.difference_update(orbit)
        orbit_histogram[len(orbit)] += 1

    expected_orbits = Counter({32: 967, 16: 83, 8: 14, 4: 4})
    assert orbit_histogram == expected_orbits
    assert sum(orbit_histogram.values()) == burnside == 1068
    assert sum(size * count for size, count in orbit_histogram.items()) == len(states)
    assert max(orbit_histogram) == 32
    return fixed_histogram, orbit_histogram


def audit():
    mapping = _fixture()
    quarks, antiquarks = mapping[0], mapping[1]
    field_relabelings = _induced_field_relabelings(mapping)
    expected_characters = (-1, -1, -1, -1, 1)
    induced_characters = tuple(
        _permutation_sign(q_map) * _permutation_sign(anti_map)
        for q_map, anti_map in field_relabelings
    )
    assert induced_characters == expected_characters
    states = tuple(action.he4_states())
    state_set = set(states)
    assert len(states) == 32400
    expected_raw = factorial(6) * factorial(6)
    assert expected_raw == 518400

    fibers = Counter()
    signs = Counter()
    masks_seen = {state: 0 for state in states}
    witness_counts = Counter()
    raw_count = 0
    for term in core.enumerate_wicks(quarks, antiquarks):
        pairing = term.pairing
        assert len(pairing) == 12
        assert all(
            quarks[q].flavor == antiquarks[a].flavor
            for q, a in enumerate(pairing)
        )
        state, mask = _decode(pairing, mapping)
        assert state in state_set
        assert _reconstruct(state, mask, mapping) == pairing
        assert _pairing_sign(pairing) == term.sign
        mask_bit = 1 << mask
        assert masks_seen[state] & mask_bit == 0
        masks_seen[state] |= mask_bit
        fibers[state] += 1
        signs[(state, term.sign)] += 1

        for name, generator, (q_map, anti_map), character in zip(
            GENERATOR_NAMES,
            action.HE4_GENERATORS,
            field_relabelings,
            induced_characters,
        ):
            transformed = _transform_pairing(pairing, q_map, anti_map)
            target_state, target_mask = _decode(transformed, mapping)
            assert target_state == generator(state)
            assert _reconstruct(target_state, target_mask, mapping) == transformed
            transformed_sign = _pairing_sign(transformed)
            assert term.sign * transformed_sign == character
            witness_counts[name] += 1
        raw_count += 1

    assert raw_count == expected_raw
    assert len(fibers) == len(states)
    assert set(fibers.values()) == {16}
    assert set(masks_seen.values()) == {(1 << 16) - 1}
    assert all(signs[(state, 1)] == 8 and signs[(state, -1)] == 8 for state in states)
    assert all(witness_counts[name] == expected_raw for name in GENERATOR_NAMES)

    fixed_histogram, orbit_histogram = _audit_group(states)
    return {
        "raw_wicks": raw_count,
        "quotient_states": len(states),
        "fiber_size": 16,
        "signs_per_fiber": "8 positive / 8 negative",
        "group_order": 32,
        "induced_characters": dict(zip(GENERATOR_NAMES, induced_characters)),
        "pairing_generator_witnesses": dict(witness_counts),
        "burnside_orbits": 1068,
        "fixed_point_histogram": dict(sorted(fixed_histogram.items())),
        "orbit_size_histogram": dict(sorted(orbit_histogram.items())),
    }


def main():
    result = audit()
    print("ppnn exact Wick-state quotient audit: PASS")
    for key, value in result.items():
        print("{}: {}".format(key, value))
    print("scope: combinatorial quotient and label-group action only")
    print("paper comparison: this 1,068-orbit count does not establish the paper's 1,107")


if __name__ == "__main__":
    main()
