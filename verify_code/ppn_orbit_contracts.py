#!/usr/bin/env python3
"""Emit ppn contract candidates from the proposed 93 label-action orbits.

The generated calls are orbit representatives, not certified reducer basis
elements. No orbit coefficients or final summed correlator are emitted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from verify_code import ppnn_group_action as action
from kernels.wick_reducer import expression_output
from kernels.wick_reducer.operator_model import fields_for_operators, operator_library
from kernels.wick_reducer.reducer import reduce_case


def _state_pairing(state: action.He3State, operators) -> Tuple[int, ...]:
    fields, quarks, antiquarks = fields_for_operators(operators)
    anti_by_label_slot = {
        (field.label, field.slot): index for index, field in enumerate(antiquarks)
    }
    quark_by_label_slot = {
        (field.label, field.slot): index for index, field in enumerate(quarks)
    }
    up_anti = {
        1: anti_by_label_slot[("source_p_1", 0)],
        2: anti_by_label_slot[("source_p_1", 2)],
        3: anti_by_label_slot[("source_p_2", 0)],
        4: anti_by_label_slot[("source_p_2", 2)],
        5: anti_by_label_slot[("source_n", 1)],
    }
    down_anti = {
        1: anti_by_label_slot[("source_p_1", 1)],
        2: anti_by_label_slot[("source_p_2", 1)],
        3: anti_by_label_slot[("source_n", 0)],
        4: anti_by_label_slot[("source_n", 2)],
    }
    pairing: List[int] = [-1] * len(quarks)

    for sink_label, source_pair in zip(("sink_p_1", "sink_p_2"), state.up_protons):
        for slot, source_label in zip((0, 1), source_pair):
            pairing[quark_by_label_slot[(sink_label, slot)]] = up_anti[source_label]
    pairing[quark_by_label_slot[("sink_n", 2)]] = up_anti[state.up_neutron]

    for sink_label, source_label in zip(("sink_p_1", "sink_p_2"), state.down_protons):
        pairing[quark_by_label_slot[(sink_label, 2)]] = down_anti[source_label]
    for slot, source_label in zip((0, 1), state.down_neutron):
        pairing[quark_by_label_slot[("sink_n", slot)]] = down_anti[source_label]

    if sorted(pairing) != list(range(len(antiquarks))):
        raise ValueError(f"state does not define a complete Wick pairing: {state!r}")
    return tuple(pairing)


def _state_pairings(state: action.He3State, operators):
    """Return the eight raw Wick terms represented by a quotient state."""
    _fields, quarks, _antiquarks = fields_for_operators(operators)
    quark_by_label_slot = {
        (field.label, field.slot): index for index, field in enumerate(quarks)
    }
    canonical = list(_state_pairing(state, operators))
    up_slot_pairs = (
        (quark_by_label_slot[("sink_p_1", 0)], quark_by_label_slot[("sink_p_1", 1)]),
        (quark_by_label_slot[("sink_p_2", 0)], quark_by_label_slot[("sink_p_2", 1)]),
    )
    down_neutron_slots = (
        quark_by_label_slot[("sink_n", 0)], quark_by_label_slot[("sink_n", 1)]
    )
    variants = set()
    for mask in range(8):
        pairing = list(canonical)
        for bit, (left, right) in enumerate((*up_slot_pairs, down_neutron_slots)):
            if mask & (1 << bit):
                pairing[left], pairing[right] = pairing[right], pairing[left]
        variants.add(tuple(pairing))
    if len(variants) != 8:
        raise AssertionError("within-baryon slot orbit did not contain eight Wick terms")
    return tuple(sorted(variants))


def _state_orbits(states: Iterable[action.He3State]):
    generators = (action.he3_sink_proton_swap, action.he3_source_proton_swap)
    unseen = set(states)
    orbits = []
    while unseen:
        seed = min(unseen)
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
        orbits.append(tuple(sorted(orbit)))
    return orbits


def _contract_call(operators, pairing: Sequence[int]) -> str:
    equation, operands, _output = expression_output.contraction_expression(
        operators, pairing, isospin_degenerate=True,
    )
    return "contract(" + ", ".join((repr(equation), *operands)) + ")"


def render_candidate_file() -> Tuple[str, Mapping[str, int]]:
    operators = operator_library()["ppn"]
    _fields, quarks, antiquarks = fields_for_operators(operators)
    wick_terms = list(core.enumerate_wicks(quarks, antiquarks))
    wick_id_by_pairing = {term.pairing: index for index, term in enumerate(wick_terms, 1)}
    states = tuple(action.he3_states())
    if len(states) != 360:
        raise AssertionError(f"expected 360 quotient states; found {len(states)}")

    state_pairs = {state: _state_pairing(state, operators) for state in states}
    if len(set(state_pairs.values())) != len(states):
        raise AssertionError("distinct quotient states did not map to distinct pairings")
    state_wick_ids = {
        state: wick_id_by_pairing[pairing]
        for state, pairing in state_pairs.items()
    }
    state_raw_wick_ids = {
        state: tuple(sorted(wick_id_by_pairing[pairing] for pairing in _state_pairings(state, operators)))
        for state in states
    }
    all_raw_ids = [wick_id for values in state_raw_wick_ids.values() for wick_id in values]
    if len(all_raw_ids) != 2880 or len(set(all_raw_ids)) != 2880:
        raise AssertionError("quotient states and within-baryon swaps must cover 2880 Wicks")

    production = reduce_case("ppn", operators, mode="independent")
    production_basis_ids = {
        int(item["wick_index"]) for item in production["independent_wicks"]
    }

    orbit_records = []
    for orbit in _state_orbits(states):
        orbit_raw_ids = tuple(sorted({
            wick_id for state in orbit for wick_id in state_raw_wick_ids[state]
        }))
        independent_ids = [wick_id for wick_id in orbit_raw_ids if wick_id in production_basis_ids]
        if not independent_ids:
            raise AssertionError(
                "a label-action orbit has no representative in the production basis"
            )
        wick_id = min(independent_ids)
        representative = next(state for state in orbit if wick_id in state_raw_wick_ids[state])
        orbit_records.append((wick_id, representative, orbit, orbit_raw_ids))
    orbit_records.sort(key=lambda record: record[0])
    if len(orbit_records) != 93:
        raise AssertionError(f"expected 93 label-action orbits; found {len(orbit_records)}")

    lines = [
        "# Auto-generated by verify_code/ppn_orbit_contracts.py.",
        "# These are one contract candidate per proposed ppn label-action orbit.",
        "# Orbit membership alone does not certify tensor equality or coefficients.",
        "# No summed correlator is provided until signs, axes, and stabilizers are verified.",
        "from opt_einsum import contract",
        "",
        "# Elemental convention matches wick_reducer's built-in ppn fixture.",
        "#   VVV_sink and conj_VVV_source use the shared default momentum.",
        "ppn_orbit_candidate_wick_ids = (",
    ]
    lines.extend(f"    {wick_id}," for wick_id, _state, _orbit, _raw_ids in orbit_records)
    lines.extend((")", "", "# Canonical quotient-state Wick IDs in each label-action orbit."))
    lines.append("ppn_orbit_candidate_members = {")
    for index, (wick_id, _state, orbit, orbit_raw_ids) in enumerate(orbit_records):
        member_ids = tuple(sorted(state_wick_ids[state] for state in orbit))
        lines.append(
            f"    {index}: {{'representative_wick_id': {wick_id}, "
            f"'canonical_state_wick_ids': {member_ids}, 'raw_wick_ids': {orbit_raw_ids}}},"
        )
    lines.extend((
        "}",
        "",
        "# One explicit contract call per orbit representative; keys are raw Wick IDs.",
        "# Each representative is also an independent Wick in the current reducer.",
        "# The orbit action still needs a separate proof before any coefficient aggregation.",
    ))
    lines.append("ppn_orbit_candidate_basis_contract = {}")
    signs = {index: term.sign for index, term in enumerate(wick_terms, 1)}
    pairing_by_id = {wick_id: pairing for pairing, wick_id in wick_id_by_pairing.items()}
    for index, (wick_id, _state, _orbit, _raw_ids) in enumerate(orbit_records):
        lines.append(
            f"ppn_orbit_candidate_basis_contract[{wick_id}] = "
            f"{_contract_call(operators, pairing_by_id[wick_id])}  # orbit {index}"
        )
    lines.extend(("", "# Project-style signed basis entries; no coefficients or result are asserted."))
    lines.append("ppn_orbit_candidate_basis = {}")
    for basis_index, (wick_id, _state, _orbit, _raw_ids) in enumerate(orbit_records):
        expression = f"ppn_orbit_candidate_basis_contract[{wick_id}]"
        if signs[wick_id] < 0:
            expression = f"(-{expression})"
        lines.append(f"ppn_orbit_candidate_basis[{basis_index}] = {expression}")
    return "\n".join(lines) + "\n", {
        "raw_wicks": len(wick_terms),
        "quotient_states": len(states),
        "orbits": len(orbit_records),
        "contract_calls": len(orbit_records),
        "production_basis_representatives": len(orbit_records),
    }


def main() -> None:
    output = PROJECT_ROOT / "verify_code" / "ppn_orbit_candidate_contracts.py"
    text, counts = render_candidate_file()
    output.write_text(text, encoding="utf-8")
    print(", ".join(f"{key}={value}" for key, value in counts.items()))
    print(output)


if __name__ == "__main__":
    main()
