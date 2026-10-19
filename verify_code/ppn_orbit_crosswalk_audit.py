#!/usr/bin/env python3
"""Crosswalk ppn group orbits to reducer basis Wick IDs and audit one orbit."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from verify_code import ppnn_group_action as action
from kernels.wick_reducer import expression_output, operator_model, reducer


@dataclass
class Crosswalk:
    terms: List[core.WickTerm]
    states: List[action.He3State]
    state_to_orbit: Dict[action.He3State, int]
    orbit_states: List[Tuple[action.He3State, ...]]
    basis_by_orbit: Dict[int, List[int]]
    basis_by_state: Dict[action.He3State, List[int]]
    reduction: Mapping[str, object]


def state_for_pairing(
    term: core.WickTerm,
    quarks: Sequence[core.Field],
    antiquarks: Sequence[core.Field],
) -> action.He3State:
    """Map an exact ppn Wick pairing to the 360-state label convention."""
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
    quark_by_operator_slot = {
        (field.operator, field.slot): index for index, field in enumerate(quarks)
    }

    def up_label(operator_index: int, slot: int) -> int:
        anti_index = term.pairing[quark_by_operator_slot[(operator_index, slot)]]
        return up_labels[antiquarks[anti_index].index]

    def up_pair(operator_index: int, slots: Tuple[int, int]) -> Tuple[int, int]:
        values = tuple(
            up_label(operator_index, slot) for slot in slots
        )
        return tuple(sorted(values))

    def down_label(operator_index: int, slot: int) -> int:
        anti_index = term.pairing[quark_by_operator_slot[(operator_index, slot)]]
        return down_labels[antiquarks[anti_index].index]

    return action.He3State(
        up_protons=(up_pair(0, (0, 1)), up_pair(1, (0, 1))),
        up_neutron=up_label(2, 2),
        down_protons=(down_label(0, 2), down_label(1, 2)),
        down_neutron=tuple(sorted((down_label(2, 0), down_label(2, 1)))),
    )


def _orbits(
    states: Iterable[action.He3State],
) -> Tuple[Dict[action.He3State, int], List[Tuple[action.He3State, ...]]]:
    state_to_orbit: Dict[action.He3State, int] = {}
    orbit_states: List[Tuple[action.He3State, ...]] = []
    generators = (action.he3_sink_proton_swap, action.he3_source_proton_swap)

    for seed in states:
        if seed in state_to_orbit:
            continue
        orbit_index = len(orbit_states)
        orbit = {seed}
        pending = [seed]
        while pending:
            state = pending.pop()
            for generator in generators:
                image = generator(state)
                if image not in orbit:
                    orbit.add(image)
                    pending.append(image)
        ordered_orbit = tuple(sorted(orbit))
        orbit_states.append(ordered_orbit)
        for state in orbit:
            state_to_orbit[state] = orbit_index
    return state_to_orbit, orbit_states


def build_crosswalk() -> Crosswalk:
    operators = operator_model.operator_library()["ppn"]
    _fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    terms = list(core.enumerate_wicks(quarks, antiquarks))
    states = [state_for_pairing(term, quarks, antiquarks) for term in terms]
    state_to_orbit, orbit_states = _orbits(action.he3_states())
    if set(states) != set(action.he3_states()):
        raise AssertionError("Wick pairings do not cover the declared He3 state space")

    reduction = reducer.reduce_case("ppn", operators, mode="independent")
    basis_by_orbit: Dict[int, List[int]] = defaultdict(list)
    basis_by_state: Dict[action.He3State, List[int]] = defaultdict(list)
    for item in reduction["independent_wicks"]:
        wick_id = int(item["wick_index"])
        state = states[wick_id - 1]
        basis_by_orbit[state_to_orbit[state]].append(wick_id)
        basis_by_state[state].append(wick_id)

    return Crosswalk(
        terms=terms,
        states=states,
        state_to_orbit=state_to_orbit,
        orbit_states=orbit_states,
        basis_by_orbit=dict(basis_by_orbit),
        basis_by_state=dict(basis_by_state),
        reduction=reduction,
    )


def _eps3() -> np.ndarray:
    tensor = np.zeros((3, 3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                tensor[i, j, k] = ((i - j) * (j - k) * (k - i)) / 2
    return tensor


def _random_antisymmetric_vvv(dimension: int, rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((dimension,) * 3) + 1j * rng.standard_normal((dimension,) * 3)
    tensor = np.zeros_like(raw)
    for permutation in (
        (0, 1, 2), (0, 2, 1), (1, 0, 2),
        (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )
        tensor += (-1 if inversions % 2 else 1) * raw.transpose(permutation)
    return tensor / 6.0


def _contract_parts(
    operators: Sequence[core.Operator], pairing: Sequence[int]
) -> Tuple[Tuple[str, ...], List[str], Tuple[str, ...]]:
    equation, operands, output_axes = expression_output.contraction_expression(
        operators, pairing, isospin_degenerate=True
    )
    factor_indices = equation.split("->", 1)[0].split(",")
    if len(factor_indices) != len(operands):
        raise AssertionError("contract equation and operand list have different lengths")
    return operands, factor_indices, output_axes


def _remap_open_axes(
    factor_indices: Sequence[str],
    output_axes: Sequence[str],
    output_permutation: Sequence[int],
) -> List[str]:
    inverse = [0] * len(output_permutation)
    for output_axis, input_axis in enumerate(output_permutation):
        inverse[input_axis] = output_axis
    mapping = {
        output_axes[axis]: output_axes[inverse[axis]]
        for axis in range(len(output_axes))
    }
    return ["".join(mapping.get(index, index) for index in factor) for factor in factor_indices]


def _alpha_signature(
    operands: Sequence[str],
    factor_indices: Sequence[str],
    output_axes: Sequence[str],
) -> Tuple[object, ...]:
    dummy_ids: Dict[str, int] = {}
    factors = []
    for operand, indices in zip(operands, factor_indices):
        canonical_indices = []
        for index in indices:
            if index in output_axes:
                canonical_indices.append(("open", index))
            else:
                if index not in dummy_ids:
                    dummy_ids[index] = len(dummy_ids)
                canonical_indices.append(("dummy", dummy_ids[index]))
        factors.append((operand, tuple(canonical_indices)))
    return tuple(output_axes), tuple(factors)


def _factor_incidence(
    operands: Sequence[str],
    factor_indices: Sequence[str],
    output_axes: Sequence[str],
) -> Tuple[Tuple[Tuple[str, int], ...], List[List[Tuple[object, ...]]]]:
    colors = tuple((operand, len(indices)) for operand, indices in zip(operands, factor_indices))
    occurrences: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for factor_id, indices in enumerate(factor_indices):
        for slot, index in enumerate(indices):
            occurrences[index].append((factor_id, slot))

    ports: List[List[Tuple[object, ...]]] = [[] for _ in factor_indices]
    open_set = set(output_axes)
    for index, endpoints in occurrences.items():
        if index in open_set:
            if len(endpoints) != 1:
                raise AssertionError(f"open index {index} has {len(endpoints)} occurrences")
            factor_id, slot = endpoints[0]
            ports[factor_id].append((slot, ("open", index)))
            continue
        if len(endpoints) != 2:
            raise AssertionError(f"dummy index {index} has {len(endpoints)} occurrences")
        (left_factor, left_slot), (right_factor, right_slot) = endpoints
        ports[left_factor].append((left_slot, ("dummy", right_factor, right_slot)))
        ports[right_factor].append((right_slot, ("dummy", left_factor, left_slot)))

    ordered_ports: List[List[Tuple[object, ...]]] = []
    for factor_ports in ports:
        by_slot = {slot: target for slot, target in factor_ports}
        ordered_ports.append([by_slot[slot] for slot in range(len(factor_ports))])
    return colors, ordered_ports


def _factor_graph_isomorphic(
    left_operands: Sequence[str],
    left_indices: Sequence[str],
    right_operands: Sequence[str],
    right_indices: Sequence[str],
    output_axes: Sequence[str],
) -> bool:
    """Compare contraction graphs, allowing identical operand factors to swap."""
    left_colors, left_ports = _factor_incidence(left_operands, left_indices, output_axes)
    right_colors, right_ports = _factor_incidence(right_operands, right_indices, output_axes)
    if Counter(left_colors) != Counter(right_colors):
        return False

    def local_signature(
        factor_id: int,
        colors: Sequence[Tuple[str, int]],
        ports: Sequence[Sequence[Tuple[object, ...]]],
    ) -> Tuple[object, ...]:
        signature = []
        for target in ports[factor_id]:
            if target[0] == "open":
                signature.append(target)
            else:
                other_id, other_slot = int(target[1]), int(target[2])
                signature.append(("dummy", colors[other_id], other_slot, other_id == factor_id))
        return colors[factor_id], tuple(signature)

    left_signatures = [
        local_signature(index, left_colors, left_ports)
        for index in range(len(left_colors))
    ]
    right_signatures = [
        local_signature(index, right_colors, right_ports)
        for index in range(len(right_colors))
    ]
    candidates = {
        left_id: [
            right_id
            for right_id, signature in enumerate(right_signatures)
            if signature == left_signatures[left_id]
        ]
        for left_id in range(len(left_colors))
    }
    if any(not values for values in candidates.values()):
        return False

    order = sorted(candidates, key=lambda factor_id: (len(candidates[factor_id]), factor_id))
    mapping: Dict[int, int] = {}
    inverse_mapping: Dict[int, int] = {}

    def compatible(left_id: int, right_id: int) -> bool:
        for left_slot, left_target in enumerate(left_ports[left_id]):
            right_target = right_ports[right_id][left_slot]
            if left_target[0] == "open":
                if right_target != left_target:
                    return False
                continue
            if right_target[0] != "dummy" or int(left_target[2]) != int(right_target[2]):
                return False
            left_neighbor, right_neighbor = int(left_target[1]), int(right_target[1])
            if left_neighbor == left_id and right_neighbor != right_id:
                return False
            if left_neighbor in mapping and mapping[left_neighbor] != right_neighbor:
                return False
            if right_neighbor in inverse_mapping and inverse_mapping[right_neighbor] != left_neighbor:
                return False
        return True

    def search(position: int) -> bool:
        if position == len(order):
            return True
        left_id = order[position]
        for right_id in candidates[left_id]:
            if right_id in inverse_mapping or not compatible(left_id, right_id):
                continue
            mapping[left_id] = right_id
            inverse_mapping[right_id] = left_id
            if search(position + 1):
                return True
            del mapping[left_id]
            del inverse_mapping[right_id]
        return False

    return search(0)


def _allowed_axis_permutations() -> Dict[str, Tuple[int, ...]]:
    return {
        "identity": (0, 1, 2, 3, 4, 5),
        "sink_proton_swap": (1, 0, 2, 3, 4, 5),
        "source_proton_swap": (0, 1, 2, 4, 3, 5),
        "both_proton_swaps": (1, 0, 2, 4, 3, 5),
    }


def _formal_structure_audit(
    crosswalk: Crosswalk,
    basis_ids: Sequence[int],
) -> Dict[str, object]:
    operators = operator_model.operator_library()["ppn"]
    parsed = {}
    for wick_id in basis_ids:
        term = crosswalk.terms[wick_id - 1]
        operands, indices, axes = _contract_parts(operators, term.pairing)
        parsed[wick_id] = (operands, indices, axes)

    permutations = _allowed_axis_permutations()
    ordered_matches: Dict[int, List[int]] = {wick_id: [] for wick_id in basis_ids}
    graph_matches: Dict[int, List[int]] = {wick_id: [] for wick_id in basis_ids}
    alpha_hashes = {}
    for wick_id, (operands, indices, axes) in parsed.items():
        signature = _alpha_signature(operands, indices, axes)
        alpha_hashes[str(wick_id)] = hashlib.sha256(repr(signature).encode("ascii")).hexdigest()[:12]

    for left_id in basis_ids:
        left_operands, left_indices, left_axes = parsed[left_id]
        for right_id in basis_ids:
            right_operands, right_indices, right_axes = parsed[right_id]
            if left_axes != right_axes:
                raise AssertionError("basis contracts disagree on open axes")
            ordered_match = False
            graph_match = False
            for permutation in permutations.values():
                transformed = _remap_open_axes(right_indices, right_axes, permutation)
                if _alpha_signature(right_operands, transformed, right_axes) == _alpha_signature(
                    left_operands, left_indices, left_axes
                ):
                    ordered_match = True
                if _factor_graph_isomorphic(
                    left_operands,
                    left_indices,
                    right_operands,
                    transformed,
                    left_axes,
                ):
                    graph_match = True
                if ordered_match and graph_match:
                    break
            if ordered_match:
                ordered_matches[left_id].append(right_id)
            if graph_match:
                graph_matches[left_id].append(right_id)

    def equivalence_classes(matches: Mapping[int, Sequence[int]]) -> List[List[int]]:
        remaining = set(basis_ids)
        classes = []
        while remaining:
            seed = min(remaining)
            related = set(matches[seed]) & remaining
            classes.append(sorted(related))
            remaining.difference_update(related)
        return classes

    return {
        "basis_wick_ids": list(basis_ids),
        "canonical_alpha_signature_sha256": alpha_hashes,
        "ordered_factor_alpha_classes": equivalence_classes(ordered_matches),
        "factor_permutation_graph_classes": equivalence_classes(graph_matches),
        "allowed_axis_permutations": {
            name: list(permutation) for name, permutation in permutations.items()
        },
    }


def _sink_proton_swap_witness(crosswalk: Crosswalk) -> Dict[str, object]:
    """Verify one full sink-proton block exchange as a tensor-network identity."""
    operators = operator_model.operator_library()["ppn"]
    left_id, right_id = 1, 781
    left = crosswalk.terms[left_id - 1]
    right = crosswalk.terms[right_id - 1]
    left_operands, left_indices, left_axes = _contract_parts(operators, left.pairing)
    right_operands, right_indices, right_axes = _contract_parts(operators, right.pairing)
    if left_axes != right_axes or left_axes != tuple("ABCDEF"):
        raise AssertionError("sink proton swap must preserve the ppn open-axis schema")

    sink_swap = (1, 0, 2, 3, 4, 5)
    swapped_left_indices = _remap_open_axes(left_indices, left_axes, sink_swap)
    if not _factor_graph_isomorphic(
        left_operands,
        swapped_left_indices,
        right_operands,
        right_indices,
        right_axes,
    ):
        raise AssertionError("Wick 1 and 781 are not related by the expected sink-block swap")

    signed_coefficient = right.sign / left.sign
    return {
        "wick_ids": [left_id, right_id],
        "wick_signs": [left.sign, right.sign],
        "open_axis_permutation": list(sink_swap),
        "unsigned_network_isomorphic": True,
        "signed_relation_coefficient": signed_coefficient,
    }


def _tensor_sample(
    crosswalk: Crosswalk,
    raw_ids: Sequence[int],
    basis_ids: Sequence[int],
    *,
    name: str,
    seed: int,
    color_dimension: int,
    use_epsilon: bool,
) -> Dict[str, object]:
    operators = operator_model.operator_library()["ppn"]
    rng = np.random.default_rng(seed)
    gamma = core._euclidean_gamma_matrices()
    peram = rng.standard_normal((4, color_dimension, color_dimension, 4)) + 1j * rng.standard_normal(
        (4, color_dimension, color_dimension, 4)
    )
    if use_epsilon:
        epsilon = _eps3()
        sink_vvv = (rng.standard_normal() + 1j * rng.standard_normal()) * epsilon
        source_vvv = (rng.standard_normal() + 1j * rng.standard_normal()) * epsilon
    else:
        sink_vvv = _random_antisymmetric_vvv(color_dimension, rng)
        source_vvv = _random_antisymmetric_vvv(color_dimension, rng)
    operands = {
        "C_g5": np.asarray(gamma["Cgamma5"]),
        "C_g5_dagger": np.asarray(gamma["Cgamma5_dagger"]),
        "peram_u_sin_sou": peram,
        "VVV_sink": sink_vvv,
        "conj_VVV_source": np.conjugate(source_vvv),
    }

    tensors: Dict[int, np.ndarray] = {}
    for wick_id in raw_ids:
        term = crosswalk.terms[wick_id - 1]
        equation, names, axes = expression_output.contraction_expression(
            operators, term.pairing, isospin_degenerate=True
        )
        if axes != tuple("ABCDEF"):
            raise AssertionError(f"unexpected free-axis order for Wick {wick_id}: {axes}")
        tensors[wick_id] = term.sign * np.einsum(
            equation,
            *[operands[operand] for operand in names],
            optimize=True,
        )

    basis_matrix = np.stack([tensors[wick_id].reshape(-1) for wick_id in basis_ids], axis=1)
    singular_values = np.linalg.svd(basis_matrix, compute_uv=False)
    basis_rank = int(np.count_nonzero(singular_values > singular_values[0] * 1.0e-10))
    all_matrix = np.stack([tensors[wick_id].reshape(-1) for wick_id in raw_ids], axis=1)
    all_singular_values = np.linalg.svd(all_matrix, compute_uv=False)
    all_rank = int(np.count_nonzero(all_singular_values > all_singular_values[0] * 1.0e-10))

    permutations = _allowed_axis_permutations()
    tolerance = 1.0e-12
    pairwise_matches: Dict[str, List[str]] = {}
    pairwise_residuals: Dict[str, Dict[str, float]] = {}
    for left_id in basis_ids:
        pairwise_matches[str(left_id)] = []
        pairwise_residuals[str(left_id)] = {}
        left = tensors[left_id]
        left_norm = float(np.linalg.norm(left))
        for right_id in basis_ids:
            right = tensors[right_id]
            best = float("inf")
            for permutation in permutations.values():
                view = np.transpose(right, permutation)
                for phase in (1, -1):
                    residual = float(np.linalg.norm(left - phase * view) / left_norm)
                    if residual < best:
                        best = residual
            pairwise_residuals[str(left_id)][str(right_id)] = best
            if best <= tolerance:
                pairwise_matches[str(left_id)].append(str(right_id))

    remaining = set(basis_ids)
    equivalence_classes: List[List[int]] = []
    while remaining:
        first = min(remaining)
        related = set(map(int, pairwise_matches[str(first)])) & remaining
        equivalence_classes.append(sorted(related))
        remaining.difference_update(related)

    return {
        "name": name,
        "seed": seed,
        "color_dimension": color_dimension,
        "vvv_kind": "complex-scaled epsilon3" if use_epsilon else "random fully antisymmetric rank-3 tensor",
        "basis_tensor_rank": basis_rank,
        "all_raw_tensor_rank": all_rank,
        "singular_values_basis": [float(value) for value in singular_values],
        "axis_equivalence_classes": equivalence_classes,
        "relative_residuals_to_wick_1": pairwise_residuals[str(basis_ids[0])],
        "normalized_pairwise_residuals": pairwise_residuals,
        "pairwise_matches": pairwise_matches,
        "tolerance": tolerance,
    }


def _numeric_audit(crosswalk: Crosswalk) -> Dict[str, object]:
    target = action.He3State(
        up_protons=((1, 2), (3, 4)),
        up_neutron=5,
        down_protons=(1, 2),
        down_neutron=(3, 4),
    )
    raw_ids = [index + 1 for index, state in enumerate(crosswalk.states) if state == target]
    basis_ids = crosswalk.basis_by_state[target]
    if raw_ids != [1, 3, 25, 27, 121, 123, 145, 147] or basis_ids != [1, 3, 27, 121, 123, 147]:
        raise AssertionError(f"target-state Wick IDs changed: raw={raw_ids}, basis={basis_ids}")

    samples = [
        _tensor_sample(
            crosswalk,
            raw_ids,
            basis_ids,
            name=f"epsilon3_seed_{seed}",
            seed=seed,
            color_dimension=3,
            use_epsilon=True,
        )
        for seed in (721, 20261007, 20261008)
    ]
    samples.append(
        _tensor_sample(
            crosswalk,
            raw_ids,
            basis_ids,
            name="antisymmetric6_seed_20261009",
            seed=20261009,
            color_dimension=6,
            use_epsilon=False,
        )
    )
    for sample in samples:
        if sample["basis_tensor_rank"] != 6 or sample["all_raw_tensor_rank"] != 8:
            raise AssertionError(f"unexpected numeric rank in sample {sample['name']}: {sample}")
        if len(sample["axis_equivalence_classes"]) != 6:
            raise AssertionError(f"axis-action equivalence changed in sample {sample['name']}")

    formal = _formal_structure_audit(crosswalk, basis_ids)
    if len(formal["ordered_factor_alpha_classes"]) != 6:
        raise AssertionError(f"ordered alpha canonicalization failed: {formal}")
    if len(formal["factor_permutation_graph_classes"]) != 6:
        raise AssertionError(f"factor-permutation graph audit failed: {formal}")
    block_swap = _sink_proton_swap_witness(crosswalk)

    return {
        "target_state": repr(target),
        "raw_wick_ids": raw_ids,
        "basis_wick_ids": basis_ids,
        "relative_singular_value_threshold": 1.0e-10,
        "samples": samples,
        "formal_structure": formal,
        "sink_proton_swap_witness": block_swap,
    }


def audit() -> Tuple[Dict[str, object], Dict[str, object]]:
    crosswalk = build_crosswalk()
    raw_state_multiplicity = Counter(crosswalk.states)
    orbit_sizes = Counter(len(states) for states in crosswalk.orbit_states)
    basis_per_orbit = Counter(len(items) for items in crosswalk.basis_by_orbit.values())
    basis_per_orbit_size = Counter(
        (len(crosswalk.orbit_states[index]), len(items))
        for index, items in crosswalk.basis_by_orbit.items()
    )
    basis_per_state = Counter(len(items) for items in crosswalk.basis_by_state.values())
    states_with_basis_by_orbit = Counter(
        len({state for state in states if state in crosswalk.basis_by_state})
        for states in crosswalk.orbit_states
    )

    summary = {
        "raw_wicks": len(crosswalk.terms),
        "quotient_states": len(set(crosswalk.states)),
        "raw_multiplicity_per_state": dict(sorted(Counter(raw_state_multiplicity.values()).items())),
        "orbit_count": len(crosswalk.orbit_states),
        "orbit_size_histogram": dict(sorted(orbit_sizes.items())),
        "reducer_basis_count": int(crosswalk.reduction["independent_count"]),
        "orbits_with_basis_histogram_by_state_count": dict(sorted(states_with_basis_by_orbit.items())),
        "basis_wicks_per_orbit_histogram": dict(sorted(basis_per_orbit.items())),
        "orbit_size_and_basis_count_histogram": dict(sorted(basis_per_orbit_size.items())),
        "basis_wicks_per_quotient_state_histogram": dict(sorted(basis_per_state.items())),
        "candidate_first_basis_wick_per_orbit": [
            min(crosswalk.basis_by_orbit[index])
            for index in range(len(crosswalk.orbit_states))
        ],
    }
    if summary["raw_wicks"] != 2880 or summary["quotient_states"] != 360:
        raise AssertionError(f"unexpected raw/state count: {summary}")
    if summary["orbit_count"] != 93 or summary["orbit_size_histogram"] != {2: 6, 4: 87}:
        raise AssertionError(f"unexpected orbit count: {summary}")
    if summary["reducer_basis_count"] != 698 or len(crosswalk.basis_by_orbit) != 93:
        raise AssertionError(f"unexpected reducer/orbit coverage: {summary}")
    if summary["basis_wicks_per_orbit_histogram"] != {2: 1, 4: 2, 6: 14, 7: 4, 8: 72}:
        raise AssertionError(f"unexpected basis occupancy: {summary}")
    if summary["orbit_size_and_basis_count_histogram"] != {
        (2, 2): 1,
        (2, 4): 2,
        (2, 6): 3,
        (4, 6): 11,
        (4, 7): 4,
        (4, 8): 72,
    }:
        raise AssertionError(f"unexpected orbit-size/basis cross table: {summary}")

    numeric = _numeric_audit(crosswalk)
    return summary, numeric


def main() -> None:
    summary, numeric = audit()
    print("ppn orbit/reducer crosswalk")
    for key, value in summary.items():
        if key == "candidate_first_basis_wick_per_orbit":
            print(f"{key}: {len(value)} candidates")
            continue
        print(f"{key}: {value}")
    print("\nfirst-state numerical tensor audit")
    print(f"target_state: {numeric['target_state']}")
    print(f"raw_wick_ids: {numeric['raw_wick_ids']}")
    print(f"basis_wick_ids: {numeric['basis_wick_ids']}")
    print("sample ranks and residuals versus Wick 1:")
    for sample in numeric["samples"]:
        residuals = sample["relative_residuals_to_wick_1"]
        formatted = ", ".join(f"{wick_id}:{float(residuals[str(wick_id)]):.6g}" for wick_id in numeric["basis_wick_ids"])
        print(
            f"  {sample['name']}: rank={sample['basis_tensor_rank']}/"
            f"{sample['all_raw_tensor_rank']}, color={sample['color_dimension']}, "
            f"residuals=[{formatted}]"
        )
    formal = numeric["formal_structure"]
    print("formal structure classes under dummy renaming/open-axis swaps:")
    print(f"  ordered factor list: {len(formal['ordered_factor_alpha_classes'])}")
    print(f"  identical-factor graph isomorphism: {len(formal['factor_permutation_graph_classes'])}")
    print(f"  canonical signature hashes: {formal['canonical_alpha_signature_sha256']}")
    print(f"sink proton block-swap witness: {numeric['sink_proton_swap_witness']}")


if __name__ == "__main__":
    main()
