#!/usr/bin/env python3
"""Test whether projected ppn spin components admit exact 93-orbit weights.

This is a standalone numerical audit. It compares projected orbit sums with
both a single raw representative and the representative state's sum of eight
signed raw Wicks, using independent random peram inputs.
"""

from __future__ import annotations

import ast
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import opt_einsum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from kernels.wick_reducer import expression_output, operator_model
from verify_code.ppn_orbit_crosswalk_audit import build_crosswalk


def _candidate_members() -> dict:
    path = PROJECT_ROOT / "verify_code" / "ppn_orbit_candidate_contracts.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    value = next(
        node.value for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "ppn_orbit_candidate_members"
            for target in node.targets
        )
    )
    return ast.literal_eval(value)


ppn_orbit_candidate_members = _candidate_members()


def _epsilon3() -> np.ndarray:
    tensor = np.zeros((3, 3, 3), dtype=np.complex128)
    for i, j, k in itertools.permutations(range(3)):
        inversions = (i > j) + (i > k) + (j > k)
        tensor[i, j, k] = -1 if inversions % 2 else 1
    return tensor


def _inputs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def random_complex(shape):
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ) / np.sqrt(2.0)

    gamma = core._euclidean_gamma_matrices()
    epsilon = _epsilon3()
    source_vvv = random_complex(())
    return {
        "C_g5": np.asarray(gamma["Cgamma5"], dtype=np.complex128),
        "C_g5_dagger": np.asarray(gamma["Cgamma5_dagger"], dtype=np.complex128),
        "peram_u_sin_sou": random_complex((3, 4, 3, 4)),
        "VVV_sink": random_complex(()) * epsilon,
        "conj_VVV_source": source_vvv.conjugate() * epsilon,
    }


def _spin_projectors() -> dict[str, np.ndarray]:
    """Return normalized three-nucleon J=1/2, M=1/2 spin states.

    Dirac components 0 and 1 are treated as nonrelativistic up/down; lower
    components vanish. These are candidate spin projections, not a claim that
    either is the paper's full spin-isospin wavefunction.
    """
    pair_singlet = np.zeros((4, 4, 4), dtype=np.complex128)
    pair_singlet[0, 1, 0] = 1 / np.sqrt(2)
    pair_singlet[1, 0, 0] = -1 / np.sqrt(2)

    proton_triplet = np.zeros_like(pair_singlet)
    proton_triplet[0, 0, 1] = 2 / np.sqrt(6)
    proton_triplet[0, 1, 0] = -1 / np.sqrt(6)
    proton_triplet[1, 0, 0] = -1 / np.sqrt(6)

    return {
        "pp_spin0_n_up": pair_singlet,
        "pp_spin1_coupled_J1/2": proton_triplet,
    }


def _contract_tensor(operators, pairing, values) -> np.ndarray:
    equation, operands, axes = expression_output.contraction_expression(
        operators, pairing, isospin_degenerate=True,
    )
    if axes != tuple("ABCDEF"):
        raise AssertionError(f"unexpected free-axis order {axes}")
    return opt_einsum.contract(equation, *(values[name] for name in operands))


def _orbit_lookup(crosswalk):
    state_to_orbit = crosswalk.state_to_orbit
    return {
        wick_id: state_to_orbit[state]
        for wick_id, state in enumerate(crosswalk.states, start=1)
    }


def run(seed: int, crosswalk, projectors):
    operators = operator_model.operator_library()["ppn"]
    values = _inputs(seed)
    orbit_by_wick = _orbit_lookup(crosswalk)
    state_multiplicity = Counter(crosswalk.states)
    if len(state_multiplicity) != 360 or set(state_multiplicity.values()) != {8}:
        raise AssertionError("expected 360 label states with 8 raw Wicks per state")
    actual_members = {}
    for wick_id, orbit in orbit_by_wick.items():
        actual_members.setdefault(orbit, []).append(wick_id)
    candidates = {}
    candidate_state_by_orbit = {}
    for item in ppn_orbit_candidate_members.values():
        representative = int(item["representative_wick_id"])
        orbit = orbit_by_wick[representative]
        raw_ids = tuple(int(w) for w in item["raw_wick_ids"])
        if set(raw_ids) != set(actual_members.get(orbit, ())):
            raise AssertionError(
                f"candidate orbit represented by Wick {representative} does not match crosswalk"
            )
        candidates[orbit] = representative
        candidate_state_by_orbit[orbit] = crosswalk.states[representative - 1]
    flattened_ids = [
        int(w)
        for item in ppn_orbit_candidate_members.values()
        for w in item["raw_wick_ids"]
    ]
    if len(candidates) != 93 or set(candidates) != set(actual_members):
        raise AssertionError("candidate representatives do not cover exactly 93 crosswalk orbits")
    if len(flattened_ids) != len(set(flattened_ids)) or set(flattened_ids) != set(orbit_by_wick):
        raise AssertionError("candidate raw member lists do not partition all Wick IDs")
    candidate_values = {name: {} for name in projectors}
    signed_scalars = {name: {} for name in projectors}
    state_scalars = {name: {} for name in projectors}
    started = time.perf_counter()
    for wick_id, term in enumerate(crosswalk.terms, start=1):
        orbit = orbit_by_wick[wick_id]
        tensor = _contract_tensor(operators, term.pairing, values)
        state = crosswalk.states[wick_id - 1]
        for name, spin in projectors.items():
            signed_scalars[name].setdefault(orbit, 0j)
            state_scalars[name].setdefault(state, 0j)
            scalar = complex(np.einsum("abc,def,abcdef->", spin.conj(), spin, tensor))
            signed_scalars[name][orbit] += term.sign * scalar
            state_scalars[name][state] += term.sign * scalar
            if wick_id == candidates[orbit]:
                candidate_values[name][orbit] = term.sign * scalar
    elapsed = time.perf_counter() - started

    audit = {}
    for name in projectors:
        ratios = {}
        block_ratios = {}
        absolute_errors = []
        block_absolute_errors = []
        full = sum(signed_scalars[name].values())
        integer_block_sum = sum(
            len(crosswalk.orbit_states[orbit])
            * state_scalars[name][candidate_state_by_orbit[orbit]]
            for orbit in candidates
        )
        candidate_block_l1 = float(sum(
            abs(state_scalars[name][state])
            for state in candidate_state_by_orbit.values()
        ))
        orbit_sum_l1 = float(sum(abs(value) for value in signed_scalars[name].values()))
        integer_block_error = abs(full - integer_block_sum)
        for orbit in candidates:
            candidate = candidate_values[name][orbit]
            numerator = signed_scalars[name][orbit]
            if abs(candidate) > 1e-11:
                ratios[orbit] = numerator / candidate
            else:
                ratios[orbit] = None
                absolute_errors.append(abs(numerator))
            block = state_scalars[name][candidate_state_by_orbit[orbit]]
            if abs(block) > 1e-11:
                block_ratios[orbit] = numerator / block
            else:
                block_ratios[orbit] = None
                block_absolute_errors.append(abs(numerator))
        audit[name] = {
            "full": full,
            "ratios": ratios,
            "block_ratios": block_ratios,
            "candidate_values": candidate_values[name],
            "candidate_state_values": {
                orbit: state_scalars[name][state]
                for orbit, state in candidate_state_by_orbit.items()
            },
            "candidate_l1": float(sum(abs(value) for value in candidate_values[name].values())),
            "candidate_block_l1": candidate_block_l1,
            "orbit_sum_l1": orbit_sum_l1,
            "integer_block_reconstruction_error": float(integer_block_error),
            "integer_block_reconstruction_scaled_error": float(
                integer_block_error / max(candidate_block_l1, orbit_sum_l1, 1e-30)
            ),
            "candidate_zero_orbit_sum_norm": float(np.linalg.norm(absolute_errors)),
            "block_zero_orbit_sum_norm": float(np.linalg.norm(block_absolute_errors)),
            "block_orbit_size_residual": float(max(
                (
                    abs(block_ratios[orbit] - len(crosswalk.orbit_states[orbit]))
                    for orbit in candidates
                    if block_ratios[orbit] is not None
                ),
                default=float("nan"),
            )),
            "nonzero_candidate_ratios": sum(value is not None for value in ratios.values()),
        }
    return audit, elapsed


def main() -> None:
    crosswalk = build_crosswalk()
    projectors = _spin_projectors()
    seeds = (20261021, 20261022, 20261023)
    results = []
    print("ppn projected 93-orbit reconstruction audit")
    print("raw Wick sum uses the current built-in ppn operator, Nc=3 epsilon VVV, and isospin-degenerate peram")
    print("spin projection uses upper Dirac components 0/1 and two normalized J=1/2, M=1/2 states")
    print(f"counts: raw={len(crosswalk.terms)}, orbits={len(crosswalk.orbit_states)}, seeds={seeds}")
    print(f"orbit-size histogram: {dict(sorted(Counter(map(len, crosswalk.orbit_states)).items()))}")
    for seed in seeds:
        result, elapsed = run(seed, crosswalk, projectors)
        results.append(result)
        print(f"seed={seed} runtime={elapsed:.2f}s")
        for name, item in result.items():
            ratios = [v for v in item["ratios"].values() if v is not None]
            print(
                f"  {name}: nonzero-ratios={len(ratios)}/93 "
                f"|full|={abs(item['full']):.3e} "
                f"candidate-L1={item['candidate_l1']:.3e} "
                f"orbit-sum-L1={item['orbit_sum_l1']:.3e} "
                f"ratio-range=({min(r.real for r in ratios):.6g},{max(r.real for r in ratios):.6g}) "
                f"max-imag-ratio={max(abs(r.imag) for r in ratios):.3e} "
                f"zero-candidate/orbit-sum-norm={item['candidate_zero_orbit_sum_norm']:.3e}"
            )
            block_ratios = [v for v in item["block_ratios"].values() if v is not None]
            print(
                f"    8-term-state-block ratios: usable={len(block_ratios)}/93 "
                f"range=({min((r.real for r in block_ratios), default=float('nan')):.6g},"
                f"{max((r.real for r in block_ratios), default=float('nan')):.6g}) "
                f"max-imag={max((abs(r.imag) for r in block_ratios), default=float('nan')):.3e} "
                f"max-deviation-from-orbit-size={item['block_orbit_size_residual']:.3e} "
                f"block-L1={item['candidate_block_l1']:.3e} "
                f"integer-weight scaled reconstruction-error="
                f"{item['integer_block_reconstruction_scaled_error']:.3e} "
                f"zero-block/orbit-sum-norm={item['block_zero_orbit_sum_norm']:.3e}"
            )
    for name in projectors:
        first = results[0][name]["ratios"]
        for seed_index in range(1, len(seeds)):
            second = results[seed_index][name]["ratios"]
            usable = [
                orbit for orbit in first
                if first[orbit] is not None and second[orbit] is not None
            ]
            differences = [abs(first[orbit] - second[orbit]) for orbit in usable]
            scale = max(1.0, max(abs(first[orbit]) for orbit in usable))
            cross_prediction = sum(
                first[orbit] * results[seed_index][name]["candidate_values"][orbit]
                for orbit in usable
            )
            cross_error = abs(results[seed_index][name]["full"] - cross_prediction)
            error_scale = max(
                results[seed_index][name]["candidate_l1"],
                results[seed_index][name]["orbit_sum_l1"],
                1e-30,
            )
            cross_residual = cross_error / error_scale
            print(
                f"{name}: raw-term seed-{seeds[0]} weights on seed-{seeds[seed_index]} "
                f"scaled error={cross_residual:.3e}; "
                f"max-ratio-difference={max(differences, default=float('nan')):.3e}"
            )

            first_blocks = results[0][name]["block_ratios"]
            second_blocks = results[seed_index][name]["block_ratios"]
            block_usable = [
                orbit for orbit in first_blocks
                if first_blocks[orbit] is not None and second_blocks[orbit] is not None
            ]
            block_differences = [
                abs(first_blocks[orbit] - second_blocks[orbit])
                for orbit in block_usable
            ]
            block_prediction = sum(
                first_blocks[orbit]
                * results[seed_index][name]["candidate_state_values"][orbit]
                for orbit in block_usable
            )
            block_error = abs(results[seed_index][name]["full"] - block_prediction)
            block_scale = max(
                results[seed_index][name]["candidate_block_l1"],
                results[seed_index][name]["orbit_sum_l1"],
                1e-30,
            )
            block_weight_scale = max(
                1.0,
                max((abs(first_blocks[orbit]) for orbit in block_usable), default=0.0),
            )
            print(
                f"{name}: 8-term-block seed-{seeds[0]} weights on seed-{seeds[seed_index]} "
                f"scaled error={block_error / block_scale:.3e}; "
                f"max-ratio-difference="
                f"{max(block_differences, default=float('nan')):.3e}; "
                f"relative={max(block_differences, default=float('nan')) / block_weight_scale:.3e}"
            )


if __name__ == "__main__":
    main()
