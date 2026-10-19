#!/usr/bin/env python3
"""Numerically audit ppnn projected block invariance under the 32-element action.

The calculation uses the project's baryon operator conventions, an
isospin-degenerate random perambulator, common zero-momentum baryon elementals,
and the fixed ppnn spin-isospin slice of the alpha-particle singlet. It checks
each orbit representative against its five generator images. It does not
evaluate the full 518,400-term sum.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import math
import os
from pathlib import Path
import string
import sys
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Keep each worker's small einsum operations from starting a nested BLAS pool.
for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from verify_code import ppnn_group_action as action
from kernels.wick_reducer import expression_output, operator_model


_WORKER_CONTEXT = None
_PATH_CACHE: Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], object] = {}


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


def _slot_indices(operators):
    _fields, quarks, antiquarks = operator_model.fields_for_operators(operators)
    quark_index = {
        (field.label, field.slot): index for index, field in enumerate(quarks)
    }
    antiquark_index = {
        (field.label, field.slot): index for index, field in enumerate(antiquarks)
    }

    # These label orders match He4State: p up-pairs, n up-singles, n
    # down-pairs, then p down-singles. Each flavor label is assigned once.
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

    def label_map(slots):
        return {label: antiquark_index[slot] for label, slot in enumerate(slots, 1)}

    up_source = label_map(up_source_slots)
    down_source = label_map(down_source_slots)
    up_sink = tuple(quark_index[slot] for slot in up_sink_slots)
    down_sink = tuple(quark_index[slot] for slot in down_sink_slots)

    if len(quarks) != 12 or len(antiquarks) != 12:
        raise AssertionError("ppnn fixture must have 12 quarks and 12 antiquarks")
    if set(up_source) != set(range(1, 7)) or set(down_source) != set(range(1, 7)):
        raise AssertionError("each source flavor must expose six state labels")
    return quarks, antiquarks, up_sink, down_sink, up_source, down_source


def _state_pairings(state, mapping):
    quarks, antiquarks, up_sink, down_sink, up_source, down_source = mapping
    pairings = []
    for up_p1_flip, up_p2_flip, down_n1_flip, down_n2_flip in itertools.product(
        (False, True), repeat=4
    ):
        up_p1 = state.up_protons[0][::-1] if up_p1_flip else state.up_protons[0]
        up_p2 = state.up_protons[1][::-1] if up_p2_flip else state.up_protons[1]
        down_n1 = (
            state.down_neutrons[0][::-1]
            if down_n1_flip
            else state.down_neutrons[0]
        )
        down_n2 = (
            state.down_neutrons[1][::-1]
            if down_n2_flip
            else state.down_neutrons[1]
        )

        assignments = (
            (up_sink[0], up_source[up_p1[0]]),
            (up_sink[1], up_source[up_p1[1]]),
            (up_sink[2], up_source[up_p2[0]]),
            (up_sink[3], up_source[up_p2[1]]),
            (up_sink[4], up_source[state.up_neutrons[0]]),
            (up_sink[5], up_source[state.up_neutrons[1]]),
            (down_sink[0], down_source[state.down_protons[0]]),
            (down_sink[1], down_source[state.down_protons[1]]),
            (down_sink[2], down_source[down_n1[0]]),
            (down_sink[3], down_source[down_n1[1]]),
            (down_sink[4], down_source[down_n2[0]]),
            (down_sink[5], down_source[down_n2[1]]),
        )
        pairing = [-1] * 12
        for quark_index, antiquark_index in assignments:
            pairing[quark_index] = antiquark_index
        if sorted(pairing) != list(range(12)):
            raise AssertionError("state assignment is not a permutation")
        if any(quarks[q].flavor != antiquarks[a].flavor for q, a in enumerate(pairing)):
            raise AssertionError("state assignment connected unequal exact flavors")
        pairings.append(tuple(pairing))

    if len(set(pairings)) != 16:
        raise AssertionError("each ppnn quotient state must contain 16 distinct Wicks")
    return tuple(pairings)


def _wick_sign(pairing, quarks, antiquarks):
    original_positions = tuple(field.index for field in quarks) + tuple(
        antiquarks[index].index for index in pairing
    )
    return core._permutation_sign(original_positions)


def _epsilon3():
    tensor = np.zeros((3, 3, 3), dtype=np.complex128)
    for permutation in itertools.permutations(range(3)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )
        tensor[permutation] = -1 if inversions % 2 else 1
    return tensor


def _spin_isospin_projector():
    """Build the fixed ppnn component of the alpha S=T=0 state, up to scale."""
    spin_state = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    for indices, amplitude in (
        ((0, 1, 0, 1), 0.5),
        ((0, 1, 1, 0), -0.5),
        ((1, 0, 0, 1), -0.5),
        ((1, 0, 1, 0), 0.5),
    ):
        spin_state[indices] = amplitude
    return np.einsum("abcd,efgh->abcdefgh", spin_state, spin_state.conjugate())


def _random_inputs(seed: int):
    rng = np.random.default_rng(seed)

    def random_complex(shape):
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ) / np.sqrt(2.0)

    gamma = core._euclidean_gamma_matrices()
    epsilon = _epsilon3()
    return {
        "C_g5": np.asarray(gamma["Cgamma5"], dtype=np.complex128),
        "C_g5_dagger": np.asarray(gamma["Cgamma5_dagger"], dtype=np.complex128),
        "peram_u_sin_sou": random_complex((3, 4, 3, 4)),
        "VVV_sink": epsilon,
        "conj_VVV_source": epsilon.conjugate(),
    }


def _canonical_equation(equation: str) -> str:
    """Canonicalize labels so equivalent einsum topologies share a path."""
    lhs, rhs = equation.split("->", 1)
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    labels: Dict[str, str] = {}
    for symbol in lhs:
        if symbol != "," and symbol not in labels:
            if len(labels) >= len(alphabet):
                raise ValueError("einsum equation exceeds NumPy's label alphabet")
            labels[symbol] = alphabet[len(labels)]
    canonical_lhs = "".join(labels[symbol] if symbol != "," else "," for symbol in lhs)
    canonical_rhs = "".join(labels[symbol] for symbol in rhs)
    return canonical_lhs + "->" + canonical_rhs


def _worker_init(seed: int):
    global _WORKER_CONTEXT, _PATH_CACHE
    operators = _operators()
    quarks, antiquarks, up_sink, down_sink, up_source, down_source = _slot_indices(
        operators
    )
    values = _random_inputs(seed)
    projector = _spin_isospin_projector()
    _WORKER_CONTEXT = (
        operators,
        (quarks, antiquarks, up_sink, down_sink, up_source, down_source),
        values,
        projector,
    )
    _PATH_CACHE = {}


def _projected_wick(pairing, sign: int, context) -> complex:
    operators, _mapping, values, projector = context
    equation, operands, output = expression_output.contraction_expression(
        operators, pairing, isospin_degenerate=True
    )
    if output != tuple("ABCDEFGH"):
        raise AssertionError(f"unexpected ppnn open-axis order {output}")
    projected_equation = equation.split("->", 1)[0] + ",ABCDEFGH->"
    tensors = tuple(values[name] for name in operands) + (projector,)
    cache_key = (
        _canonical_equation(projected_equation),
        tuple(tensor.shape for tensor in tensors),
    )
    path = _PATH_CACHE.get(cache_key)
    if path is None:
        path, _ = np.einsum_path(
            cache_key[0], *tensors, optimize="greedy"
        )
        _PATH_CACHE[cache_key] = path
    value = np.einsum(
        projected_equation, *tensors, optimize=path
    )
    return sign * complex(value)


def _evaluate_state(state):
    context = _WORKER_CONTEXT
    mapping = context[1]
    quarks, antiquarks = mapping[0], mapping[1]
    result = sum(
        _projected_wick(
            pairing,
            _wick_sign(pairing, quarks, antiquarks),
            context,
        )
        for pairing in _state_pairings(state, mapping)
    )
    return state, result


def _orbit_records(states):
    unseen = set(states)
    records = []
    while unseen:
        representative = min(unseen)
        orbit = {representative}
        pending = [representative]
        while pending:
            state = pending.pop()
            for generator in action.HE4_GENERATORS:
                image = generator(state)
                if image not in orbit:
                    orbit.add(image)
                    pending.append(image)
        unseen.difference_update(orbit)
        records.append((representative, tuple(sorted(orbit))))
    records.sort(key=lambda item: item[0])
    return records


def _scaled_difference(left: complex, right: complex) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-30)


def _validated_orbit_records():
    states = tuple(action.he4_states())
    if len(states) != 32400:
        raise AssertionError(f"expected 32400 ppnn quotient states, got {len(states)}")
    records = _orbit_records(states)
    histogram = Counter(len(orbit) for _representative, orbit in records)
    if len(records) != 1068 or histogram != Counter({4: 4, 8: 14, 16: 83, 32: 967}):
        raise AssertionError(f"unexpected ppnn orbit structure: {dict(histogram)}")
    if sum(size * count for size, count in histogram.items()) != len(states):
        raise AssertionError("ppnn orbits do not partition the quotient states")
    return states, records, histogram


def _evaluate_states(states, seed: int, workers: int):
    ordered_states = sorted(states)
    started = time.perf_counter()
    if workers == 1:
        _worker_init(seed)
        state_values = dict(map(_evaluate_state, ordered_states))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(seed,),
        ) as pool:
            chunksize = max(1, len(ordered_states) // (workers * 16))
            state_values = dict(
                pool.map(_evaluate_state, ordered_states, chunksize=chunksize)
            )
    return state_values, time.perf_counter() - started


def reduced_sum(seed: int, workers: int, max_orbits: int | None = None) -> dict:
    states, records, histogram = _validated_orbit_records()
    selected = records if max_orbits is None else records[:max_orbits]
    state_values, evaluation_seconds = _evaluate_states(
        (representative for representative, _orbit in selected), seed, workers
    )
    weighted_values = [
        (len(orbit), state_values[representative])
        for representative, orbit in selected
    ]
    weighted_sum = complex(
        math.fsum(size * value.real for size, value in weighted_values),
        math.fsum(size * value.imag for size, value in weighted_values),
    )
    covered_states = sum(size for size, _value in weighted_values)
    return {
        "seed": seed,
        "workers": workers,
        "state_count": len(states),
        "raw_wicks_per_state": 16,
        "orbit_count": len(records),
        "orbit_histogram": dict(sorted(histogram.items())),
        "computed_orbits": len(selected),
        "covered_states": covered_states,
        "full_state_coverage": len(selected) == len(records),
        "evaluated_state_blocks": len(state_values),
        "evaluated_raw_contracts": len(state_values) * 16,
        "weighted_projected_sum": weighted_sum,
        "evaluation_seconds": evaluation_seconds,
    }


def audit(seed: int, workers: int, max_orbits: int | None = None) -> dict:
    states, records, histogram = _validated_orbit_records()

    selected = records if max_orbits is None else records[:max_orbits]
    required_states = {
        image
        for representative, _orbit in selected
        for image in (
            representative,
            *(generator(representative) for generator in action.HE4_GENERATORS),
        )
    }
    ordered_required = sorted(required_states)
    state_values, evaluation_seconds = _evaluate_states(
        ordered_required, seed, workers
    )

    generator_names = (
        "sink_protons",
        "sink_neutrons",
        "source_protons",
        "source_neutrons",
        "isospin_flip",
    )
    residuals = defaultdict(list)
    for representative, _orbit in selected:
        base = state_values[representative]
        for name, generator in zip(generator_names, action.HE4_GENERATORS):
            image_value = state_values[generator(representative)]
            residuals[name].append(_scaled_difference(image_value, base))

    return {
        "seed": seed,
        "workers": workers,
        "raw_wick_count_formula": "6! * 6! = 518400 (not directly evaluated as a full sum)",
        "state_count": len(states),
        "raw_wicks_per_state": 16,
        "orbit_count": len(records),
        "orbit_histogram": dict(sorted(histogram.items())),
        "checked_orbits": len(selected),
        "checked_generator_edges": len(selected) * len(generator_names),
        "evaluated_state_blocks": len(state_values),
        "evaluated_raw_contracts": len(state_values) * 16,
        "max_generator_edge_residuals": {
            name: max(values, default=0.0) for name, values in residuals.items()
        },
        "rms_generator_edge_residuals": {
            name: math.sqrt(sum(value * value for value in values) / len(values))
            if values
            else 0.0
            for name, values in residuals.items()
        },
        "evaluation_seconds": evaluation_seconds,
        "full_representative_edge_audit": max_orbits is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="run the reduced sum for each listed deterministic random seed",
    )
    parser.add_argument(
        "--reduced-sum",
        action="store_true",
        help="compute the orbit-size-weighted projected sum from representatives",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--max-orbits",
        type=int,
        default=None,
        help="limit numerical work to the first N orbits for a quick smoke test",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_orbits is not None and args.max_orbits < 1:
        parser.error("--max-orbits must be at least 1")

    seeds = args.seeds or [args.seed if args.seed is not None else 20261009]
    if args.seeds is not None and not args.reduced_sum:
        parser.error("--seeds can only be used with --reduced-sum")

    if args.reduced_sum:
        for seed in seeds:
            result = reduced_sum(seed, args.workers, args.max_orbits)
            scope = (
                "full 1,068-orbit weighted sum"
                if result["full_state_coverage"]
                else f"partial weighted sum ({result['computed_orbits']} orbits)"
            )
            print(f"ppnn projected orbit-weighted sum, seed={seed}")
            print(
                f"quotient states: {result['state_count']} x "
                f"{result['raw_wicks_per_state']} Wicks/state; "
                f"orbits: {result['orbit_count']} {result['orbit_histogram']}"
            )
            print(
                f"evaluated: {result['evaluated_state_blocks']} representative blocks, "
                f"{result['evaluated_raw_contracts']} raw contract evaluations; "
                f"orbit-weighted state coverage: {result['covered_states']}"
            )
            value = result["weighted_projected_sum"]
            print(f"weighted projected sum: {value.real:.17e}{value.imag:+.17e}j")
            print(f"evaluation seconds: {result['evaluation_seconds']:.3f}")
            print(f"scope: {scope}; direct 518,400-term sum not evaluated")
        return

    seed = seeds[0]
    result = audit(seed, args.workers, args.max_orbits)
    print("ppnn projected orbit-generator numerical audit")
    print(f"raw Wick count: {result['raw_wick_count_formula']}")
    print(
        f"quotient states: {result['state_count']} x "
        f"{result['raw_wicks_per_state']} Wicks/state; "
        f"orbits: {result['orbit_count']} {result['orbit_histogram']}"
    )
    print(
        f"numerically checked: {result['checked_orbits']} representatives, "
        f"{result['checked_generator_edges']} generator edges, "
        f"{result['evaluated_state_blocks']} distinct state blocks, "
        f"{result['evaluated_raw_contracts']} raw contract evaluations"
    )
    print("max scaled residual by generator:")
    for name, residual in result["max_generator_edge_residuals"].items():
        print(f"  {name}: {residual:.6e}")
    print("RMS scaled residual by generator:")
    for name, residual in result["rms_generator_edge_residuals"].items():
        print(f"  {name}: {residual:.6e}")
    print(f"evaluation seconds: {result['evaluation_seconds']:.3f}")
    if not result["full_representative_edge_audit"]:
        print("scope: smoke test only; all 1,068 representative edges were not evaluated")
    else:
        print("scope: all 1,068 representatives and their five generator edges")
    print("The 518,400-term full projected Wick sum was not directly evaluated.")


if __name__ == "__main__":
    main()
