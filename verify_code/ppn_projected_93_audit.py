#!/usr/bin/env python3
"""Audit a 93-block reconstruction of a projected ppn correlator.

Each representative block is independently evaluated from its eight signed
raw Wick terms. Its exact orbit-size multiplicity reconstructs the direct
2,880-term projected Wick sum.
"""

from __future__ import annotations

import ast
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import opt_einsum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from verify_code import ppnn_group_action as action
from verify_code import ppn_orbit_contracts
from kernels.wick_reducer import expression_output
from kernels.wick_reducer.operator_model import operator_library
from kernels.wick_reducer.reducer import reduce_case


def _orbit_members() -> Dict[int, Mapping[str, object]]:
    path = PROJECT_ROOT / "verify_code" / "ppn_orbit_candidate_contracts.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "ppn_orbit_candidate_members"
            for target in node.targets
        ):
            result = ast.literal_eval(node.value)
            return {int(key): value for key, value in result.items()}
    raise ValueError("the generated 93-orbit member table is missing")


def _projector_pp_singlet_n_up() -> np.ndarray:
    """Project the fixed ppn spin slice of the ^3He operator in Eq. (6).

    In the ordering (p1,p2,n), this is (|+,-,+>-|-,+,+>)/sqrt(6), with the
    same state at source and sink. The overall normalization does not affect
    the exact orbit reconstruction.
    Axis order is sink p1,p2,n followed by source p1bar,p2bar,nbar.
    """
    state = np.zeros((2, 2, 2), dtype=np.complex128)
    state[0, 1, 0] = 1.0 / np.sqrt(6.0)
    state[1, 0, 0] = -1.0 / np.sqrt(6.0)
    spin_projector = np.einsum("abc,def->abcdef", state, state.conjugate())
    projector = np.zeros((4, 4, 4, 4, 4, 4), dtype=np.complex128)
    projector[:2, :2, :2, :2, :2, :2] = spin_projector
    return projector


def _random_inputs(seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def random_complex(shape=()):
        return (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ) / np.sqrt(2.0)

    vvv = np.zeros((3, 3, 3), dtype=np.complex128)
    for permutation in itertools.permutations(range(3)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )
        vvv[permutation] = (-1 if inversions % 2 else 1) * random_complex()

    gamma = core._euclidean_gamma_matrices()
    return {
        "C_g5": np.asarray(gamma["Cgamma5"], dtype=np.complex128),
        "C_g5_dagger": np.asarray(gamma["Cgamma5_dagger"], dtype=np.complex128),
        "peram_u_sin_sou": random_complex((3, 4, 3, 4)),
        "VVV_sink": vvv,
        "conj_VVV_source": vvv.conjugate(),
    }


def _projected_contract(operators, pairing: Sequence[int], values, projector) -> complex:
    equation, operands, output = expression_output.contraction_expression(
        operators, pairing, isospin_degenerate=True,
    )
    if output != tuple("ABCDEF"):
        raise AssertionError(f"unexpected open-axis order: {output}")
    lhs = equation.split("->", 1)[0]
    projected_equation = lhs + ",ABCDEF->"
    result = opt_einsum.contract(
        projected_equation,
        *(values[name] for name in operands),
        projector,
        optimize="greedy",
    )
    return complex(result)


def _fit_ratio(representative: np.ndarray, orbit_sum: np.ndarray) -> Tuple[complex, float]:
    denominator = float(np.vdot(representative, representative).real)
    if denominator <= 1.0e-28:
        return 0.0j, float(np.linalg.norm(orbit_sum))
    coefficient = np.vdot(representative, orbit_sum) / denominator
    residual = np.linalg.norm(orbit_sum - coefficient * representative)
    scale = max(float(np.linalg.norm(orbit_sum)), 1.0e-28)
    return complex(coefficient), float(residual / scale)


def audit(seeds: Sequence[int]) -> dict:
    operators = operator_library()["ppn"]
    report = reduce_case("ppn", operators, mode="detailed")
    graph_by_id = {
        int(item["wick_index"]): item for item in report["graph_records"]
    }
    orbits = _orbit_members()
    raw_candidate_by_orbit = {
        orbit: int(record["representative_wick_id"])
        for orbit, record in orbits.items()
    }
    all_ids = [
        int(wick_id)
        for record in orbits.values()
        for wick_id in record["raw_wick_ids"]
    ]
    if len(orbits) != 93 or len(all_ids) != 2880 or set(all_ids) != set(graph_by_id):
        raise AssertionError("93 orbit records must partition all 2,880 raw Wick IDs")

    wick_id_by_pairing = {
        tuple(item["pairing"]): int(item["wick_index"])
        for item in report["graph_records"]
    }
    states = tuple(action.he3_states())
    state_raw_ids = {
        state: tuple(sorted(
            wick_id_by_pairing[pairing]
            for pairing in ppn_orbit_contracts._state_pairings(state, operators)
        ))
        for state in states
    }
    if len(states) != 360 or any(len(ids) != 8 for ids in state_raw_ids.values()):
        raise AssertionError("expected 360 quotient states with eight raw Wick terms each")
    state_wick_ids = {
        state: wick_id_by_pairing[ppn_orbit_contracts._state_pairing(state, operators)]
        for state in states
    }
    state_orbits = ppn_orbit_contracts._state_orbits(states)
    state_orbits.sort(key=lambda orbit: min(state_wick_ids[state] for state in orbit))
    block_records = []
    for orbit_id, orbit_states in enumerate(state_orbits):
        representative_state = min(orbit_states, key=lambda state: state_wick_ids[state])
        member_raw_ids = tuple(sorted(
            wick_id
            for state in orbit_states
            for wick_id in state_raw_ids[state]
        ))
        candidate_raw_ids = [
            wick_id
            for wick_id in member_raw_ids
            if wick_id in {item["wick_index"] for item in report["independent_wicks"]}
        ]
        block_records.append({
            "orbit_id": orbit_id,
            "states": tuple(orbit_states),
            "canonical_state_wick_ids": tuple(sorted(
                state_wick_ids[state] for state in orbit_states
            )),
            "representative_state": representative_state,
            "representative_raw_ids": state_raw_ids[representative_state],
            "representative_wick_id": state_wick_ids[representative_state],
            "member_raw_ids": member_raw_ids,
            "member_state_count": len(orbit_states),
        })
    if len(block_records) != 93 or set(
        wick_id for record in block_records for wick_id in record["member_raw_ids"]
    ) != set(graph_by_id):
        raise AssertionError("93 state-block orbits must partition all 2,880 raw Wicks")

    candidate_by_members = {
        frozenset(int(wick_id) for wick_id in record["raw_wick_ids"]):
        (orbit_id, record)
        for orbit_id, record in orbits.items()
    }
    for record in block_records:
        candidate = candidate_by_members.get(frozenset(record["member_raw_ids"]))
        if candidate is None:
            raise AssertionError("state-block orbit does not match generated 93-orbit table")
        candidate_orbit_id, candidate_record = candidate
        if int(candidate_record["representative_wick_id"]) not in candidate_record["raw_wick_ids"]:
            raise AssertionError("generated orbit representative is not one of its raw members")
        record["candidate_orbit_id"] = candidate_orbit_id

    projector = _projector_pp_singlet_n_up()
    sample_values: Dict[int, Dict[int, complex]] = {}
    sample_raw_orbit_sums: Dict[int, Dict[int, complex]] = {}
    sample_block_orbit_sums: Dict[int, Dict[int, complex]] = {}
    sample_block_representatives: Dict[int, Dict[int, complex]] = {}
    sample_independent_block_representatives: Dict[int, Dict[int, complex]] = {}
    reduced_totals: Dict[int, complex] = {}
    independent_block_evaluation_counts: Dict[int, int] = {}
    totals = {}
    started = time.perf_counter()
    for seed in seeds:
        values = _random_inputs(seed)
        by_wick: Dict[int, complex] = {}
        for wick_id, graph in graph_by_id.items():
            by_wick[wick_id] = int(graph["wick_sign"]) * _projected_contract(
                operators, graph["pairing"], values, projector,
            )
        sample_values[seed] = by_wick
        raw_orbit_sums = {
            orbit: sum(by_wick[int(wick_id)] for wick_id in record["raw_wick_ids"])
            for orbit, record in orbits.items()
        }
        sample_raw_orbit_sums[seed] = raw_orbit_sums
        state_block_sums = {
            state: sum(by_wick[wick_id] for wick_id in ids)
            for state, ids in state_raw_ids.items()
        }
        sample_block_orbit_sums[seed] = {
            record["orbit_id"]: sum(
                state_block_sums[state] for state in record["states"]
            )
            for record in block_records
        }
        sample_block_representatives[seed] = {
            record["orbit_id"]: state_block_sums[record["representative_state"]]
            for record in block_records
        }
        independent_representatives = {}
        for record in block_records:
            independent_representatives[record["orbit_id"]] = sum(
                int(graph_by_id[wick_id]["wick_sign"])
                * _projected_contract(
                    operators,
                    graph_by_id[wick_id]["pairing"],
                    values,
                    projector,
                )
                for wick_id in record["representative_raw_ids"]
            )
        sample_independent_block_representatives[seed] = independent_representatives
        independent_block_evaluation_counts[seed] = sum(
            len(record["representative_raw_ids"]) for record in block_records
        )
        reduced_totals[seed] = sum(
            record["member_state_count"]
            * independent_representatives[record["orbit_id"]]
            for record in block_records
        )
        totals[seed] = sum(by_wick.values())
        print(
            f"seed={seed} projected_full={totals[seed]:.12g} "
            f"seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )

    raw_coefficients: Dict[int, complex] = {}
    raw_orbit_residuals: Dict[int, float] = {}
    block_coefficients: Dict[int, complex] = {
        record["orbit_id"]: record["member_state_count"]
        for record in block_records
    }
    block_orbit_residuals: Dict[int, float] = {
        record["orbit_id"]: 0.0 for record in block_records
    }
    for orbit_id, record in enumerate(orbits.values()):
        rep_id = raw_candidate_by_orbit[orbit_id]
        rep_raw = np.asarray([sample_values[seed][rep_id] for seed in seeds])
        raw_sum = np.asarray([sample_raw_orbit_sums[seed][orbit_id] for seed in seeds])
        coefficient, residual = _fit_ratio(rep_raw, raw_sum)
        raw_coefficients[orbit_id] = coefficient
        raw_orbit_residuals[orbit_id] = residual

    for record in block_records:
        orbit_id = record["orbit_id"]
        weight = record["member_state_count"]
        for seed in seeds:
            orbit_sum = sample_block_orbit_sums[seed][orbit_id]
            reconstructed_orbit = (
                weight * sample_independent_block_representatives[seed][orbit_id]
            )
            residual = abs(orbit_sum - reconstructed_orbit) / max(
                abs(orbit_sum), abs(reconstructed_orbit), 1.0e-28
            )
            block_orbit_residuals[orbit_id] = max(
                block_orbit_residuals[orbit_id], residual
            )

    raw_reconstructed = {
        seed: sum(
            raw_coefficients[orbit] * sample_values[seed][raw_candidate_by_orbit[orbit]]
            for orbit in orbits
        )
        for seed in seeds
    }
    block_reconstructed = {
        seed: reduced_totals[seed] for seed in seeds
    }
    raw_full_residuals = {
        seed: abs(totals[seed] - raw_reconstructed[seed])
        / max(abs(totals[seed]), 1.0e-28)
        for seed in seeds
    }
    block_full_residuals = {
        seed: abs(totals[seed] - block_reconstructed[seed])
        / max(abs(totals[seed]), 1.0e-28)
        for seed in seeds
    }
    elapsed = time.perf_counter() - started
    return {
        "seeds": tuple(seeds),
        "raw_candidate_ids": raw_candidate_by_orbit,
        "block_records": block_records,
        "raw_coefficients": raw_coefficients,
        "raw_orbit_residuals": raw_orbit_residuals,
        "block_coefficients": block_coefficients,
        "block_orbit_residuals": block_orbit_residuals,
        "totals": totals,
        "raw_reconstructed": raw_reconstructed,
        "block_reconstructed": block_reconstructed,
        "reduced_totals": reduced_totals,
        "independent_block_evaluation_counts": independent_block_evaluation_counts,
        "independent_representative_blocks": sample_independent_block_representatives,
        "raw_wick_records": {
            wick_id: {
                "wick_sign": int(graph["wick_sign"]),
                "pairing": list(graph["pairing"]),
            }
            for wick_id, graph in graph_by_id.items()
        },
        "raw_full_residuals": raw_full_residuals,
        "block_full_residuals": block_full_residuals,
        "elapsed": elapsed,
        "projector": "Eq. (6) fixed ppn slice (|+,-,+>-|-,+,+>)/sqrt(6), upper components, at source and sink",
        "raw_wicks": report["wick_count"],
        "candidate_count": len(orbits),
    }


def _format_complex(value: complex) -> str:
    return f"({value.real:.12g}{value.imag:+.12g}j)"


def write_report(result: dict) -> Path:
    seeds = result["seeds"]
    max_raw_orbit_residual = max(result["raw_orbit_residuals"].values())
    max_raw_total_residual = max(result["raw_full_residuals"].values())
    max_block_orbit_residual = max(result["block_orbit_residuals"].values())
    max_block_total_residual = max(result["block_full_residuals"].values())
    passed = max_block_orbit_residual < 1.0e-10 and max_block_total_residual < 1.0e-10
    histogram = {
        size: sum(record["member_state_count"] == size for record in result["block_records"])
        for size in sorted({record["member_state_count"] for record in result["block_records"]})
    }
    lines = [
        "# ppn: What the 93 Projected Contractions Are",
        "",
        "## Finding",
        "",
        "In this reconstruction, **93 is the number of projected, pre-summed state blocks**, not 93 individual raw Wick contractions and not 93 open six-Dirac-index tensors. A block first combines eight signed raw Wick terms that differ by the three same-flavor slot exchanges internal to the three nucleons. The 360 resulting label states are then grouped by independently exchanging the two proton baryons at sink and source.",
        "",
        "The exact reduced formula audited by the script is",
        "",
        "$$F_{ppn}=\\sum_{s\\in S}b_s=\\sum_{O\\in S/G}|O|b_{s_O},\\qquad b_s=\\sum_{w:s(w)=s}\\operatorname{sgn}(w)\\,\\langle P,C_w\\rangle.$$",
        "",
        "Here `C_w` is the local code's open-index Wick tensor; `P` projects its six spin axes onto the fixed ppn charge slice of the \\({}^3\\mathrm{He}\\) operator. Each `b_s` sums 8 raw Wick terms. The orbit multiplicity `|O|` is 2 or 4, so the 93 representative blocks stand for `6*2 + 87*4 = 360` state blocks and therefore `8*360 = 2880` raw terms.",
        "",
        "## Why The Blocks Are Equal",
        "",
        "The spin slice is `(|+,-,+> - |-,+,+>)/sqrt(6)` in the ordering (p1,p2,n), at both source and sink. It is the fixed ppn component of the six-term \\({}^3\\mathrm{He}\\) interpolator in Eq. (6) of [Yamazaki (arXiv:1012.0410)](https://arxiv.org/abs/1012.0410). For either one-sided proton-block exchange, the signed Wick block contributes a minus sign and the spin projector contributes another minus sign, so `b_(g s)=b_s`. The group is `S2_sink x S2_source`; its 93 orbits have sizes " + str(histogram) + ".",
        "",
        "## Why 698 And 93 Differ",
        "",
        "The local 698-term audit preserves the full open tensor `T_{ABCDEF}=sum_w sign(w) C_w`, with all six Dirac indices available, and reproduces it to about `1.4e-15`. The 93-block identity applies only after contracting `T` with the fixed spin-flavor projector `P`, producing one scalar correlator. Projection discards tensor components and permits cancellations, so a basis size for the open tensor and a count for projected blocks are different targets. In the earlier open-tensor test, the span of 93 one-raw-term candidates left residuals of `0.82-0.86`; see [ppn_orbit_numeric_audit.md](ppn_orbit_numeric_audit.md).",
        "",
        "The [1205.0585 paper](https://arxiv.org/abs/1205.0585) also separates these notions: it cites 93 as an earlier symmetry-based permutation count, while its own unified contraction algorithm has separate `N_perm` and `N_contr` counts (for the nonrelativistic 3He setup, 360 and 3240 in Table 8). Thus neither 93 nor 3240 can be compared directly with the local 698 without matching the exact object and contraction stage.",
        "",
        "## Audit",
        "",
        f"- Projector: `{result['projector']}`.",
        "- All six open Dirac axes are projected before terms are compared; only upper components 0 and 1 survive.",
        "- Perams are independent generic complex isospin-degenerate matrices, and color tensors use `N_c=3` epsilon symbols.",
        f"- Direct reference: {result['raw_wicks']} signed Wick contractions per seed.",
        f"- Reduced evaluation: 93 representative blocks, independently reevaluating {next(iter(result['independent_block_evaluation_counts'].values()))} underlying raw contractions per seed (8 per block).",
        "- Orbit weights are the exact integer sizes 2 or 4; they are not fitted from numerical data.",
        f"- Raw single-Wick representative max orbit residual: `{max_raw_orbit_residual:.6e}`; fitted raw-93 full-sum residual: `{max_raw_total_residual:.6e}`.",
        f"- Eight-Wick block max orbit residual: `{max_block_orbit_residual:.6e}`; independently recomputed block-93 full-sum residual: `{max_block_total_residual:.6e}`.",
        f"- Full reconstruction at tolerance `1e-10`: **{'PASS' if passed else 'FAIL'}**.",
        f"- Orbit histogram: `{histogram}`; generated member table: [ppn_projected_93_mapping.md](ppn_projected_93_mapping.md).",
        f"- Seeds: `{', '.join(map(str, seeds))}`; elapsed time: `{result['elapsed']:.3f} s`.",
        "",
        "## Seed-Wise Values",
        "",
        "| Seed | Direct projected sum | raw-93 residual | block-93 residual |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for seed in seeds:
        lines.append(
            f"| {seed} | `{_format_complex(result['totals'][seed])}` | "
            f"`{result['raw_full_residuals'][seed]:.6e}` | "
            f"`{result['block_full_residuals'][seed]:.6e}` |"
        )
    lines.extend((
        "",
        "## Relation To The Papers",
        "",
        "The [0912.1383v1 paper](https://arxiv.org/abs/0912.1383v1) states that computer scrutiny leaves 93 contractions for the \\({}^3\\mathrm{He}\\) correlator and reports agreement with the full result on one gauge configuration. It does not print the 93 survivor list or a term-by-term map. The same discussion separately says its zero-momentum sink baryon block can evaluate `2^N_N=8` contractions simultaneously. Our reconstruction explains a concrete way 93 projected blocks can account for 2,880 terms, but the missing published list prevents identifying our IDs as the authors' exact 93 entries.",
        "",
        "This audit verifies the algebra for the repository's built-in ppn quark operators, the specified fixed-charge spin projection, isospin-degenerate propagators, and its modeled color/spin contraction. It is not a bit-for-bit reproduction of the original gauge-configuration calculation or of the paper's unpublished survivor table. The 93 representative blocks expand to 744 ordinary raw `contract` evaluations in this reference script; a fused baryon-block kernel may evaluate those combinations more efficiently.",
        "",
        "## Reproduction",
        "",
        "```sh",
        "python3 -B verify_code/ppn_projected_93_audit.py",
        "```",
        "",
        "The script checks exact member coverage (`2880` IDs, once each), state-block sizes (`8` each), orbit weights, independent representative reevaluation, and direct full-sum equality.",
    ))
    path = PROJECT_ROOT / "verify_code" / "ppn_projected_93_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_mapping(result: dict) -> Path:
    records = result["block_records"]
    covered = [wick_id for record in records for wick_id in record["member_raw_ids"]]
    if len(records) != 93 or len(covered) != 2880 or len(set(covered)) != 2880:
        raise AssertionError("mapping must list 93 blocks partitioning 2,880 raw Wicks")
    histogram = {
        size: sum(record["member_state_count"] == size for record in records)
        for size in (2, 4)
    }
    lines = [
        "# ppn 93-Block Mapping",
        "",
        "This table is generated by `ppn_projected_93_audit.py`. Wick IDs are the one-based IDs from the repository's built-in detailed ppn reducer. Each representative block is the signed sum over its 8 `representative raw IDs`; its `weight` multiplies that block in the reduced projected correlator. `all raw IDs covered` records the full orbit represented by that weighted term. Full per-Wick signs and pairings are in [ppn_projected_93_blocks.json](ppn_projected_93_blocks.json).",
        "",
        "The table is an independently reconstructed mapping for the specified projected ppn calculation, not a recovered list from arXiv:0912.1383v1, which does not publish the survivor IDs.",
        "",
        "$$F_{ppn}=\\sum_{r=0}^{92}w_r\\,b_r,\\qquad b_r=\\sum_{j=1}^{8}\\epsilon_{rj}\\langle P,C_{rj}\\rangle,\\qquad w_r\\in\\{2,4\\}.$$",
        "",
        "| Orbit | Representative state Wick ID | Orbit size / weight | 8 representative raw IDs | All raw IDs covered |",
        "| ---: | ---: | ---: | --- | --- |",
    ]
    for record in sorted(records, key=lambda item: item["candidate_orbit_id"]):
        lines.append(
            f"| {record['candidate_orbit_id']} | {record['representative_wick_id']} | "
            f"{record['member_state_count']} | "
            f"{', '.join(map(str, record['representative_raw_ids']))} | "
            f"{', '.join(map(str, record['member_raw_ids']))} |"
        )
    lines.extend((
        "",
        "## Coverage Check",
        "",
        f"- Representative blocks: `{len(records)}`.",
        f"- Orbit-size histogram: `{histogram}`.",
        f"- Raw Wick IDs listed: `{len(covered)}` unique IDs, from `{min(covered)}` through `{max(covered)}`.",
        f"- Raw Wick terms evaluated directly by the reduced reference implementation: `{sum(len(record['representative_raw_ids']) for record in records)}` per input sample.",
    ))
    path = PROJECT_ROOT / "verify_code" / "ppn_projected_93_mapping.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_machine_mapping(result: dict) -> Path:
    blocks = []
    wick_records = result["raw_wick_records"]
    for record in sorted(
        result["block_records"], key=lambda item: item["candidate_orbit_id"]
    ):
        blocks.append({
            "orbit_id": record["candidate_orbit_id"],
            "representative_state_wick_id": record["representative_wick_id"],
            "weight": record["member_state_count"],
            "representative_wicks": [
                {
                    "wick_id": wick_id,
                    "wick_sign": wick_records[wick_id]["wick_sign"],
                    "pairing": wick_records[wick_id]["pairing"],
                }
                for wick_id in record["representative_raw_ids"]
            ],
            "state_wick_ids": list(record["canonical_state_wick_ids"]),
            "covered_raw_wick_ids": list(record["member_raw_ids"]),
        })

    payload = {
        "format": "ppn_projected_93_blocks/v1",
        "operator": "built-in ppn source/sink operators",
        "projection": {
            "state_order": ["p1", "p2", "n"],
            "upper_dirac_spin": {"0": "up", "1": "down"},
            "spin_state_components": [
                {"indices": [0, 1, 0], "amplitude": "1/sqrt(6)"},
                {"indices": [1, 0, 0], "amplitude": "-1/sqrt(6)"},
            ],
            "applied_at": ["sink", "source"],
        },
        "pairing_convention": {
            "index_base": 0,
            "sink_quark_order": ["p1[0:3]", "p2[0:3]", "n[0:3]"],
            "source_antiquark_order": ["pbar1[0:3]", "pbar2[0:3]", "nbar[0:3]"],
            "meaning": "pairing[k] is the source antiquark index contracted with sink quark index k",
        },
        "counts": {
            "raw_wick_terms": 2880,
            "state_blocks": 360,
            "representative_blocks": 93,
            "wicks_per_representative_block": 8,
            "raw_contract_calls_in_reference_evaluation": 744,
        },
        "formula": "result = sum(block.weight * sum(w.wick_sign * projected_contract(w.pairing) for w in block.representative_wicks) for block in blocks)",
        "blocks": blocks,
    }
    path = PROJECT_ROOT / "verify_code" / "ppn_projected_93_blocks.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_explicit_contracts() -> Path:
    """Write executable scalar contract calls for the reconstructed 93 blocks."""
    mapping_path = PROJECT_ROOT / "verify_code" / "ppn_projected_93_blocks.json"
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    operators = operator_library()["ppn"]
    expected_operands = (
        "C_g5", "C_g5", "C_g5",
        "C_g5_dagger", "C_g5_dagger", "C_g5_dagger",
        "peram_u_sin_sou", "peram_u_sin_sou", "peram_u_sin_sou",
        "peram_u_sin_sou", "peram_u_sin_sou", "peram_u_sin_sou",
        "peram_u_sin_sou", "peram_u_sin_sou", "peram_u_sin_sou",
        "VVV_sink", "VVV_sink", "VVV_sink",
        "conj_VVV_source", "conj_VVV_source", "conj_VVV_source",
    )
    lines = [
        '"""Explicit scalar `contract` expressions for the reconstructed ppn 93 blocks.',
        "",
        "The input arrays use the same shapes and conventions as the ppn operator",
        "fixture in this repository. Each `block_values[r]` is a signed sum of the",
        "eight Wick terms in one representative projected block; `block_weights[r]`",
        "is its exact proton-exchange orbit size (2 or 4). This is a reconstruction",
        "for the stated ppn projection, not a published survivor list from the paper.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "from opt_einsum import contract",
        "",
        "",
        "def ppn_projected_93(",
        "    C_g5, C_g5_dagger, peram_u_sin_sou, VVV_sink, conj_VVV_source,",
        "    P_ppn=None,",
        "):",
        '    """Evaluate the 93 weighted ppn blocks as scalar index contractions."""',
        "    if P_ppn is None:",
        "        psi = np.zeros((2, 2, 2), dtype=np.complex128)",
        "        psi[0, 1, 0] = 1.0 / np.sqrt(6.0)",
        "        psi[1, 0, 0] = -1.0 / np.sqrt(6.0)",
        '        spin_projector = np.einsum("abc,def->abcdef", psi, psi.conjugate())',
        "        P_ppn = np.zeros((4, 4, 4, 4, 4, 4), dtype=np.complex128)",
        "        P_ppn[:2, :2, :2, :2, :2, :2] = spin_projector",
        '    projected_operands = (',
        "        C_g5, C_g5, C_g5, C_g5_dagger, C_g5_dagger, C_g5_dagger,",
        "        peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou,",
        "        peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou,",
        "        peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou,",
        "        VVV_sink, VVV_sink, VVV_sink,",
        "        conj_VVV_source, conj_VVV_source, conj_VVV_source,",
        "        P_ppn,",
        "    )",
        "    block_values = []",
    ]

    weights = []
    for block in payload["blocks"]:
        orbit_id = int(block["orbit_id"])
        weight = int(block["weight"])
        weights.append(weight)
        lines.extend((
            f"    # Orbit {orbit_id}: representative state Wick ID "
            f"{block['representative_state_wick_id']}; weight {weight}.",
            "    block_values.append(",
            "        ",
        ))
        for index, wick in enumerate(block["representative_wicks"]):
            equation, operand_names, output = expression_output.contraction_expression(
                operators,
                wick["pairing"],
                isospin_degenerate=True,
            )
            if output != tuple("ABCDEF") or tuple(operand_names) != expected_operands:
                raise AssertionError("unexpected output axes or operand order")
            projected_equation = equation.split("->", 1)[0] + ",ABCDEF->"
            sign = int(wick["wick_sign"])
            operator = ("-" if sign < 0 else "+") if index else ("-" if sign < 0 else "")
            lines.extend((
                f"        {operator}contract(",
                f"            {projected_equation!r},",
                "            *projected_operands,",
                '            optimize="greedy",',
                f"        )  # Wick ID {wick['wick_id']}",
            ))
        lines.extend((
            "    )",
            "",
        ))

    if len(weights) != 93 or any(weight not in (2, 4) for weight in weights):
        raise AssertionError("expected 93 blocks with orbit weights 2 or 4")
    lines.append("    block_weights = (")
    for start in range(0, len(weights), 16):
        lines.append("        " + ", ".join(map(str, weights[start:start + 16])) + ",")
    lines.extend((
        "    )",
        "    return sum(",
        "        (weight * block for weight, block in zip(block_weights, block_values)),",
        "        0j,",
        "    )",
        "",
    ))
    path = PROJECT_ROOT / "verify_code" / "ppn_projected_93_contracts.py"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    result = audit((20261021, 20261022, 20261023, 20261024))
    print(
        f"raw-max-orbit-residual={max(result['raw_orbit_residuals'].values()):.6e} "
        f"raw-max-full-residual={max(result['raw_full_residuals'].values()):.6e} "
        f"block-max-orbit-residual={max(result['block_orbit_residuals'].values()):.6e} "
        f"block-max-full-residual={max(result['block_full_residuals'].values()):.6e}"
    )
    print(f"independent representative raw evaluations per seed={next(iter(result['independent_block_evaluation_counts'].values()))}")
    print(write_report(result))
    print(write_mapping(result))
    print(write_machine_mapping(result))
    print(write_explicit_contracts())


if __name__ == "__main__":
    main()
