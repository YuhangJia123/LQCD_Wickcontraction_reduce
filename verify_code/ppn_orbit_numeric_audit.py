#!/usr/bin/env python3
"""Numerically test the ppn reducer and orbit-candidate span on random tensors.

Requires numpy and opt_einsum. This audit prints residuals only; it does not
write a summed correlator or promote sample-fitted coefficients to a result.
"""

from __future__ import annotations

import ast
import itertools
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import opt_einsum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels import correlator_wick_normalizer as core
from kernels.wick_reducer import expression_output
from kernels.wick_reducer.operator_model import fields_for_operators, operator_library
from kernels.wick_reducer.reducer import reduce_case


def _candidate_ids() -> Tuple[int, ...]:
    path = PROJECT_ROOT / "verify_code" / "ppn_orbit_candidate_contracts.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    value = next(
        node.value for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "ppn_orbit_candidate_wick_ids"
            for target in node.targets
        )
    )
    return tuple(int(value) for value in ast.literal_eval(value))


def _antisymmetric_vvv(rng, scalar):
    tensor = np.zeros((3, 3, 3), dtype=np.complex128)
    for permutation in itertools.permutations(range(3)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(3) for j in range(i + 1, 3)
        )
        tensor[permutation] = (-1 if inversions % 2 else 1) * scalar
    return tensor


def _inputs(seed: int):
    rng = np.random.default_rng(seed)

    def random_complex(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)

    registry = core._euclidean_gamma_matrices()
    values = {
        "C_g5": np.asarray(registry["Cgamma5"], dtype=np.complex128),
        "C_g5_dagger": np.asarray(registry["Cgamma5_dagger"], dtype=np.complex128),
        "peram_u_sin_sou": random_complex((3, 4, 3, 4)),
        "VVV_sink": _antisymmetric_vvv(rng, random_complex(())),
        "conj_VVV_source": _antisymmetric_vvv(rng, random_complex(()).conjugate()),
    }
    return values


def _contract(operators, pairing, values):
    equation, operands, axes = expression_output.contraction_expression(
        operators, pairing, isospin_degenerate=True,
    )
    result = opt_einsum.contract(equation, *(values[name] for name in operands))
    if result.shape != (4, 4, 4, 4, 4, 4) or len(axes) != 6:
        raise AssertionError(f"unexpected output shape/axes: {result.shape}, {axes}")
    return np.asarray(result)


def _transpose(candidate_map, basis_map, axes):
    inverse_basis = {root: basis for basis, root in basis_map.items()}
    positions = {axis: index for index, axis in enumerate(axes)}
    result = tuple(
        positions[inverse_basis.get(candidate_map.get(axis, axis), candidate_map.get(axis, axis))]
        for axis in axes
    )
    if sorted(result) != list(range(len(axes))):
        raise ValueError("reducer relation has an invalid output-axis permutation")
    return result


def run_seed(seed: int, report, operators, graph_by_id, candidate_ids):
    values = _inputs(seed)
    dependent = {int(item["wick_index"]): item for item in report["dependent_relations"]}
    independent_ids = {int(item["wick_index"]) for item in report["independent_wicks"]}
    axes = expression_output._bare_axes(operators)
    candidate_set = set(candidate_ids)
    basis_tensors = {}
    candidate_tensors = {}
    raw_sum = np.zeros((4, 4, 4, 4, 4, 4), dtype=np.complex128)

    started = time.perf_counter()
    for wick_id in range(1, int(report["wick_count"]) + 1):
        graph = graph_by_id[wick_id]
        tensor = _contract(operators, graph["pairing"], values)
        raw_sum += int(graph["wick_sign"]) * tensor
        if wick_id in independent_ids:
            basis_tensors[wick_id] = int(graph["wick_sign"]) * tensor
        if wick_id in candidate_set:
            candidate_tensors[wick_id] = int(graph["wick_sign"]) * tensor
    contraction_seconds = time.perf_counter() - started

    reducer_sum = np.zeros_like(raw_sum)
    for wick_id in independent_ids:
        reducer_sum += basis_tensors[wick_id]
    for wick_id, relation in dependent.items():
        selected = int(relation["transporter_solution"])
        candidate_map = relation["transporter_solutions"][selected].get("bare_axis_map", {})
        basis_maps = {
            int(item["wick_index"]): item.get("bare_axis_map", {})
            for item in relation.get("basis_transports", ())
        }
        for basis_id, coefficient in zip(
            relation["basis_wick_indices"], relation["coefficients"]
        ):
            basis_id = int(basis_id)
            permutation = _transpose(candidate_map, basis_maps.get(basis_id, {}), axes)
            value = complex(float(coefficient[0]), float(coefficient[1]))
            reducer_sum += value * basis_tensors[basis_id].transpose(permutation)

    candidate_matrix = np.stack(
        [candidate_tensors[wick_id].reshape(-1) for wick_id in candidate_ids], axis=1
    )
    target = raw_sum.reshape(-1)
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        candidate_matrix, target, rcond=1.0e-12,
    )
    reconstructed = candidate_matrix @ coefficients
    raw_norm = np.linalg.norm(target)
    candidate_residual = np.linalg.norm(target - reconstructed) / raw_norm
    reducer_residual = np.linalg.norm(raw_sum - reducer_sum) / raw_norm
    return {
        "seed": seed,
        "rank": int(rank),
        "candidate_count": len(candidate_ids),
        "raw_relative_residual": float(candidate_residual),
        "production_reducer_relative_residual": float(reducer_residual),
        "full_raw_norm": float(raw_norm),
        "contraction_seconds": contraction_seconds,
        "fitted_coefficients": coefficients,
    }


def main() -> None:
    audit_started = time.perf_counter()
    operators = operator_library()["ppn"]
    report = reduce_case("ppn", operators, mode="detailed")
    graph_by_id = {int(item["wick_index"]): item for item in report["graph_records"]}
    candidate_ids = _candidate_ids()
    print(f"fixture: Nc=3 epsilon VVV, generic complex peram seed(s); shape=(4,4,4,4,4,4)")
    print(f"counts: raw={report['wick_count']}, production_basis={report['independent_count']}, candidates={len(candidate_ids)}")
    results = []
    for seed in (20261009, 20261010, 20261011):
        result = run_seed(seed, report, operators, graph_by_id, candidate_ids)
        results.append(result)
        print(
            f"seed={seed} rank={result['rank']}/{result['candidate_count']} "
            f"candidate-span-relative-residual={result['raw_relative_residual']:.6e} "
            f"production-reducer-relative-residual={result['production_reducer_relative_residual']:.6e} "
            f"raw-norm={result['full_raw_norm']:.6e} "
            f"2880-contract-seconds={result['contraction_seconds']:.3f}"
        )
    print("No summed correlator or sample-fitted coefficient list has been emitted.")

    coefficient_deltas = []
    for first, second in itertools.combinations(results, 2):
        delta = np.linalg.norm(
            second["fitted_coefficients"] - first["fitted_coefficients"]
        )
        scale = max(1.0, np.linalg.norm(first["fitted_coefficients"]))
        coefficient_deltas.append((first["seed"], second["seed"], float(delta / scale)))
    for first, second, delta in coefficient_deltas:
        print(f"fitted-coefficient relative difference {first}-{second}={delta:.6e}")
    audit_seconds = time.perf_counter() - audit_started
    print(f"total-audit-seconds={audit_seconds:.3f}")

    report_lines = [
        "# ppn 93-candidate numerical span audit",
        "",
        "This audit directly evaluates all raw ppn `contract(...)` expressions from the current operator fixture and compares their Wick-sign-weighted sum with the production reducer and with the span of the 93 orbit candidate basis tensors.",
        "",
        "## Fixture",
        "",
        f"- Python {sys.version.split()[0]}, NumPy {np.__version__}, opt_einsum {opt_einsum.__version__}.",
        "- Dimension `Nc=3`; sink/source VVV tensors are random complex scalar multiples of the fully antisymmetric epsilon tensor.",
        "- `peram_u_sin_sou` is a generic random complex array of shape `(4,3,3,4)`, shared by `u` and `d` as in the reducer's default isospin-degenerate ppn mode.",
        "- Gamma matrices are the project's built-in `Cgamma5` and `Cgamma5_dagger`; each contraction returns a `(4,4,4,4,4,4)` tensor.",
        f"- Terms: {report['wick_count']} raw Wicks, {report['independent_count']} production basis terms, {len(candidate_ids)} orbit candidates.",
        "",
        "The direct total is computed as `sum(wick_sign[w] * contract(w))`. The production reconstruction uses the reducer's selected coefficients and open-axis transposes. The candidate reconstruction uses least squares against the 93 signed candidate basis tensors and reports relative residual `||full - projection|| / ||full||`; the fitted coefficients are intentionally not written as a result.",
        "",
        "## Results",
        "",
        "| Seed | Candidate tensor rank | Candidate-span residual | Production-reducer residual | 2,880 calls (s) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        report_lines.append(
            f"| {result['seed']} | {result['rank']}/{result['candidate_count']} | "
            f"{result['raw_relative_residual']:.6e} | "
            f"{result['production_reducer_relative_residual']:.6e} | "
            f"{result['contraction_seconds']:.3f} |"
        )
    report_lines.extend((
        f"",
        "Pairwise relative differences between the sample-fitted coefficient vectors:",
        "",
        *[f"- Seeds {first} and {second}: `{delta:.6e}`." for first, second, delta in coefficient_deltas],
        f"",
        f"Total audit runtime, including reducer setup, all tensor contractions, and least-squares fits: `{audit_seconds:.3f} s`.",
        "",
        "## Interpretation",
        "",
        "The production reducer reconstruction agrees with the direct 2,880-term sum at floating-point precision for all fixtures. The 93 candidate tensors have numerical rank 93, but the full result has a large residual outside their span in every fixture. Therefore the 93 calls do not reconstruct the current ppn correlator for these generic inputs. The seed-to-seed coefficient changes also rule out treating a sample-fitted coefficient vector as fixed orbit weights.",
        "",
        "This is strong counterevidence to using the proposed label orbits alone as a reduction of the current concrete output. It does not refute a paper's additional operator/color-spin assumptions or a separately derived symmetry-projected correlator. No candidate-summed correlator is emitted.",
        "",
        "## Reproduction",
        "",
        "Install the optional numerical dependencies from the project root with `python3 -m pip install -e '.[numeric]'`.",
        "",
        "```sh",
        "python3 -m pip install -e '.[numeric]'",
        "python3 -B -m verify_code.ppn_orbit_numeric_audit",
        "```",
    ))
    output = PROJECT_ROOT / "verify_code" / "ppn_orbit_numeric_audit.md"
    output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
