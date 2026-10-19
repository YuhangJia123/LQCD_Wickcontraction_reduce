#!/usr/bin/env python3
"""Cross-reference the proposed 93 ppn orbit representatives to the reducer."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.wick_reducer import expression_output
from kernels.wick_reducer.operator_model import operator_library
from kernels.wick_reducer.reducer import reduce_case


def _candidate_ids() -> Tuple[int, ...]:
    generated = PROJECT_ROOT / "verify_code" / "ppn_orbit_candidate_contracts.py"
    module = ast.parse(generated.read_text(encoding="utf-8"))
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


def _transpose(candidate_map: Mapping[str, str], basis_map: Mapping[str, str], axes):
    inverse_basis = {root: basis for basis, root in basis_map.items()}
    positions = {axis: index for index, axis in enumerate(axes)}
    result = tuple(
        positions[inverse_basis.get(candidate_map.get(axis, axis), candidate_map.get(axis, axis))]
        for axis in axes
    )
    if sorted(result) != list(range(len(axes))):
        raise ValueError("candidate relation does not give a permutation of open axes")
    return result


def render_crosswalk() -> Tuple[str, Dict[str, int]]:
    operators = operator_library()["ppn"]
    report = reduce_case("ppn", operators, mode="detailed")
    candidates = _candidate_ids()
    independent = {int(item["wick_index"]): item for item in report["independent_wicks"]}
    dependent = {int(item["wick_index"]): item for item in report["dependent_relations"]}
    graph_records = {int(item["wick_index"]): item for item in report["graph_records"]}
    axes = expression_output._bare_axes(operators)
    if len(candidates) != 93 or len(set(candidates)) != len(candidates):
        raise AssertionError("candidate file must contain 93 unique Wick IDs")

    rows = []
    dependent_count = 0
    identity = tuple(range(len(axes)))
    for orbit_id, wick_id in enumerate(candidates):
        if wick_id in independent:
            graph = graph_records[wick_id]
            rows.append({
                "orbit": orbit_id,
                "wick_id": wick_id,
                "status": "basis",
                "candidate_wick_sign": int(graph["wick_sign"]),
                "basis_wick_id": wick_id,
                "basis_wick_sign": int(graph["wick_sign"]),
                "coefficient": 1,
                "transpose": identity,
                "residual_norm2": 0,
            })
            continue

        relation = dependent.get(wick_id)
        if relation is None:
            raise AssertionError(f"candidate Wick {wick_id} is absent from reducer output")
        dependent_count += 1
        selected = int(relation["transporter_solution"])
        solutions = relation["transporter_solutions"]
        if selected >= len(solutions):
            raise AssertionError(f"Wick {wick_id} has no selected transporter witness")
        candidate_map = solutions[selected].get("bare_axis_map", {})
        basis_transports = {
            int(item["wick_index"]): item.get("bare_axis_map", {})
            for item in relation.get("basis_transports", ())
        }
        candidate_sign = int(graph_records[wick_id]["wick_sign"])
        terms = []
        for basis_id, coefficient in zip(
            relation["basis_wick_indices"], relation["coefficients"]
        ):
            basis_id = int(basis_id)
            basis_sign = int(graph_records[basis_id]["wick_sign"])
            permutation = _transpose(
                candidate_map, basis_transports.get(basis_id, {}), axes,
            )
            value = complex(float(coefficient[0]), float(coefficient[1]))
            if value.imag:
                formatted = f"{value.real:g}{value.imag:+g}j"
            else:
                formatted = f"{value.real:g}"
            terms.append({
                "basis_id": basis_id,
                "basis_sign": basis_sign,
                "coefficient": formatted,
                "transpose": permutation,
            })
        if len(terms) != 1:
            # Preserve full exact expansion for unusual multi-basis candidates.
            mapping = "; ".join(
                f"{term['coefficient']} * ({term['basis_sign']}*C[{term['basis_id']}])^T{term['transpose']}"
                for term in terms
            )
        else:
            term = terms[0]
            mapping = (
                f"{term['coefficient']} * ({term['basis_sign']}*C[{term['basis_id']}])"
                f"^T{term['transpose']}"
            )
        rows.append({
            "orbit": orbit_id,
            "wick_id": wick_id,
            "status": "dependent",
            "candidate_wick_sign": candidate_sign,
            "basis_wick_id": "; ".join(str(term["basis_id"]) for term in terms),
            "basis_wick_sign": "; ".join(str(term["basis_sign"]) for term in terms),
            "coefficient": "; ".join(term["coefficient"] for term in terms),
            "transpose": "; ".join(str(term["transpose"]) for term in terms),
            "residual_norm2": float(relation["residual_norm2"]),
            "mapping": mapping,
        })

    lines = [
        "# ppn orbit representative crosswalk",
        "",
        "This table compares each of the 93 deterministic label-action representatives with the current `mode='detailed'` production reducer. `C[w]` denotes the raw `contract(...)` tensor for Wick ID `w`; the production basis view includes that basis Wick's `wick_sign`. A mapping is stated for the signed Wick expression:",
        "",
        "`wick_sign[w] * C[w] = coefficient * transpose(wick_sign[b] * C[b], axes)`.",
        "",
        "An identity transpose is `(0, 1, 2, 3, 4, 5)`. Independent candidates map to themselves. Reducer dependent relations report zero residual in the current numerical Dirac projection; this is fixture evidence, not a proof for arbitrary tensors.",
        "",
        f"Reducer count: {report['wick_count']} raw Wicks, {report['independent_count']} basis Wicks, {report['dependent_count']} dependent Wicks. Orbit representatives: {len(candidates)}; already dependent: {dependent_count}.",
        "",
        "| Orbit | Candidate Wick | Reducer status | Mapping to production basis | Residual norm squared |",
        "| ---: | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        if row["status"] == "basis":
            mapping = f"`{row['candidate_wick_sign']}*C[{row['wick_id']}]` (basis representative)"
        else:
            mapping = f"`{row['candidate_wick_sign']}*C[{row['wick_id']}] = {row['mapping']}`"
        lines.append(
            f"| {row['orbit']} | {row['wick_id']} | {row['status']} | {mapping} | {row['residual_norm2']:.3g} |"
        )
    lines.extend((
        "",
        "The generator chooses the lowest Wick ID classified independent by the production reducer from the full raw-Wick members of each proposed label orbit, including the eight within-baryon slot variants per quotient state. All 93 selected calls are therefore production basis elements, each with coefficient 1 and identity transpose to itself. This membership does not show that the full raw Wick sum lies in their span; the random-tensor span audit is separate.",
        "",
        "Reproduce with:",
        "",
        "```sh",
        "python3 -B verify_code/ppn_orbit_reducer_crosswalk.py",
        "```",
    ))
    return "\n".join(lines) + "\n", {
        "raw_wicks": int(report["wick_count"]),
        "production_basis": int(report["independent_count"]),
        "production_dependent": int(report["dependent_count"]),
        "orbit_representatives": len(candidates),
        "orbit_representatives_already_dependent": dependent_count,
    }


def main() -> None:
    text, counts = render_crosswalk()
    output = PROJECT_ROOT / "verify_code" / "ppn_orbit_reducer_crosswalk.md"
    output.write_text(text, encoding="utf-8")
    print(", ".join(f"{key}={value}" for key, value in counts.items()))
    print(output)


if __name__ == "__main__":
    main()
