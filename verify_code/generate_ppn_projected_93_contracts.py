#!/usr/bin/env python3
"""Generate explicit scalar contract calls from the ppn 93-block mapping."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.wick_reducer import expression_output
from kernels.wick_reducer.operator_model import operator_library


MAPPING_PATH = PROJECT_ROOT / "verify_code" / "ppn_projected_93_blocks.json"
OUTPUT_PATH = PROJECT_ROOT / "verify_code" / "ppn_projected_93_contracts.py"


def generate() -> Path:
    payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    blocks = payload["blocks"]
    covered = [wick_id for block in blocks for wick_id in block["covered_raw_wick_ids"]]
    if (
        len(blocks) != 93
        or len(covered) != 2880
        or len(set(covered)) != 2880
        or any(len(block["representative_wicks"]) != 8 for block in blocks)
    ):
        raise AssertionError("mapping must contain 93 blocks covering 2,880 Wicks")
    operators = operator_library()["ppn"]
    expected_operands = (
        "C_g5", "C_g5", "C_g5",
        "C_g5_dagger", "C_g5_dagger", "C_g5_dagger",
        *("peram_u_sin_sou",) * 9,
        "VVV_sink", "VVV_sink", "VVV_sink",
        "conj_VVV_source", "conj_VVV_source", "conj_VVV_source",
    )
    lines = [
        '"""Explicit scalar `contract` expressions for the reconstructed ppn 93 blocks.',
        "",
        "The input arrays use the same shapes and conventions as the ppn operator",
        "fixture in this repository. Each block is a signed sum of eight Wick terms;",
        "its weight is the exact proton-exchange orbit size (2 or 4). This is a",
        "reconstruction for the stated ppn projection, not a published survivor list.",
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
        '    """Evaluate 93 weighted ppn blocks as scalar index contractions."""',
        "    if P_ppn is None:",
        "        psi = np.zeros((2, 2, 2), dtype=np.complex128)",
        "        psi[0, 1, 0] = 1.0 / np.sqrt(6.0)",
        "        psi[1, 0, 0] = -1.0 / np.sqrt(6.0)",
        '        spin_projector = np.einsum("abc,def->abcdef", psi, psi.conjugate())',
        "        P_ppn = np.zeros((4, 4, 4, 4, 4, 4), dtype=np.complex128)",
        "        P_ppn[:2, :2, :2, :2, :2, :2] = spin_projector",
        "    projected_operands = (",
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
    for block in blocks:
        orbit_id = int(block["orbit_id"])
        weight = int(block["weight"])
        weights.append(weight)
        lines.extend((
            f"    # Orbit {orbit_id}: representative state Wick ID "
            f"{block['representative_state_wick_id']}; weight {weight}.",
            "    block_values.append(",
        ))
        for index, wick in enumerate(block["representative_wicks"]):
            equation, operand_names, output = expression_output.contraction_expression(
                operators,
                wick["pairing"],
                isospin_degenerate=True,
            )
            if output != tuple("ABCDEF") or tuple(operand_names) != expected_operands:
                raise AssertionError("unexpected output axes or operand order")
            scalar_equation = equation.split("->", 1)[0] + ",ABCDEF->"
            sign = int(wick["wick_sign"])
            operator = ("-" if sign < 0 else "+") if index else ("-" if sign < 0 else "")
            lines.extend((
                f"        {operator}contract(",
                f"            {scalar_equation!r},",
                "            *projected_operands,",
                '            optimize="greedy",',
                f"        )  # Wick ID {wick['wick_id']}",
            ))
        lines.extend(("    )", ""))

    if (
        len(weights) != 93
        or any(weight not in (2, 4) for weight in weights)
        or sum(weights) != 360
    ):
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
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return OUTPUT_PATH


if __name__ == "__main__":
    print(generate())
