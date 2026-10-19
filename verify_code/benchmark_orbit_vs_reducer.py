#!/usr/bin/env python3
"""Benchmark the verify_code orbit path against the production reducer.

The two implementations expose different mathematical objects.  This script
therefore reports both the measured wall time/memory and the number of
contraction calls implied by each representation.  The default command runs
both families in one process; use the isolated commands in
``benchmark_orbit_vs_reducer.md`` when a clean per-run ``ru_maxrss`` is needed.
A complete ppnn production reduction is optional because the repository's only
known measurement is a historical 539 second, 1.21 GiB run.

Examples::

    python3 verify_code/benchmark_orbit_vs_reducer.py --output verify_code/benchmark_orbit_vs_reducer.json
    python3 verify_code/benchmark_orbit_vs_reducer.py --contract-sample 32
    python3 verify_code/benchmark_orbit_vs_reducer.py --production-ppnn-mode coarse

The optional numerical contract sample uses shape-correct random perams and
three-dimensional antisymmetric VVV arrays.  It times actual
``opt_einsum.contract`` calls, but it does not claim that a fused block is a
valid tensor identity; fusion remains a separate, explicitly audited step.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# An optional dependency directory may be supplied by the caller.  Keeping
# this opt-in makes the project portable when the whole directory is moved.
_LOCAL_DEPS = os.environ.get("WICK_DISS_REDUCE_DEPS")
if _LOCAL_DEPS:
    dependency_path = Path(_LOCAL_DEPS).expanduser()
    if dependency_path.is_dir() and str(dependency_path) not in sys.path:
        sys.path.insert(0, str(dependency_path))


HISTORICAL_PPNN_REDUCER = {
    "source": "docs/reference/wick-api.md historical ppnn row",
    "raw_wick_count": 518400,
    "production_basis_count": 31092,
    "production_dependent_count": 487308,
    "wall_seconds": 539.4,
    "peak_memory_bytes": int(1.21 * 1024 ** 3),
    "mode": "independent",
    "measured_in_this_run": False,
}


def _rss_bytes() -> int:
    """Return the current process peak RSS in bytes on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes, Linux reports KiB.  Keep the branch explicit so a
    # JSON report cannot silently mix the two units.
    if sys.platform == "darwin":
        return value
    return value * 1024


def _measure(callable_object, *, label: str) -> Dict[str, Any]:
    started = time.perf_counter()
    result = callable_object()
    elapsed = time.perf_counter() - started
    return {
        "label": label,
        "wall_seconds": elapsed,
        "python_tracemalloc_peak_bytes": None,
        "process_peak_rss_bytes": _rss_bytes(),
        "result": result,
    }


def _ppnn_operators():
    """Return the exact ppnn baryon fixture used by the verify_code quotient."""

    from verify_code.ppnn_state_quotient_audit import _operators

    return tuple(_operators())


def _orbit_measurement(family: str) -> Dict[str, Any]:
    if family == "ppn":
        from verify_code.orbit_ppn import ppn_orbit_system

        measured = _measure(ppn_orbit_system, label="verify_code orbit ppn")
    elif family == "ppnn":
        from verify_code.orbit_ppnn import ppnn_orbit_system

        measured = _measure(ppnn_orbit_system, label="verify_code orbit ppnn")
    else:
        raise ValueError(f"unknown family: {family}")
    system = measured.pop("result")
    state_block_calls = sum(
        len(system.state_to_raw[orbit.representative])
        for orbit in system.state_orbits
    )
    internal_group_order = {
        "ppn": 2 ** 3,
        "ppnn": 2 ** 4,
    }[family]
    return {
        **measured,
        "family": family,
        "raw_wick_count": system.raw_wick_count,
        "state_count": system.state_count,
        "orbit_count": system.orbit_count,
        "state_orbit_size_histogram": dict(sorted(system.orbit_size_histogram.items())),
        "raw_orbit_size_histogram": dict(sorted(system.raw_orbit_size_histogram.items())),
        "state_block_contract_calls": state_block_calls,
        "fused_block_calls_if_certified": system.orbit_count,
        "internal_exchange_group_order": internal_group_order,
        "measured_in_this_run": True,
    }


def _production_measurement(family: str, mode: str) -> Dict[str, Any]:
    if family == "ppnn" and mode == "historical":
        return dict(HISTORICAL_PPNN_REDUCER)
    if mode == "skip":
        return {
            "source": "not run (use --production-ppnn-mode coarse or independent)",
            "family": family,
            "measured_in_this_run": False,
        }

    from kernels.wick_reducer.reducer import reduce_case

    operators = _ppnn_operators() if family == "ppnn" else None
    measured = _measure(
        lambda: reduce_case(
            family,
            operators,
            mode=mode,
            isospin_degenerate=True,
        ),
        label=f"production reducer {family} {mode}",
    )
    report = measured.pop("result")
    return {
        **measured,
        "family": family,
        "mode": mode,
        "raw_wick_count": report["wick_count"],
        "production_basis_count": report["independent_count"],
        "production_dependent_count": report.get("dependent_count"),
        "reducer_timing_seconds": report.get("timing_seconds"),
        "measured_in_this_run": True,
    }


def _epsilon3():
    import numpy as np

    epsilon = np.zeros((3, 3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len({i, j, k}) == 3:
                    inversions = (i > j) + (i > k) + (j > k)
                    epsilon[i, j, k] = -1 if inversions % 2 else 1
    return epsilon


def _contract_values(operators, equations: Iterable[Tuple[str, Tuple[str, ...]]], seed: int):
    import numpy as np
    from kernels import correlator_wick_normalizer as core

    rng = np.random.default_rng(seed)
    gamma = core._euclidean_gamma_matrices()
    values: Dict[str, Any] = {
        name.replace("Cgamma", "C_g").replace("gamma", "g"): np.asarray(matrix)
        for name, matrix in gamma.items()
    }
    # The generated expression names C_g5 and C_g5_dagger; the replacement
    # above also handles gamma5/gamma5_dagger.
    for equation, operands in equations:
        for operand in operands:
            if operand in values:
                continue
            if operand.startswith("peram_"):
                values[operand] = (
                    rng.standard_normal((3, 4, 3, 4))
                    + 1j * rng.standard_normal((3, 4, 3, 4))
                ) / np.sqrt(2.0)
            elif operand.startswith("VVV_") or operand.startswith("conj_VVV_"):
                scalar = rng.standard_normal() + 1j * rng.standard_normal()
                values[operand] = scalar * _epsilon3()
            elif operand.startswith("VDV_") or operand.startswith("conj_VDV_"):
                # This benchmark currently targets ppn/ppnn baryons.  The
                # branch keeps the helper useful for a meson fixture.
                values[operand] = (
                    rng.standard_normal((3, 3, 3))
                    + 1j * rng.standard_normal((3, 3, 3))
                ) / np.sqrt(2.0)
            else:
                raise KeyError(f"no benchmark tensor generated for {operand!r}")
    return values


def _contract_sample(family: str, seed: int, sample: int) -> Dict[str, Any]:
    """Time a deterministic sample of actual opt_einsum contractions."""

    import opt_einsum
    from kernels.wick_reducer import expression_output, operator_model

    if family == "ppn":
        from verify_code.orbit_ppn import ppn_orbit_system

        system = ppn_orbit_system()
        operators = operator_model.operator_library()["ppn"]
    else:
        from verify_code.orbit_ppnn import ppnn_orbit_system

        system = ppnn_orbit_system()
        operators = _ppnn_operators()

    raw_ids = tuple(system.raw_pairings)
    if not raw_ids:
        raise AssertionError("orbit system contains no raw Wick pairings")
    if sample <= 0:
        raise ValueError("contract sample must be positive")
    if sample < len(raw_ids):
        positions = [round(index * (len(raw_ids) - 1) / (sample - 1)) for index in range(sample)]
        selected = tuple(raw_ids[position] for position in positions)
    else:
        selected = raw_ids
    equations = []
    for raw_id in selected:
        equation, operands, _axes = expression_output.contraction_expression(
            operators, system.raw_pairings[raw_id], isospin_degenerate=True,
        )
        equations.append((equation, operands))
    values = _contract_values(operators, equations, seed)
    # Warm one path so the timing describes repeated numerical contractions,
    # while expression construction and random allocation remain outside.
    equation, operands = equations[0]
    opt_einsum.contract(equation, *(values[name] for name in operands))
    started = time.perf_counter()
    outputs = []
    for equation, operands in equations:
        outputs.append(opt_einsum.contract(equation, *(values[name] for name in operands)))
    elapsed = time.perf_counter() - started
    call_seconds = elapsed / len(equations)
    return {
        "family": family,
        "seed": seed,
        "sample_calls": len(equations),
        "contract_seconds": elapsed,
        "seconds_per_contract": call_seconds,
        "sample_output_shapes": [tuple(getattr(value, "shape", ())) for value in outputs[:3]],
        "estimated_seconds_for_state_block_calls": call_seconds * (
            system.orbit_count * len(system.state_to_raw[system.state_orbits[0].representative])
        ),
        "estimated_seconds_for_fused_block_calls": call_seconds * system.orbit_count,
        "note": "estimates multiply a sample mean; they are not a full ppnn run",
    }


def build_report(
    *,
    contract_sample: int = 0,
    seed: int = 20261009,
    production_ppnn_mode: str = "historical",
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "metadata": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "seed": seed,
            "contract_sample": contract_sample,
            "production_ppnn_mode": production_ppnn_mode,
            "memory_measurement": "resource.ru_maxrss; RSS is process peak (macOS bytes)",
        },
        "systems": {},
    }
    for family in ("ppn", "ppnn"):
        production_mode = "independent" if family == "ppn" else production_ppnn_mode
        if family == "ppnn" and production_mode == "historical":
            production = _production_measurement(family, "historical")
        else:
            production = _production_measurement(family, production_mode)
        report["systems"][family] = {
            "orbit_audit": _orbit_measurement(family),
            "production_reducer": production,
        }
        if contract_sample:
            report["systems"][family]["contract_sample"] = _contract_sample(
                family, seed, contract_sample,
            )
    return report


def _worker(args: argparse.Namespace) -> None:
    # Kept for callers that want one report in a fresh process.  The public
    # command runs both families in one process for convenience; use the
    # per-family commands in benchmark_orbit_vs_reducer.md for isolated RSS.
    report = build_report(
        contract_sample=args.contract_sample,
        seed=args.seed,
        production_ppnn_mode=args.production_ppnn_mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON report")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--contract-sample", type=int, default=0,
        help="actual opt_einsum calls per family (0 disables; 32 is a useful smoke sample)",
    )
    parser.add_argument(
        "--production-ppnn-mode",
        choices=("historical", "skip", "coarse", "independent"),
        default="historical",
        help="historical avoids rerunning the 539 s/1.21 GiB ppnn reducer",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    report = build_report(
        contract_sample=args.contract_sample,
        seed=args.seed,
        production_ppnn_mode=args.production_ppnn_mode,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
