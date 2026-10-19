"""Measure normal ppn block construction, then validate its full output.

Run this module in a fresh interpreter with BLAS threads fixed to one. The
reported construction time includes generation of the independent tables,
adaptation, pairing-family construction, block planning and its exact proof,
and execution-table serialization. Numerical validation follows the reported
time/RSS checkpoint and is recorded separately.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

from kernels.block_reduction import block_reduce, evaluate_expression, evaluate_plan
from verify_code.audit_global_block import _arrays, _backend, _error
from verify_code.audit_global_ppnn import _rss_bytes, _save_plan, _sha256, _write_json
from verify_code.normal_block_validation import generate_normal_fixture
from kernels.block_reduction.output_adapter import load_normal_output, pairing_family_from_operators
from kernels.wick_reducer.operator_model import operator_library


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def benchmark(directory: Path, output: Path, *, seeds=(20261015,),
              search_seconds=5.0, search_budget=40):
    report = {
        "schema_version": 1,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": "ppn",
        "input": "normal",
        "mode": "wick",
        "output": "block",
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                             "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
        },
        "parameters": {"max_w_types": 1, "search_seconds": search_seconds,
                       "search_budget": search_budget, "seeds": list(seeds)},
        "measurement_scope": {
            "time": (
                "Fresh normal independent-table generation, output adapter and "
                "raw pairing family, block_reduce including its exact symbolic "
                "certificate, and portable block execution-table write. "
                "Interpreter startup/imports and subsequent random numerical "
                "validation are excluded from the wall time."
            ),
            "memory": (
                "Process ru_maxrss sampled immediately after execution-table "
                "write, before numerical validation; includes interpreter and "
                "imports. Bytes on macOS, KiB converted to bytes elsewhere."
            ),
            "numerical_target": "All six open Dirac axes, each of dimension four.",
        },
        "phase_seconds": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "normal_ppn_independent.py"
    phase = perf_counter()
    generated = generate_normal_fixture("ppn", source_path)
    report["phase_seconds"]["generate_independent_tables"] = perf_counter() - phase
    phase = perf_counter()
    adapted = load_normal_output(source_path)
    operators = tuple(replace(operator, mode="normal")
                      for operator in operator_library()["ppn"])
    family = pairing_family_from_operators(operators, tensor_specs=adapted.tensor_specs)
    report["phase_seconds"]["adapter_and_pairing_family"] = perf_counter() - phase
    counts = (len(adapted.basis_contract), len(adapted.basis), len(family.terms))
    if counts != (219, 2760, 2880):
        raise AssertionError(f"normal ppn input counts changed: {counts}")
    phase = perf_counter()
    plan = block_reduce(
        adapted.basis_contract, adapted.basis, adapted.coefficients,
        tensor_specs=adapted.tensor_specs, pairing_family=family,
        max_w_types=1, search_seconds=search_seconds, search_budget=search_budget,
    )
    report["phase_seconds"]["block_plan_and_symbolic_proof"] = perf_counter() - phase
    if len(plan.networks) != 48 or not plan.stats["symbolic_verified"]:
        raise AssertionError("expected 48 networks with the exact certificate verified")
    phase = perf_counter()
    execution_output = _save_plan(plan, directory)
    report["phase_seconds"]["write_block_execution_tables"] = perf_counter() - phase
    wall_seconds = perf_counter() - started
    peak_rss_bytes = _rss_bytes()
    report.update({
        "status": "construction_complete_validation_pending",
        "generator": generated,
        "raw_wick_count": len(family.terms),
        "basis_contract_count": len(adapted.basis_contract),
        "basis_view_count": len(adapted.basis),
        "block_network_count": len(plan.networks),
        "w_definition_count": len(plan.w_definitions),
        "symbolic_verified": bool(plan.stats["symbolic_verified"]),
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_mib": peak_rss_bytes / (1024 * 1024),
        "source_output": {"path": str(source_path.resolve()),
                          "sha256": _sha256(source_path),
                          "bytes": source_path.stat().st_size},
        "execution_output": execution_output,
        "plan_stats": dict(plan.stats),
    })
    _write_json(output, report)
    print(json.dumps({"event": "timed_construction_finished",
                      "wall_seconds": wall_seconds,
                      "peak_rss_mib": report["peak_rss_mib"],
                      "networks": len(plan.networks)}, ensure_ascii=False), flush=True)

    validation_started = perf_counter()
    samples = []
    for seed in seeds:
        arrays = _arrays(adapted, seed)
        reference_backend, reference_calls = _backend()
        phase = perf_counter()
        expected = evaluate_expression(
            adapted.basis_contract, adapted.basis, adapted.coefficients,
            arrays, contract=reference_backend,
        )
        reference_seconds = perf_counter() - phase
        block_backend, block_calls = _backend()
        phase = perf_counter()
        actual = evaluate_plan(plan, arrays, contract=block_backend)
        block_seconds = perf_counter() - phase
        if expected.shape != (4,) * 6 or actual.shape != expected.shape:
            raise AssertionError("validation must retain all six open Dirac axes")
        error = _error(expected, actual)
        if error["maximum_relative"] > 1e-10:
            raise AssertionError(f"full-tensor numerical mismatch at seed {seed}: {error}")
        samples.append({
            "seed": seed, "output_shape": list(actual.shape),
            "reference_contract_calls": len(reference_calls),
            "block_contract_calls": len(block_calls),
            "reference_evaluation_seconds": reference_seconds,
            "block_evaluation_seconds": block_seconds,
            "error": error,
        })
    report["validation"] = {
        "status": "passed", "samples": samples,
        "wall_seconds": perf_counter() - validation_started,
        "random_inputs": "Complex standard-normal perambulators by tensor identity.",
        "fixed_inputs": "Exact epsilon_color, delta_color, and registered Dirac matrices.",
        "normalization": "max(abs(reference), abs(actual), 1e-30) over the full tensor",
        "tolerance": 1e-10,
        "vvv_randomized": False,
        "full_open_tensor": True,
    }
    report["status"] = "completed"
    _write_json(output, report)
    print(json.dumps({"event": "validation_finished", "report": str(output),
                      "samples": samples}, ensure_ascii=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path("/private/tmp/ppn_block_report_20261015"))
    parser.add_argument("--json", type=Path,
                        default=PROJECT_ROOT / "report/data/benchmark_20261015/ppn_block.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261015])
    parser.add_argument("--search-seconds", type=float, default=5.0)
    parser.add_argument("--search-budget", type=int, default=40)
    args = parser.parse_args()
    benchmark(args.directory, args.json, seeds=tuple(args.seeds),
              search_seconds=args.search_seconds, search_budget=args.search_budget)


if __name__ == "__main__":
    main()
