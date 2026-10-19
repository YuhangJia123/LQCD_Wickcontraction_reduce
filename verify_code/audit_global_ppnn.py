"""Long-running acceptance of normal ppnn final tables and block_reduce.

The supplied file is parsed as data.  Its complete weighted eight-axis target
is compared against the returned block plan; no spin projection or raw-Wick
numerical enumeration is used.  The planner performs its own exact symbolic
certificate check, which this driver does not redundantly repeat.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
import sys
import threading
from time import perf_counter
import traceback

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.block_reduction import block_reduce, evaluate_expression, evaluate_plan
from kernels.block_reduction.model import GaussianRational
from kernels.block_reduction.output_adapter import load_normal_output, pairing_family_from_operators
from verify_code.ppnn_state_quotient_audit import _operators


def _jsonable(value):
    if isinstance(value, (Fraction, GaussianRational, complex)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


class _Progress:
    def __init__(self, directory):
        self.directory = directory
        self.started = perf_counter()
        self.phase = "initializing"
        self.phase_started = self.started
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.details = {}
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def emit(self, event, *, phase=None, **details):
        with self.lock:
            if phase is not None and phase != self.phase:
                self.phase = phase
                self.phase_started = perf_counter()
                self.details = {}
            self.details.update(details)
            row = dict(self.details, event=event, phase=self.phase,
                       elapsed_seconds=perf_counter() - self.started,
                       phase_elapsed_seconds=perf_counter() - self.phase_started,
                       max_rss_bytes=_rss_bytes())
            _write_json(self.directory / "progress.json", row)
            print(json.dumps(_jsonable(row), ensure_ascii=False), flush=True)

    def _heartbeat(self):
        while not self.stop.wait(30):
            self.emit("heartbeat")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _collect_progress(progress):
    """Observe consumed rows without modifying normalization or its caches."""
    import kernels.block_reduction.planner as planner

    original = planner._collect
    calls = 0

    def observed(terms, specs):
        nonlocal calls
        calls += 1
        call_id = calls
        total = len(terms) if hasattr(terms, "__len__") else None
        monitored = total is None or total >= 50000
        started = perf_counter()
        count = 0
        if monitored:
            progress.emit("collect_started", collect_call=call_id,
                          collect_total=total, collect_consumed=0)

        def rows():
            nonlocal count
            for term in terms:
                yield term
                count += 1
                if count % 50000 == 0:
                    progress.emit("collect_progress", collect_call=call_id,
                                  collect_total=total, collect_consumed=count,
                                  collect_seconds=perf_counter() - started)

        result = original(rows(), specs)
        if monitored:
            progress.emit("collect_completed", collect_call=call_id,
                          collect_total=total, collect_consumed=count,
                          collect_ledger_size=len(result),
                          collect_seconds=perf_counter() - started)
        return result

    planner._collect = observed
    try:
        yield
    finally:
        planner._collect = original


@contextmanager
def _execution_progress(progress, expected):
    """Keep the default evaluator and its path cache while counting calls."""
    import kernels.block_reduction.evaluate as evaluator

    original = evaluator._execute
    counts = {"calls": 0}

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        counts["calls"] += 1
        if counts["calls"] % 100 == 0 or counts["calls"] == expected:
            progress.emit("contraction_progress", contractions=counts["calls"],
                          expected_contractions=expected)
        return result

    evaluator._execute = observed
    try:
        yield counts
    finally:
        evaluator._execute = original


def _arrays(adapted, seed):
    rng = np.random.default_rng(seed)
    arrays = dict(adapted.constant_arrays)
    by_identity = {}
    for name, spec in adapted.tensor_specs.items():
        if spec.kind == "peram":
            identity = spec.identity or name
            if identity not in by_identity:
                by_identity[identity] = (rng.standard_normal(spec.shape)
                                         + 1j * rng.standard_normal(spec.shape)) / np.sqrt(2)
            arrays[name] = by_identity[identity]
    return arrays


def _save_plan(plan, directory):
    """Save portable execution tables, not the large replay certificate."""
    target = directory / "plan_execution.json"
    _write_json(target, {
        "format": "block_reduce_execution_v1",
        "coefficient_encoding": "exact scalar strings",
        "certificate_included": False,
        "w_definitions": plan.w_definitions,
        "networks": plan.networks,
        "reconstruction": [{"network_id": term.network_id,
                             "coefficient": str(term.coefficient),
                             "permutation": term.permutation}
                            for term in plan.reconstruction],
        "tensor_specs": plan.tensor_specs,
        "stats": plan.stats,
        "pairing_layout": None if plan.family_certificate is None else {
            "pairs": plan.family_certificate.pairs,
            "factor_positions": plan.family_certificate.factor_positions,
            "fused_terms": len(plan.family_certificate.fused_terms),
            "raw_terms": len(plan.family_certificate.raw_terms),
        },
    })
    return {"path": str(target.resolve()), "sha256": _sha256(target),
            "bytes": target.stat().st_size, "certificate_included": False}


def audit(*, source, directory, search_seconds=1800, search_budget=200, seed=20261014):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    progress = _Progress(directory)
    report = {"status": "running", "source": str(source),
              "directory": str(directory), "seed": seed,
              "parameters": {"max_w_types": 1, "search_seconds": search_seconds,
                             "search_budget": search_budget,
                             "compact_pairing_family": True},
              "phase_seconds": {}, "operator_mode": "normal",
              "dirac_components": 4, "raw_numeric_evaluation": False}
    started = perf_counter()
    try:
        progress.emit("phase_started", phase="load_input", source=str(source))
        phase_at = perf_counter()
        report["source_sha256"] = _sha256(source)
        report["source_bytes"] = source.stat().st_size
        adapted = load_normal_output(source)
        report["phase_seconds"]["load_input"] = perf_counter() - phase_at
        report["basis_contract_count"] = len(adapted.basis_contract)
        report["basis_view_count"] = len(adapted.basis)
        report["nonzero_coefficient_count"] = sum(bool(c) for c in adapted.coefficients)
        progress.emit("phase_completed", basis_contract_count=len(adapted.basis_contract),
                      basis_view_count=len(adapted.basis), source_sha256=report["source_sha256"])

        progress.emit("phase_started", phase="build_pairing_family")
        phase_at = perf_counter()
        operators = tuple(replace(operator, mode="normal") for operator in _operators())
        family = pairing_family_from_operators(operators, tensor_specs=adapted.tensor_specs,
                                               compact=True)
        report["raw_wick_count"] = len(family.terms)
        report["phase_seconds"]["build_pairing_family"] = perf_counter() - phase_at
        progress.emit("phase_completed", raw_wick_count=len(family.terms),
                      physical_exchange_classes=family.terms.endpoint_classes)

        progress.emit("phase_started", phase="block_reduce",
                      search_seconds=search_seconds, search_budget=search_budget)
        phase_at = perf_counter()
        with _collect_progress(progress):
            plan = block_reduce(adapted.basis_contract, adapted.basis, adapted.coefficients,
                                tensor_specs=adapted.tensor_specs, max_w_types=1,
                                search_seconds=search_seconds, search_budget=search_budget,
                                pairing_family=family)
        report["phase_seconds"]["block_reduce"] = perf_counter() - phase_at
        if not plan.stats.get("symbolic_verified"):
            raise AssertionError("block_reduce did not confirm its built-in exact symbolic proof")
        report.update(block_network_count=len(plan.networks),
                      w_definition_count=len(plan.w_definitions),
                      reconstruction_view_count=len(plan.reconstruction),
                      symbolic_verified=True, plan_stats=dict(plan.stats))
        report["plan_execution"] = _save_plan(plan, directory)
        _write_json(directory / "report.json", report)
        progress.emit("phase_completed", block_network_count=len(plan.networks),
                      w_definition_count=len(plan.w_definitions), symbolic_verified=True)

        arrays = _arrays(adapted, seed)
        progress.emit("phase_started", phase="evaluate_input",
                      expected_contractions=len(adapted.basis_contract), seed=seed)
        phase_at = perf_counter()
        with _execution_progress(progress, len(adapted.basis_contract)) as target_calls:
            reference = evaluate_expression(adapted.basis_contract, adapted.basis,
                                            adapted.coefficients, arrays)
        report["phase_seconds"]["evaluate_input"] = perf_counter() - phase_at
        report["target_contractions"] = target_calls["calls"]
        progress.emit("phase_completed", output_shape=reference.shape,
                      contractions=target_calls["calls"])

        progress.emit("phase_started", phase="evaluate_plan",
                      expected_contractions=len(plan.networks), seed=seed)
        phase_at = perf_counter()
        with _execution_progress(progress, len(plan.networks)) as block_calls:
            actual = evaluate_plan(plan, arrays)
        report["phase_seconds"]["evaluate_plan"] = perf_counter() - phase_at
        report["block_contractions"] = block_calls["calls"]
        if reference.shape != (4,) * 8 or actual.shape != reference.shape:
            raise AssertionError(f"expected full eight-axis ppnn output, got {reference.shape} and {actual.shape}")
        if not np.isfinite(reference).all() or not np.isfinite(actual).all():
            raise AssertionError("nonfinite numerical result")
        delta = reference - actual
        maximum = float(np.max(np.abs(delta)))
        scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(actual))), 1e-30)
        report["numerical_validation"] = {
            "seed": seed, "output_shape": actual.shape,
            "maximum_absolute_error": maximum,
            "maximum_normalized_error": maximum / scale,
            "norm_relative_error": float(np.linalg.norm(delta)) / max(
                float(np.linalg.norm(reference)), float(np.linalg.norm(actual)), 1e-30),
            "tolerance": 1e-10, "full_open_tensor": True,
        }
        if maximum / scale > 1e-10:
            raise AssertionError(f"full ppnn numerical mismatch: {maximum / scale:.6g}")
        report["status"] = "completed"
        progress.emit("phase_completed", **report["numerical_validation"])
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        progress.emit("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["wall_seconds"] = perf_counter() - started
        report["max_rss_bytes"] = _rss_bytes()
        _write_json(directory / "report.json", report)
        progress.emit("run_finished", status=report["status"],
                      report=str(directory / "report.json"))
        progress.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/private/tmp/ppnn_block_run/normal_ppnn_independent.py"))
    parser.add_argument("--directory", type=Path,
                        default=Path("/private/tmp/ppnn_block_run/block_audit"))
    parser.add_argument("--search-seconds", type=float, default=1800)
    parser.add_argument("--search-budget", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args(argv)
    audit(source=args.source, directory=args.directory,
          search_seconds=args.search_seconds, search_budget=args.search_budget, seed=args.seed)


if __name__ == "__main__":
    main()
