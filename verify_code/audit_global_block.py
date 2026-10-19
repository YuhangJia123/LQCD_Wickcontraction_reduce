"""Full-open-spin normal ppn audit of raw Wick, final tables, and W planning."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np
import opt_einsum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.block_reduction import BlockPlan, block_reduce, evaluate_expression, evaluate_plan
from kernels.block_reduction.model import FamilyCertificate, WeightedNetwork
from kernels.block_reduction.planner import _block_specs, _collect, _reconstruct, normalize_inputs
from verify_code.normal_block_validation import generate_normal_fixture
from kernels.block_reduction.output_adapter import load_normal_output, pairing_family_from_operators


def _backend():
    calls = []

    def contract(*args, **kwargs):
        calls.append(1)
        kwargs.setdefault("optimize", "greedy")
        return opt_einsum.contract(*args, **kwargs)

    return contract, calls


def _error(reference, actual):
    if reference.shape != actual.shape:
        raise AssertionError(f"output shape changed: {reference.shape} versus {actual.shape}")
    delta = reference - actual
    maximum = float(np.max(np.abs(delta)))
    scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(actual))), 1e-30)
    return {"maximum_absolute": maximum, "maximum_relative": maximum / scale,
            "norm_relative": float(np.linalg.norm(delta)) /
                             max(float(np.linalg.norm(reference)),
                                 float(np.linalg.norm(actual)), 1e-30)}


def _raw_plan(family, specs, raw_ledger):
    # Exact fixed-output normalization first; the free-output pass shares
    # contractions across transposed views without changing the raw sum.
    networks, reconstruction = _reconstruct(raw_ledger, specs, {})
    return BlockPlan((), networks, reconstruction, specs, family.terms, (),
                     {"output_shape": (4,) * 6})


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


def _alternative_layout_plan(family, specs, target_terms, pairs):
    """Rebuild a discovered layout independently from its pairing orbits."""
    from kernels.block_reduction.global_layout import _prepare, _orbit, _fused

    pairings, _sink, _source, allowed, *_rest = _prepare(family, specs)
    pairs = tuple(tuple(pair) for pair in pairs)
    if any(pair not in allowed for pair in pairs):
        raise AssertionError("a discovered alternative uses an invalid pair")
    expanded, definitions, by_propagator = _block_specs(specs)
    seen = set()
    fused = []
    for pairing in sorted(pairings):
        if pairing in seen:
            continue
        orbit = _orbit(pairing, pairs)
        term = pairings[pairing]
        for mask, image in enumerate(orbit):
            if image not in pairings or pairings[image].coefficient != (
                    term.coefficient * (-1) ** bin(mask).count("1")):
                raise AssertionError("the alternative layout has an invalid exchange orbit")
        seen.update(orbit)
        fused.append(WeightedNetwork(
            _fused(term.network, pairs, family.factor_positions, expanded, by_propagator),
            term.coefficient))
    if len(seen) != len(pairings):
        raise AssertionError("alternative layout does not cover the raw family")
    active = tuple(definitions[name] for name in sorted({
        factor.tensor for term in fused for factor in term.network.factors
        if factor.tensor in definitions}))
    ledger = _collect(fused, expanded)
    networks, reconstruction = _reconstruct(ledger, expanded, {})
    certificate = FamilyCertificate(family.terms, tuple(fused), pairs,
                                     family.factor_positions, active)
    result = BlockPlan(active, networks, reconstruction, expanded, tuple(target_terms), (),
                       {"output_shape": (4,) * 6}, certificate)
    if not result.certificate.verify():
        raise AssertionError("alternative layout symbolic certificate failed")
    return result


def audit_global_ppn(*, directory: Path | None = None, seeds=(20261013, 20261014),
                     search_seconds=5.0, search_budget=40, raw_samples=6,
                     direct_raw_first_seed=True):
    from kernels.wick_reducer.operator_model import operator_library
    from kernels.block_reduction.evaluate import _execute

    temporary = None
    if directory is None:
        temporary = tempfile.TemporaryDirectory()
        directory = Path(temporary.name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "normal_ppn_independent.py"
    try:
        generated_at = perf_counter()
        generated = generate_normal_fixture("ppn", path)
        generation_seconds = perf_counter() - generated_at
        source = path.read_text(encoding="utf-8")
        adapted_at = perf_counter()
        adapted = load_normal_output(source)
        operators = tuple(replace(operator, mode="normal")
                          for operator in operator_library()["ppn"])
        family = pairing_family_from_operators(operators, tensor_specs=adapted.tensor_specs)
        adapter_seconds = perf_counter() - adapted_at
        if (len(adapted.basis_contract), len(adapted.basis), len(family.terms)) != (219, 2760, 2880):
            raise AssertionError("ppn three-table or raw family count changed")
        normalized_at = perf_counter()
        _contracts, _views, target_terms, specs = normalize_inputs(
            adapted.basis_contract, adapted.basis, adapted.coefficients, adapted.tensor_specs)
        target_ledger = _collect(target_terms, specs)
        raw_ledger = _collect(family.terms, specs)
        target_coefficients = {key: term.coefficient for key, term in target_ledger.items()}
        raw_coefficients = {key: term.coefficient for key, term in raw_ledger.items()}
        if raw_coefficients != target_coefficients:
            raise AssertionError("raw Wick family does not map exactly to final weighted tables")
        raw_plan = _raw_plan(family, specs, raw_ledger)
        mapping_seconds = perf_counter() - normalized_at
        planned_at = perf_counter()
        plan = block_reduce(adapted.basis_contract, adapted.basis, adapted.coefficients,
                            tensor_specs=adapted.tensor_specs,
                            pairing_family=family, max_w_types=1,
                            search_seconds=search_seconds, search_budget=search_budget)
        planning_seconds = perf_counter() - planned_at
        if not plan.certificate.verify():
            raise AssertionError("W symbolic certificate failed")
        alternative_started = perf_counter()
        selected_pairs = plan.family_certificate.pairs if plan.family_certificate else ()
        alternative_plans = []
        for layout in plan.stats.get("global_best_layouts", ()):
            pairs = tuple(tuple(pair) for pair in layout["pairs"])
            if pairs == selected_pairs:
                continue
            alternative = _alternative_layout_plan(family, specs, target_terms, pairs)
            if len(alternative.networks) != layout["network_count"]:
                raise AssertionError("alternative reconstruction count differs from search score")
            alternative_plans.append((pairs, alternative))
        alternative_proof_seconds = perf_counter() - alternative_started
        samples = []
        for sample_index, seed in enumerate(seeds):
            arrays = _arrays(adapted, seed)
            raw_backend, raw_calls = _backend()
            started = perf_counter()
            raw_value = evaluate_plan(raw_plan, arrays, contract=raw_backend)
            raw_seconds = perf_counter() - started
            target_backend, target_calls = _backend()
            started = perf_counter()
            target_value = evaluate_expression(adapted.basis_contract, adapted.basis,
                                               adapted.coefficients, arrays,
                                               contract=target_backend)
            target_seconds = perf_counter() - started
            block_backend, block_calls = _backend()
            started = perf_counter()
            block_value = evaluate_plan(plan, arrays, contract=block_backend)
            block_seconds = perf_counter() - started
            if tuple(raw_value.shape) != (4,) * 6:
                raise AssertionError("ppn did not retain all six open Dirac axes")
            raw_error = _error(raw_value, target_value)
            block_error = _error(target_value, block_value)
            if max(raw_error["maximum_relative"], block_error["maximum_relative"]) > 1e-10:
                raise AssertionError(f"normal ppn numerical mismatch at seed {seed}")
            alternative_errors = []
            for pairs, alternative in alternative_plans:
                alternative_backend, alternative_calls = _backend()
                started = perf_counter()
                alternative_value = evaluate_plan(alternative, arrays, contract=alternative_backend)
                seconds = perf_counter() - started
                error = _error(target_value, alternative_value)
                if error["maximum_relative"] > 1e-10:
                    raise AssertionError(f"alternative W layout mismatch at seed {seed}")
                alternative_errors.append({"pairs": pairs, "calls": len(alternative_calls),
                                           "seconds": seconds, "target_vs_alternative": error})
            direct_error = None
            direct_calls = None
            direct_seconds = None
            if direct_raw_first_seed and sample_index == 0:
                # Independently execute all original Wick rows; this path
                # does not use canonicalization to compute their sum.
                direct_backend, calls = _backend()
                direct_started = perf_counter()
                direct = np.zeros((4,) * 6, dtype=np.complex128)
                for term in family.terms:
                    direct += complex(term.coefficient) * _execute(term.network, arrays,
                                                                    direct_backend)
                direct_seconds = perf_counter() - direct_started
                direct_calls = len(calls)
                direct_error = _error(direct, raw_value)
                if direct_error["maximum_relative"] > 1e-10:
                    raise AssertionError(f"direct 2880 Wick sum mismatch at seed {seed}")
            # Independent spot check: execute selected unsimplified Wick rows
            # and compare their individual fixed-output canonical witnesses.
            sampled_indices = np.linspace(0, len(family.terms) - 1,
                                          num=min(raw_samples, len(family.terms)),
                                          dtype=int)
            row_errors = []
            for index in sampled_indices:
                term = family.terms[int(index)]
                witness = next(iter(_collect((term,), specs).values()))
                value = complex(term.coefficient) * _execute(term.network, arrays,
                                                             opt_einsum.contract)
                canonical = complex(witness.coefficient) * _execute(witness.network, arrays,
                                                                    opt_einsum.contract)
                row_errors.append(_error(value, canonical)["maximum_relative"])
            if any(value > 1e-10 for value in row_errors):
                raise AssertionError(f"raw row normalization differs at seed {seed}")
            samples.append({
                "seed": seed, "output_shape": list(raw_value.shape),
                "raw_canonicalized_calls": len(raw_calls), "target_calls": len(target_calls),
                "block_calls": len(block_calls), "raw_canonicalized_seconds": raw_seconds,
                "target_seconds": target_seconds, "block_seconds": block_seconds,
                "raw_vs_target": raw_error, "target_vs_block": block_error,
                "alternative_layout_errors": alternative_errors,
                "direct_raw_calls": direct_calls, "direct_raw_seconds": direct_seconds,
                "direct_raw_vs_canonicalized": direct_error,
                "raw_spotcheck_maximum_relative": max(row_errors, default=0.0),
            })
        return {
            "mode": "normal", "target": "complete six-axis four-Dirac tensor",
            "source_path": str(path),
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "generator": generated, "raw_wick_count": len(family.terms),
            "basis_contract_count": len(adapted.basis_contract),
            "basis_view_count": len(adapted.basis),
            "exact_mapping": True, "fixed_output_ledger_size": len(raw_ledger),
            "raw_canonicalized_network_count": len(raw_plan.networks),
            "generation_seconds": generation_seconds,
            "adapter_seconds": adapter_seconds, "mapping_seconds": mapping_seconds,
            "planning_seconds": planning_seconds, "plan_stats": dict(plan.stats),
            "alternative_proof_seconds": alternative_proof_seconds,
            "alternative_layouts": [{"pairs": pairs, "network_count": len(alternative.networks),
                                      "symbolic_verified": True}
                                     for pairs, alternative in alternative_plans],
            "block_network_count": len(plan.networks),
            "w_definition_count": len(plan.w_definitions), "samples": samples,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--search-seconds", type=float, default=5.0)
    parser.add_argument("--search-budget", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261013, 20261014])
    parser.add_argument("--skip-direct-raw", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = audit_global_ppn(directory=args.directory, seeds=tuple(args.seeds),
                              search_seconds=args.search_seconds,
                              search_budget=args.search_budget,
                              direct_raw_first_seed=not args.skip_direct_raw)
    content = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
