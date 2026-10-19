"""Audit one already selected ppnn block layout.

This driver deliberately skips layout scoring.  It builds the first verified
288-network layout, checks its exact symbolic certificate, and compares only
the final full eight-open-axis Wick sum with the block plan.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from kernels.block_reduction import evaluate_expression, evaluate_plan
from kernels.block_reduction.canonical import canonicalization_cache
from kernels.block_reduction.model import BlockPlan, FamilyCertificate, WeightedNetwork
from kernels.block_reduction import planner
from kernels.block_reduction.global_layout import (_fused, _prepare, _endpoint_actions,
                                           CompactPairingTerms)
from verify_code.audit_global_ppnn import (_arrays, _execution_progress, _jsonable,
                                     _Progress, _save_plan, _sha256, _write_json)
from kernels.block_reduction.output_adapter import load_normal_output, pairing_family_from_operators
from verify_code.ppnn_state_quotient_audit import _operators


PAIRS = ((0, 1), (2, 5), (3, 4), (6, 7), (8, 11), (9, 10))


class _NoProgress:
    def emit(self, *args, **kwargs):
        return None


def audit(source: Path, directory: Path, seed: int = 20261014):
    started = perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "source": str(source.resolve()),
              "source_sha256": _sha256(source), "seed": seed,
              "operator_mode": "normal", "dirac_components": 4,
              "selected_pairs": PAIRS, "layout_selection": "first_verified_288",
              "other_layouts_not_scored": True, "phase_seconds": {}}

    phase = perf_counter()
    adapted = load_normal_output(source)
    report["basis_contract_count"] = len(adapted.basis_contract)
    report["basis_view_count"] = len(adapted.basis)
    report["phase_seconds"]["load_input"] = perf_counter() - phase

    operators = tuple(replace(operator, mode="normal") for operator in _operators())
    family = pairing_family_from_operators(operators, tensor_specs=adapted.tensor_specs,
                                           compact=True)
    if not isinstance(family.terms, CompactPairingTerms):
        raise TypeError("ppnn adapter must return a CompactPairingTerms family")
    report["raw_wick_count"] = len(family.terms)

    phase = perf_counter()
    with canonicalization_cache():
        token = planner._canonicalizers.set([])
        try:
            contracts, views, original_terms, input_specs = planner.normalize_inputs(
                adapted.basis_contract, adapted.basis, adapted.coefficients,
                adapted.tensor_specs)
            specs, definitions, by_propagator = planner._block_specs(input_specs)
            initial = planner._collect_family(family, specs)
            pairings, sink_types, source_types, _edges, sinks, sources, identities = _prepare(
                family, input_specs)
            witnesses = _endpoint_actions(
                family, input_specs, sinks, sources, sink_types, source_types,
                identities, pairings)
            if any(tuple(sorted(pair)) not in {
                    tuple(sorted(edge)) for edge in _edges} for pair in PAIRS):
                raise ValueError("selected layout contains an illegal same-flavor pair")
            # The complete compact family is partitioned into 2**6 W exchange
            # orbits.  Materialize exactly one fused representative per orbit.
            fused_terms = []
            for pairing in family.terms.representatives(PAIRS):
                term = pairings[pairing]
                fused_terms.append(WeightedNetwork(
                    _fused(term.network, PAIRS, family.factor_positions,
                           specs, by_propagator), term.coefficient))
            fused_terms = tuple(fused_terms)
            candidate = planner._collect(fused_terms, specs)
            free_cache = planner._BoundedCache(4096)
            networks, reconstruction = planner._reconstruct(candidate, specs, free_cache)
            active = planner._used_types(candidate, definitions)
            certificate = FamilyCertificate(
                family.terms, fused_terms, PAIRS, family.factor_positions,
                tuple(definitions[name] for name in sorted(active)),
            )
            stats = {
                "input_contracts": len(contracts), "input_views": len(views),
                "initial_network_count": len(initial),
                "network_count": len(networks), "w_type_count": len(active),
                "search_states": 0, "search_complete": True,
                "budget_exhausted": False, "optimality_scope": "selected_layout",
                "search_strategy": "fixed_first_288_layout", "max_w_types": 1,
                "pairing_family_supplied": True, "global_layouts_tested": 0,
                "global_layout_complete": False, "selected_layout_network_count": len(networks),
                "symbolic_verified": False,
            }
            plan = BlockPlan(tuple(definitions[name] for name in sorted(active)),
                             networks, reconstruction,
                             {name: spec for name, spec in specs.items()
                              if spec.kind != "wedge" or name in active},
                             original_terms, (), stats, certificate)
            proof_started = perf_counter()
            if not planner._verify_plan(
                    plan, verified_target=initial,
                    verified_family_terms=family.terms):
                raise AssertionError("selected fixed layout failed exact symbolic proof")
            plan.stats["symbolic_verified"] = True
            plan.stats["proof_seconds"] = perf_counter() - proof_started
        finally:
            planner._canonicalizers.reset(token)
    report["phase_seconds"]["fixed_layout_plan"] = perf_counter() - phase
    report["block_network_count"] = len(plan.networks)
    report["fused_term_count"] = len(fused_terms)
    report["w_definition_count"] = len(plan.w_definitions)
    report["symbolic_verified"] = bool(plan.stats["symbolic_verified"])
    report["plan_execution"] = _save_plan(plan, directory)

    arrays = _arrays(adapted, seed)
    progress = _Progress(directory)
    progress.emit("phase_started", phase="evaluate_final_wick_sum",
                  expected_contractions=len(adapted.basis_contract))
    phase = perf_counter()
    with _execution_progress(progress, len(adapted.basis_contract)):
        reference = evaluate_expression(adapted.basis_contract, adapted.basis,
                                        adapted.coefficients, arrays)
    report["phase_seconds"]["evaluate_final_wick_sum"] = perf_counter() - phase
    progress.emit("phase_completed", phase="evaluate_final_wick_sum",
                  output_shape=reference.shape)
    progress.emit("phase_started", phase="evaluate_fixed_block_sum",
                  expected_contractions=len(plan.networks))
    phase = perf_counter()
    with _execution_progress(progress, len(plan.networks)):
        actual = evaluate_plan(plan, arrays)
    report["phase_seconds"]["evaluate_fixed_block_sum"] = perf_counter() - phase
    progress.emit("phase_completed", phase="evaluate_fixed_block_sum",
                  output_shape=actual.shape)
    if reference.shape != (4,) * 8 or actual.shape != reference.shape:
        raise AssertionError(f"expected full eight-axis output, got {reference.shape} and {actual.shape}")
    delta = reference - actual
    maximum = float(np.max(np.abs(delta)))
    scale = max(float(np.max(np.abs(reference))), float(np.max(np.abs(actual))), 1e-30)
    report["numerical_validation"] = {
        "seed": seed, "comparison": "final weighted Wick sum vs final block sum",
        "output_shape": actual.shape, "maximum_absolute_error": maximum,
        "maximum_normalized_error": maximum / scale,
        "norm_relative_error": float(np.linalg.norm(delta)) / max(
            float(np.linalg.norm(reference)), float(np.linalg.norm(actual)), 1e-30),
        "tolerance": 1e-10, "full_open_tensor": True,
    }
    if maximum / scale > 1e-10:
        raise AssertionError("selected fixed layout does not reproduce final Wick sum")
    report["status"] = "completed"
    report["wall_seconds"] = perf_counter() - started
    _write_json(directory / "report.json", report)
    progress.emit("run_finished", status=report["status"], report=str(directory / "report.json"))
    progress.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("/private/tmp/ppnn_block_run/normal_ppnn_independent.py"))
    parser.add_argument("--directory", type=Path,
                        default=Path("/private/tmp/ppnn_block_run/fixed_288_audit"))
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args(argv)
    result = audit(args.source, args.directory, args.seed)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
