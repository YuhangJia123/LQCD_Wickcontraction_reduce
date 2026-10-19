"""Compare streaming expression execution with retained-array evaluation.

Run from the repository root with ``.venv/bin/python -B -m
verify_code.benchmark_block_execution``. This tests execution memory, not planner
time or the contraction cost of a physical ppnn network. All eight axes have
four components; no spin projection is used. ``tracemalloc`` tracks Python and
NumPy allocations seen by its allocator, not process-wide resident memory.
Each identity contraction is materialized to isolate retention of complete
network arrays; this benchmark deliberately does not time path-cache reuse.
"""

from __future__ import annotations

import gc
import json
from time import perf_counter
import tracemalloc

import numpy as np
from opt_einsum import contract

from kernels.block_reduction import BasisView, Factor, Network, evaluate_expression


def _fixture():
    labels = tuple("abcdefgh")
    contracts = {
        index: Network((Factor("X", labels),), labels)
        for index in range(24)
    }
    basis = {
        6 * index + view: BasisView(
            index,
            scalar=-1 if view % 2 else 1,
            coefficient=view + 1,
            permutation=tuple(range(view, 8)) + tuple(range(view)),
        )
        for index in contracts
        for view in range(6)
    }
    rng = np.random.default_rng(20261014)
    source = rng.standard_normal((4,) * 8) + 1j * rng.standard_normal((4,) * 8)
    return contracts, basis, {"X": source}


def _retained_reference(contracts, basis, arrays):
    """Retain all network results and weighted views, as the old evaluator did."""
    computed = {
        index: _materialized_contract(
            arrays["X"], tuple(range(8)), tuple(range(8)), optimize="greedy"
        )
        for index in contracts
    }
    weighted = [
        complex(view.scalar) * complex(view.coefficient)
        * computed[view.contract_id].transpose(view.permutation)
        for view in basis.values()
    ]
    result = np.array(weighted[0], copy=True)
    for value in weighted[1:]:
        result = result + value
    return result


def _materialized_contract(*args, **kwargs):
    return np.array(contract(*args, **kwargs), copy=True)


def _measure(operation):
    gc.collect()
    tracemalloc.start()
    start = perf_counter()
    try:
        value = operation()
        seconds = perf_counter() - start
        peak_bytes = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return value, seconds, peak_bytes


def main():
    contracts, basis, arrays = _fixture()
    reference, reference_seconds, reference_peak = _measure(
        lambda: _retained_reference(contracts, basis, arrays)
    )
    streamed, streamed_seconds, streamed_peak = _measure(
        lambda: evaluate_expression(contracts, basis, None, arrays, _materialized_contract)
    )
    error = np.abs(reference - streamed)
    maximum = float(np.max(error))
    scale = max(float(np.max(np.abs(reference))), 1e-30)
    print(json.dumps({
        "networks": len(contracts),
        "views": len(basis),
        "output_shape": streamed.shape,
        "full_spin_components_per_axis": 4,
        "retained_reference": {
            "wall_seconds": reference_seconds,
            "peak_tracked_bytes": reference_peak,
        },
        "streamed": {
            "wall_seconds": streamed_seconds,
            "peak_tracked_bytes": streamed_peak,
        },
        "max_absolute_error": maximum,
        "max_relative_error": maximum / scale,
        "addition_order_caveat": (
            "Grouping views by network may change floating-point addition order "
            "when input views interleave network IDs."
        ),
    }, indent=2))
    if not np.allclose(reference, streamed, rtol=1e-12, atol=1e-12):
        raise AssertionError("streamed result differs from retained reference")


if __name__ == "__main__":
    main()
