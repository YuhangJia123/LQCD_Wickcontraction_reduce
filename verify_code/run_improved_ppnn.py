#!/usr/bin/env python3
"""Compatibility benchmark for the historical four-baryon fixture.

The reducer is invoked through ``reduce_operators`` and receives only a policy
object.  The fixture name is retained in the verify_code data for reproducibility;
no system name controls the reduction path.  New arbitrary inputs should use
``run_factor_graph.py`` or call ``reduce_operators`` directly.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verify_code.ppnn_state_quotient_audit import _operators
from kernels.wick_reducer.reducer import ReductionOptions, reduce_operators


def main() -> None:
    start = time.perf_counter()
    report = reduce_operators(
        _operators(),
        label="generic_fixture",
        mode="independent",
        options=ReductionOptions(isospin_degenerate=True),
    )
    print(json.dumps({
        "raw": report["wick_count"],
        "factor_buckets": report["peram_incidence_buckets"],
        "basis_contract": report["independent_count"],
        "dependent": report["dependent_count"],
        "timing_seconds": report["timing_seconds"],
        "wall_seconds": time.perf_counter() - start,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
