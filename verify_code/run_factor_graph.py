#!/usr/bin/env python3
"""Run the generic factor-graph reducer on a selected operator fixture.

The reducer accepts any ``Sequence[Operator]``.  This command-line helper
only supplies the small built-in library as convenient input; the algorithm
does not branch on the fixture name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kernels.wick_reducer import ReductionOptions, operator_library, reduce_operators


def run_fixture(
    operators,
    *,
    label: str = "correlator",
    isospin_degenerate: bool = True,
    mode: str = "independent",
):
    """Reduce arbitrary operators under the factor-graph policy."""
    options = ReductionOptions(
        isospin_degenerate=isospin_degenerate,
    )
    return reduce_operators(operators, label=label, mode=mode, options=options)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", default="pn", choices=tuple(operator_library()))
    parser.add_argument("--mode", choices=("coarse", "independent", "detailed"), default="independent")
    parser.add_argument("--no-isospin", action="store_true")
    args = parser.parse_args(argv)
    report = run_fixture(
        operator_library()[args.example],
        label=args.example,
        isospin_degenerate=not args.no_isospin,
        mode=args.mode,
    )
    print(json.dumps({
        "name": report["name"],
        "raw": report["wick_count"],
        "factor_buckets": report.get("peram_incidence_buckets"),
        "basis_contract": report.get("independent_count"),
        "dependent": report.get("dependent_count"),
        "factor_graph": report["factor_graph"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
