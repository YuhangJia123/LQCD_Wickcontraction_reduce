#!/usr/bin/env python3
"""Reproduce the four complete p+n+n colour--Dirac dictionary measurements.

Each sample uses a fresh process. Dictionary construction is timed separately
from SQLite serialization and exact row-by-row verification. The reference
implementations are snapshots from the sibling project's report benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "report" / "benchmarks" / "reference"
FIXTURE = ROOT / "report" / "data" / "benchmark_20261014" / "pnn_special_input_20261014.json"
EXPECTED_ENTRIES = 3_775_680
SOURCE_SPINS = (0, 0, 1)
METHODS = (
    ("uca_p1_direct", "UCA-P1 direct local reproduction"),
    ("qcd_colour_first", "qcd colour-first local implementation"),
    ("merged_normal_special_independent", "This project normal-special-independent"),
    ("recursive_antisymmetric_product", "Local sparse antisymmetric-product recursion"),
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REFERENCE))


def _peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _prepare(method: str):
    if method in {"uca_p1_direct", "qcd_colour_first"}:
        import uca_3h_reference as uca

        if method == "uca_p1_direct":
            return lambda: (uca.accumulate_direct(True, SOURCE_SPINS), {})

        def colour_first():
            buckets = uca.accumulate_colour_first(True, SOURCE_SPINS)
            return uca.flatten_buckets(buckets), {"colour_buckets": len(buckets)}

        return colour_first

    if method == "merged_normal_special_independent":
        from kernels.wick_reducer.unified import load_unified_input, run_unified

        config, operators, matrices = load_unified_input(FIXTURE)

        def merged():
            result = run_unified(config, operators, gamma_matrices=matrices)
            return result["coefficients"], result["stats"]

        return merged

    if method == "recursive_antisymmetric_product":
        from recursive_3h_reference import (
            build_recursive_tensor,
            expand_recursive_tensor,
            make_fixture,
        )

        correlator = make_fixture()

        def recursive():
            tensor, details = build_recursive_tensor(correlator)
            values = expand_recursive_tensor(tensor, correlator)
            return values, {**details, "canonical_tensor_components": len(tensor)}

        return recursive

    raise ValueError(f"unsupported method: {method}")


def _integer_row(key: int, value: complex) -> tuple[int, int]:
    number = complex(value)
    integer = int(number.real)
    if number.imag != 0 or number.real != integer:
        raise ValueError("the selected fixture must yield exactly integer coefficients")
    return int(key), integer


def _json_stats(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_stats(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_stats(item) for item in value]
    return value


def _write_dictionary(database: Path, table: str, values: dict) -> None:
    if table not in {"reference", "candidate"}:
        raise ValueError("invalid table")
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA cache_size = -8192")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "full_key INTEGER PRIMARY KEY, coefficient INTEGER NOT NULL) WITHOUT ROWID"
        )
        connection.execute(f"DELETE FROM {table}")
        connection.executemany(
            f"INSERT INTO {table}(full_key, coefficient) VALUES (?, ?)",
            (_integer_row(key, value) for key, value in values.items()),
        )


def _worker(method: str, database: Path, table: str) -> dict[str, Any]:
    run_method = _prepare(method)
    start = time.perf_counter()
    values, details = run_method()
    seconds = time.perf_counter() - start
    peak = _peak_rss_mib()
    entries = len(values)
    _write_dictionary(database, table, values)
    return {
        "method": method,
        "seconds": seconds,
        "peak_rss_mib": peak,
        "entries": entries,
        "details": _json_stats(details),
        "python": sys.version.split()[0],
    }


def _run_worker(method: str, database: Path, table: str) -> dict[str, Any]:
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
               method, str(database), table]
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
             "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"},
    )
    if completed.returncode:
        raise RuntimeError(f"{method} worker failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


def _compare(database: Path) -> dict[str, Any]:
    with sqlite3.connect(database) as connection:
        reference_count = connection.execute("SELECT COUNT(*) FROM reference").fetchone()[0]
        candidate_count = connection.execute("SELECT COUNT(*) FROM candidate").fetchone()[0]
        reference = connection.execute("SELECT full_key, coefficient FROM reference ORDER BY full_key")
        candidate = connection.execute("SELECT full_key, coefficient FROM candidate ORDER BY full_key")
        left, right = reference.fetchone(), candidate.fetchone()
        missing = extra = mismatched = 0
        while left is not None and right is not None:
            if left[0] < right[0]:
                missing += 1
                left = reference.fetchone()
            elif right[0] < left[0]:
                extra += 1
                right = candidate.fetchone()
            else:
                mismatched += left[1] != right[1]
                left, right = reference.fetchone(), candidate.fetchone()
        while left is not None:
            missing += 1
            left = reference.fetchone()
        while right is not None:
            extra += 1
            right = candidate.fetchone()
    return {
        "reference_entries": reference_count,
        "candidate_entries": candidate_count,
        "missing_keys": missing,
        "extra_keys": extra,
        "coefficient_mismatches": mismatched,
        "exact_dictionary_equal": (
            missing == extra == mismatched == 0
            and reference_count == candidate_count == EXPECTED_ENTRIES
        ),
    }


def _provenance() -> dict[str, Any]:
    paths = [FIXTURE, Path(__file__).resolve(), REFERENCE / "uca_3h_reference.py",
             REFERENCE / "recursive_3h_reference.py", ROOT / "kernels" / "wick_reducer" / "unified.py"]
    paths.extend(sorted((ROOT / "kernels" / "wick_colour_orbit").glob("*.py")))
    return {
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths},
        "reference_snapshot_origin": {
            "uca_3h_reference.py": "wick_colour_orbit_improve/report/benchmarks/reference/uca_p1/benchmark_3h_methods.py",
            "recursive_3h_reference.py": "wick_colour_orbit_improve/report/benchmarks/recursive_3h_reference.py",
        },
    }


def _validate_fixture() -> None:
    from kernels.wick_colour_orbit.examples import make_pnn
    from kernels.wick_colour_orbit.user_api import normalize_fast_correlator
    from kernels.wick_reducer.unified import _special_correlator, load_unified_input

    config, operators, matrices = load_unified_input(FIXTURE)
    if (config.input, config.mode, config.output) != ("normal", "special", "independent"):
        raise ValueError("the report fixture must request normal-special-independent")
    actual = normalize_fast_correlator(_special_correlator(operators), gamma_matrices=matrices)
    expected = make_pnn()
    if actual.line_count != expected.line_count:
        raise ValueError("the report fixture has a different line count")
    for side in ("sink", "source"):
        left, right = getattr(actual, side), getattr(expected, side)
        if tuple(field.flavor for field in left.fields) != tuple(field.flavor for field in right.fields):
            raise ValueError(f"the report fixture has different {side} flavors")
        if len(left.vertices) != len(right.vertices):
            raise ValueError(f"the report fixture has different {side} vertices")
        for a, b in zip(left.vertices, right.vertices):
            if (a.spin_terms, a.phase, a.colour_kind) != (b.spin_terms, b.phase, b.colour_kind):
                raise ValueError(f"the report fixture has different {side} tensors")


def run(repeats: int, output: Path) -> dict[str, Any]:
    _validate_fixture()
    samples: dict[str, list[dict[str, Any]]] = {method: [] for method, _ in METHODS}
    result: dict[str, Any] = {
        "benchmark": "3H = p+n+n, relativistic C gamma5, source external spins (0,0,1)",
        "fixture": {"path": str(FIXTURE.relative_to(ROOT)), "source_spins": SOURCE_SPINS,
                    "wick_maps": 2880, "expected_dictionary_entries": EXPECTED_ENTRIES},
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "repeats": repeats,
        "timing_scope": (
            "Fresh process per sample; complete in-memory coefficient dictionary construction. "
            "Includes colour-first flattening and the merged normal-special cross-time kernel, and "
            "recursive tensor expansion. Excludes process startup, preparation/imports, fixture "
            "loading, SQLite serialization, and exact comparison. Samples execute sequentially."
        ),
        "memory_scope": (
            "Peak process RSS sampled immediately after complete dictionary generation, before "
            "SQLite output. Includes interpreter imports and algorithm temporaries."
        ),
        "thread_limits": "OMP/OpenBLAS/MKL/vecLib/NumExpr thread counts set to 1 for every worker",
        "dictionary_key": "packed (colour_code, spin_code) -> exact integer coefficient",
        "comparison_scope": (
            "Every candidate's full dictionary is compared row-by-row against the first "
            "UCA direct dictionary, including later UCA samples. No float tolerance, "
            "hash-only check, sampling, or count-only equality is used."
        ),
        "recursive_method_scope": (
            "Local canonical antisymmetric-tuple recursion plus full dictionary expansion; "
            "not the published gauge-field-dependent, sink-pruned Lambda operation list."
        ),
        "provenance": _provenance(),
        "complete": False,
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wick_special_three_runs_") as temporary:
        database = Path(temporary) / "comparison.sqlite3"
        for repetition in range(repeats):
            for method, _ in METHODS:
                is_reference = method == "uca_p1_direct" and repetition == 0
                print(f"run {repetition + 1}/{repeats}: {method}", file=sys.stderr, flush=True)
                sample = _run_worker(method, database, "reference" if is_reference else "candidate")
                sample["repetition"] = repetition + 1
                if is_reference:
                    sample["verification"] = {
                        "role": "reference dictionary",
                        "entry_count_matches_expected": sample["entries"] == EXPECTED_ENTRIES,
                    }
                    passed = sample["entries"] == EXPECTED_ENTRIES
                else:
                    sample["verification"] = _compare(database)
                    passed = sample["verification"]["exact_dictionary_equal"]
                samples[method].append(sample)
                output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(f"  {sample['seconds']:.6f} s; {sample['peak_rss_mib']:.2f} MiB; verified={passed}",
                      file=sys.stderr, flush=True)
                if not passed:
                    raise RuntimeError(f"exact dictionary validation failed for {method}")

    summaries = []
    for method, label in METHODS:
        runs = samples[method]
        checks = [item["verification"] for item in runs if "exact_dictionary_equal" in item["verification"]]
        summaries.append({
            "id": method, "label": label, "repeats": repeats,
            "median_seconds": statistics.median(item["seconds"] for item in runs),
            "min_seconds": min(item["seconds"] for item in runs),
            "max_seconds": max(item["seconds"] for item in runs),
            "peak_rss_mib_max": max(item["peak_rss_mib"] for item in runs),
            "entries": runs[0]["entries"],
            "dictionary_verification": {
                "complete_dictionary": all(item["entries"] == EXPECTED_ENTRIES for item in runs),
                "exact_comparisons": len(checks),
                "all_compared_runs_exactly_equal": all(item["exact_dictionary_equal"] for item in checks),
                "role": "reference plus repeated-reference checks" if method == "uca_p1_direct" else "candidate",
            },
        })
    result.update({"complete": True, "all_dictionary_checks_passed": True, "methods": summaries})
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "report/data/benchmark_20261015/special_three_runs.json")
    parser.add_argument("--worker", nargs=3, metavar=("METHOD", "DATABASE", "TABLE"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(_worker(args.worker[0], Path(args.worker[1]), args.worker[2]), separators=(",", ":")))
    else:
        if args.runs < 1:
            parser.error("--runs must be positive")
        result = run(args.runs, args.output)
        print(json.dumps(result["methods"], indent=2))
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
