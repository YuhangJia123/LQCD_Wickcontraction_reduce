"""Command-line runner for JSON operator inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys

from .input_json import load_input
from .output import write_count_output, write_output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reduce an ordinary-colour Wick correlator from JSON input."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON operator document with explicit sink/source sides")
    parser.add_argument("--output", required=True, type=Path,
                        help="single JSON result file to write")
    parser.add_argument("--mode", choices=("independent", "check", "count"),
                        help="override the mode declared in the input file")
    parser.add_argument("--wick-scope", choices=("complete", "cross_time"),
                        help="override the input Wick scope (default: complete)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="SQLite contribution batch size for count mode")
    args = parser.parse_args(argv)

    try:
        settings = load_input(args.input)
        mode = args.mode or settings.pop("mode")
        operators = settings.pop("operators")
        input_wick_scope = settings.pop("wick_scope")
        shared = {
            "name": settings.pop("name"),
            "normal_ordered": settings.pop("normal_ordered"),
            "wick_scope": args.wick_scope or input_wick_scope,
            "isospin_degenerate_perams": settings.pop("isospin_degenerate_perams"),
            "tolerance": settings.pop("tolerance"),
            "gamma_matrices": settings.pop("gamma_matrices"),
        }
        if mode == "count":
            summary = write_count_output(
                operators, args.output, batch_size=args.batch_size, **shared,
            )
        else:
            summary = write_output(
                operators,
                args.output,
                mode=mode,
                **shared,
            )
    except (OSError, sqlite3.Error, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))
    if mode == "count":
        print(f"count: {summary['nlist']} nonzero coordinates from "
              f"{summary['raw_wicks']} Wick -> {args.output}")
    else:
        print(
            f"{mode}: {summary['raw_wicks']} Wick, "
            f"{summary['independent_wicks']} independent, "
            f"{summary['dependent_wicks']} dependent -> {args.output}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
