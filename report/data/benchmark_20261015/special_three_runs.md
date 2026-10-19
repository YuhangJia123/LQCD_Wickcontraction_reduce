# Complete p+n+n dictionary comparison, 2026-10-15

Run from the repository root with Python 3.11.17 and the project dependencies:

```sh
python3 -B verify_code/benchmark_special_report.py --runs 3
```

The recorded interpreter was Python 3.11.17 (`python3`); the machine-specific
absolute path is intentionally omitted from this report.
The runner uses the tracked `report/data/benchmark_20261014/pnn_special_input_20261014.json`
for this project's `normal-special` cross-time kernel. The two local reference
implementations are preserved under `report/benchmarks/reference/` as byte-for-byte
snapshots from the sibling colour-orbit project's benchmark sources. Source
SHA-256 digests, per-run measurements, and verification records are in
`special_three_runs.json`.

Each of the four methods ran three times, sequentially, in a fresh subprocess.
The table reports the median construction time and the maximum peak process RSS
across those three runs. OMP, OpenBLAS, MKL, vecLib, and NumExpr thread counts were
set to one. Timing includes construction of the complete in-memory dictionary,
including colour-first flattening and recursive tensor expansion. It excludes
interpreter startup, preparation/imports, fixture loading, SQLite serialization,
and the later dictionary comparison. Peak RSS was read immediately after
dictionary construction, before SQLite output; it includes interpreter imports
and algorithm temporaries.

| Method | Median time (s) | Maximum peak RSS (MiB) | Complete entries per run |
| --- | ---: | ---: | ---: |
| UCA direct local reproduction | 32.019433 | 843.27 | 3,775,680 |
| qcd colour-first local implementation | 25.548473 | 874.16 | 3,775,680 |
| This project normal-special-independent | 5.749668 | 739.61 | 3,775,680 |
| Local sparse antisymmetric recursion | 22.691168 | 273.33 | 3,775,680 |

Every candidate dictionary was compared row by row against the first UCA
dictionary. All three runs of each candidate had zero missing keys, zero extra
keys, and zero coefficient mismatches. The second and third UCA runs were also
exactly equal to the first. Coefficients were checked as exact integers, with no
floating-point tolerance, sampling, or hash-only comparison. This establishes
three-run equality of the complete colour--Dirac dictionary for the merged
project, colour-first implementation, and recursive implementation.

This fixture is **p+n+n (3H)** with relativistic `Cgamma5` and source external
spins `(0, 0, 1)`, yielding 2,880 Wick maps. These timings and equality claims do
not establish ppn or ppnn performance. The recursive reference constructs a
canonical antisymmetric tensor and then expands the complete dictionary; it is
not the paper's gauge-field-dependent, sink-pruned Lambda operation list. UCA
and recursion are local reproductions, not measurements of the original authors'
software. The refactored cross-time kernel retains the exact dictionary and remains
in the same time order of magnitude; its three measured runs were 5.61--6.34 s
with 709.78--739.61 MiB peak RSS.
