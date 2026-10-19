# Fresh normal-wick-block measurement for ppn

The report's previously unrecorded 48-network row is measured by
`verify_code/benchmark_ppn_block_report.py`. The recorded run starts a fresh Python
process and fixes all supported BLAS thread controls to one:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 python3 -B \
  -m verify_code.benchmark_ppn_block_report
```

The authoritative machine-readable record is `ppn_block.json`.

| Quantity | Recorded value |
| --- | ---: |
| Raw Wick pairings | 2,880 |
| Independent input contracts | 219 |
| Output block contracts | 48 |
| End-to-end construction and output time | 15.850817791 s |
| Peak resident memory before numerical validation | 176.65625 MiB |
| Exact symbolic certificate | Passed |
| Full-output maximum normalized numerical error | 4.161336738530926e-15 |

The recorded interpreter is Python 3.11.17, the same interpreter as the fresh
three-run dictionary comparison. Its machine-specific absolute path is
intentionally omitted; the portable command is `python3`.

The wall time includes fresh normal independent-table generation, output
adaptation and raw pairing-family construction, `block_reduce` and its built-in
exact symbolic proof, and writing portable execution tables. It excludes
interpreter startup/imports and the subsequent numerical validation. Peak memory
is the fresh process's `ru_maxrss` captured after writing the block output and
before numerical validation; it includes interpreter/import memory.

The planner uses `max_w_types=1`, a 5-second search budget, and 40 search states.
Normalization, materialization, and proof are outside the search time budget.
The saved result is the best 48-network layout found in that budget; the run
does not claim an exhaustive search or global optimality.

Validation uses random complex normal perambulators at seed `20261015`, fixed
color epsilon/delta tensors, and the registered Dirac matrices. No random VVV
is generated. The complete 4-by-4-by-4-by-4-by-4-by-4 output from all 219
independent contracts agrees with the 48-network block output. The maximum
absolute error is `2.441754138272187e-08`; dividing it by the larger maximum
absolute value of the two output tensors gives `4.161336738530926e-15`, below
the `1e-10` tolerance. Numerical execution timing is recorded separately in
the JSON and is not used as construction timing.

Generated independent Python output and block execution JSON are kept under
`/private/tmp/ppn_block_report_py311_20261015`; their SHA-256 digests and sizes are
recorded in the JSON. Rerunning the command regenerates both files.
