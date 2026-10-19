# Factor-Graph Optimization Results

This report compares the original checkout (`a3b76f8`) with the optimized
working tree. Each small fixture ran in its own process. Peak RSS is
`resource.getrusage(RUSAGE_SELF).ru_maxrss`; times are single-run measurements
and are intended to show scale, not stable microbenchmark precision.

Both measurements used Python 3.9.6 and NumPy 2.0.2 on macOS 14.8.4 arm64.
The optimized environment also has `opt_einsum` 3.4.0. The original baseline
did not have `opt_einsum`, but the reducer-only benchmarks do not import or
call it; it is used by the numerical audits.

## Changes

1. Canonicalization now returns and stores the operator-map witness during the
   first pass. The bucket reducer no longer canonicalizes every raw term a
   second time. The ppnn witness buffer adds about 4 MiB.
2. A per-reduction context precomputes operator groups, endpoint routes and
   edge colors; permutation tables are cached and reused.
3. Compact pairings and witnesses are read directly from shared buffers with
   cached `Struct.unpack_from`, avoiding a temporary `bytes` slice per record.
4. Two-color factor graphs use exact color refinement when it assigns a unique
   invariant color to every vertex. Any unresolved symmetry follows the
   existing exhaustive canonicalization path.

## Small Fixtures

| Fixture | Raw -> basis / dependent | Baseline (s) | Optimized (s) | Time change | Peak RSS baseline -> optimized (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pipi` | 24 -> 10 / 14 | 0.011745 | 0.006747 | -42.6% | 22.59 -> 20.50 |
| `pn` | 36 -> 18 / 18 | 0.018669 | 0.013393 | -28.3% | 21.14 -> 20.17 |
| `ppn` | 2,880 -> 219 / 2,661 | 1.653858 | 1.265962 | -23.4% | 28.75 -> 27.63 |
| `delta` | 6 -> 2 / 4 | 0.002432 | 0.001777 | -26.9% | 25.25 -> 21.84 |
| `delta++delta-` | 36 -> 3 / 33 | 0.018650 | 0.014868 | -20.3% | 21.84 -> 21.28 |
| `pip` | 24 -> 24 / 0 | 0.007863 | 0.005935 | -24.5% | 25.00 -> 22.42 |
| `pip-p` | 6 -> 6 / 0 | 0.001796 | 0.001656 | -7.8% | 25.88 -> 21.20 |
| `pn_omega_pi+` | 3,456 -> 908 / 2,548 | 2.342970 | 1.798358 | -23.2% | 40.27 -> 39.73 |

## ppnn

| Measure | Original | Optimized | Change |
| --- | ---: | ---: | ---: |
| Raw Wick terms | 518,400 | 518,400 | unchanged |
| Factor buckets / basis contracts | 4,298 | 4,298 | unchanged |
| Dependent terms | 514,102 | 514,102 | unchanged |
| Peram bucket stage | 140.655 s | 65.164 s | -53.7% |
| Transporter, sink and source stage | 322.928 s | 223.523 s | -30.8% |
| Total reducer time | 463.584 s | 288.688 s | -37.7% (1.61x faster) |
| Peak RSS | 406.25 MiB | 410.23 MiB | +3.98 MiB (+0.98%) |

The optimized run keeps the same reduction counts and lowers the measured
ppnn reducer time by 174.9 seconds. Peak RSS rises by about 1%, consistent
with storing one compact operator witness for each raw term.

The supplied historical chat also records 915.216 seconds and about
407.4 MiB for a 518,400 -> 4,298/514,102 run. Its exact source revision and
measurement environment are not identified, so it is preserved as a
historical observation and is not used to calculate the direct speedup above.
The reproducible comparison in this report is the committed original snapshot
`a3b76f8` (463.584 seconds) against this optimized checkout (288.688 seconds).

Recreate the original snapshot benchmark with:

```sh
mkdir -p /tmp/wick-reduce-original
git archive --format=tar a3b76f8 | tar -xf - -C /tmp/wick-reduce-original
cd /tmp/wick-reduce-original
python3 verify_code/run_improved_ppnn.py
```

## Correctness Checks

The eight small fixtures were checked with `opt_einsum.contract`, random
complex perambulators, random antisymmetric VVV tensors and random dense VDV
arrays at color dimensions 3 and 6. Every signed raw Wick contraction was
compared against the generated basis reconstruction using a shared random
projection for open spin axes. All 16 fixture/dimension checks passed; the
maximum per-Wick relative residual was `4.9873e-13` for `ppn` at C=6. The
relative residual for each Wick uses `|direct - reconstructed| / max(|direct|,
|reconstructed|, 1e-30)`, matching the full ppnn check below.

The ppnn check source contains all 518,400 raw contracts and all 518,400
`origen_wick_to_independent` mappings. The full random-peram audit evaluates
all direct raw contractions at C=3 and C=6, uses random antisymmetric VVV
arrays, and compares every generated check mapping against its basis views.
Both dimensions and all 518,400 mapping slots passed. Each eight-axis open
spin identity is evaluated with a fixed random projection, avoiding a dense
`4**8` output tensor for every raw term.

| Color dimension | Raw contracts | Mapping rows checked | Maximum relative residual | Total relative residual |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 518,400 | 518,400 | `1.1053e-12` | `3.6964e-14` |
| 6 | 518,400 | 518,400 | `6.5752e-13` | `2.8591e-14` |

The C=6 maximum absolute residual was `1.8971e-4` on contractions with values
around `1e12`; its relative residual was `6.5752e-13`. The two direct sums and
basis-reconstructed sums also agree to the total relative residuals shown
above. The full audit took 6,121.8 seconds across both dimensions.

The independent canonicalization audit exhaustively tested all 19,683
3-by-3 matrices with three cell states under 708,588 row/column relabelings;
the canonical signatures were invariant. Stored operator witnesses were
checked as bijections that reconstruct the canonical matrix for all 36 `pn`
and 2,880 `ppn` terms.

The complete unit suite passed: 70 tests. A distributable wheel also built
successfully as `wick_diss_reduce-0.1.0-py3-none-any.whl`.

## Reproduction

Install the optional numerical dependencies and run the structural suite:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[numeric]'
.venv/bin/python -B -m unittest discover -s tests -t . -p 'test_*.py'
```

Run the eight-fixture numeric audit and generate the small check files:

```sh
.venv/bin/python verify_code/check_basis_vs_original.py \
  --without-ppnn --include-eighth --dimensions 3,6 \
  --output-dir verify_output/optimized-check \
  --json verify_output/optimized-check/summary.json
```

Generate and fully audit ppnn check data:

```sh
.venv/bin/python verify_code/ppnn_check_stream.py \
  --output verify_output/optimized-check/ppnn_check.py
.venv/bin/python verify_code/ppnn_random_end_to_end.py \
  --check verify_output/optimized-check/ppnn_check.py \
  --color-dimension 3 --color-dimension 6
```

Run the reducer-only ppnn benchmark:

```sh
.venv/bin/python verify_code/run_improved_ppnn.py
```
