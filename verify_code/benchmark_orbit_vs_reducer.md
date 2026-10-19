# Orbit versus production reducer benchmark

This file records a local benchmark for the verify_code orbit representation and
the existing production `wick_reducer`.  The two paths reduce different
objects, so the numbers must be read together with the mathematical scope in
[`group_orbit_theory.md`](group_orbit_theory.md): the orbit path reduces a
fixed ppn/ppnn label-state action, while the production reducer preserves the
open Dirac tensor and its transporter relations.

The measurements below were run on 2026-10-09 in the repository checkout with
the system Python 3.9. The commands were run in separate Python processes;
`resource.getrusage(...).ru_maxrss` is therefore a clean process peak. On
macOS the value is reported in bytes.

## Measured construction and reduction

| path | raw | state | orbit / basis | wall seconds | peak RSS | contract calls represented |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| verify_code ppn orbit | 2,880 | 360 | 93 | 0.4898 | 26,361,856 B (25.1 MiB) | 744 state-block calls, or 93 if fused and certified |
| production ppn independent | 2,880 | - | 698 basis | 2.9853 | 28,884,992 B (27.5 MiB) | 698 basis calls |
| verify_code ppnn orbit | 518,400 | 32,400 | 1,068 | 90.5676 | 649,199,616 B (619.2 MiB) | 17,088 state-block calls, or 1,068 if fused and certified |
| production ppnn independent | 518,400 | - | 31,092 basis | 539.4 (historical) | about 1.21 GiB (historical) | 31,092 basis calls |

The production ppn timing is a direct `reduce_case("ppn", mode="independent")`
run. The verify_code measurements build and validate the complete raw-to-state
and state-to-orbit mappings, including partition coverage and equivariance;
they are not just the final orbit count. The ppnn production row is the
existing historical measurement recorded in `docs/reference/wick-api.md`; the
full 518,400-term production reduction was not rerun for the 2026-10-09 orbit
benchmark. A later same-checkout comparison of the generic reducer is in
[`optimization_benchmark.md`](../report/benchmarks/optimization_benchmark.md); its
4,298 open-Dirac basis is a different object from the 1,068 orbit-state count.

The direct commands that produced the three fresh measurements were:

```text
python3 -B -c 'import resource,time; t=time.perf_counter(); from verify_code.orbit_ppn import ppn_orbit_system; s=ppn_orbit_system(); print({"counts":(s.raw_wick_count,s.state_count,s.orbit_count),"wall":time.perf_counter()-t,"rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})'
python3 -B -c 'import resource,time; t=time.perf_counter(); from kernels.wick_reducer.reducer import reduce_case; x=reduce_case("ppn",mode="independent"); print({"counts":(x["wick_count"],x["independent_count"]),"wall":time.perf_counter()-t,"rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"timing":x["timing_seconds"]})'
python3 -B -c 'import resource,time; t=time.perf_counter(); from verify_code.orbit_ppnn import ppnn_orbit_system; s=ppnn_orbit_system(); print({"counts":(s.raw_wick_count,s.state_count,s.orbit_count),"wall":time.perf_counter()-t,"rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})'
```

The reusable driver is
`[benchmark_orbit_vs_reducer.py](benchmark_orbit_vs_reducer.py)`. Its default
ppnn production mode is `historical`, so a normal run does not unexpectedly
start the GiB-scale production job. Use `--production-ppnn-mode coarse` or
`independent` only when an explicit fresh production measurement is desired.
The driver can also time a small sample of actual `opt_einsum.contract`
calls with `--contract-sample N`; these samples estimate the cost of 744,
17,088, 93, or 1,068 calls, but they do not prove that an opaque fused tensor
is a valid identity.

## Interpretation and limits

For ppn, the measured verify_code construction is faster than production
independent reduction (about 0.49 s versus 2.99 s), but it still represents
744 ordinary contractions when each 8-term state block is expanded. The
potential 93-call path requires a separately certified block fusion.

For ppnn, the verify_code construction is much lighter than the historical
production reduction in represented basis calls (1,068 versus 31,092), but
the complete verify_code audit itself used about 90.6 s and 619 MiB. This is not
an end-to-end speedup claim: the production timing is historical, and the
verify_code timing includes full map construction and validation. A fair
production-versus-verify_code wall-time comparison requires rerunning both on the
same machine, Python runtime, input fixture, and validation scope.

The ordinary state-block workload is 8 calls per ppn orbit and 16 per ppnn
orbit. Consequently the un-fused orbit path is 744 and 17,088 calls. A fused
path can be counted as 93 and 1,068 high-level calls only when the Gamma,
VVV, open-axis, momentum, and projector transformations have been proven for
the chosen fixture; otherwise explicit block expansion is the correct
fallback.
