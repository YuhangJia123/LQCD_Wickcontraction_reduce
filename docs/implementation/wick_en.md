# Distillation Wick reduction algorithm

This document describes the production reducer in `kernels/wick_reducer/`. The
implementation is local to this project: Wick enumeration and the small Dirac
contraction primitives are provided by `kernels/correlator_wick_normalizer.py`, while
the reducer itself is exposed as `kernels.wick_reducer.reduce_operators()`.

## Input conventions

Wick pairings are enumerated with exact flavors. In the default isospin policy,
`u` and `d` are considered one numerical peram class only after the valid raw
pairings have been generated; other flavors remain distinct. Baryon flavors
are three ordered quark slots. Meson flavors are `(antiquark, quark)`.

Each operator may define one Gamma matrix acting on two slots, an open bare
spin slot for baryons, an `external_tag`, and optional contraction metadata.
These fields are part of the equivalence policy. A hadron species label is
metadata and is not itself a generic factor-graph color.

## Reduction stages

1. Enumerate every flavor-compatible raw Wick pairing and its permutation sign.
2. Build a compact, deterministic factor-graph key from operator colors,
   directed peram incidence, endpoint roles, and metadata. Terms with the same
   key are placed in a candidate bucket.
3. Enumerate every allowed transporter in a bucket. A transporter contains the
   operator permutation, line map, VVV parity, and bare-axis map.
4. Construct the sink Dirac payload first. Zero overlaps are rejected before
   source payloads are built; nonzero candidates are screened by incremental
   Gram/rank projection.
5. For sink-compatible transporters, project the source payload and retain the
   complex coefficients, basis Wick IDs, and any bare-axis transpose. A term is
   dependent only when this projection proves the relation.
6. Emit independent representatives and their basis views. The output writer
   performs one `contract(...)` call per representative, then applies signs,
   transposes, and the coefficient vector.

The graph key is only a candidate filter. Final equivalence still checks line
direction, endpoint, open/dummy role, Gamma slots, VVV parity, and Dirac
payloads. The bounded transporter search fails explicitly above its configured
limit instead of silently merging incomplete results.

## Current fixtures

The generic reducer currently reports these raw/basis/dependent counts:

| Example | Raw Wick | Basis contracts | Dependent |
| --- | ---: | ---: | ---: |
| `pipi` | 24 | 10 | 14 |
| `pn` | 36 | 18 | 18 |
| `ppn` | 2,880 | 219 | 2,661 |
| `delta` | 6 | 2 | 4 |
| `delta++delta-` | 36 | 3 | 33 |
| `pip` | 24 | 24 | 0 |
| `pip-p` | 6 | 6 | 0 |
| `pn_omega_pi+` | 3,456 | 908 | 2,548 |

The separate verify_code `ppnn` audit has 518,400 raw pairings and is intentionally
not part of the small committed independent examples. The Chinese companion
is [`algorithm.md`](wick.md).
