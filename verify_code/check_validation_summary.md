# Generic Check Validation

All outputs in this note use the one generic `reduce_operators` path. The species and factor-graph switches are not part of the package API.

The numerical comparison is

\[
  s_w\,C_w^{\rm direct}
  \;\stackrel{?}{=}
  \sum_{b,\pi} a_{w b\pi}\,s_b\,
  \operatorname{transpose}_{\pi}(C_b^{\rm basis\_contract}),
\]

where `basis_contract` is deliberately unsigned, `s` is the exact Wick sign,
and `a` and `pi` are the reducer's recorded coefficient and open-axis
permutation. Random perambulator arrays have shape `(4,N,N,4)`, baryon VVV
arrays are fully antisymmetric `(N,N,N)` tensors, and meson VDV arrays have
shape `(N,N)`. The seven small fixtures were checked at `N=3` and `N=6`.

| fixture | raw Wick | basis_contract | dependent | max relative residual |
|---|---:|---:|---:|---:|
| `pipi` | 24 | 10 | 14 | `6.10e-16` |
| `pn` | 36 | 18 | 18 | `3.63e-16` |
| `ppn` | 2880 | 219 | 2661 | `4.56e-16` |
| `delta` | 6 | 2 | 4 | `0` |
| `delta++delta-` | 36 | 3 | 33 | `1.32e-16` |
| `pip` | 24 | 24 | 0 | `0` |
| `pip-p` | 6 | 6 | 0 | `0` |

The complete streamed ppnn check file has:

| object | count |
|---|---:|
| `ppnn_origen_wick_raw` | 518400 |
| `ppnn_basis_contract` | 4298 |
| `ppnn_origen_wick` | 518400 |
| `ppnn_origen_wick_to_independent` | 518400 |
| dependent relations | 514102 |

Thus `518400 = 4298 + 514102`. The ppnn file is intentionally streamed and
is about 463 MB. A 130-row uniform sample of its actual relation map was
evaluated with fresh random inputs at both color dimensions:

| color dimension | sampled rows | max absolute residual | max relative residual |
|---:|---:|---:|---:|
| 3 | 130 | `5.33e-7` | `9.52e-16` |
| 6 | 130 | `1.88e-1` | `1.97e-15` |

The ppnn sample checks the exact generated coefficients and transposes from
the complete check file. It does not claim that all 518400 expensive direct
einsum evaluations were completed; the complete check coverage and reducer
relation coverage are full, while the numerical ppnn check is sampled.

Artifacts:

- `../verify_output/check/seven_summary.json`
- `../verify_output/check/pipi_check.py`, `pn_check.py`, `ppn_check.py`, `delta_check.py`, `delta_plus_plusdelta-_check.py`, `pip_check.py`, `pip-p_check.py`
- `../verify_output/check/ppnn_check.py`
- `../verify_output/check/ppnn_sample_summary.json`
