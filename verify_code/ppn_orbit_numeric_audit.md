# ppn 93-candidate numerical span audit

This audit directly evaluates all raw ppn `contract(...)` expressions from the current operator fixture and compares their Wick-sign-weighted sum with the production reducer and with the span of the 93 orbit candidate basis tensors.

## Fixture

- Python 3.12.14, NumPy 2.3.5, opt_einsum 3.4.0.
- Dimension `Nc=3`; sink/source VVV tensors are random complex scalar multiples of the fully antisymmetric epsilon tensor.
- `peram_u_sin_sou` is a generic random complex array of shape `(4,3,3,4)`, shared by `u` and `d` as in the reducer's default isospin-degenerate ppn mode.
- Gamma matrices are the project's built-in `Cgamma5` and `Cgamma5_dagger`; each contraction returns a `(4,4,4,4,4,4)` tensor.
- Terms: 2880 raw Wicks, 698 production basis terms, 93 orbit candidates.

The direct total is computed as `sum(wick_sign[w] * contract(w))`. The production reconstruction uses the reducer's selected coefficients and open-axis transposes. The candidate reconstruction uses least squares against the 93 signed candidate basis tensors and reports relative residual `||full - projection|| / ||full||`; the fitted coefficients are intentionally not written as a result.

## Results

| Seed | Candidate tensor rank | Candidate-span residual | Production-reducer residual | 2,880 calls (s) |
| ---: | ---: | ---: | ---: | ---: |
| 20261009 | 93/93 | 8.563398e-01 | 1.422936e-15 | 9.996 |
| 20261010 | 93/93 | 8.201924e-01 | 1.431869e-15 | 10.856 |
| 20261011 | 93/93 | 8.283801e-01 | 1.373800e-15 | 10.815 |

Pairwise relative differences between the sample-fitted coefficient vectors:

- Seeds 20261009 and 20261010: `8.943728e-01`.
- Seeds 20261009 and 20261011: `8.619349e-01`.
- Seeds 20261010 and 20261011: `8.284470e-01`.

Total audit runtime, including reducer setup, all tensor contractions, and least-squares fits: `33.497 s`.

## Interpretation

The production reducer reconstruction agrees with the direct 2,880-term sum at floating-point precision for all fixtures. The 93 candidate tensors have numerical rank 93, but the full result has a large residual outside their span in every fixture. Therefore the 93 calls do not reconstruct the current ppn correlator for these generic inputs. The seed-to-seed coefficient changes also rule out treating a sample-fitted coefficient vector as fixed orbit weights.

This is strong counterevidence to using the proposed label orbits alone as a reduction of the current concrete output. It does not refute a paper's additional operator/color-spin assumptions or a separately derived symmetry-projected correlator. No candidate-summed correlator is emitted.

## Reproduction

Install the optional numerical dependencies from the project root with `python3 -m pip install '.[numeric]'`.

```sh
python3 -m pip install '.[numeric]'
python3 -B -m verify_code.ppn_orbit_numeric_audit
```
