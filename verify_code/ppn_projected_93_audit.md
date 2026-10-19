# ppn: What the 93 Projected Contractions Are

## Finding

In this reconstruction, **93 is the number of projected, pre-summed state blocks**, not 93 individual raw Wick contractions and not 93 open six-Dirac-index tensors. A block first combines eight signed raw Wick terms that differ by the three same-flavor slot exchanges internal to the three nucleons. The 360 resulting label states are then grouped by independently exchanging the two proton baryons at sink and source.

The exact reduced formula audited by the script is

$$F_{ppn}=\sum_{s\in S}b_s=\sum_{O\in S/G}|O|b_{s_O},\qquad b_s=\sum_{w:s(w)=s}\operatorname{sgn}(w)\,\langle P,C_w\rangle.$$

Here `C_w` is the local code's open-index Wick tensor; `P` projects its six spin axes onto the fixed ppn charge slice of the \({}^3\mathrm{He}\) operator. Each `b_s` sums 8 raw Wick terms. The orbit multiplicity `|O|` is 2 or 4, so the 93 representative blocks stand for `6*2 + 87*4 = 360` state blocks and therefore `8*360 = 2880` raw terms.

## Why The Blocks Are Equal

The spin slice is `(|+,-,+> - |-,+,+>)/sqrt(6)` in the ordering (p1,p2,n), at both source and sink. It is the fixed ppn component of the six-term \({}^3\mathrm{He}\) interpolator in Eq. (6) of [Yamazaki (arXiv:1012.0410)](https://arxiv.org/abs/1012.0410). For either one-sided proton-block exchange, the signed Wick block contributes a minus sign and the spin projector contributes another minus sign, so `b_(g s)=b_s`. The group is `S2_sink x S2_source`; its 93 orbits have sizes {2: 6, 4: 87}.

## Why 698 And 93 Differ

The local 698-term audit preserves the full open tensor `T_{ABCDEF}=sum_w sign(w) C_w`, with all six Dirac indices available, and reproduces it to about `1.4e-15`. The 93-block identity applies only after contracting `T` with the fixed spin-flavor projector `P`, producing one scalar correlator. Projection discards tensor components and permits cancellations, so a basis size for the open tensor and a count for projected blocks are different targets. In the earlier open-tensor test, the span of 93 one-raw-term candidates left residuals of `0.82-0.86`; see [ppn_orbit_numeric_audit.md](ppn_orbit_numeric_audit.md).

The [1205.0585 paper](https://arxiv.org/abs/1205.0585) also separates these notions: it cites 93 as an earlier symmetry-based permutation count, while its own unified contraction algorithm has separate `N_perm` and `N_contr` counts (for the nonrelativistic 3He setup, 360 and 3240 in Table 8). Thus neither 93 nor 3240 can be compared directly with the local 698 without matching the exact object and contraction stage.

## Audit

- Projector: `Eq. (6) fixed ppn slice (|+,-,+>-|-,+,+>)/sqrt(6), upper components, at source and sink`.
- All six open Dirac axes are projected before terms are compared; only upper components 0 and 1 survive.
- Perams are independent generic complex isospin-degenerate matrices, and color tensors use `N_c=3` epsilon symbols.
- Direct reference: 2880 signed Wick contractions per seed.
- Reduced evaluation: 93 representative blocks, independently reevaluating 744 underlying raw contractions per seed (8 per block).
- Orbit weights are the exact integer sizes 2 or 4; they are not fitted from numerical data.
- Raw single-Wick representative max orbit residual: `9.969288e-01`; fitted raw-93 full-sum residual: `1.908259e+00`.
- Eight-Wick block max orbit residual: `7.042687e-15`; independently recomputed block-93 full-sum residual: `6.033386e-15`.
- Full reconstruction at tolerance `1e-10`: **PASS**.
- Orbit histogram: `{2: 6, 4: 87}`; generated member table: [ppn_projected_93_mapping.md](ppn_projected_93_mapping.md).
- Seeds: `20261021, 20261022, 20261023, 20261024`; elapsed time: `15.931 s`.

## Seed-Wise Values

| Seed | Direct projected sum | raw-93 residual | block-93 residual |
| ---: | ---: | ---: | ---: |
| 20261021 | `(244998.796741+137576.749537j)` | `1.908259e+00` | `6.033386e-15` |
| 20261022 | `(27702.8581181+42154.9936415j)` | `9.523263e-01` | `3.761353e-15` |
| 20261023 | `(223480.555512+1362868.81863j)` | `4.973646e-01` | `3.368580e-15` |
| 20261024 | `(-147860.87923+125315.435498j)` | `9.933060e-01` | `2.103550e-15` |

## Relation To The Papers

The [0912.1383v1 paper](https://arxiv.org/abs/0912.1383v1) states that computer scrutiny leaves 93 contractions for the \({}^3\mathrm{He}\) correlator and reports agreement with the full result on one gauge configuration. It does not print the 93 survivor list or a term-by-term map. The same discussion separately says its zero-momentum sink baryon block can evaluate `2^N_N=8` contractions simultaneously. Our reconstruction explains a concrete way 93 projected blocks can account for 2,880 terms, but the missing published list prevents identifying our IDs as the authors' exact 93 entries.

This audit verifies the algebra for the repository's built-in ppn quark operators, the specified fixed-charge spin projection, isospin-degenerate propagators, and its modeled color/spin contraction. It is not a bit-for-bit reproduction of the original gauge-configuration calculation or of the paper's unpublished survivor table. The 93 representative blocks expand to 744 ordinary raw `contract` evaluations in this reference script; a fused baryon-block kernel may evaluate those combinations more efficiently.

## Reproduction

```sh
python3 -B verify_code/ppn_projected_93_audit.py
```

The script checks exact member coverage (`2880` IDs, once each), state-block sizes (`8` each), orbit weights, independent representative reevaluation, and direct full-sum equality.
