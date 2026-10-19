# Generic Wick reducer design and validation

This directory is the standalone implementation of the generic Wick reducer.
The parent checkout is not needed at runtime. Current counts below match the
standalone project's root `README.md`; report tables are scoped to the
reduction object named in each section.

## Reduction rule

The exact field content is used only while generating legal raw Wick pairings.
After raw generation, `reduce_operators()` applies the same generic
factor-graph reduction to an arbitrary operator sequence; the policy is
independent of a correlator name.  Operator species are absent from the
reduction key, which keeps only the contraction data:

- the peram direction and endpoint role (`sin_sin`, `sin_sou`, ... and bare or
  dummy endpoints);
- the VVV side, operator kind, Gamma matrix and Gamma ports;
- the external/momentum tag;
- the canonical connection pattern between all sink and source factor vertices
  present in the input.

The last item is the index graph.  It cannot be discarded: two contractions
with the same factor names but different index connections are different
`contract(...)` calls.  For a two-sided baryon input, the graph key is the
canonical table of peram edge colors under independent permutations of the two
vertex color classes.  The same function accepts arbitrary colored operator
graphs and uses its bounded generic permutation path when the input is not
bipartite.  The corresponding operator map is stored only as a
canonicalization witness.  It is not a separate p/n isospin transporter.

Inside each bucket the existing Dirac Gram and bare-axis transport logic is
used.  Wick signs, coefficients and bare-axis transposes remain metadata in
the dependency relations.  The exact raw generator is unchanged, so flavor
selection still prevents an illegal u/d pairing before reduction.

The generic API is enabled with:

```python
from kernels.wick_reducer import ReductionOptions, reduce_operators

report = reduce_operators(
    operators,
    label="arbitrary_input",
    options=ReductionOptions(),
)
```

`label` is output metadata only.  The supplied VVV/VDV, Gamma, momentum and
normalization factors are the factors being compared; species names are not
part of the key.  There is no second species-aware reduction path.

The factor-graph policy is used for every example.  The exact raw generator
and downstream transporter, Dirac and Gram checks remain unchanged.

## Results

The built-in fixtures below were run with the current generic reducer:

| example | raw Wick | independent | dependent |
| --- | ---: | ---: | ---: |
| pipi | 24 | 10 | 14 |
| pn | 36 | 18 | 18 |
| ppn | 2,880 | 219 | 2,661 |
| delta | 6 | 2 | 4 |
| delta++delta- | 36 | 3 | 33 |
| pip | 24 | 24 | 0 |
| pip-p | 6 | 6 | 0 |
| pn_omega_pi+ | 3,456 | 908 | 2,548 |

For the historical four-baryon fixture, the generic species-blind factor
bucketing produced:

```text
raw Wick                 518,400
factor buckets             4,298
independent basis_contract 4,298
dependent terms          514,102
```

The coverage check is exact:


\[
4,298+514,102=518,400.
\]

The run used Python 3.9.6 and took about 242 s on the local machine.  The
result is an open-index `basis_contract` count for this direct factor-graph
rule.  It is not the orbit count 1,068 and it is not the orbit expansion count
17,088; those numbers use different equivalence relations.

The previous production reducer is a historical comparison baseline for the
same fixture:

```text
518,400 raw Wick -> 31,092 basis_contract
```

The new 4,298 result therefore reflects the additional quotient that removes
hadron names after exact raw Wick generation, while retaining the
peram/VVV/Gamma connection graph and the open-axis reduction.

The end-to-end random audit in
`verify_code/ppnn_random_end_to_end.py` evaluates all 518,400 raw terms with one
fixed random projection of the eight open spin axes and reconstructs the same
scalar from the 4,298 basis terms and 514,102 dependency relations.  For seed
`20261011`, the relative residual is `1.29e-14` (absolute residual
`8.32e-03`).  The raw total is supplied by the full raw run; the reduced side
evaluates each basis tensor once and reuses its 489,966 axis views.  This is a
full raw-coverage scalar check, not a materialized element-by-element
comparison of the dense `4^8` open tensor.

In the current model the VVV and Gamma factors do not depend on hadron name.
The isospin-specific numerical condition is that `peram_u` and `peram_d` are
the same numerical propagator.  Momentum, projection, `external_tag`, Clebsch
factors and extra phases remain part of the key whenever they differ.

The remaining index symmetries are the original reducer's responsibility and
are still applied after graph canonicalization: peram line matching, dummy
index renaming, bare-axis maps/transposes, VVV parity, sink projective checks
and source Dirac Gram projection.  The new key only removes redundant input
labels before that existing tensor reduction.

See [original_vs_improved_flowchart.md](../../../verify_code/original_vs_improved_flowchart.md)
for a side-by-side flowchart and output comparison.
