# Contraction output format

`expression_output.py` accepts `[O1_sink, O2_source]` and writes a readable
Python module using `opt_einsum.contract`. The source and sink operator order,
flavor slots, Gamma slots, and open slots are taken exactly from the input.

## Generated names

The file header documents the elemental names used by the input. Baryons use
`VVV_sink` and `conj_VVV_source`; mesons use `VDV_sink` and
`conj_VDV_source`. Distinct momentum labels are supplied with
`external_tag`, producing names such as `VVV_sink_1`.

Independent output contains four layers:

1. `<name>_coefficients[i]` stores the final coefficient for basis view `i`.
2. `<name>_basis_contract[wick_id]` evaluates each independent representative
   exactly once with `contract(...)`; Wick IDs start at 1.
3. `<name>_basis[basis_id]` applies the representative sign and any open-axis
   `.transpose(...)`; basis IDs start at 0.
4. `<name>_result` sums nonzero coefficients times the corresponding basis
   views.

Check output adds the raw Wick contracts, signs, and the raw-to-basis relation
map. It is useful for numerical reconstruction tests and can be much larger
than independent output.

## API example

```python
from kernels.wick_reducer import make_operator, write_output

sink = [make_operator(
    "sink", "baryon", "p", "sink_p", ("u", "u", "d"),
    "Cgamma5", (1, 2), open_slot=0,
)]
source = [make_operator(
    "source", "baryon", "p", "source_p", ("u", "d", "u"),
    "Cgamma5_dagger", (1, 2), open_slot=0,
)]
write_output([sink, source], "verify_output/pn_independent.py",
             name="pn", mode="independent")
```

The generated module still needs the Gamma matrices and numerical peram,
VVV, and VDV arrays named in its header. The Chinese companion is
[`output_format.md`](wick-output.md).
