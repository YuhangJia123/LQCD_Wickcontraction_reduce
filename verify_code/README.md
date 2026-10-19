# Orbit reduction and validation code

`verify_code/` contains reproducible orbit-reduction, contraction, and numerical
validation code for fixed `ppn` and `ppnn` Wick fixtures. It is deliberately
separate from `kernels/`:
importing or running these modules does not change the production reducer or
its output format.

From a copied project folder, create the root Conda environment before running
the commands below:

```sh
conda env create -f environment.yml
conda activate wick-diss-reduce
python -m pip install .
```

The commands below use `python3` for compatibility with an already activated
environment; `python` is equivalent inside the Conda environment. A Chinese
version of this guide is available in [`README.md`](README.md).

Some audit notes in this directory compare against earlier reducer runs and
therefore retain historical counts such as 698 and 31,092. The maintained
generic reducer counts are listed in the project root `README.md`.

For the generic contraction reducer, use `run_factor_graph.py` or the
`kernels.wick_reducer.reduce_operators()` API.  The historical orbit scripts below
still use ppn/ppnn names because they audit those published fixtures; those
names do not select the factor-graph algorithm.

The reduction has two visible stages:

\[
  \Omega_{\rm raw}\;\longrightarrow\;
  \Omega_{\rm raw}/K\;\longrightarrow\;
  (\Omega_{\rm raw}/K)/G .
\]

`K` removes internal same-flavour slot permutations.  `G` acts on the
quotient states by the declared proton/neutron block permutations (and the
ppnn isospin flip).  This gives 93 ppn state orbits and 1068 ppnn state
orbits.  The numbers are orbit counts, not the number of calls needed by a
generic open-Dirac tensor contraction.

## Quick start

Run commands from this project root. The root directory is the one containing
`verify_code/`, `kernels/`, and `pyproject.toml`; its folder name does not matter.
The `PYTHONPATH` setting is needed when the modules are invoked with `-m`
outside an installed package:

```sh
cd /path/to/project
PYTHONPATH=. python3 -m verify_code.orbit_ppn
PYTHONPATH=. python3 -m verify_code.orbit_ppnn
```

The ppnn construction enumerates 518,400 raw pairings and is intentionally a
heavier audit.  The commands print the raw Wick count, quotient-state count,
orbit count, orbit-size histograms, and the Burnside count.

Orbit scripts in this directory provide supplementary group-action and numerical
checks. They are separate from the production `normal-wick-block` writer; normal
output uses the production writer described below.

For the generic reducer's check representation, use
`check_basis_vs_original.py`.  It runs the detailed generic path, writes the
small fixture check files through `write_case_output(mode="check")`, and
compares every signed original Wick value with the corresponding
`basis_contract -> basis -> origen_wick` reconstruction.  Perambulator arrays
have shape `(4, N, N, 4)`; baryon VVV arrays are random antisymmetric
`(N, N, N)` tensors, and meson VDV arrays are `(N, N)`.  The open spin axes
are tested with a shared random projection, so ppnn does not require a dense
`4**8` tensor for each of its 518,400 terms.

```sh
cd /path/to/project
PYTHONPATH=. python3 verify_code/check_basis_vs_original.py \
  --without-ppnn --dimensions 3,6 \
  --output-dir verify_output/check --json verify_output/check_summary.json
```

The default fixture set is the seven historical examples.  Add
`--include-eighth` for `pn_omega_pi+`; ppnn is included by default.  Its
518,400-line check source is written only with `--write-ppnn-check`.

For ppnn, `ppnn_check_stream.py` writes the same four layers without retaining
all 518,400 detailed graph records.  `ppnn_sample_numeric.py` then samples
the actual generated mapping at C=3 and C=6, evaluates the direct signed raw
contract and the basis reconstruction with the same random peram/VVV input,
and reports the maximum residual.  For a complete numerical audit, use
`ppnn_random_end_to_end.py --check`; it contracts all 518,400 raw terms for
each requested color dimension and validates every generated mapping slot
against those values.  This is intentionally a multi-minute run per
dimension.

```sh
PYTHONPATH=. python3 verify_code/ppnn_check_stream.py \
  --output verify_output/check/ppnn_check.py
PYTHONPATH=. python3 verify_code/ppnn_random_end_to_end.py \
  --check verify_output/check/ppnn_check.py \
  --color-dimension 3 --color-dimension 6
```

The archived ppn orbit output contains:

* independent mode: 93 representative state blocks and 744 raw state-block
  contract assignments;
* check mode: the complete 2,880-term raw expansion.

## Python API

The two built-in systems are cached after their first construction:

```python
from verify_code.orbit_ppn import ppn_orbit_system, summary as ppn_summary
from verify_code.orbit_ppnn import ppnn_orbit_system, summary as ppnn_summary

ppn = ppn_orbit_system()
ppnn = ppnn_orbit_system()
print(ppn_summary())
print(ppnn_summary())
```

An `OrbitSystem` exposes the audit data needed to reproduce the maps:

```python
ppnn.raw_pairings          # raw Wick id -> quark/antiquark pairing
ppnn.raw_signs             # raw Wick id -> Wick sign
ppnn.raw_to_state          # raw Wick id -> quotient state
ppnn.state_to_raw          # quotient state -> its raw Wick fibre
ppnn.state_orbits          # OrbitRecord objects for the 1068 state orbits
ppnn.raw_orbits            # OrbitRecord objects for the extended raw action
ppnn.state_to_orbit        # quotient state -> orbit id
ppnn.raw_to_orbit          # raw Wick id -> raw orbit id
```

Useful checks are:

```python
ppnn.validate()                 # partition and coverage invariants
ppnn.check_quotient_equivariance()  # q(g.w) == g.q(w)
ppnn.state_fixed_point_histogram()
ppnn.raw_fixed_point_histogram()
```

`verify_code.orbit_common` contains the reusable pieces:
`OrbitRecord`, `OrbitPartition`, `OrbitSystem`, `enumerate_orbits`,
`fixed_point_histogram`, `burnside_count`, and
`action_from_generators`.

## Archived orbit output

`ppn_orbit_independent.py` and `ppn_orbit_check.py` record the earlier orbit
experiment. They preserve signed raw contractions and orbit mappings for
comparison; they are not the production normal-wick-block writer.

## Operator input

`verify_code.operator_input` normalizes the project’s existing operator forms:

```python
from verify_code.operator_input import load_operators, enumerate_raw_wicks

flat = load_operators([sink_operator, source_operator])
pair = load_operators([[sink_operator], [source_operator]])
operators, fields, quarks, antiquarks, terms = enumerate_raw_wicks(pair)
```

It also accepts a JSON path or mapping.  A JSON object may use a library name:

```json
{"name": "ppn"}
```

or explicitly list operators:

```json
{
  "operators": [
    {"side": "sink", "kind": "baryon", "species": "p",
     "label": "sink_p", "flavors": ["u", "u", "d"],
     "gamma": "Cgamma5", "gamma_slots": [1, 2], "open_slot": 0},
    {"side": "source", "kind": "baryon", "species": "p",
     "label": "source_p", "flavors": ["u", "d", "u"],
     "gamma": "Cgamma5_dagger", "gamma_slots": [1, 2], "open_slot": 0,
     "external_tag": "source-0"}
  ]
}
```

Flavor order, Gamma slots, open slots, and `external_tag` are retained. This
normalizer enumerates raw Wick terms; it does not infer a symmetry group for a
new arbitrary operator list. For `normal-wick-block`, use the production
writer; it constructs and certifies the raw pairing family automatically.

## Normal Wick block validation

Call the production writer directly from normal operators:

```python
from dataclasses import replace
from kernels.wick_reducer import expression_output, operator_model

operators = tuple(replace(op, mode="normal")
                  for op in operator_model.operator_library()["ppn"])
result = expression_output.write_output(
    expression_output.split_operators(operators),
    "result/normal_wick_block/ppn_normal_wick_block.py",
    name="ppn", mode="block",
)
assert result["block_network_count"] == 48
```

`block` is only the `normal-wick-block` mode. The W reduction operates on the
complete raw pairing family, preserves output transposes, and verifies the
weighted expression. ppn validates to 48 networks and ppnn to 288. See
[`docs/reference/block-api.md`](../docs/reference/block-api.md) for the
planner interface and its exact certificate.

## Validation and limitations

The unit tests cover raw/state/orbit coverage, Burnside counts, equivariance,
output syntax, automatic normal-wick-block planning, and numerical checks:

```sh
python3 -B -m unittest discover -s tests -t . -p 'test_*.py'
```

`verify_code.numeric_validation.validate_seeds` requires at least three seeds:

```python
from verify_code.numeric_validation import validate_seeds
records = validate_seeds(ppnn_orbit_system(), (20261009, 20261010, 20261011))
assert max(item.max_relative_residual for item in records) < 1e-12
```

The audit uses shape-correct complex perams `(3, 4, 3, 4)` and an
antisymmetric `(3, 3, 3)` VVV tensor.  For each seed, the complete raw sum and
the orbit reconstruction use the same peram/VVV sample through a shared input
factor, while assigning symmetry-compatible random signed contributions to
orbit fibres.  It verifies the exact raw, explicit-block and weighted-block
identities under the declared action.  It does not claim that an unconstrained
random peram automatically has that symmetry.

Current scope is intentionally narrow:

* built-in orbit actions are implemented only for ppn and ppnn;
* the code does not automatically discover all Gamma, momentum, transporter,
  open-Dirac-axis, projector, or VVV symmetries;
* 93 and 1068 are state-block orbit counts, not a replacement for the
  generic reducer’s open-Dirac basis (219 for ppn and 4,298 for ppnn in the
  current standalone project; report files may record scoped comparison data);
* performance comparisons must measure the same numerical contract workload.
  Orbit enumeration alone is not evidence that the final contracted result
  is faster or uses less memory.

See [`group_orbit_theory.md`](group_orbit_theory.md) for the mathematical
derivation and [`implementation_report.md`](implementation_report.md)
for the current audit numbers and checkpoint history.

For a timing/RSS reference run, use:

```sh
python3 verify_code/benchmark_orbit_vs_reducer.py \
  --contract-sample 8 \
  --output verify_code/benchmark_orbit_vs_reducer.json
```

The JSON distinguishes measured orbit construction, measured production runs,
and sampled `opt_einsum.contract` extrapolations.  The extrapolated fused
times are conditional estimates; they are not a claim that a generic tensor
contract can be factorized.
