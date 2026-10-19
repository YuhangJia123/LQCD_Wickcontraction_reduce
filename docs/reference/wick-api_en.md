# `wick_reducer`

This directory contains the production reducer package. The project entry
point is the project root's [`README.md`](../../README.md); this file is a
package-level index for readers who open `kernels/wick_reducer/` directly.

## Public API

```python
from kernels.wick_reducer import ReductionOptions, reduce_operators

report = reduce_operators(
    operators,
    label="my_input",
    options=ReductionOptions(),
)
```

The reducer first enumerates exact-flavour Wick pairings and then applies one
generic factor-graph path.  The path preserves peram direction and endpoints,
VVV/VDV structure, Gamma slots, open axes, external tags, and connection
topology.  Isospin-degenerate mode may identify `u` and `d` perams when the
caller declares that they are numerically equal; it does not add a second
species-specific reducer.

The main modules are:

- `operator_model.py`: operator and field definitions;
- `peram_incidence.py`: peram-line incidence profiles;
- `reducer.py`: generic candidate grouping, transporter checks, and source
  Gram reduction;
- `expression_output.py`: `independent` and `check` Python writers;
- `run_examples.py`: built-in fixture runner.

## Running from the project root

Run these commands from the project root, one level above `kernels/`,
which is the directory that
contains `kernels/`, `verify_code/`, `tests/`, and `pyproject.toml`:

```sh
python3 kernels/wick_reducer/run_examples.py --example pn --mode independent
python3 -B -m unittest discover -s tests -t . -p 'test_*.py'
```

Install the project when using it from another directory:

```sh
python3 -m pip install .
```

For a reproducible setup from a copied project folder, use the bundled Conda
environment first:

```sh
conda env create -f environment.yml
conda activate wick-diss-reduce
python -m pip install .
```

The optional numerical audit dependencies are installed with:

```sh
python3 -m pip install '.[numeric]'
```

## Output

`write_output(..., mode="independent")` emits the reduced
`basis_contract`, coefficients, basis views, and result expression.
`write_output(..., mode="check")` additionally emits the original Wick
records, their signs, the raw-to-basis mapping, and the reconstruction data.
See [`output_format_en.md`](wick-output_en.md) for the English variable
schema, or [`output_format.md`](wick-output.md) for Chinese.

## Current counts

The maintained counts are listed in the parent
[`README.md`](../../README.md). Reports under `report/` contain scoped
benchmark runs and orbit-audit counts; those numbers describe different
reduction objects and should not be mixed with the current generic
`basis_contract` counts.

## Operator construction modes

`make_operator(..., mode="distillation")` is the default and preserves the
existing VVV/VDV input and propagator names. `mode="normal"` describes
unsmeared fields. Every operator in one correlator must use the same mode.
This construction setting is independent of the reducer's
`coarse`/`independent`/`detailed` setting and the writer's
`independent`/`check` setting.

```python
from kernels.wick_reducer import make_operator, reduce_operators, write_output

sink = [
    make_operator("sink", "baryon", "p", "sink_p", ("u", "u", "d"),
                  "Cgamma5", mode="normal", momentum=(1, 0, 0)),
    make_operator("sink", "meson", "pi", "sink_pi", ("u", "u"),
                  "gamma5", mode="normal", external_tag="pi_p"),
]
source = [
    make_operator("source", "baryon", "p", "source_p", ("u", "u", "d"),
                  "Cgamma5_dagger", mode="normal", momentum=(1, 0, 0)),
    make_operator("source", "meson", "pi", "source_pi", ("u", "u"),
                  "gamma5", mode="normal", external_tag="pi_p"),
]
report = reduce_operators(sink + source, mode="detailed")
write_output([sink, source], "normal_check.py", name="normal", mode="check")
```

In normal mode, `external_tag` and `momentum` identify each quark or
antiquark endpoint of that operator. Their ordered pair enters the peram
class and output symbol; the epsilon/delta operand carries no p label.
The writer does not add position axes, Fourier phases, or spatial sums.
Other contraction metadata retains its existing role.

Supply the arrays named by the generated document:

| Operand | Meaning and shape |
| --- | --- |
| `epsilon_color` | Real Levi-Civita tensor, `(3, 3, 3)`, with `epsilon_color[0,1,2] = 1` |
| `delta_color` | Real identity matrix, `(3, 3)` |
| `peram_<flavor>_<direction>_x_x` | Unsmeared propagator, `(V_sink, D_sink, V_source, D_source) = (3, 4, 3, 4)` |

Directions include `sin_sin`, `sin_sou`, `sou_sin`, and `sou_sou`. A labeled
propagator appends `_p_<sha256>`; its header declaration records the two
endpoint labels in quark-to-antiquark order. Equivalent list and tuple
momenta produce the same symbol; explicit momentum `0` remains distinct
from an omitted momentum. With the default isospin policy, u/d propagators
share the `peram_u` name; disable `ReductionOptions.isospin_degenerate` and
the writer's `isospin_degenerate` together when supplying different arrays.
Normal propagators always have three color components. The numeric audit
helpers use the actual epsilon and identity tensors in this mode, even
when a distillation eigenvector dimension of six is requested.

## Unified facade

`kernels.wick_reducer.unified` normalizes `input=distillation|normal`, `mode=wick|special`, and `output=independent|check|block|count`. Use `python -m kernels --input-file input.json`; output is grouped under `result/<input>_<mode>_<output>/` and keeps the existing writer schemas.
