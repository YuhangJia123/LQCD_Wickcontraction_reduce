# wick_reduce

This standalone Python project enumerates and reduces Wick contractions for custom meson and three-quark baryon operator lists. It combines a generic factor-graph Wick reducer, a normal-color incidence + global-color S3 fusion kernel, and the `normal-wick-block` W-fusion path.

The primary entry point is `python -m kernels`. Choose `input`, `mode`, and `output` before defining the operators. Keep the operators in one JSON file and override the selectors from the command line.

Start with the [English Quickstart](QUICKSTART_en.md). The primary Chinese pages are [README](README.md) and [Quickstart](QUICKSTART.md).

## Unified interface

| Field | Meaning | Values |
| --- | --- | --- |
| `input` | Tensor representation | `distillation` uses VVV/VDV elementals and perams; `normal` uses three-color propagators and epsilon/delta tensors. |
| `mode` | Reduction path | `wick` uses the generic graph and Dirac/source reducer; `special` is the normal-color path, selecting a cross-time or complete graph kernel through `wick_scope`. |
| `output` | Requested result | `independent`, `check`, `block`, or `count`, with the restrictions below. |

There are eight valid combinations:

| Input | Mode | Output | Result |
| --- | --- | --- | --- |
| `distillation` | `wick` | `independent` | Python representative contractions, signed/transposed basis views, coefficients, and final sum. |
| `distillation` | `wick` | `check` | Those tables plus raw Wick contractions, signs, and reconstruction mappings. |
| `normal` | `wick` | `independent` | Python contractions retaining normal color tensors and open spin axes. |
| `normal` | `wick` | `check` | Normal independent tables and per-Wick reconstruction data. |
| `normal` | `wick` | `block` | Automatically builds and certifies the raw Wick pairing family before W fusion; the current ppn acceptance is 48 networks and ppnn is 288. |
| `normal` | `special` | `independent` | JSON v3 independent Wick vectors and explicit propagator monomials. |
| `normal` | `special` | `check` | JSON basis records plus raw-Wick linear relations and vector digests. |
| `normal` | `special` | `count` | SQLite-merged nonzero coordinate count `nlist` and statistics, without the full basis or coefficient dictionary. |

The code-level default is `distillation-wick-independent`; `wick_scope` defaults to `complete`, and the repository's `default.json` demonstrates `normal-special-independent`. The accepted aliases `distrilation` and `indenpent` normalize to `distillation` and `independent`.

### File output and special kernels

The CLI calls `write_unified`. Wick uses its Python writer; special independent/check uses the original color-orbit JSON v3 writer, whose `records` array contains `basis` and optional `raw_wick_relation` records.

The special in-memory calculation has two sibling kernels: `cross_time` uses the compact source-coefficient fusion, while `complete` uses the general graph fusion with both sink and source vertices. Both share the incidence/spin/S3 core. Check mode disables S3 orbit merging in the same graph construction and compares the result term by term. CLI file output still follows the JSON schemas in the interface table.

## Operators and input representation

Use `operators = [sink_list, source_list]`. Both lists must be nonempty. Nested records can omit `side`; flat records require `side=sink|source`. All records must use the top-level input representation. A per-record `mode`, if present, must match `input`.

| Operator field | Meaning |
| --- | --- |
| `kind` | `meson` or `baryon`. |
| `species`, `label` | Particle and instance labels, not specialized algorithm switches. |
| `flavors` | Three ordered quark flavors for baryons; `[antiquark, quark]` for mesons. One flavor is shorthand for a diagonal meson. |
| `gamma`, `gamma_slots` | Matrix name and its ordered two slots. Specify source daggering and slot order explicitly. |
| `open_slot` | Baryon open spin slot, default 0; the default Gamma slots are the remaining two. |
| `external_spin` | In special, fixes the open baryon slot to 0 through 3. When omitted, that implementation enumerates and sums the slot; this differs from wick's open spin tensor. |
| `external_tag`, `momentum` | Separate endpoint/momentum labels, placed according to `input`. |

The full runnable JSON is maintained in [Quickstart](QUICKSTART_en.md#create-an-input-and-run).

### Endpoint and momentum labels

| Input | Color/spatial factor | Propagator representation |
| --- | --- | --- |
| `distillation` | VVV for baryons and VDV for mesons; conjugated source elementals. | Axes `(V_q,D_q,V_bar,D_bar)`, usually `(N_v,4,N_v,4)`. Tags and metadata distinguish the elementals. |
| `normal` | `epsilon_color` and `delta_color`, color dimension 3. | Shape `(3,4,3,4)`. Wick places `external_tag` and `momentum` on peram endpoints. Special preserves those labels and other metadata except `external_spin` in endpoint `point` labels. |

These labels do not add coordinate axes, Fourier transforms, or spatial sums. The unified JSON has no separate `p` or `x` field: use `momentum` and `external_tag`. Metadata such as `projector`, `clebsch`, and `phase` is handled differently by the paths and does not automatically construct numeric tensors or expand physical-state superpositions.

## Environment and installation

Python 3.11 is recommended and was used for this documentation validation. Package metadata declares Python >= 3.9. The symbolic Wick/special paths run directly from the source root using the standard library. Normal block generation needs NumPy; evaluating generated Python and running numeric audits needs NumPy and opt_einsum.

| Use | Setup |
| --- | --- |
| Generate symbolic output from source | Follow [Quickstart](QUICKSTART_en.md). |
| Install the package with numeric dependencies | Run `python -m pip install '.[numeric]'` from the repository root in your virtual environment. |
| Use Conda | `environment.yml` supplies Python 3.11, NumPy, opt_einsum, and build tools; the environment name remains `wick-diss-reduce`. |

See [environment setup](QUICKSTART_en.md#prepare-the-environment) for complete commands and validation limits.

## Results

The unified writer uses:

```text
result/<input>_<mode>_<output>/
    <name>_<input>_<mode>_<output>.py     # wick
    <name>_<input>_<mode>_<output>.json   # special
    manifest.json
```

Only the applicable result extension is created. A combination directory can contain results for several particle `name` values; the filename starts with that particle name. Repeating the same name and combination replaces that result. A combination's `manifest.json` describes the most recent write, not every result stored in the directory. Use `--result-root PATH` to change the root.

The distribution includes four small format examples under `examples/example_result/`. The first level remains strictly grouped by `input_mode_output`:

```text
examples/example_result/
├── normal_special_count/pipi_normal_special_count.json
├── distillation_wick_independent/pip_distillation_wick_independent.py
├── normal_wick_check/pn_normal_wick_check.py
└── normal_wick_block/pip-p_normal_wick_block.py
```

The four small result snapshots are in [`examples/example_result`](examples/README.md), with matching inputs in [`examples/sample_json`](examples/README.md). They demonstrate file organization and schemas without lattice peram data. New runs still write to the repository-root `result/`; `example_result/` is a read-only bundled snapshot.

Wick `basis_contract` evaluates representatives; `basis` adds signs and open-axis transposes; `coefficients` weights the views; `result` sums them. To evaluate the generated Python, supply the Gamma, peram, VVV/VDV, or epsilon/delta arrays listed in its header. This project does not load or generate lattice data.

`result/` is the only result location. Validation scripts use the system temporary directory or `verify_output/` for temporary files.

## Algorithms and limits

Wick enumerates exact-flavor pairings, groups directed perams and factor graphs, retains valid transporters, and reduces sink/source Dirac relations. The special kernels combine incidence/connectivity classification with the shared global-color S3 fusion core. Count merges nonzero coordinates through temporary SQLite storage. Normal-wick-block retains the complete raw pairing family, certifies its equality to the normal tables, and then searches reusable antisymmetric two-propagator structures.

Current limits that affect use:

- `wick_scope=complete|cross_time`, `normal_ordered`, and `tolerance` are special settings; the default is `complete`. The Wick writer does not use `wick_scope`; set `cross_time` explicitly when that scope is intended.
- `isospin_degenerate_perams=true` assumes numerically equal u/d propagators after exact-flavor enumeration. Set it to `false` for different arrays.
- The CLI file writer does not use `--max-entries` as a memory bound; the special kernels forward it to their accumulators.
- Custom Gamma matrices must be numeric `4 x 4` values. An operator supports one two-slot Gamma, not arbitrary multiport spin tensors or matrix-product strings.
- Custom patterns are supported within the meson/baryon model and available computational resources; pairing counts can grow rapidly.

This documentation pass completed 153 project tests and direct-Wick versus reconstructed-basis checks for eight distillation fixtures at dimensions 3 and 6. See [verification](QUICKSTART_en.md#verification) for commands and unverified steps.

Wick `basis_contract` counts, special JSON `independent_wicks`, fused/count `nlist`, and literature state-block counts describe different objects. Refer to the [final research report](report/final/wick_diss_reduce_algorithm_report.md) for scoped 3H/4H comparisons.

## Documentation and layout

| Path | Responsibility |
| --- | --- |
| `kernels/wick_reducer/` | Generic reducer, operator model, unified entry point, and Python writer. |
| `kernels/wick_colour_orbit/` | Incidence/S3 fusion kernel, exact ordinary-color backend, and JSON v3 writer. |
| `kernels/block_reduction/` | Normal block planning and exact reconstruction. |
| `kernels/correlator_wick_normalizer.py` | Wick enumeration and Dirac normalization primitives. |
| `verify_code/` | Numeric validation, audits, benchmarks, and adapters. |
| `tests/` | Unit and regression tests. |
| `report/` | Algorithms, research reports, and experiment data. |

`verify_code/` is limited to correctness comparisons, numeric validation, performance benchmarks, and ppn/ppnn audit records. Production entry points do not depend on this directory; `python -m kernels` and unified `result/` output run independently. The normal block output adapter is part of the production interface at `kernels/block_reduction/output_adapter.py`.

- [English Quickstart](QUICKSTART_en.md): first result, mode changes, and troubleshooting.
- [Legacy Python API](docs/reference/wick-api_en.md): each API's `mode` parameter has its own meaning; it is not necessarily the unified top-level mode.
- [Wick output format](docs/reference/wick-output_en.md) and [algorithm](docs/implementation/wick_en.md).
- [Block interface](docs/reference/block-api.md) and [verify_code](verify_code/README.md).
- [Documentation structure](docs/README.md), [calculation kernels](docs/kernels/README.md), [reports](report/README.md), and [eight-fixture and ppnn data](report/data/benchmark_20261014/README.md).
