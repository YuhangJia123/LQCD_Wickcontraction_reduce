# wick_reduce Quickstart

Use `python -m kernels` to generate your first symbolic result. Choose three selectors before defining the operators:

| Field | Meaning | Valid choices |
| --- | --- | --- |
| `input` | Tensor representation | `distillation` or `normal`. |
| `mode` | Reduction path | `wick`, or normal-only `special`. |
| `output` | Result type | wick: `independent`/`check`, with `block` for normal-wick; normal-special: `independent`/`check`/`count`. |

The JSON stores defaults; command-line selectors override them. Keep one operator list. See the [README interface table](README_en.md#unified-interface) for output meanings, or the [Chinese Quickstart](QUICKSTART.md).

## Prepare the environment

Python 3.11 is recommended. The commands below target macOS/Linux. Start at the project root containing `pyproject.toml`, `kernels/`, and `examples/`; replace the example path with your checkout:

```sh
cd /path/to/wick_reduce
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python --version
```

The first symbolic run works directly from source without third-party libraries. For normal block generation, evaluating contraction files, or numeric checks, install the project with numeric dependencies:

```sh
python -m pip install '.[numeric]'
python -c "import numpy, opt_einsum; print('numeric dependencies OK')"
```

Optional: use the supplied environment when Conda is available:

```sh
conda env create -f environment.yml
conda activate wick-diss-reduce
python -m pip install .
```

The Conda environment is named `wick-diss-reduce`. This pass verified execution in the existing Python environment and an offline local package install. Creating a new Conda environment and downloading numeric dependencies were not verified.

## Create an input and run

Create `default.json` at the project root. This diagonal meson starts with `normal-special-independent`:

```json
{
  "input": "normal",
  "mode": "special",
  "output": "independent",
  "name": "pion",
  "wick_scope": "complete",
  "operators": [
    [
      {
        "side": "sink",
        "kind": "meson",
        "species": "pi0_uubar",
        "label": "sink_pi",
        "flavors": ["u", "u"],
        "gamma": "gamma5"
      }
    ],
    [
      {
        "side": "source",
        "kind": "meson",
        "species": "pi0_uubar",
        "label": "source_pi",
        "flavors": ["u", "u"],
        "gamma": "gamma5"
      }
    ]
  ]
}
```

`operators[0]` is sink and `operators[1]` is source; both must be nonempty. Meson flavors are ordered as antiquark, then quark. This is only the `u_bar u` component of pi0; the program does not expand its physical flavor superposition.

Run:

```sh
python -m kernels --input-file default.json
```

The command prints one JSON summary. Its relevant fields are:

```json
{
  "input": "normal",
  "mode": "special",
  "output": "independent",
  "raw_wicks": 2,
  "independent_wicks": 2,
  "basis_terms": 288
}
```

It creates:

```text
result/normal_special_independent/
    pion_normal_special_independent.json
    manifest.json
```

Inspect the format and counts:

```sh
python -c "import json; from pathlib import Path; p=Path('result/normal_special_independent'); d=json.loads((p/'pion_normal_special_independent.json').read_text()); print(d['format']); print(d['summary'])"
```

Expect `wick-colour-orbit-json-v3` and the same summary counts. Its `records` contain independent basis data. The manifest records the top-level selectors and filename. The result JSON's `mode=independent` describes the result representation; the manifest records the top-level selectors.

## Inspect the bundled result examples

The distribution keeps four small cases under `examples/example_result/`. Directory names always use `input_mode_output`; the particle name appears in the result filename. New runs still write to the repository-root `result/`:

| Particle case | Directory | Output file |
| --- | --- | --- |
| `pipi` | `result/normal_special_count/` | `pipi_normal_special_count.json` |
| `pip` | `result/distillation_wick_independent/` | `pip_distillation_wick_independent.py` |
| `pn` | `result/normal_wick_check/` | `pn_normal_wick_check.py` |
| `pip-p` | `result/normal_wick_block/` | `pip-p_normal_wick_block.py` |

The matching inputs are in [`examples/sample_json`](examples/README.md), with bundled snapshots in [`examples/example_result`](examples/README.md). These files demonstrate output schemas and directory organization rather than a full numerical benchmark. New runs write to the repository-root `result/`. To keep another particle in the same combination directory, keep the directory selectors and change the JSON `name`.

## Change modes from the command line

`--input-file` selects the file; `--input-type` selects the representation. CLI selectors override JSON fields; omitted selectors retain their JSON values.

Generate special check output with the same file:

```sh
python -m kernels --input-file default.json --output check
```

This writes `result/normal_special_check/` and adds `raw_wick_relation` records. The CLI exports relations and digests; it does not automatically run the fused/direct-oracle comparison.

All eight valid selections can use this file:

```sh
python -m kernels --input-file default.json --input-type distillation --mode wick --output independent
python -m kernels --input-file default.json --input-type distillation --mode wick --output check
python -m kernels --input-file default.json --input-type normal --mode wick --output independent
python -m kernels --input-file default.json --input-type normal --mode wick --output check
python -m kernels --input-file default.json --input-type normal --mode wick --output block
python -m kernels --input-file default.json --input-type normal --mode special --output independent
python -m kernels --input-file default.json --input-type normal --mode special --output check
python -m kernels --input-file default.json --input-type normal --mode special --output count
```

Block generation requires NumPy. Production normal-wick-block automatically builds and certifies the raw pairing family; the accepted ppn and ppnn results are 48 and 288 networks. Count writes a summary rather than Python `contract` calls; its `nlist` follows `wick_scope`: this default `complete` example has 288, while explicit `cross_time` has 144.

Special supports `wick_scope=complete` and `cross_time`, with `complete` as the default. `complete` includes same-side contractions; `cross_time` keeps only sink/source-crossing pairings. Wick output does not read this field. Compare modes only at the same pairing scope.

Use `--result-root /path/to/run` for another result root. Repeating a name and combination updates its file and the combination's manifest.

## Define your particles

Replace or add operator records. One source-baryon record can look like:

```json
{
  "side": "source",
  "kind": "baryon",
  "species": "p",
  "label": "source_p",
  "flavors": ["u", "d", "u"],
  "gamma": "Cgamma5_dagger",
  "gamma_slots": [1, 2],
  "open_slot": 0,
  "external_spin": 0,
  "momentum": [0, 0, 0],
  "external_tag": "source_origin"
}
```

This is an illustrative single record, not a standalone correlator: pair it with a balanced sink. Special uses `external_spin` to fix the open slot; see README for Wick open-axis behavior. Source slots are not reversed and Gamma matrices are not daggered automatically.

Keep `momentum` and `external_tag` separate from particle names. Distillation places labels on VVV/VDV; normal uses propagator endpoints. The program does not perform momentum projection or spatial sums.

For custom Gamma matrices, add a top-level `gamma_matrices` object, such as `{"G_custom": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}`, and set the operator's `gamma` to `G_custom`. Use numeric `4 x 4` matrices, not expression strings such as `gamma1 * gamma5`.

## Troubleshoot

| Symptom | Resolution |
| --- | --- |
| `No module named kernels` or `No module named kernels.wick_reducer` | Run from the project root or install the package in the active Python environment. |
| Missing `numpy` or `opt_einsum` | Run `python -m pip install '.[numeric]'` with the active interpreter. NumPy is also required to generate block output. |
| `mode='special' is available only for normal input` | Set `--input-type normal`, or use `--mode wick`. |
| `output='block' is available only for normal-wick input` | Set `--input-type normal --mode wick`. |
| `output='count' is available only for special mode` | Set `--input-type normal --mode special`. |
| `operators must contain nonempty sink and source lists` | Supply both sides; flat records require `side`. |
| `all operator modes must match top-level input` | Remove redundant per-operator `mode` values or match them to `input`. |
| `Gamma matrix ... is not registered` | Use a registered name or supply the matrix in `gamma_matrices`. |
| `unbalanced quark/antiquark count` or `unbalanced flavor` | Check each flavor's quark/antiquark counts and ordered meson slots. |

## Verification

This documentation pass executed:

- All eight CLI combinations extracted from the documented input, with output-format checks.
- The `normal-special` `complete` and `cross_time` kernels against the direct reference.
- `python -B -m unittest discover -s tests -t . -p 'test_*.py'`: 153 tests passed.
- Direct Wick versus reconstructed-basis checks for eight fixtures at dimensions 3 and 6. Reproduce with:

```sh
python verify_code/check_basis_vs_original.py \
  --without-ppnn --include-eighth --dimensions 3,6 \
  --output-dir result/check \
  --json result/check_summary.json
```

Validation output was written to temporary directories. Source code was not changed. Full ppnn enumeration, fresh online installation, and other Python versions were not verified in this documentation pass; the test suite does not replace those checks.

Continue with the [overview](README_en.md), [interface reference](docs/reference/input-mode-output.md), [verify_code](verify_code/README.md), [calculation kernels](docs/kernels/README.md), or [reports](report/README.md).
