# Unified Input/Mode/Output Reference

The public command-line contract has three selectors:

| Selector | Values | Meaning |
| --- | --- | --- |
| `input` | `distillation`, `normal` | Tensor representation and endpoint data model. |
| `mode` | `wick`, `special` | Generic Wick reduction, or normal-color incidence/S3 fusion. `special` accepts `wick_scope=complete` (default) or `cross_time`. |
| `output` | `independent`, `check`, `block`, `count` | Requested result representation, subject to the valid combinations in the root README. |

The selectors may be stored in the input JSON or overridden by the command line:

```sh
python -m kernels \
  --input-file examples/sample_json/pipi.json \
  --input-type normal --mode special --output count
```

Results are written below `result/<input>_<mode>_<output>/`. The result filename starts with the input `name`; `manifest.json` records the canonical selectors and the most recent result file.

For `normal-special`, the two scopes select sibling production kernels:

- `complete` keeps sink and source vertices in the general graph and uses `accumulate_general_fused()`.
- `cross_time` uses the compact cross-time source-coefficient kernel in `cross_time.py`.

Both kernels share the incidence, spin, and global-color S3 fusion primitives. In `check`, the selected kernel is compared with a same-graph reference that disables S3 orbit merging.

See the [Chinese overview](../../README.md#统一-input-mode-output-接口), [English overview](../../README_en.md#unified-interface), and [Quickstart](../../QUICKSTART.md) for the full operator schema and runnable examples.
