# 示例输入与结果

`sample_json/` 保存四个可直接运行的轻量输入；`example_result/` 保存与它们对应的结果快照。示例只用于查看字段、文件名和目录组织，不包含个人数据或大型格点数组。

| 输入文件 | 示例结果目录 | 组合 | 结果文件 |
| --- | --- | --- | --- |
| `sample_json/pipi.json` | `example_result/normal_special_count/` | `normal-special-count` | `pipi_normal_special_count.json` |
| `sample_json/pip.json` | `example_result/distillation_wick_independent/` | `distillation-wick-independent` | `pip_distillation_wick_independent.py` |
| `sample_json/pn.json` | `example_result/normal_wick_check/` | `normal-wick-check` | `pn_normal_wick_check.py` |
| `sample_json/pip-p.json` | `example_result/normal_wick_block/` | `normal-wick-block` | `pip-p_normal_wick_block.py` |

在项目根目录运行单个样例时，将输入文件和命令行选择一起传给统一入口。例如：

```sh
python -m kernels \
  --input-file examples/sample_json/pipi.json \
  --input-type normal --mode special --output count
```

命令会把新结果写入项目根目录 `result/normal_special_count/`，不会覆盖 `examples/example_result/`。附带结果是小型符号格式样例，不包含实际格点 peram 数值，也不是完整性能基准。

结果目录的命名规则固定为 `result/<input>_<mode>_<output>/`；同一目录下可放多个粒子名称的结果文件。`normal-special-count` 通常只写计数和统计字段，`wick` 输出为 Python 表达式，`special-independent/check` 输出为 JSON。
