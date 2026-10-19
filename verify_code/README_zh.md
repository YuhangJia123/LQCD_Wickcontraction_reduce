# 轨道约化与验证工具

**范围说明：** `verify_code/` 中的 `orbit` 脚本属于固定 `ppn`/`ppnn` Wick 夹具的历史/研究审计工具，不是本项目当前的生产 reducer 模块，也不替代 `kernels/wick_reducer/`。本项目的正式复现入口只覆盖 Wick reducer 工作流；下文的轨道构造属于独立的研究审计，不属于主复现流程。

`verify_code/` 还包含收缩和数值验证工具。它与 `kernels/` 分开：导入或运行这些模块不会改变生产 reducer 或其输出格式。

从复制后的项目目录运行下面命令前，先使用根目录的 Conda 环境文件：

```sh
conda env create -f environment.yml
conda activate wick-diss-reduce
python -m pip install .
```

后续命令使用 `python3`，在已激活的 Conda 环境中也可以直接写成 `python`。
英文说明见 [`README.md`](README.md)。

本目录中的部分审计说明会与较早的 reducer 运行结果比较，因此仍保留 698 和 31,092 等历史计数。当前通用 reducer 的计数见项目根目录的 `README.md`。

使用通用收缩 reducer 时，请运行 `run_factor_graph.py`，或使用 `kernels.wick_reducer.reduce_operators()` API。下文的历史轨道脚本仍使用 ppn/ppnn 名称，因为它们审计的是这些已发表的夹具；这些名称不会选择 factor-graph 算法。

约化分为两个可见阶段：

\[
  \Omega_{\rm raw}\;\longrightarrow\;
  \Omega_{\rm raw}/K\;\longrightarrow\;
  (\Omega_{\rm raw}/K)/G .
\]

`K` 消去同一 flavor 内部槽位的排列。`G` 通过声明的质子/中子块置换（以及 ppnn 的同位旋翻转）作用于商状态。这得到 93 个 ppn 状态轨道和 1068 个 ppnn 状态轨道。这些数字是轨道数，不是通用开放 Dirac 张量收缩所需的调用数。

## 快速开始

从项目根目录运行命令。项目根目录是包含 `verify_code/`、`kernels/` 和 `pyproject.toml` 的目录；目录名称不限。在尚未安装为软件包的情况下，从 `-m` 调用模块需要设置 `PYTHONPATH`：

```sh
cd /path/to/project
PYTHONPATH=. python3 -m verify_code.orbit_ppn
PYTHONPATH=. python3 -m verify_code.orbit_ppnn
```

ppnn 构造会枚举 518,400 个 raw 配对，属于计算量较大的审计。命令会输出 raw Wick 数、商状态数、轨道数、轨道大小直方图和 Burnside 计数。

orbit 分析文件用于补充验证和文献口径对照，不是生产入口。normal 输出应使用下文的 `normal-wick-block` writer。

对于通用 reducer 的 check 表示，请使用 `check_basis_vs_original.py`。它运行通用 detailed 路径，通过 `write_case_output(mode="check")` 写出小型夹具的 check 文件，并逐项比较带符号的原始 Wick 值与 `basis_contract -> basis -> origen_wick` 回代结果。Perambulator 数组形状为 `(4, N, N, 4)`；重子 VVV 数组是随机反对称 `(N, N, N)` 张量，介子 VDV 数组形状为 `(N, N)`。开放自旋轴使用共享的随机投影进行测试，因此 ppnn 不需要为 518,400 项中的每一项构造稠密的 `4**8` 张量。

```sh
cd /path/to/project
PYTHONPATH=. python3 verify_code/check_basis_vs_original.py \
  --without-ppnn --dimensions 3,6 \
  --output-dir verify_output/check --json verify_output/check_summary.json
```

默认夹具集合包含七个历史例子。加入 `--include-eighth` 可纳入 `pn_omega_pi+`；默认会纳入 ppnn。只有指定 `--write-ppnn-check` 时，才会写出含 518,400 行的 check 源文件。

对于 ppnn，`ppnn_check_stream.py` 会写出同样的四层数据，但不会在内存中保留全部 518,400 条详细图记录。随后，`ppnn_sample_numeric.py` 会在 C=3 和 C=6 下抽样检查实际生成的映射，使用相同的随机 peram/VVV 输入计算带符号的 raw 直接收缩和 basis 回代，并报告最大残差。完整数值审计请运行 `ppnn_random_end_to_end.py --check`；该脚本会对每个请求的颜色维度计算全部 518,400 个 raw 项，并逐项验证每个生成的映射槽位。每个维度都需要运行数分钟。

```sh
PYTHONPATH=. python3 verify_code/ppnn_check_stream.py \
  --output verify_output/check/ppnn_check.py
PYTHONPATH=. python3 verify_code/ppnn_random_end_to_end.py \
  --check verify_output/check/ppnn_check.py \
  --color-dimension 3 --color-dimension 6
```

归档的 ppn orbit 输出包括：

* `independent` 模式：93 个代表状态块，以及 744 条状态块内 raw contract 赋值；
* `check` 模式：完整的 2,880 项 raw 展开。

## Python API

两个内置系统会在第一次构造后缓存：

```python
from verify_code.orbit_ppn import ppn_orbit_system, summary as ppn_summary
from verify_code.orbit_ppnn import ppnn_orbit_system, summary as ppnn_summary

ppn = ppn_orbit_system()
ppnn = ppnn_orbit_system()
print(ppn_summary())
print(ppnn_summary())
```

`OrbitSystem` 提供重现映射所需的审计数据：

```python
ppnn.raw_pairings          # raw Wick id -> 夸克/反夸克配对
ppnn.raw_signs             # raw Wick id -> Wick 符号
ppnn.raw_to_state           # raw Wick id -> 商状态
ppnn.state_to_raw           # 商状态 -> 对应的 raw Wick 集合
ppnn.state_orbits           # 1068 个状态轨道的 OrbitRecord 对象
ppnn.raw_orbits             # 扩展 raw 作用的 OrbitRecord 对象
ppnn.state_to_orbit         # 商状态 -> 轨道 id
ppnn.raw_to_orbit            # raw Wick id -> raw 轨道 id
```

可用的检查包括：

```python
ppnn.validate()                 # 分区与覆盖不变量
ppnn.check_quotient_equivariance()  # q(g.w) == g.q(w)
ppnn.state_fixed_point_histogram()
ppnn.raw_fixed_point_histogram()
```

`verify_code.orbit_common` 包含可复用组件：`OrbitRecord`、`OrbitPartition`、`OrbitSystem`、`enumerate_orbits`、`fixed_point_histogram`、`burnside_count` 和 `action_from_generators`。

## Orbit 验证输出

`ppn_orbit_independent.py` 和 `ppn_orbit_check.py` 提供带符号 raw 收缩及轨道映射，用于补充群作用与数值验证。生产 normal 输出使用 normal reducer 的符号输出和 `block_reduce`。

## 算符输入

`verify_code.operator_input` 会规范化项目现有的算符表示：

```python
from verify_code.operator_input import load_operators, enumerate_raw_wicks

flat = load_operators([sink_operator, source_operator])
pair = load_operators([[sink_operator], [source_operator]])
operators, fields, quarks, antiquarks, terms = enumerate_raw_wicks(pair)
```

它也接受 JSON 文件路径或映射。JSON 对象可以使用算符库名称：

```json
{"name": "ppn"}
```

也可以显式列出算符：

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

Flavor 顺序、Gamma 槽位、开放槽位和 `external_tag` 均会保留。该规范化器会枚举 raw Wick 项，但不会为任意新算符列表推断对称群。`normal-wick-block` 由生产入口自动构造 raw pairing family，再与 normal 表进行精确证书核对。

## normal-wick-block 验证

直接从 normal 算符调用生产入口；不需要事先提供 pairing family：

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

`block` 只代表 `normal-wick-block`。当前 W 约化作用于完整 raw pairing family，保留输出转置，并以最终加权表达式的证书验证等价性。ppn 验收为 48 个网络，ppnn 验收为 288 个网络。接口与证书说明见[Block 文档](../docs/reference/block-api.md)。

## 验证与限制

单元测试覆盖 raw/state/orbit 覆盖范围、Burnside 计数、等变性、输出语法、自动 normal-wick-block 规划与数值检查：

```sh
python3 -B -m unittest discover -s tests -t . -p 'test_*.py'
```

`verify_code.numeric_validation.validate_seeds` 至少需要三个随机种子：

```python
from verify_code.numeric_validation import validate_seeds
records = validate_seeds(ppnn_orbit_system(), (20261009, 20261010, 20261011))
assert max(item.max_relative_residual for item in records) < 1e-12
```

审计使用形状正确的复数 peram `(3, 4, 3, 4)` 和反对称 `(3, 3, 3)` VVV 张量。对每个随机种子，完整 raw 和与轨道重构通过共享输入因子使用同一组 peram/VVV 样本，同时向轨道纤维分配满足对称性的随机带符号贡献。该审计验证声明的作用下，raw 和、显式块和与加权块之间的精确恒等关系；它不声称无约束的随机 peram 会自动满足该对称性。

当前范围有意保持狭窄：

* 内置轨道作用只针对 ppn 和 ppnn 实现；
* 代码不会自动发现全部 Gamma、动量、transporter、开放 Dirac 轴、投影算符或 VVV 对称性；
* 93 和 1068 是状态块轨道数，不能替代通用 reducer 的开放 Dirac basis（当前独立项目中 ppn 为 219、ppnn 为 4,298；较早报告可能记录先前算法的数字）；
* 性能比较必须测量相同的数值 contract 工作量。只枚举轨道不能证明最终收缩结果运行更快或占用更少内存。

数学推导见[群轨道理论](group_orbit_theory.md)，当前审计数字和检查点历史见[实现报告](implementation_report.md)。

如需测量时间/RSS，可运行：

```sh
python3 verify_code/benchmark_orbit_vs_reducer.py \
  --contract-sample 8 \
  --output verify_code/benchmark_orbit_vs_reducer.json
```

JSON 会分别列出实测轨道构造、实测生产版运行，以及根据 `opt_einsum.contract` 样本外推的结果。外推的融合时间是有条件的估算；它不代表通用张量 contract 可以因子分解。
