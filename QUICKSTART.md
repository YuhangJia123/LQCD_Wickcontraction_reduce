# wick_reduce 快速开始

本指南使用统一入口 `python -m kernels` 完成首次符号结果生成。先选择三个字段，再定义算符：

| 字段 | 含义 | 合法选择 |
| --- | --- | --- |
| `input` | 张量表示 | `distillation` 或 `normal`。 |
| `mode` | 约化路径 | `wick`；或仅用于 normal 的 `special`。 |
| `output` | 结果类型 | wick：`independent`/`check`，normal-wick 另有 `block`；normal-special：`independent`/`check`/`count`。 |

这三项写在 JSON 顶部作为默认配置，也能由命令行覆盖。算符列表只维护一份。各输出的完整含义见 [README 接口表](README.md#统一-input-mode-output-接口)；英文版见 [English Quickstart](QUICKSTART_en.md)。

## 准备环境

推荐 Python 3.11。以下命令适用于 macOS/Linux；从包含 `pyproject.toml`、`kernels/`、`examples/` 的项目根目录开始。将示例路径替换为你的实际项目路径：

```sh
cd /path/to/wick_reduce
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python --version
```

首次符号输出可直接从源码运行，不需要安装第三方库。生成 normal block、执行收缩文件或运行数值检查时，安装项目和数值依赖：

```sh
python -m pip install '.[numeric]'
python -c "import numpy, opt_einsum; print('numeric dependencies OK')"
```

可选：已有 Conda 时，使用仓库提供的环境文件：

```sh
conda env create -f environment.yml
conda activate wick-diss-reduce
python -m pip install .
```

Conda 环境名是 `wick-diss-reduce`。本轮验证了现有 Python 环境中的运行及本地离线包安装；联网创建 Conda 环境和全新下载数值依赖未验证。

## 创建输入并运行

在项目根目录创建 `default.json`。这是一个单味道介子例子，首次选择为 `normal-special-independent`：

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

`operators[0]` 为 sink，`operators[1]` 为 source，两侧都必须非空。介子味道槽按“反夸克、夸克”排列。这个例子只是 `pi0` 的 `u_bar u` 分量，不会自动展开完整物理 `pi0` 的味道叠加态。

运行：

```sh
python -m kernels --input-file default.json
```

终端输出一行 JSON 摘要；重要字段应为：

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

程序自动生成：

```text
result/normal_special_independent/
    pion_normal_special_independent.json
    manifest.json
```

检查结果格式和计数：

```sh
python -c "import json; from pathlib import Path; p=Path('result/normal_special_independent'); d=json.loads((p/'pion_normal_special_independent.json').read_text()); print(d['format']); print(d['summary'])"
```

应显示 `wick-colour-orbit-json-v3`，摘要中的计数与命令行一致。结果的 `records` 包含独立基底记录；`manifest.json` 记录顶层选择和结果文件名。结果 JSON 内部的 `mode=independent` 表示结果表示方式；顶层选择写在 `manifest.json` 中。

## 查看随包结果样例

分发包的 `examples/example_result/` 目录只保留四个轻量案例。目录名始终是 `input_mode_output`，粒子名位于目录中的结果文件名；实际运行结果仍写入根目录 `result/`：

| 粒子案例 | 目录 | 输出文件 |
| --- | --- | --- |
| `pipi` | `result/normal_special_count/` | `pipi_normal_special_count.json` |
| `pip` | `result/distillation_wick_independent/` | `pip_distillation_wick_independent.py` |
| `pn` | `result/normal_wick_check/` | `pn_normal_wick_check.py` |
| `pip-p` | `result/normal_wick_block/` | `pip-p_normal_wick_block.py` |

对应输入在 [`examples/sample_json`](examples/README.md)，配套结果快照在 [`examples/example_result`](examples/README.md)。这些文件只展示输出格式和目录组织，不是完整数值 benchmark。实际运行时，程序会在项目根目录 `result/` 中写入结果；需要保存同一组合下的其他粒子时，保持目录名不变，改变 JSON 的 `name`。

## 用命令行切换模式

`--input-file` 是文件路径；改变输入类型用 `--input-type`。命令行选择项优先于 JSON，省略的选择项沿用 JSON。

例如，同一份 `default.json` 生成 special check：

```sh
python -m kernels --input-file default.json --output check
```

结果进入 `result/normal_special_check/`。JSON 会增加 `raw_wick_relation` 记录，保存原始 Wick 到基底的关系和摘要；这条 CLI 命令不会自动执行融合/direct oracle 的对照。

同一输入的八种合法选择可直接运行：

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

block 命令需要 NumPy。生产 block 会自动从 normal 算符构造 raw pairing family 并验证等价性；ppn 和 ppnn 的验收分别为 48 和 288 个网络。count 输出是计数 JSON，不是 Python `contract` 数；它的 `nlist` 随 `wick_scope` 改变，本例的默认 `complete` 为 288，显式 `cross_time` 时为 144。

`special` 的 `wick_scope` 有 `complete` 和 `cross_time` 两种选择，缺省为 `complete`。`complete` 包含同侧配对；`cross_time` 只保留跨 sink/source 的配对。Wick 输出不读取这个字段。比较不同模式时必须使用相同的配对范围。

用 `--result-root /path/to/run` 指定其他结果根目录。重复运行同名组合会更新结果文件，该组合目录的 manifest 也会更新。

## 换成自己的粒子

在两侧列表中添加或替换算符即可。重子记录例如：

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

这是单条算符的结构示例，需要配上味道与夸克数平衡的 sink；不能独立作为完整输入运行。`external_spin` 在 special 中固定开放槽，Wick writer 的开放轴行为见 README。源端槽顺序不会自动反转，也不会自动取 dagger。

`momentum` 和 `external_tag` 单独描述端点：distillation 把标签放在 VVV/VDV，normal 把标签用于 peram 端点。程序不会自动做动量投影或空间求和。

自定义 Gamma 放入顶层 `gamma_matrices`，例如 `{"G_custom": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}`，并在算符中写 `"gamma": "G_custom"`。必须提供数值 `4 x 4` 矩阵，不能使用 `"gamma1 * gamma5"` 作为计算表达式。

## Python 线性组合输入

需要精确系数或一个算符在乘积中出现多次时，按下面四步构造输入：

```text
原子 Operator
    ↓
OperatorProductTerm()
    ↓
OperatorLinearCombination(terms=(...))
    ↓
一个 sink combination + 一个 source combination
```

这里的“输入”指 Python 输入模块，不是前面的 JSON CLI 格式。第四步直接调用
`reduce_linear_combination(sink_combination, source_combination)`：一次调用只描述一个 sink
线性组合和一个 source 线性组合。若有两个独立 sink（例如 Lambda1p、Lambda2p），就分别构造
sink combination、分别调用约化；它们不会在输入层相加。原子 `Operator` 的 `label` 可以省略，
程序会保留一个内部诊断标签；`flavors` 是算符定义的一部分，必须提供。

### pipi：两个相同介子因子

```python
from kernels.wick_reducer import (
    make_operator, OperatorProductTerm, OperatorLinearCombination,
    reduce_linear_combination,
)

# 1. 原子 Operator
pi_sink = make_operator(
    side="sink", kind="meson", species="pi0",
    flavors=("u", "u"), gamma="gamma5", gamma_slots=(0, 1),
)
pi_source = make_operator(
    side="source", kind="meson", species="pi0",
    flavors=("u", "u"), gamma="gamma5", gamma_slots=(0, 1),
)

# 2. 一个有序乘积项；同一对象出现两次
pipi_sink_term = OperatorProductTerm((pi_sink, pi_sink), "1", "pipi sink")
pipi_source_term = OperatorProductTerm((pi_source, pi_source), "1", "pipi source")

# 3. 两个组合
pipi_sink = OperatorLinearCombination(terms=(pipi_sink_term,))
pipi_source = OperatorLinearCombination(terms=(pipi_source_term,))

# 4. 约化
pipi_result = reduce_linear_combination(pipi_sink, pipi_source, sink_name="pipi")
print(pipi_result["wick_count"], pipi_result["global_basis_count"])
```

### pn：一个乘积项包含两个不同 baryon

```python
p_sink = make_operator(
    side="sink", kind="baryon", species="p",
    flavors=("u", "u", "d"), gamma="Cgamma5",
    gamma_slots=(1, 2), open_slot=0,
)
n_sink = make_operator(
    side="sink", kind="baryon", species="n",
    flavors=("d", "d", "u"), gamma="Cgamma5",
    gamma_slots=(1, 2), open_slot=0,
)
p_source = make_operator(
    side="source", kind="baryon", species="p",
    flavors=("u", "d", "u"), gamma="Cgamma5_dagger",
    gamma_slots=(1, 2), open_slot=0,
)
n_source = make_operator(
    side="source", kind="baryon", species="n",
    flavors=("d", "u", "d"), gamma="Cgamma5_dagger",
    gamma_slots=(1, 2), open_slot=0,
)

pn_sink = OperatorLinearCombination(terms=(
    OperatorProductTerm((p_sink, n_sink), "1", "pn sink"),
))
pn_source = OperatorLinearCombination(terms=(
    OperatorProductTerm((p_source, n_source), "1", "pn source"),
))
pn_result = reduce_linear_combination(pn_sink, pn_source, sink_name="pn")
```

### Sigma-N source 与 Lambda-p sinks

物理 source 为

\[
S=\sqrt{2/3}\,|\Sigma^0_{norm}p\rangle+
\sqrt{1/3}\,|\Sigma^+n_{phys}\rangle.
\]

项目约定 `Sigma0_norm=Sigma0_raw/sqrt(2)`，且 `n_phys=-n_raw`。所以输入 raw 算符时，Sigma0 raw 的两个插值项系数各为 `+sqrt(1/3)`，`Sigma+_raw n_raw` 的系数为 `-sqrt(1/3)`。下面代码不再重复乘 Sigma0 的归一化因子；两个 Lambda sink 各自单独运行。

```python
from kernels.wick_reducer import (
    make_operator, OperatorProductTerm, OperatorLinearCombination,
    reduce_linear_combination,
)

# 1. 逐个构造原子 Operator。source 的场顺序和 Gamma dagger 显式给出。
p_sink = make_operator(
    side="sink", kind="baryon", species="p",
    flavors=("u", "u", "d"), gamma="Cgamma5",
    gamma_slots=(1, 2), open_slot=0,
)
lambda1_sink_operator = make_operator(
    side="sink", kind="baryon", species="Lambda1",
    flavors=("s", "u", "d"), gamma="Cgamma5",
    gamma_slots=(1, 2), open_slot=0,
)
lambda2_dus_sink_operator = make_operator(
    side="sink", kind="baryon", species="Lambda2_dus",
    flavors=("d", "u", "s"), gamma="Cgamma5",
    gamma_slots=(1, 2), open_slot=0,
)
lambda2_uds_sink_operator = make_operator(
    side="sink", kind="baryon", species="Lambda2_uds",
    flavors=("u", "d", "s"), gamma="Cgamma5",
    gamma_slots=(1, 2), open_slot=0,
)

p_source = make_operator(
    side="source", kind="baryon", species="pbar",
    flavors=("u", "d", "u"), gamma="Cgamma5_dagger",
    gamma_slots=(1, 2), open_slot=0,
)
sigma0_dus_source = make_operator(
    side="source", kind="baryon", species="Sigma0_raw_dus",
    flavors=("d", "s", "u"), gamma="Cgamma5_dagger",
    gamma_slots=(1, 2), open_slot=0,
)
sigma0_uds_source = make_operator(
    side="source", kind="baryon", species="Sigma0_raw_uds",
    flavors=("u", "s", "d"), gamma="Cgamma5_dagger",
    gamma_slots=(1, 2), open_slot=0,
)
sigma_plus_source = make_operator(
    side="source", kind="baryon", species="Sigma_plus",
    flavors=("u", "s", "u"), gamma="Cgamma5_dagger",
    gamma_slots=(1, 2), open_slot=0,
)
n_source = make_operator(
    side="source", kind="baryon", species="nbar",
    flavors=("d", "u", "d"), gamma="Cgamma5_dagger",
    gamma_slots=(1, 2), open_slot=0,
)

# 2. Sigma-N source 的三个 raw 项。Sigma0_raw 已展开成前两个项。
sigma0_dus_p_term = OperatorProductTerm(
    operators=(p_source, sigma0_dus_source),
    coefficient="sqrt(1/3)",
    source_label="Sigma0_raw: d(u^T Cgamma5 s) p",
)
sigma0_uds_p_term = OperatorProductTerm(
    (p_source, sigma0_uds_source), "sqrt(1/3)",
    "Sigma0_raw: u(d^T Cgamma5 s) p",
)
sigma_plus_n_term = OperatorProductTerm(
    (n_source, sigma_plus_source), "-sqrt(1/3)",
    "Sigma_plus n_raw (project phase -1)",
)

# 3. 一套 source combination；Lambda1p 和 Lambda2p 分别建立 sink combination。
source = OperatorLinearCombination(terms=(
    sigma0_dus_p_term, sigma0_uds_p_term, sigma_plus_n_term,
))

lambda1_term = OperatorProductTerm(
    operators=(lambda1_sink_operator, p_sink),
    coefficient="1",
    source_label="Lambda1 p",
)
lambda1_sink = OperatorLinearCombination(terms=(lambda1_term,))

lambda2_dus_term = OperatorProductTerm(
    operators=(lambda2_dus_sink_operator, p_sink),
    coefficient="1",
    source_label="Lambda2: + d(u^T Cgamma5 s) p",
)
lambda2_uds_term = OperatorProductTerm(
    (lambda2_uds_sink_operator, p_sink), "-1",
    "Lambda2: - u(d^T Cgamma5 s) p",
)
lambda2_sink = OperatorLinearCombination(terms=(lambda2_dus_term, lambda2_uds_term))

# 4. 每次传入一个 sink combination 和共同的 source combination。
lambda1_result = reduce_linear_combination(lambda1_sink, source, sink_name="Lambda1p")
lambda2_result = reduce_linear_combination(lambda2_sink, source, sink_name="Lambda2p")

print(lambda1_result["wick_count"], lambda1_result["global_basis_count"],
      lambda1_result["exactness"]["all_rows_exact_zero"])
print(lambda2_result["wick_count"], lambda2_result["global_basis_count"],
      lambda2_result["exactness"]["all_rows_exact_zero"])
```

在项目当前约定下，两行应分别显示 `36 18 True` 和 `72 33 True`。source 两项与每个 sink 项先做笛卡尔展开，再批量映射到本次约化的基底；原始 Wick 符号和来源标签保存在结果的 `wick_contributions`。约化器用浮点 Dirac/Gram 关系发现基底关系（固定秩容差 `1e-10`），输入根式与整数关系系数再精确汇总；`all_rows_exact_zero` 表示每个报告基底系数均精确为零，不只是小于数值阈值。

## 常见错误

| 现象 | 处理 |
| --- | --- |
| `No module named kernels` 或 `No module named kernels.wick_reducer` | 回到项目根目录，或在当前 Python 环境安装项目。 |
| 缺少 `numpy` / `opt_einsum` | 在当前解释器运行 `python -m pip install '.[numeric]'`。NumPy 也是 block 生成依赖。 |
| `mode='special' is available only for normal input` | 设置 `--input-type normal`，或改用 `--mode wick`。 |
| `output='block' is available only for normal-wick input` | 同时设置 `--input-type normal --mode wick`。 |
| `output='count' is available only for special mode` | 设置 `--input-type normal --mode special`。 |
| `operators must contain nonempty sink and source lists` | 补齐两侧；扁平列表每条都要有 `side`。 |
| `all operator modes must match top-level input` | 删去算符内冗余的 `mode`，或使其与顶层 `input` 相同。 |
| `Gamma matrix ... is not registered` | 使用已注册名称，或在 `gamma_matrices` 提供矩阵。 |
| `unbalanced quark/antiquark count` / `unbalanced flavor` | 核对每个味道的夸克/反夸克数，及介子有序味道槽。 |

## 验证记录

本次已执行：

- 从文档配置提取的 unified CLI 八种合法组合及输出格式检查。
- `normal-special` 的 `complete` 与 `cross_time` 内核结果和 direct reference 对照。
- `python -B -m unittest discover -s tests -t . -p 'test_*.py'`：153 个测试通过。
- 八例数值核对：维度 3 和 6 全部通过。复现命令如下，输出位置可自行替换。

```sh
python verify_code/check_basis_vs_original.py \
  --without-ppnn --include-eighth --dimensions 3,6 \
  --output-dir result/check \
  --json result/check_summary.json
```

验证时生成文件均放在临时目录，没有改动源码。完整 ppnn 全量枚举、全新联网安装和其他 Python 版本未在本次文档工作中验证；153 个测试通过不能代替这些任务。

更多资料见 [项目总览](README.md)、[统一接口参考](docs/reference/input-mode-output.md)、[工具说明](verify_code/README_zh.md)、[计算内核](docs/kernels/README.md)和[报告目录](report/README_zh.md)。
