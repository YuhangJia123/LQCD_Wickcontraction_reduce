# wick_diss_reduce：蒸馏关联函数 Wick 约化

本文件是 `kernels/wick_reducer/` 包内的中文使用说明。项目根目录的
[`README.md`](../../README.md) 是当前目录结构、安装方式和维护中计数的唯一入口；
本文件保留较完整的 API 和数学说明。

`wick_diss_reduce` 从 sink/source 算符定义出发枚举 Wick 收缩，按 peram、sink 和 Dirac/source 指标关系约化收缩图，并导出可供后续数值计算使用的 Python 文件。

这份指南从手工构造 `pn`、`delta` 和 `pip` 三个关联函数开始，依次说明如何运行、读取输出和运行测试。示例不调用内置算符库。

## 通用因子图约化

真正的约化入口是 `reduce_operators(operators, ...)`，因此不需要把输入命名成
`ppn` 或 `ppnn`。精确的 p/n、u/d 味道只在生成合法 raw Wick 配对时使用；之后的
key 只由 peram 的方向和端点、VVV/VDV 类型、Gamma 及槽位、动量/`external_tag`、
开放轴和连接图决定。

```python
from kernels.wick_reducer import ReductionOptions, reduce_operators

report = reduce_operators(
    arbitrary_operators,
    label="my_input",                 # 仅是输出标签
    options=ReductionOptions(),
)
```

这个通用模式不把介子或重子的 species 名字放进约化 key；真正参与比较的是
peram 边、VVV/VDV、Gamma、开放槽和 `external_tag`。调用方必须保证这些因子就是
要比较的数值对象。项目不再提供按 species 分桶的第二条约化路径。

## 目录

- [运行前准备](#运行前准备)
- [构造并导出三个例子](#构造并导出三个例子)
- [读取生成文件](#读取生成文件)
- [自定义输入规则](#自定义输入规则)
- [测试与更多文档](#测试与更多文档)

## 运行前准备

以下命令从项目根目录运行，也就是同时包含 `kernels/`、`tests/`、
`kernels/correlator_wick_normalizer.py` 和 `pyproject.toml` 的目录。项目可以直接从源码运行，
也可以安装到当前虚拟环境：

先确认 Python 可用：

```sh
python3 --version
```

生成 Wick 约化文件只使用项目源码和 Python 标准库。生成文件中的 `contract(...)` 要执行数值收缩时，需安装可选依赖：

```sh
python3 -m pip install '.[numeric]'
```

从复制后的项目目录开始，也可以使用根目录提供的 Conda 环境文件：

```sh
conda env create -f environment.yml
conda activate wick-diss-reduce
python -m pip install .
```

如果使用虚拟环境，可以先在仓库根目录运行：

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。

## 构造并导出三个例子

将下面的完整程序保存为项目根目录的 `run_correlator_examples.py`，并从项目根目录运行 `python3 run_correlator_examples.py`。它逐个手工构造 sink/source 算符，并为每个关联函数生成 `independent` 和 `check` 两种输出。`write_output` 会自动建立输出目录。

```python
from pathlib import Path

from kernels.wick_reducer import make_operator, write_output


OUTPUT_DIR = Path("verify_output/readme_examples")


def export_both_modes(name, sink, source):
    for mode in ("independent", "check"):
        output_path = OUTPUT_DIR / f"{name}_{mode}.py"
        summary = write_output(
            [sink, source], output_path,
            name=name,
            mode=mode,
        )
        print(
            f"{name:5s} {mode:11s}: "
            f"{summary['wick_count']} Wick, "
            f"{summary['independent_count']} independent, "
            f"{summary['dependent_count']} dependent -> {output_path}"
        )


# 1. pn = p + n
# p sink: VVV[u v w] u[u] (C gamma5)[a b] u[v] d[w]
# n sink: VVV[x y z] d[x] (C gamma5)[c d] d[y] u[z]
pn_sink = [
    make_operator(
        "sink", "baryon", "p", "sink_p", ("u", "u", "d"),
        "Cgamma5", (1, 2), open_slot=0,
    ),
    make_operator(
        "sink", "baryon", "n", "sink_n", ("d", "d", "u"),
        "Cgamma5", (1, 2), open_slot=0,
    ),
]
pn_source = [
    # Source slots are explicit. They are not automatically reversed.
    make_operator(
        "source", "baryon", "p", "source_pbar", ("u", "d", "u"),
        "Cgamma5_dagger", (1, 2), open_slot=0,
    ),
    make_operator(
        "source", "baryon", "n", "source_nbar", ("d", "u", "d"),
        "Cgamma5_dagger", (1, 2), open_slot=0,
    ),
]


# 2. delta = Delta++ sink and source
delta_sink = [
    make_operator(
        "sink", "baryon", "Delta++", "sink_Delta++", ("u", "u", "u"),
        "Cgamma3", (1, 2), open_slot=0,
    ),
]
delta_source = [
    make_operator(
        "source", "baryon", "Delta++", "source_Delta++", ("u", "u", "u"),
        "Cgamma3_dagger", (1, 2), open_slot=0,
    ),
]


# 3. pip = pi0 (u-bar u component) + p
pip_sink = [
    make_operator(
        "sink", "meson", "pi0_uubar", "sink_pi0", ("u", "u"),
        "gamma5", (0, 1),
    ),
    make_operator(
        "sink", "baryon", "p", "sink_p", ("u", "u", "d"),
        "Cgamma5", (1, 2), open_slot=0,
    ),
]
pip_source = [
    make_operator(
        "source", "meson", "pi0_uubar", "source_pi0", ("u", "u"),
        "gamma5", (0, 1),
    ),
    make_operator(
        "source", "baryon", "p", "source_pbar", ("u", "d", "u"),
        "Cgamma5_dagger", (1, 2), open_slot=0,
    ),
]


for example_name, sink_ops, source_ops in (
    ("pn", pn_sink, pn_source),
    ("delta", delta_sink, delta_source),
    ("pip", pip_sink, pip_source),
):
    export_both_modes(example_name, sink_ops, source_ops)
```

从仓库根目录运行：

```sh
python3 run_correlator_examples.py
```

程序会在 `verify_output/readme_examples/` 下生成六个文件：

```text
pn_independent.py     pn_check.py
delta_independent.py  delta_check.py
pip_independent.py    pip_check.py
```

按这组明确给定的槽位和矩阵，摘要应显示：

| 关联函数 | 原始 Wick 图 | 独立图 | 依赖图 |
|---|---:|---:|---:|
| `pn` | 36 | 35 | 1 |
| `delta` (`Delta++`) | 6 | 2 | 4 |
| `pip` (`pi0_uubar + p`) | 24 | 24 | 0 |

改变 source 槽位、Gamma 或算符组成后，约化结果也可能改变。这里的 `pip` 使用 `pi0` 的 `u_bar u` 分量；物理 `pi0` 的完整同位旋线性组合需要分别输入各 flavor 分量并在外部组合，程序不会自动展开 flavor 叠加态。

## 读取生成文件

两个模式导出用途不同。

### `independent`

- `<name>_coefficients[i]` 是第 `i` 个基底的约化系数，系数数组只输出一次。
- `<name>_basis_contract[wick_id]` 对每个独立代表调用一次 `contract(...)`。`wick_id` 从 1 起。
- `<name>_basis[basis_id]` 在收缩结果上加入 Wick 符号，并按需执行 `.transpose(...)`。`basis_id` 从 0 起。
- `<name>_result` 将非零系数乘以对应基底并求和。

同一个独立收缩的不同裸轴排列复用 `basis_contract`，只在 `basis` 中创建转置视图。每一条生成的 `contract(...)` 都包含完整指标，并保持在一行。

### `check`

- `<name>_origen_wick_raw[slot]` 保存每个原始 Wick 收缩，`slot` 从 0 起。
- `<name>_origen_wick_sign[slot]` 保存对应的 Wick 符号；原始 Wick 编号为 `slot + 1`。
- `<name>_origen_wick[slot]` 直接写出由基底恢复该原始图的表达式，例如 `pn_origen_wick[30] = 1 * pn_basis[1]`。
- `<name>_origen_wick_to_independent` 是映射字典，不是函数。每个 slot 对应的 `terms` 记录 basis 下标、系数、代表 Wick 编号和裸轴转置。一个原始图可能由多个 basis 项组合得到。

逐图验证时比较：

```python
name_origen_wick_sign[slot] * name_origen_wick_raw[slot]
```

与相同 slot 的 `name_origen_wick[slot]`。例如 pn 中 Wick 31 位于 slot 30，由 Wick 2 对应的基底视图重构，系数为 1，转置为恒等置换。

### 实际数值收缩

生成的文件包含 Python `contract(...)` 调用，但 reducer 不会读取格点数据，也不会自动提供 Gamma、peram、VVV 或 VDV 数组。执行输出前，需要先定义生成文件头部列出的所有数组变量，再在同一命名空间中执行该文件。例如：

```python
from pathlib import Path

# 先在这里定义 C_g5、C_g5_dagger、peram_u_sin_sou、VVV_sink 等实际数组。
namespace = globals()
source = Path("verify_output/readme_examples/pn_independent.py").read_text()
exec(compile(source, "pn_independent.py", "exec"), namespace)

pn_value = namespace["pn_result"]
```

`opt_einsum` 可使用 NumPy、CuPy 等兼容数组后端。输出文件里的变量名以文件开头的约定为准；自定义动量标签会产生不同的 `VVV`/`VDV` 名称。

## 自定义输入规则

`make_operator` 的参数按以下方式使用：

| 参数 | 含义 |
|---|---|
| `side` | `"sink"` 或 `"source"` |
| `kind` | `"baryon"` 或 `"meson"` |
| `species` | 算符类别标签，例如 `"p"`、`"Delta++"`、`"pi0_uubar"` |
| `label` | 本次关联函数中该算符的实例名 |
| `flavors` | 重子为 3 个有序味道槽；介子为 `(antiquark, quark)` |
| `gamma` | 已注册的 Gamma 名称，或 `gamma_matrices` 中提供的矩阵名 |
| `gamma_slots` | Gamma 矩阵连接的两个槽位，顺序决定矩阵指标顺序 |
| `open_slot` | 重子的开放裸 spin 槽；默认是槽 0。介子没有开放槽 |
| `external_tag` | 动量/elemental 标签；省略表示共享默认零动量张量 |

source 的味道顺序和 Gamma 名称由调用者明确输入，构造器不会自动生成共轭算符，也不会自动反排 source 槽位。`gamma_slots` 和 source 的矩阵是否需要 dagger 同样由输入决定。

默认同位旋模式开启：Wick 配对仍按精确 flavor 枚举，后续 peram 等价类会把 `u` 和 `d` 视为同一类。其他 flavor 仍各自分类。关闭同位旋模式可调用 `write_output(..., isospin_degenerate=False)`。

自定义数值 Gamma 矩阵通过 `write_output(..., gamma_matrices={"G": matrix})` 传入；矩阵必须为数值 `4×4`。当前算符结构支持每个算符一个作用于两个 spin 槽位的 Gamma 矩阵，不表示任意多端口 spin tensor。einsum 标签最多支持 52 个。

## 测试与更多文档

从仓库根目录运行此子项目的测试：

```sh
python3 -B -m unittest discover -s tests -t . -p 'test_*.py'
```

其他文档：

- [算法说明](../implementation/wick.md)：Wick 图、peram、sink/source 与 Dirac 约化流程。
- [输出文件格式](wick-output.md)：导出变量命名、完整指标表达式和 check 映射结构。
- [English README](wick-api_en.md)：项目摘要和文档入口。

## 历史单次结果与耗时

以下数据是补充测量记录，用于说明通用 factor-graph
reducer 的 `basis_contract` 计数。当前计数请以项目根目录 `README.md` 为准。

下表记录一次可复现的运行结果；ppn 等价性报告提供对应的比较数据。`ppnn` 一行沿用此前完整运行的数据，本轮没有重算。

| 例子 | Wick 图 | 独立图 | 依赖图 | Coarse (s) | Independent (s) | Detailed (s) | 峰值内存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pipi` | 24 | 10 | 14 | 0.000461 | 0.004961 | 0.005520 | — |
| `pn` | 36 | 35 | 1 | 0.000804 | 0.005694 | 0.006091 | — |
| `ppn` | 2,880 | 698 | 2,182 | 0.077593 | 0.916977 | 1.081582 | — |
| `delta` | 6 | 2 | 4 | 0.002257 | 0.001245 | 0.001356 | — |
| `delta++delta-` | 36 | 4 | 32 | 0.000690 | 0.009134 | 0.009464 | — |
| `pip` | 24 | 24 | 0 | 0.000467 | 0.003734 | 0.004127 | — |
| `pip-p` | 6 | 6 | 0 | 0.000152 | 0.000931 | 0.000966 | — |
| `ppnn`（此前实测） | 518,400 | 31,092 | 487,308 | 25.3 | 539.4 | 未记录 | 1.21 GiB |

本轮七例的环境为 Python 3.9.6、macOS 14.8.4 arm64。每种模式只计时一次，尤其是毫秒以下的数值会受系统负载影响，适合作为本机参考，不代表稳定基准。历史 `ppnn` 数据来自此前完整运行：coarse 枚举约 25.3 秒，independent 完整约化约 539.4 秒，峰值常驻内存约 1.21 GiB；当时没有记录 detailed 模式耗时。

该表和历史 `ppnn` 行不是当前 factor-graph 优化版的性能结果。当前版与原始提交的同机 ppnn 基准、八例子计时和全量随机数值审计见[优化基准报告](../../report/benchmarks/optimization_benchmark.md)。

## ppn 等价性补充记录

以下内容与项目报告 [`ppn等价性初步报告.md`](../../report/historical/ppn等价性初步报告.md) 的记录相互参照；当前 reducer 计数以项目根目录 `README.md` 为准。每个例子当时连续运行三次，时间列取中位数；图数依次为原始 Wick 图、独立图和依赖图。

| 例子 | 原始 Wick 图 | 独立图 | 依赖图 | Coarse（秒） | Independent（秒） | Detailed（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pipi` | 24 | 10 | 14 | 0.000692 | 0.008699 | 0.007496 |
| `pn` | 36 | 35 | 1 | 0.001091 | 0.009943 | 0.010113 |
| `ppn` | 2,880 | 698 | 2,182 | 0.129983 | 1.499670 | 1.668814 |
| `delta++delta-` | 36 | 4 | 32 | 0.001281 | 0.014309 | 0.016833 |
| `delta` | 6 | 2 | 4 | 0.002113 | 0.001830 | 0.001949 |
| `pip` | 24 | 24 | 0 | 0.000833 | 0.005617 | 0.006429 |
| `pip-p` | 6 | 6 | 0 | 0.000276 | 0.001676 | 0.001827 |

### ppn 原始 Wick 与 basis 的等式

历史 ppn check 文件中有 2,880 个原始 Wick 收缩、698 个 basis 收缩，以及 2,760 个包含裸指标轴转置的 basis 视图。当前项目的通用 check 文件由 `verify_code/check_basis_vs_original.py` 生成；两者属于不同版本的输出记录，不能混用计数。

逐项检查的数学关系是：

```text
origen_wick_sign[w] * origen_wick[w]
    = relation_coefficient[w] * ppn_basis[basis_index]
```

结构和系数检查得到的具体数据如下：

| 检查项目 | 数据 | 结果 |
| --- | ---: | --- |
| 原始 Wick 收缩 | 2,880 | 全部枚举完成 |
| basis 收缩 | 698 | 每个只计算一次 |
| basis 视图（包括裸指标转置） | 2,760 | 全部生成 |
| 原始 Wick 到 basis 的映射 | 2,880 | 每个原始图对应一个 basis 项 |
| 符号关系 | 2,880 / 2,880 | 全部一致 |
| 映射系数汇总与 `ppn_coefficients` | 2,760 / 2,760 | 完全一致 |

完整数值回放的输入和误差如下：

| VVV 张量 | 随机种子 | peram 输入布局 | 比较图数 | 最大绝对误差 | 最大相对误差 |
| --- | ---: | --- | ---: | ---: | ---: |
| 三维完全反对称 epsilon | 20261006 | `(3,4,3,4)` -> `(4,3,3,4)` | 2,880 / 2,880 | `7.357e-11` | `5.170e-16` |
| 三维完全反对称 epsilon | 20261007 | `(3,4,3,4)` -> `(4,3,3,4)` | 2,880 / 2,880 | `6.000e-11` | `6.159e-16` |
| 三维完全反对称 epsilon | 20261008 | `(3,4,3,4)` -> `(4,3,3,4)` | 2,880 / 2,880 | `1.037e-10` | `9.693e-16` |
| 六维随机完全反对称张量 | 20261009 | `(6,4,6,4)` -> `(4,6,6,4)` | 2,880 / 2,880 | `6.032e-06` | `1.196e-15` |

上表记录数值审计的输入/方程轴转换；当前 normal 输出统一使用 `(V_sink,D_sink,V_source,D_source)=(3,4,3,4)`，不再要求调用方预先转置。

使用三维完全反对称 epsilon VVV 张量和随机复 peram 的三组测试，以及六维随机完全反对称 VVV 张量的一组测试，均完成 2,880 / 2,880 项比较。报告中最大相对误差分别为 `6.159e-16`、`9.693e-16` 和 `1.196e-15`，处于浮点舍入误差量级。normal peram 按统一轴顺序 `(V_sink, D_sink, V_source, D_source)=(3,4,3,4)` 输入；distillation peram 也按统一的 `(V_sink,D_sink,V_source,D_source)` 轴顺序输入。

### 代码一致性审计

当时的目录清理没有引入新的算法差异：`multi_solution/multi_solution.py` 移入 `kernels/wick_reducer/reducer.py` 时 Git 相似度为 99%，实际使用的 incidence helper 抽取到 `kernels/wick_reducer/peram_incidence.py`。当前测试状态以项目根目录中的测试套件为准。

## 未来的修正方向

- **更一般的 sink Dirac 多 transporter 判断。** 当前例子主要使用标准的 `C_g` 类 Gamma 矩阵。这类矩阵通常具有良好的正交性，以及一致的对称性或反对称性，因此在当前问题中，检查具体张量是否相等或成比例通常已经足够。未来若要支持更一般的具体矩阵，应始终保留有序的 sink Dirac 张量，并把 Gamma 转制、指标置换、内积、projective residual 和秩判断都纳入流程，不能只依赖 Gamma 名称或 `ga == gb`。

  **修正算法：**

  1. 先按 peram profile 和 incidence key 分桶，并枚举当前桶中所有允许的 transporter。
  2. 对每个 transporter 建立有序的 sink Dirac key，包含 sink line order、Gamma 槽位、具体 Gamma 矩阵、VVV 置换和开放 sink 轴；第一次遇到时计算 sink payload，之后直接复用缓存。
  3. 只对具体有序 sink payload 去重，不根据 `ga == gb` 或 Gamma 名称去重。Gamma 指标转置后得到的 payload 必须单独保留，除非具体矩阵确实相同。
  4. 对每个不同 payload 计算与基准的 overlap 和 projective residual。全部为零时当前 sink class 没有解；有非零项时先做 sink rank，再保留真正成比例的 transporter。
  5. 只有 sink 成比例的 transporter 才进入 source Gram/Gram-Schmidt。多个有效 transporter 只保留一个 basis contract，其他作为 witness，并把对应系数合并。
  6. 裸指标重命名和转置单独记录，不能为了合并而删除开放轴的顺序信息。

  这个修正不会增加时间复杂度：原算法本来就要访问每个候选 transporter；现在只是把相同 sink key 的 payload 和内积缓存起来，避免重复构造。缓存可以按需建立，因此额外内存只随唯一 sink key 数量增长，不会预先展开所有 source 或所有完整 Wick 张量。

- **按 peram 类并行计算。** 不同的 `(profile, incidence_key)` 类别之间互不影响，因此算法天然适合做类别级并行。未来可以把不同类别分配给不同 worker，每个 worker 独立维护自己的 sink/source Gram 状态，最后按 Wick 编号确定性合并结果。worker 数量需要受到限制，因为 detailed 模式会保存 transporter witness，内存占用更高。

## 算符构造的 normal / distillation 模式

`make_operator(..., mode="distillation")` 是默认设置，沿用原有 VVV/VDV
输入和传播子名称。`mode="normal"` 表示不 smear 的场。同一个关联函数
中的全部算符必须采用相同模式。此处的构造模式与 reducer 的
`coarse/independent/detailed`、输出的 `independent/check` 设置各自独立。

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

normal 模式将 `external_tag` 和 `momentum` 放到该算符的每个夸克或反夸克
端点上。peram 分类和变量名称保留有序的两端标签，顺序固定为
quark → antiquark；epsilon/delta 不携带 p 标签。此次标签迁移不增加空间
坐标轴、Fourier 计算或空间求和，其他 contraction metadata 沿用原有作用。

生成的 Python 表达式由调用者提供以下数组：

| 数组 | 含义和形状 |
| --- | --- |
| `epsilon_color` | 实 Levi-Civita 张量，`(3, 3, 3)`，规定 `epsilon_color[0,1,2] = 1` |
| `delta_color` | 实单位阵，`(3, 3)` |
| `peram_<flavor>_<direction>_x_x` | 不 smear 的传播子，轴顺序 `(V_sink, D_sink, V_source, D_source)`，形状 `(3, 4, 3, 4)` |

传播方向包括 `sin_sin`、`sin_sou`、`sou_sin` 和 `sou_sou`。有 p 标签时，
传播子名称追加 `_p_<sha256>`，文件头列出 hash 对应的有序端点信息。
等价的 list/tuple 动量得到相同名称；显式 `momentum=0` 与未提供动量
保持区别。默认同位旋设置使 u/d 共用 `peram_u`；当二者数组不同，应同时
关闭 `ReductionOptions.isospin_degenerate` 和输出函数的
`isospin_degenerate`。normal 的颜色维度始终为 3；数值审计辅助函数使用
真实 epsilon 和单位阵，即使传入的蒸馏本征向量维度为 6。

## 统一接口

`kernels.wick_reducer.unified` 统一处理 `input=distillation|normal`、`mode=wick|special` 和 `output=independent|check|block|count`。使用 `python -m kernels --input-file input.json`；输出位于 `result/<input>_<mode>_<output>/`，并保留原有 writer 格式。
