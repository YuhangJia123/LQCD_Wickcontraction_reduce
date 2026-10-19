# wick_reduce

本项目用于格点关联函数的 Wick 收缩枚举、等价约化与符号输出，支持由介子和三夸克重子组成的自定义 sink/source 算符列表。项目已经包含通用因子图 Wick reducer、normal 模式的 incidence + global-colour S3 融合内核，以及 `normal-wick-block` W 融合模块。

统一入口是 `python -m kernels`。所有计算先选择 `input`、`mode`、`output`，再定义粒子算符；算符定义保存在一份 JSON 中，每次运行可以直接用命令行切换选择项。

首次运行请阅读[中文 Quickstart](QUICKSTART.md)。英文文档见 [English README](README_en.md) 和 [English Quickstart](QUICKSTART_en.md)。

## 统一 input mode output 接口

| 选择项 | 含义 | 可用值 |
| --- | --- | --- |
| `input` | 输入的张量表示 | `distillation`：蒸馏 VVV/VDV 与 peram；`normal`：普通三色传播子与 epsilon/delta 颜色张量。 |
| `mode` | 约化路径 | `wick`：通用因子图与 Dirac/source 约化；`special`：normal 的 ordinary-colour 路径，按 `wick_scope` 选择跨时间或完整图融合。 |
| `output` | 需要的结果 | `independent`、`check`、`block`、`count`；各组合的限制如下。 |

当前共有八种合法组合：

| input | mode | output | 输出内容 |
| --- | --- | --- | --- |
| `distillation` | `wick` | `independent` | Python 独立收缩、带符号/转置的基底视图、系数和最终求和。 |
| `distillation` | `wick` | `check` | 上述 Python 表，加上原始 Wick 收缩、符号和逐项重构映射。 |
| `normal` | `wick` | `independent` | 保留普通颜色张量和开放自旋轴的 Python 收缩表。 |
| `normal` | `wick` | `check` | normal 的独立收缩表及逐项 Wick 重构数据。 |
| `normal` | `wick` | `block` | 自动从 raw Wick 配对族构造并验证 W 融合，输出 W 定义与重构表；当前 ppn 验收为 48 个网络，ppnn 为 288 个网络。 |
| `normal` | `special` | `independent` | JSON v3 的独立 Wick 向量与显式传播子单项式。 |
| `normal` | `special` | `check` | 上述 JSON 基底，加上每个原始 Wick 的线性关系与向量摘要。 |
| `normal` | `special` | `count` | SQLite 合并后的非零坐标计数 `nlist` 和统计信息，不输出完整基底/系数字典。 |

`special` 只用于 `normal`；`block` 只用于 `normal-wick`；`count` 只用于 `normal-special`。代码缺省选择是 `distillation-wick-independent`，`wick_scope` 缺省为 `complete`；仓库中的 `default.json` 演示 `normal-special-independent`。新输入使用标准拼写；历史别名 `distrilation`、`indenpent` 仍分别兼容 `distillation`、`independent`。

### 文件输出与 special 内核

统一 CLI 调用 `write_unified`。Wick 输出继续使用原 Python writer；special 的 independent/check 继续使用原 colour-orbit JSON v3 writer，其中 `records` 保存 `basis` 和可选的 `raw_wick_relation` 记录。

`special` 的内存计算由两个并列内核承担：`cross_time` 使用紧凑的跨时间源系数融合，`complete` 使用保留 sink/source 完整图的通用融合；两者都调用共享的 incidence/spin/S3 核心。`check` 在相同图构造下关闭 S3 轨道合并，再与生产结果逐项比较。CLI 文件输出仍按表格中的 JSON schema 写入 `result/`。

## 输入表示与粒子定义

统一 JSON 的结构是：

```text
input / mode / output / name
operators = [sink 算符列表, source 算符列表]
可选：gamma_matrices、wick_scope、normal_ordered 等
```

完整可运行配置放在 [Quickstart](QUICKSTART.md#创建输入并运行)，修改同一文件的选择项或使用 CLI 覆盖即可，不必为每种模式重复构造粒子。

| 算符字段 | 含义 |
| --- | --- |
| `kind` | `meson` 或 `baryon`。 |
| `species`、`label` | 粒子类别与本次输入的实例标签；名字不是专用算法开关。 |
| `flavors` | 重子：三个有序夸克味道；介子：`[反夸克味道, 夸克味道]`，单个味道表示对角介子。 |
| `gamma`、`gamma_slots` | Gamma 名称及其两个有序槽位；source 的 dagger 和场槽顺序要显式指定。 |
| `open_slot` | 重子的开放自旋槽，默认 0；默认 Gamma 连接其余两槽。 |
| `external_spin` | special 中将重子开放槽固定为 0、1、2 或 3；未指定时该实现枚举并累加该槽的自旋，不等同于 wick 的开放自旋张量。 |
| `external_tag`、`momentum` | 独立的端点/动量标签；其归属由 `input` 决定。 |

嵌套列表内可省略 `side`，程序按所属列表确定；扁平列表必须明确写 `side=sink|source`。两侧均须非空。一个关联函数内不能混用 distillation 和 normal；若单条记录写了 `mode`，它必须与顶层 `input` 一致。

### p/x 标签放在哪里

| input | 颜色/空间因子 | 传播子及标签 |
| --- | --- | --- |
| `distillation` | 重子 `VVV`、介子 `VDV`，source 使用共轭元件。 | peram 轴为 `(V_q,D_q,V_bar,D_bar)`，通常形状为 `(N_v,4,N_v,4)`；`external_tag` 和元数据区分 VVV/VDV 元件。 |
| `normal` | 重子 `epsilon_color`，介子 `delta_color`，颜色维度为 3。 | peram 形状为 `(3,4,3,4)`；Wick writer 将 `external_tag` 和 `momentum` 编入端点描述与名称。special 将这些标签及其他非 `external_spin` 元数据保留为端点 `point`。 |

标签不会自动增加坐标轴、执行 Fourier 变换或空间求和。统一 JSON 没有独立的 `p`/`x` 输入键，使用 `momentum` 与 `external_tag` 表达这些信息。`projector`、`clebsch`、`phase` 等元数据在不同路径中的处理不应混为一谈，也不表示程序会自动构造相应数值张量或展开物理态叠加。

## 环境与安装

推荐 Python 3.11；包配置声明 Python >= 3.9，本次验证使用 Python 3.11。项目可在根目录直接运行，符号 Wick/special 路径使用标准库。normal block 生成需要 NumPy；实际执行 Python 收缩文件和数值验证需要 NumPy 与 opt_einsum。

| 使用场景 | 准备方式 |
| --- | --- |
| 源码目录中生成符号结果 | Python 可用即可，按 [Quickstart](QUICKSTART.md) 运行。 |
| 安装到虚拟环境 | 在项目根目录执行 `python -m pip install '.[numeric]'`，包括项目与可选数值依赖。 |
| 使用 Conda | `environment.yml` 提供 Python 3.11、NumPy、opt_einsum 和构建工具；环境名仍为 `wick-diss-reduce`。 |

完整环境命令及已验证范围见 [Quickstart 的环境准备](QUICKSTART.md#准备环境)。

## 输出与结果组织

统一入口写入：

```text
result/<input>_<mode>_<output>/
    <name>_<input>_<mode>_<output>.py     # wick
    <name>_<input>_<mode>_<output>.json   # special
    manifest.json
```

一个组合目录可以保存多个不同 `name` 的粒子结果；每个结果文件的前缀就是粒子名。相同名称与组合再次运行会替换相应结果；同一组合目录内的 `manifest.json` 只记录最近一次写出，不是所有任务的历史清单。用 `--result-root PATH` 改变根目录。

分发包只附带四个轻量格式样例，位于 `examples/example_result/`；第一层仍严格按 `input_mode_output` 分目录：

```text
examples/example_result/
├── normal_special_count/pipi_normal_special_count.json
├── distillation_wick_independent/pip_distillation_wick_independent.py
├── normal_wick_check/pn_normal_wick_check.py
└── normal_wick_block/pip-p_normal_wick_block.py
```

四个轻量结果快照在 [`examples/example_result`](examples/README.md)，对应输入 JSON 在 [`examples/sample_json`](examples/README.md)。它们用于查看文件组织和 schema，不包含实际格点 peram 数值。运行程序时仍会把新结果写入项目根目录的 `result/`；`example_result/` 只是随项目分发的只读示例。

Wick Python 文件中的 `basis_contract` 只做代表收缩，`basis` 加入符号和开放轴转置，`coefficients` 对这些视图加权，`result` 给出最终求和。它们是表达式文件，执行前必须提供文件头列出的 Gamma、peram、VVV/VDV 或 epsilon/delta 数组。项目不会读取或生成实际格点数据。

`result/` 是统一接口的唯一结果位置。验证脚本的临时文件使用系统临时目录或 `verify_output/`。

## 算法、验证与已知限制

`wick` 先枚举精确味道配对，再按有向 peram、因子图和合法 transporter 分组，处理 sink Dirac 与 source 线性关系。`special` 的融合路径在 incidence/connectivity 分类上结合全局颜色 S3；`count` 使用 SQLite 临时存储合并非零坐标。`normal-wick-block` 自动保留完整 raw Wick 配对族，证明其与 normal 表等价，再搜索可复用的反对称双传播子结构。

当前接口限制：

- `wick_scope=complete|cross_time`、`normal_ordered`、`tolerance` 是 special 设置，缺省为 `complete`。Wick writer 不使用 `wick_scope`；需要跨时间约化时在 JSON 中明确写 `cross_time`。
- `isospin_degenerate_perams=true` 默认假设 u/d 数值传播子相同。原始 Wick 枚举仍按精确味道；u/d 数组不同就设为 `false`。
- CLI 文件输出路径不使用 `--max-entries` 作为内存上限；special 内核 API 会把它传给对应累加器。
- 自定义 Gamma 必须是数值 `4 x 4` 矩阵；每个算符目前只有一个连接两槽的 Gamma，不能直接解析矩阵乘积字符串或任意多端口自旋张量。
- 算符可自定义，但收缩数量仍会快速增长。这里的“任意模式”指支持模型内的任意算符组合与合法接口选择，不代表无资源限制。

本次文档验证已完成 153 个项目测试，以及八个蒸馏例子在维度 3、6 下的直接 Wick/基底重构比较。运行命令和未验证部分见 [Quickstart 验证记录](QUICKSTART.md#验证记录)。

计数要按对象区分：Wick 的 `basis_contract` 数、special JSON 的 `independent_wicks`、融合/计数的 `nlist` 和文献 state-block 数不是同一指标。3H/4H 对照数据见[最终研究报告](report/final/wick_diss_reduce_algorithm_report.md)，不要把其中不同对象的数量直接替换使用。

## 目录与文档导航

| 路径 | 作用 |
| --- | --- |
| `kernels/wick_reducer/` | 通用 reducer、算符模型、统一入口和 Wick Python writer。 |
| `kernels/wick_colour_orbit/` | incidence/S3 融合内核、普通颜色精确后端、JSON v3 writer。 |
| `kernels/block_reduction/` | normal block 规划与精确重构。 |
| `kernels/correlator_wick_normalizer.py` | Wick 枚举和 Dirac 归一化原语。 |
| `verify_code/` | 数值验证、审计、基准和适配工具。 |
| `tests/` | 单元与回归测试。 |
| `report/` | 算法、研究报告和实验数据。 |

`verify_code/` 只负责正确性对比、数值验证、性能基准和 ppn/ppnn 审计记录。生产入口不依赖这个子目录；`python -m kernels` 和统一 `result/` 输出独立运行。normal block 的输出适配器属于生产接口，位于 `kernels/block_reduction/output_adapter.py`。

- [中文 Quickstart](QUICKSTART.md)：准备环境、首次结果、切换模式和常见错误。
- [Python API 与 Wick 细节](docs/reference/wick-api.md)：API 的深入说明，其中函数参数 `mode` 的含义依各 API 而定，不能直接等同统一顶层 `mode`。
- [Wick 输出格式](docs/reference/wick-output.md)与[算法说明](docs/implementation/wick.md)。
- [Block 接口](docs/reference/block-api.md)与[工具说明](verify_code/README_zh.md)。
- [文档结构](docs/README.md)、[计算内核](docs/kernels/README.md)、[报告目录](report/README_zh.md)及[八例与 ppnn 验证数据](report/data/benchmark_20261014/README.md)。
