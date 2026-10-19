# 收缩输出格式

`expression_output.py` 将约化结果写成可读的 Python 文件，输入是有序二元组
`[O1_sink, O2_source]`：`O1_sink` 是 sink 算符列表，`O2_source` 是 source
算符列表。列表顺序就是张量槽位顺序。每个 source 算符的味道、Gamma 矩阵名和
`gamma_slots` 都以输入为准；构造器不会因算符位于 source 端而自动交换槽位。

## 张量命名

VVV/VDV 的名字由张量类型、端点和动量标签决定，不包含 `p`、`n` 等粒子种类。
没有 `external_tag` 的算符共用同一个默认动量张量；本项目内置例子的默认动量是
零动量。若动量不同，为算符指定不同的 `external_tag`，例如 `"1"`，则输出为
`VVV_sink_1` 或 `conj_VVV_source_1`。同一动量的不同粒子应使用相同标签。
标签也参与算符的可交换性判定，因此同一物理动量的算符必须使用相同标签。省略标签
表示共用默认零动量；若使用显式标签，所有要共用该张量的算符都应使用同一个标签。

每个生成文件开头会列出本次实际使用的 elemental 名称和标签约定。默认名称包括：

- 重子 sink：`VVV_sink`
- 重子 source：`conj_VVV_source`
- 介子 sink：`VDV_sink`
- 介子 source：`conj_VDV_source`

source 与 sink 是不同的张量端点，因此分别命名。相同动量的两个 proton 和
neutron 会引用同一个 VVV 张量名；动量不同的 proton 也应通过不同标签区分。

在 `normal` 模式中，peram 数组的轴顺序统一为 `(V_sink, D_sink, V_source, D_source)`，形状为 `(3, 4, 3, 4)`。输出方程中每个 peram 因子按有向 quark → antiquark 两个端口写成 `(V_q, D_q, V_bar, D_bar)`；这两个端口在 `TensorSpec.peram_ports` 中分别记录为 `(color, spin)`。`distillation` 模式也统一使用相同的轴顺序；若向量/eigen 维度为 N_v，形状为 `(N_v, 4, N_v, 4)`。

## 自定义算符：从 pn 开始

不需要把新关联函数注册到 `operator_library()`。先逐个构造算符，再按
`[O1_sink, O2_source]` 传给 `write_output`。当前数据结构支持重子（3 个有序夸克槽）
和介子（`(反夸克味道, 夸克味道)` 两个槽）；每个算符可指定一个作用于两个槽位的
具体 `4×4` Gamma 矩阵。重子还可指定一个开放的裸 spin 槽。`gamma_slots` 的顺序就是
矩阵两端的指标顺序。VVV/VDV 由算符类型、端点和 `external_tag` 自动命名。

`make_operator` 的位置参数依次为 `side`、`kind`、`species`、`label`、`flavors`、
`gamma`、`gamma_slots`；`open_slot` 和 `external_tag` 可用关键字指定。`species` 是
粒子/算符类别标签，`label` 标识本次输入中的算符实例。Wick 配对按实际 flavor 进行；
默认同位旋模式只把 u/d 的 peram 归入同一等价类，不会把不同味道的场改成同一个场。
source 味道槽和 Gamma 槽完全按调用者提供的顺序使用，不自动反排或生成共轭算符。

下面从组成开始定义 pn。这里明确选择 source p 的槽序为
`(u_bar, d_bar, u_bar)`，source n 为 `(d_bar, u_bar, d_bar)`；若你的共轭算符约定不同，
只需在输入中调整 tuple 和相应 Gamma 矩阵，输出器不会替你改动它们。

```python
from kernels.wick_reducer import make_operator, write_output

O1_sink = [
    make_operator(
        "sink", "baryon", "p", "sink_p", ("u", "u", "d"),
        "Cgamma5", (1, 2), open_slot=0,
    ),
    make_operator(
        "sink", "baryon", "n", "sink_n", ("d", "d", "u"),
        "Cgamma5", (1, 2), open_slot=0,
    ),
]
O2_source = [
    make_operator(
        "source", "baryon", "p", "source_pbar", ("u", "d", "u"),
        "Cgamma5_dagger", (1, 2), open_slot=0,
    ),
    make_operator(
        "source", "baryon", "n", "source_nbar", ("d", "u", "d"),
        "Cgamma5_dagger", (1, 2), open_slot=0,
    ),
]

write_output(
    [O1_sink, O2_source], "verify_output/pn_independent.py",
    name="pn", mode="independent",
)
write_output(
    [O1_sink, O2_source], "verify_output/pn_check.py",
    name="pn", mode="check",
)
```

这组明确的算符输入会生成 36 个原始 Wick 图，其中 35 个独立图、1 个依赖图。计数和
约化系数属于这组具体输入；改变 source 槽序、Gamma 或算符组成后，应以新生成文件中的
结果为准。自定义数值 Gamma 可通过 `gamma_matrices={"my_gamma": matrix}` 传入，
`matrix` 是数值 `4×4` 数组，算符的 `gamma` 字段填写 `"my_gamma"`。输出文件会引用名为
`my_gamma` 的矩阵变量，实际 Gamma、peram 和 VVV/VDV 数组由调用者提供。

## Ppn 输入

内置 `ppn` 使用 sink `p1,p2,n` 和 source `p1,p2,n`。source p 的槽位是
`(u_bar,d_bar,u_bar)`，source n 的槽位是 `(d_bar,u_bar,d_bar)`；这只是该 fixture
显式填写的算符内容。自定义 source 可采用任何槽位顺序：

```python
from kernels.wick_reducer import make_operator, write_output

O1 = [
    make_operator("sink", "baryon", "p", "p1", ("u", "u", "d"),
                  "Cgamma5", (1, 2), open_slot=0),
    make_operator("sink", "baryon", "p", "p2", ("u", "u", "d"),
                  "Cgamma5", (1, 2), open_slot=0),
    make_operator("sink", "baryon", "n", "n1", ("d", "d", "u"),
                  "Cgamma5", (1, 2), open_slot=0),
]
O2 = [
    make_operator("source", "baryon", "p", "pbar1", ("u", "d", "u"),
                  "Cgamma5_dagger", (1, 2), open_slot=0),
    make_operator("source", "baryon", "p", "pbar2", ("u", "d", "u"),
                  "Cgamma5_dagger", (1, 2), open_slot=0),
    make_operator("source", "baryon", "n", "nbar1", ("d", "u", "d"),
                  "Cgamma5_dagger", (1, 2), open_slot=0),
]
write_output([O1, O2], "verify_output/ppn.py", name="ppn", mode="independent")
```

source 的 Gamma 矩阵也由输入指定。`Cgamma5` 输出名为 `C_g5`，
`Cgamma5_dagger` 输出名为 `C_g5_dagger`，`gamma5` 输出名为 `g5`。矩阵两端
的指标顺序严格遵循 `gamma_slots`，不隐式转置。

peram 名称格式为 `peram_<flavor>_<q-side>_<anti-side>`，其中 `sink/source`
缩写为 `sin/sou`。默认同位旋模式下 `u` 和 `d` 映射到 `peram_u`；关闭该模式后
分别输出。开放裸 spin 指标依 sink 算符顺序、再 source 算符顺序分配大写字母。

## Independent 模式

文件开头用一条不换行的 `<name>_coefficients = [...]` 保存系数数组；位置 `i` 的系数
对应 `<name>_basis[i]`。check 模式中，为便于逐 Wick 重构而额外生成但不参与总关联函数的
basis 视图，其系数填 0。系数数组只出现一次，放在输出文件前部。

后续计算按两个步骤组织，每一步前都有注释说明：

1. `<name>_basis_contract[wick_id] = contract(...)` 计算每个独立 Wick 代表一次。字典键
   是该代表的 1 起始原始 Wick 编号，因此不需要另一个 contract ID 列表。
2. `<name>_basis[basis_id]` 是真正参与组合的基底视图。它由对应的
   `basis_contract[wick_id]` 加上代表图的 Wick 符号，并按需要执行 `.transpose(...)` 得到。
   符号和裸轴转置已经放在这里，不再单独保存 `basis_wick_sign`。
最终结果是系数乘基底后求和，输出中用 `<name>_result` 表示；对应的求和步骤前也有注释。

以上面的 pn 为例，`pn_independent.py` 中会看到单行 `pn_coefficients = [...]`、35 个
独立代表的 `pn_basis_contract[wick_id] = contract(...)`，以及 `pn_basis[basis_id]`。
同一套独立 Wick 图对应的约化系数会累加到系数数组；本例 `pn_coefficients[1]` 为 2，
其余系数为 1。
`pn_result` 将非零系数与相同下标的 basis 相乘并求和。`wick_id` 从 1 起，`basis_id`
从 0 起；裸指标排列不同的 basis 视图通过 `.transpose(...)` 构造，不重复执行
`contract(...)`。

下面是该 pn 输入实际生成的第一条完整收缩，指标字符串和输出轴都已展开；生成文件中每个
`contract(...)` 调用各占一行：

```python
pn_basis_contract[1] = contract('ab,cd,ef,gh,AioC,ajqf,bkpe,BlrD,cmth,dnsg,ijk,lmn,opq,rst->ABCD', C_g5, C_g5, C_g5_dagger, C_g5_dagger, peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou, peram_u_sin_sou, VVV_sink, VVV_sink, conj_VVV_source, conj_VVV_source)
```

例如，若第 1 个原始 Wick 图是代表图且符号为负：

```python
ppn_basis_contract = {}
ppn_basis_contract[1] = contract('...', C_g5, peram_u_sin_sou, VVV_sink)
ppn_basis = {}
ppn_basis[0] = -ppn_basis_contract[1]
ppn_coefficients = [1.0]
ppn_result = sum(c * ppn_basis[i] for i, c in enumerate(ppn_coefficients) if c)
```

若不同 Wick 图只差 source 裸指标顺序，它们可以共用 `basis_contract` 的收缩结果，
并在 `basis` 阶段生成相应转置视图。

## Check 模式

check 文件先按原始 Wick 编号递增输出原始图，再输出约化图。原始 Wick ID 从 1 起始，
对应的 Python 输出槽位从 0 起始：

check 保留每个原始收缩到 `<name>_origen_wick_raw[slot]`，其 sign 存在
`<name>_origen_wick_sign[slot]`；`slot` 从 0 起，原始 Wick ID 为 `slot + 1`。之后
`<name>_origen_wick[slot]` 直接写成系数乘 `basis` 的形式：

```python
ppn_origen_wick_raw[0] = contract('...', C_g5, peram_u_sin_sou, VVV_sink)
ppn_origen_wick_sign[0] = -1
ppn_origen_wick[0] = 1 * ppn_basis[0]
```

逐项核对 `origen_wick_sign[slot] * origen_wick_raw[slot]` 与 `origen_wick[slot]`。
`<name>_origen_wick_to_independent` 是映射字典，不是需要调用的函数。它的键是 0 起始的
输出槽位 `slot`，对应 1 起始的原始 Wick 编号 `slot + 1`。值中的 `terms` 列出该原始图
如何由基底重构；每项包含 0 起始的 `basis_index`、展开 `coefficient`、生成该基底收缩的
1 起始 `basis_wick_id`，以及裸轴 `transpose`。一个原始 Wick 图可能需要多个基底项的线性
组合，因此这里用 `terms` 列表。直接阅读时看 `origen_wick[slot] = coefficient *
basis[basis_index]`；需要机器读取映射、转置或多项展开时再查这个字典。它用于检查和追溯
原始图与基底的关系，不参与约化计算本身。

对上面的 pn，check 文件有 36 个 `pn_origen_wick_raw[0..35]` 和对应符号，之后有 35 个
独立代表收缩与 basis 视图，再按原始 Wick 顺序输出 36 条 `pn_origen_wick[slot] = ...`。
因此可逐项检查 `pn_origen_wick_sign[slot] * pn_origen_wick_raw[slot]` 是否等于右侧的
基底展开。本例中原始 Wick ID 31 对应槽位 30，并重构为 `1 * pn_basis[1]`；映射字典中
相应项指出它使用 Wick ID 2 生成的基底视图，系数为 1，裸轴置换为恒等置换。check 输出
全部原始 Wick 图，文件会比 independent 模式大；ppn 等图数较多的例子可按需生成。

```python
pn_origen_wick[30] = (1 * pn_basis[1])
# pn_origen_wick_to_independent 中 slot 30 的对应内容：
30: {'wick_id': 31, 'terms': [{'basis_index': 1, 'coefficient': 1, 'basis_wick_id': 2, 'transpose': (0, 1, 2, 3)}]}
```

## 命令行

```sh
python3 kernels/wick_reducer/expression_output.py \
  --example ppn --mode independent --output verify_output/ppn_independent.py
python3 kernels/wick_reducer/expression_output.py \
  --example ppn --mode check --output verify_output/ppn_check.py
```

生成文件导入 `opt_einsum.contract`，并要求调用者提供 Gamma、peram、VVV/VDV
数组。reducer 负责符号约化和表达式导出，不读取或计算具体格点上的数值张量。
