# 精确算符线性组合

输入层只保留四步：

```text
原子 Operator
    ↓
OperatorProductTerm()
    ↓
OperatorLinearCombination(terms=(...))
    ↓
一个 sink combination + 一个 source combination
```

公开约化入口是：

```python
reduce_linear_combination(sink_combination, source_combination, sink_name="correlator")
```

`OperatorProductTerm` 表示一个有序算符乘积、精确系数和来源标签。`OperatorLinearCombination` 表示一侧的项集合；sink 和 source 由调用时的两个参数位置区分，不再使用 `CorrelatorCombination`、`SinkOperator` 或 `SourceLinearCombination` 包装。一次调用只计算一个 sink。旧的 `reduce_operators(operators, ...)` 仍接受原来的单个算符乘积；`lift_legacy_operators(operators)` 返回 sink/source 两个系数为 1 的组合。

## 四步输入示例

```python
from kernels.wick_reducer import (
    make_operator, OperatorProductTerm, OperatorLinearCombination,
    reduce_linear_combination,
)

# 1. 原子 Operator。label 可省略，flavors 必须提供。
p_sink = make_operator(
    side="sink", kind="baryon", species="p",
    flavors=("u", "u", "d"), gamma="Cgamma5",
    gamma_slots=(1, 2), open_slot=0,
)
p_source = make_operator(
    side="source", kind="baryon", species="p",
    flavors=("u", "d", "u"), gamma="Cgamma5_dagger",
    gamma_slots=(1, 2), open_slot=0,
)

# 2. 有序乘积项
sink_term = OperatorProductTerm((p_sink,), "1", "p sink")
source_term = OperatorProductTerm((p_source,), "1", "p source")

# 3. 一侧一个线性组合
sink_combination = OperatorLinearCombination(terms=(sink_term,))
source_combination = OperatorLinearCombination(terms=(source_term,))

# 4. 传入一个 sink combination 和一个 source combination
result = reduce_linear_combination(
    sink_combination, source_combination, sink_name="p",
)
```

## pp 的重复因子

只构造一个 `p_sink` 和一个 `p_source`，在乘积项中重复引用：

```python
pp_sink_term = OperatorProductTerm(
    operators=(p_sink, p_sink), coefficient="1", source_label="pp sink",
)
pp_source_term = OperatorProductTerm(
    operators=(p_source, p_source), coefficient="1", source_label="pp source",
)
pp_sink = OperatorLinearCombination(terms=(pp_sink_term,))
pp_source = OperatorLinearCombination(terms=(pp_source_term,))

result = reduce_linear_combination(pp_sink, pp_source, sink_name="pp")
```

两个 tuple 位置分别代表两个质子因子，即使它们引用同一个 Python `Operator` 对象，也不会被约化器删除或合并。项目当前 pp 输入产生 48 个原始 Wick 项和 15 个基底。

## 多项组合

每个乘积项先单独构造，再以 tuple 交给 `OperatorLinearCombination`，不再嵌套额外的 source/sink 组合类：

```python
a_sink_term = OperatorProductTerm((a_sink,), "1", "A sink")
b_sink_term = OperatorProductTerm((b_sink,), "-1", "B sink")
a_source_term = OperatorProductTerm((a_source,), "sqrt(2/3)", "A source")
b_source_term = OperatorProductTerm((b_source,), "sqrt(1/3)", "B source")

sink = OperatorLinearCombination(terms=(a_sink_term, b_sink_term))
source = OperatorLinearCombination(terms=(a_source_term, b_source_term))
result = reduce_linear_combination(sink, source)
```

约化先形成至多 (mn) 个 sink/source 乘积，再枚举每个完整乘积的合法 Wick 配对。相同有序乘积会先合并精确系数，乘积内部的重复算符位置保留。按兼容拓扑分批后，每个批次只调用一次旧 reducer，因此批内 basis、transporter families 和 Dirac/peram 缓存会保留到下一项；不同拓扑批次不会直接拼接局部 basis 编号。

## Lambda-p 单独输入

Lambda1p 和 Lambda2p 是两个分别运行的单 sink 输入：

```python
from kernels.wick_reducer import make_sigma_lambda_example

lambda1_sink, source = make_sigma_lambda_example("Lambda1p")
lambda1 = reduce_linear_combination(lambda1_sink, source, sink_name="Lambda1p")

lambda2_sink, source = make_sigma_lambda_example("Lambda2p")
lambda2 = reduce_linear_combination(lambda2_sink, source, sink_name="Lambda2p")
```

Sigma-N source 的原始算符映射仍为：

\[
\sqrt{2/3}\,|\Sigma^0_{norm}p\rangle+
\sqrt{1/3}\,|\Sigma^+n_{phys}\rangle
=\frac1{\sqrt3}\left(\Sigma^0_{raw}p_{raw}-\Sigma^+_{raw}n_{raw}\right).
\]

其中 `Sigma0_norm=Sigma0_raw/sqrt(2)` 只在转换时使用一次；项目原始中子相位给出 `n_phys=-n_raw`。两个 Lambda 输入在 u/d 同位旋简并约定下分别逐基底精确消去，结果不是把两个 sink 相加后的一个值。

## 系数和 source 伴随

系数接受整数、`Fraction`、`ExactScalar` 和 `sqrt(2/3)` 等安全表达式，不接受浮点近似。source 通过伴随进入关联函数时，source 系数取复共轭；当前精确标量只表示实有理数和实平方根，所以共轭对当前支持的系数没有数值变化。source 的场顺序、Gamma dagger 和费米子重排符号需要在 source 原子 `Operator` 及其项系数中按项目约定明确给出。

每个报告的 `components` 保存展开的 sink/source 项、输入系数和原始 Wick 配对；`wick_contributions` 保存每个原始图的符号、来源标签和本次结果内的基底映射。`global_basis_id` 只对当前一次调用有效。

关系发现沿用旧 reducer 的浮点 Dirac/Gram 投影，固定秩判定容差为 `1e-10`。报告中的 `exact_zero` 是在已接受的整数关系系数上对输入根式做精确求和；`zero_within_tolerance` 使用报告的零判定容差，不能把后者单独称为代数恒等式。
