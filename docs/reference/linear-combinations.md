# 精确算符线性组合

`kernels.wick_reducer` 提供 `reduce_linear_combination`，用于先展开全部输入项，再把结构相同的多个算符乘积拼成一批，交给原有 reducer 一次约化。这样批内的 Wick basis、transporter families 和 Dirac/peram cache 会保留到下一项，约化完成整批后才生成结果。旧的 `reduce_operators(operators, ...)` 仍接受原来的单个算符乘积；`lift_legacy_operators(operators)` 可将它提升为 sink/source 各一个系数为 1 的单项组合。

```python
from kernels.wick_reducer import (
    ExactScalar, OperatorProductTerm, OperatorLinearCombination,
    SourceLinearCombination, SinkOperator, CorrelatorCombination,
    reduce_linear_combination,
)

request = CorrelatorCombination(
    sinks=(SinkOperator("channel-A", OperatorLinearCombination((
        OperatorProductTerm((sink_baryon, sink_proton), 1, "sink A"),
    ))),),
    source=SourceLinearCombination(OperatorLinearCombination((
        OperatorProductTerm((source_proton_bar, source_sigma_bar),
                            "sqrt(2/3)", "source component 1"),
        OperatorProductTerm((source_neutron_bar, source_sigma_plus_bar),
                            "sqrt(1/3)", "source component 2"),
    ))),
)
result = reduce_linear_combination(request)
```

`OperatorProductTerm` 表示一个有序算符乘积、精确系数和来源标签。sink 的每项是独立计算的 sink channel；`SinkOperator` 里的项才会相加。`SourceLinearCombination` 表示 source 态的项。相同有序算符乘积会先按精确系数合并，零系数项会移除，来源标签会保留。可用整数、`Fraction`、`ExactScalar`，或 `sqrt(2/3)`、`1/3` 这类安全解析的表达式；不支持浮点输入或复数精确数。系数支持有理数与平方根项的加减乘，以及除以单个有理/平方根项；一般多项式代数数的求逆尚未实现。

## source 伴随约定

source 输入的 `Operator` 必须显式列出 source 端反夸克场、Gamma dagger、槽位顺序和开放自旋槽；现有 `Operator` API 不会自动对算符做伴随。若 source 态写成 \(S=\sum_j w_j S_j\)，关联函数中使用 \(S^\dagger\)，因此 source 权重变为 \(w_j^*\)。sink 权重按原样乘入。当前 `ExactScalar` 只表示实有理数和实平方根组合，所以 source 共轭在数值上不改变系数；复精确系数尚未实现。

算符乘积的场顺序遵从用户声明的因子和场槽顺序。若从一个有序重子乘积写出伴随，需显式反转乘积因子顺序、按项目约定反转重子源端二夸克槽并使用 dagger Gamma；任何再排回显示顺序的费米子交换符号都必须显式包含在对应系数中。Wick 符号则由项目既有规则根据完整声明场序计算。不要把“source 系数共轭”与这些场/费米子重排混为一项。

## Sigma-N 到 Lambda-p 验证输入

项目的原始重子定义是
\[
p_{raw}=\epsilon^{abc}u^a(u_b^T C\gamma_5 d_c),\qquad
n_{raw}=\epsilon^{abc}d^a(d_b^T C\gamma_5 u_c).
\]
在该定义下 \((C\gamma_5)^T=-C\gamma_5\)，并由 epsilon 反对称性得到 \(I_+n_{raw}=-p_{raw}\)。因此 Condon–Shortley 相位的物理中子为 \(n_{phys}=-n_{raw}\)。项目 Sigma 约定为 \(\Sigma^0_{norm}=\Sigma^0_{raw}/\sqrt2\)。将用户给出的物理态
\[
\sqrt{2/3}\,|\Sigma^0_{norm}p\rangle+
\sqrt{1/3}\,|\Sigma^+n_{phys}\rangle
\]
换成项目原始算符后，实际输入应为
\[
\frac1{\sqrt3}\left(\Sigma^0_{raw}p_{raw}
-\Sigma^+_{raw}n_{raw}\right).
\]
每个 \(\Sigma^0_{raw}p\) 的两个 raw Sigma 项各取 \(+1/\sqrt3\)；\(\Sigma^+_{raw}n_{raw}\) 取 \(-1/\sqrt3\)。Sigma 归一化因子只用于这一步转换，不再乘第二次。`make_sigma_lambda_example()` 按该映射构造 source；它把 \(\Lambda_1p\) 与 \(\Lambda_2p\) 作为两个独立 sink channel，不把它们相加。

示例运行后，`components` 保存每个 sink/source 算符乘积和其每个原始 Wick 配对；`wick_contributions` 逐图记录 pairing、Wick 符号、VVV 符号、精确输入权重、批内基底映射和 relation residual；`global_basis[*].rows[sink]` 保存每个 sink 在所列结果基底坐标上的汇总系数。`global_basis_id` 只在该次结果内有效，不能与单独调用旧 API 得到的局部基底编号混用。

在 u/d 同位旋简并且使用上述归一化和相位时，示例的两个 sink channel 都应逐基底系数消去。`exact_zero` 表示输入根式权重在已接受的整数关系系数下精确求和为零；请同时阅读 `exactness`：关系发现仍沿用旧 reducer 的浮点 Dirac/Gram 投影，固定秩判定容差 `numeric_tolerance=1e-10`。因此这不是独立符号消元器给出的代数证明。`zero_within_tolerance` 是另一个数值判定，使用结果字段 `zero_reporting_tolerance`，默认也为 `1e-10`；传入 `tolerance=` 只改变该零判定，不会改变旧 reducer 的秩阈值。

当前 reducer 先按完整算符端点、Gamma、味道模式等兼容拓扑把展开项分批；每批只调用一次旧 reducer，批内按原有 `(profile, peram graph)` 桶保留 basis。不同拓扑批次不会强行共享局部 basis；结果中的 `global_basis_id` 只是本次结果的输出编号，`global_independence_certified=false` 表示没有宣称跨批次的完整张量线性独立。这个实现避免了分别输出后再拼接局部 basis ID，同时保持旧 reducer 的约化规则不变。

可用 `python -m unittest tests.test_linear_combination` 重复验证输入模型、批内 basis 共享、旧 API 兼容和 Sigma-N/Lambda-p 例子。

## 组合规模和缓存

若 source 有 \(m\) 个未合并项、一个 sink 有 \(n\) 个未合并项，输入先形成至多 \(mn\) 个算符乘积；每个乘积再生成其味道允许的 Wick 配对数，六夸克乘积最坏可按各味道重数的阶乘增长。精确预合并相同有序算符项可减少乘积数；相同端点/Gamma/味道模式可共享拓扑初始化，reducer 内可共享 Dirac payload 缓存。当前实现没有跨独立 `reduce_linear_combination` 调用持久缓存，且旧接口逐项计时会为每项重建 reducer 状态。
