# 通用 Wick 约化器设计与验证

> 本文为英文原文 [generic_reducer_design.md](generic_reducer_design.md) 的中文翻译。

本目录是通用 Wick 约化器的独立实现。运行时不需要父级代码库。下文的当前计数与独立项目根目录中的 `README.md` 一致；较早的报告文件可能仍保留早期约化规则下的历史计数。

## 约化规则

精确的场内容只在生成合法的原始 Wick 配对时使用。完成原始项生成后，`reduce_operators()` 会对任意算符序列应用相同的通用因子图约化；约化策略不依赖关联函数名称。约化键中不包含算符种类，只保留收缩数据：

- peram 的方向和端点角色（`sin_sin`、`sin_sou` 等，以及裸端点或哑端点）；
- VVV 侧、算符种类、Gamma 矩阵和 Gamma 端口；
- 外部/动量标签；
- 输入中所有 sink 和 source 因子顶点之间的规范连接模式。

最后一项是指标图，不能丢弃：即使因子名称相同，只要指标连接不同，就是不同的 `contract(...)` 调用。对于双侧重子输入，图键是 peram 边颜色表在 sink/source 两个顶点颜色类各自独立置换下的规范形式。同一函数也接受任意带颜色的算符图；输入不是二部图时，会使用有界的通用置换路径。相应的算符映射只作为规范化见证保存，不是独立的 p/n 同位旋传递器。

每个分桶内部仍使用现有的 Dirac Gram 和裸轴传递逻辑。Wick 符号、系数和裸轴转置仍作为依赖关系的元数据保留。精确原始项生成器没有改动，因此 flavor 选择仍会在约化前阻止不合法的 u/d 配对。

通过以下方式启用通用 API：

```python
from kernels.wick_reducer import ReductionOptions, reduce_operators

report = reduce_operators(
    operators,
    label="arbitrary_input",
    options=ReductionOptions(),
)
```

`label` 仅作为输出元数据。提供的 VVV/VDV、Gamma、动量和归一化因子才是比较对象；种类名称不属于约化键。系统没有第二条带种类感知的约化路径。

所有示例都使用因子图策略。精确原始项生成器以及下游的传递器、Dirac 和 Gram 检查均保持不变。

## 结果

下表中的内置算例均使用当前通用约化器运行：

| 算例 | 原始 Wick 项 | 独立项 | 依赖项 |
| --- | ---: | ---: | ---: |
| pipi | 24 | 10 | 14 |
| pn | 36 | 18 | 18 |
| ppn | 2,880 | 219 | 2,661 |
| delta | 6 | 2 | 4 |
| delta++delta- | 36 | 3 | 33 |
| pip | 24 | 24 | 0 |
| pip-p | 6 | 6 | 0 |
| pn_omega_pi+ | 3,456 | 908 | 2,548 |

对于历史四重子算例，通用的、与粒子种类无关的因子分桶结果为：

```text
raw Wick                 518,400
factor buckets             4,298
independent basis_contract 4,298
dependent terms          514,102
```

覆盖关系精确成立：

\[
4,298+514,102=518,400.
\]

本次运行使用 Python 3.9.6，在本机耗时约 242 秒。这里的结果是该直接因子图规则下开放指标 `basis_contract` 的数量；它不是轨道数 1,068，也不是轨道展开数 17,088，因为这些数字使用了不同的等价关系。

此前的生产约化器可作为同一算例的历史比较基线：

```text
518,400 raw Wick -> 31,092 basis_contract
```

因此，新的 4,298 结果体现了额外的商约化：精确生成原始 Wick 项后，去除了强子名称，同时保留 peram/VVV/Gamma 连接图和开放轴约化。

端到端随机审计脚本 `verify_code/ppnn_random_end_to_end.py` 使用八个开放自旋轴的一组固定随机投影，对全部 518,400 个原始项求值，并使用 4,298 个基项和 514,102 条依赖关系重构同一个标量。随机种子为 `20261011` 时，相对残差为 `1.29e-14`（绝对残差为 `8.32e-03`）。原始侧的总和来自完整原始项运行；约化侧对每个基张量只求值一次，并复用其 489,966 个轴视图。这是覆盖全部原始项的标量检查，并非对稠密 `4^8` 开放张量逐元素物化并比较。

在当前模型中，VVV 和 Gamma 因子不依赖强子名称。同位旋相关的数值条件是 `peram_u` 与 `peram_d` 为同一个数值传播子。只要动量、投影、`external_tag`、Clebsch 因子或额外相位存在差异，它们就仍属于约化键的一部分。

剩余指标对称性仍由原约化器负责，并在图规范化后继续应用：peram 线匹配、哑指标重命名、裸轴映射/转置、VVV 奇偶性、sink 射影检查以及 source Dirac Gram 投影。新的键只会在现有张量约化之前去除冗余输入标签。

对照流程图和输出比较见 [original_vs_improved_flowchart.md](../../../verify_code/original_vs_improved_flowchart.md)。
