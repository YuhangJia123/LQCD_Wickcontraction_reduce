# normal-wick-block 接口

项目中的 `block` 只代表 `normal-wick-block`。生产入口先从 normal 算符生成完整 raw Wick pairing family，再生成三表目标；规划器精确证明两者相等后自动搜索可复用的二传播子反对称块，并保留无法融合的残余网络。底层 `kernels.block_reduction` 仍是纯符号规划器，但公共 writer 会自动完成适配，不要求调用方提前提供 pairing family。

目标表达式明确写为

\[
T=\sum_i c_i s_i C_{k_i}.\operatorname{transpose}(p_i),
\]

其中 `c_i` 是系数数组的条目，`s_i` 是该 basis view 自己携带的 scalar，`k_i` 指向输入的 contract，`p_i` 是输出轴转置。仅统计 contracts 或忽略部分 views 会改变目标。输出计划保存新的收缩网络及其重构系数/转置；残余网络与融合网络共同恢复原目标。

## 输入接口

公共接口是 `normal-wick-block` writer；它从 normal 算符自动生成 raw pairing family、三表目标和 `TensorSpec`，然后调用底层 planner。调用方不需要也不应手工把 `basis_contract` 当作另一个 block 模式。底层 `block_reduce` 的数据结构只用于生产适配器、证书重放和单元测试。

`contracts` 的值也可写成 `{"equation": "...->...", "operands": (...)}`。索引是网络连接标签，不代表算符 slot、味道或 Wick ID。`Factor(tensor, indices)` 和 `Network(factors, output)` 支持直接构建符号网络。

`TensorSpec` 必须显式说明每个张量的形状和轴类型。normal propagator 有两个 `(V, D)` 端口，端口中的数字是张量原始轴位置。项目统一约定数组布局为 `(V_sink, D_sink, V_source, D_source) = (3, 4, 3, 4)`，因此上述 `S` 的 `axis_types` 是 `("color", "spin", "color", "spin")`，端口为 `peram_ports=((0,1),(2,3))`。指定完整端口比依赖推断更易审核。

同一个 `identity` 明确声明多个张量名是数值相同的别名，它们可以共享同一种 W。执行计划会检查别名数组相同。`TensorSymmetry(permutation, sign)` 声明调用方保证成立的张量对称性；排列必须保留每个轴的类型和维度，符号为 `+1` 或 `-1`。不能因为两个张量的形状相同就声明相同 identity 或额外对称性。

`BasisView` 还接受 `scalar` 和 `coefficient`，其权重与单独传入的系数相乘。通常保留 `coefficient=1`，把原文件中的 `coefficients` 数组单独传入，避免重复应用权重。

## W 定义与残余项

把每个 `(V, D)` 端口视为一个 12 维联合索引后，复用块为

\[
W_{abij}=S_{ai}S_{bj}-S_{aj}S_{bi}.
\]

W 保留两端完整的四个 Dirac 分量和三个颜色；展开后的数组形状为 `(3,4,3,4,3,4,3,4)`，共有 `12**4 = 20736` 个元素，`complex128` 占 331,776 字节。这里没有上两分量投影。反对称交换同时交换整个 `(V, D)` 端口，不能只交换 Dirac 或颜色子轴。

`max_w_types` 限制共享 W **定义**的种类数，同一个定义可以在一个最终收缩网络中出现多次。它不是每个网络中的 W 因子数上限。`evaluate_plan` 对每种 W 构造一次，再执行各最终网络，最后应用重构表中的 scalar 和转置。

## 算法的实际执行顺序

1. 展开完整 basis 引用，把外部系数、view 内系数和符号各乘一次；把输出转置写入网络，检查所有项的有序输出空间相同。
2. 固定开放轴做精确规范化并合并系数，得到目标账本。另做允许输出转置的规范化，识别可共享计算的网络；两种等价关系分开使用。
3. 对每个计算图轨道只生成一次同数值传播子的双因子切口，交换完整源端 `(V, D)` 端口，以固定开放轴的规范键寻找伙伴。使用 `aX+bY=aRW+(a+b)Y`，保留每个残余；还允许有限次显式补偿桥接，寻找后续多步融合。
4. 以整个目标的不同非零计算网络数评分，输入方案作为初始上界。小目标逐候选分支；大目标按完整视图族批量改写，缓存重复状态并限制队列。默认 `max_w_types=1` 时按数值传播子身份分组候选、复用单类型状态并跳过不兼容分支；`max_w_types>1` 使用通用的类型组合搜索。每张网络可含多个同种 W，类型总数受 `max_w_types` 限制。
5. 公共 normal-wick-block writer 自动构造 `pairing_family`；`block_reduce` 先精确证明其加权和等于三表目标，再从残余图发现带符号端点作用，以对称边轨道优先生成全局布局；组合计数负责排序，实际规范图负责最终评分。
6. 构造唯一网络与转置重构表，重放改写并完整展开 W 验证目标相等。预算用尽时返回已证等价的最好方案，并在 `stats` 中注明未证明最优。

## 返回值：`BlockPlan`

底层 `block_reduce(...)` 返回 Python 对象，不直接写 Python 收缩文件。公共 `normal-wick-block` writer 会自动构造并传入 `pairing_family`；其主要字段为：

| 字段 | 内容 |
| --- | --- |
| `w_definitions` | W 名称、所用数值传播子、原张量的两个有序 `(V,D)` 端口与完整八轴形状；默认最多一种定义。 |
| `networks` | 去除相同计算图后的唯一非零完整收缩；每项是 `Network(factors, output)`，可含多个同种 W。 |
| `reconstruction` | `ReconstructionTerm(network_id, coefficient, permutation)`；同一网络可多次以不同带符号转置视图出现。 |
| `tensor_specs` | 执行这些网络所需的张量形状、轴类型、数值别名和已声明对称性，包含实际使用的 W。 |
| `original_terms`, `rewrites`, `family_certificate` | 输入加权项、逐步局部改写与可选全局配对族证明，供 `plan.verify()` 独立重放。 |
| `stats` | 输入/输出调用数、搜索预算及完成状态、最优性范围、布局和分阶段耗时等诊断信息。 |

最终张量严格按 `sum(term.coefficient * evaluate(plan.networks[term.network_id]).transpose(term.permutation) for term in plan.reconstruction)` 重构。`network_id` 是 `networks` 的零起始下标；转置作用在该网络的**输出数组**上。没有活网络时 `reconstruction` 可为空，输出零张量的形状由 `stats["output_shape"]` 给出。`plan.certificate` 是计划自身，调用 `plan.verify()` 或 `plan.certificate.verify()` 可核对精确符号恒等式；`evaluate_plan(plan, arrays)` 执行数值计划。

若两项的系数是 `2` 和 `-1`，可以写成一个反对称块再加一个原网络；不能把系数强行变成 `1`、`-1` 并丢掉残余。缺少交换伙伴时，合法回退是保留原网络。若目标完全抵消，计划可以没有最终网络和 W 定义。保持原表达式也是一个可审核的结果。

默认允许一次补偿桥接 `X = RW + Y`：保留显式 residual 后，后续改写可能继续融合。设置 `max_bridge_steps=0` 只寻找已经存在的交换伙伴。桥接创造的是同一目标的代数表达方式，不需要新增 raw Wick 输入；最终评分和证书均包含补偿项。

## 转置约定

NumPy 的 `T.transpose(p)` 将原输出标签 `O` 改成 `O[p]`。若要把 view 的符号网络重命名到公共目标标签 `U`，应使用

\[
O_j\mapsto U_{p^{-1}(j)}.
\]

例如 `p=(1,2,0)`、原标签 `abc`、公共目标 `XYZ`，重命名是 `a→Z, b→X, c→Y`，不是直接使用 `p`。嵌套转置的合成是

```python
T.transpose(p).transpose(q) == T.transpose(tuple(p[j] for j in q))
```

不能假定排列总是自身的逆；生成文件中的非对合 views 必须完整保留。

## 生产 normal-wick-block 入口

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

生产入口在内部只读解析 AST 中的三张表达式表，不执行 import、最终结果表达式或其他代码。它支持 literal einsum、张量名字、scalar 和 transpose；不接受通过循环、动态函数、数组访问或任意 Python 求值生成的表。

文件不包含可靠的形状、模式、identity 和对称性声明，这些信息由调用方传入的 `TensorSpec` 提供。加载 distillation 文件允许做结构审计，但本模块的 W planner 不把它转换为 normal contraction。

项目适配器从真正的 normal 输出整理这些信息，公共 writer 会自动生成全局搜索所需的原始配对族：

```python
from dataclasses import replace
from kernels.block_reduction.output_adapter import load_normal_output, pairing_family_from_operators
from kernels.wick_reducer.operator_model import operator_library

adapted = load_normal_output("normal_ppn_independent.py")
operators = tuple(replace(op, mode="normal") for op in operator_library()["ppn"])
family = pairing_family_from_operators(operators, tensor_specs=adapted.tensor_specs,
                                      compact=True)
plan = block_reduce(adapted.basis_contract, adapted.basis, adapted.coefficients,
                    tensor_specs=adapted.tensor_specs, pairing_family=family)
assert plan.verify()
```

`PairingFamily` 可以包含显式的带符号 `WeightedNetwork` 序列，也可以包含 `CompactPairingTerms`。后者以一个参考网络、传播子位置、物理交换类、传播子规格和可选的 sink/source 名称表表示整个交替配对族，网络按需生成；其系数严格由参考系数和各类的置换奇偶性决定。生产适配器使用 `pairing_family_from_operators(..., compact=True)` 产生该表示。数值相同的 u/d 传播子可以复用 W 数组，但物理交换类仍分别声明，不能因此跨味配对。

紧凑表示还要求同一 sink 槽在该物理类内的数值传播子 identity 和 `peram_ports` 不随 source 改变；不同方向或端点标签导致这些属性变化时，构造器会拒绝。

底层规划器不认识项目算符。生产适配器先精确证明配对族的加权总和等于三表目标，才尝试全局布局；映射错误会拒绝输入。适配器记录端点方向、标签、显式别名和从实际 Gamma 数组验证的精确转置关系。不同名字的传播子只有调用方明确声明数值别名时才合并，同名传播子不会因此获得自身转置对称性。

`normal-wick-independent` 的 ppn 基线是 **219 个 contract、2760 个完整 basis views、2760 个非零系数**；公共 `normal-wick-block` 自动使用原始 pairing family，验收为 **48 个完整网络**。ppnn 的对应 block 验收为 **288 个网络**。不能丢掉 views，也不能把 `VVV` 静默替换成 normal 的 epsilon。

## 验证与搜索边界

`plan.certificate.verify()` 校验 W 定义和端口，按顺序重放每一步改写，再完整展开 W 核对最终目标。使用配对族时，它还复查配对位置、交换轨道及原始族到三表的映射。重复使用已消费项、冲突的 W 定义、错误的系数或转置都会使证书失败。`evaluate_expression` 是原始三张表的数值参考，`evaluate_plan` 是新计划的数值执行器。比较对象是完整 open tensor；一个投影后的 scalar 相等不能替代完整 open tensor 相等。

`plan.stats` 包括输入 contracts/views 数、初始和最终网络数、W 类型数、搜索状态数、预算耗尽标志、搜索是否完成、符号验证结果和最优性范围。配对族路径另列自动发现的端点交换见证和最好的布局。`search_budget` 是局部搜索状态预算，`search_budget=0` 返回经过核对的原网络方案。默认 `search_seconds=5`；解析、初始规范化、配对族映射、搜索、选定方案构造和证书验证分别计时。单次候选规范化不能中断，实际搜索可略超预算；整个函数的墙钟时间不受 5 秒限制。

- `optimality_scope="counting_lower_bound"` 表示已经证明目标完全抵消，需要零个网络，达到计数下界。仅剩一个网络并不额外证明目标非零，因此不会单凭该计数作全局最优声明。
- `optimality_scope="restricted_rewrite_family"` 仅表示在本模块允许的二-propagator 交换融合、声明的张量 identity/对称性和对应搜索空间内完成搜索。
- `optimality_scope="best_found"` 表示预算内找到的方案；预算耗尽不能标记为全局最优。

大型目标使用批量候选及有界搜索队列，结果会声明该限制。输入规模增大时，短搜索可以返回原始网络数，而不是必然发现融合；完整保留输入表达式并通过核对仍是合法结果。

可选全局搜索先用带类型的残余图自动识别端点群，再以循环类型动态规划和 Burnside 数对完整配对族的布局排序。该组合计数不是实际带符号网络数的下界，不能据此剪掉可能出现新图同构或抵消的布局。最终分数以实际规范化后的不同非零网络为准。默认优先保留整体交换的最大配对布局，剩余预算按需生成其他匹配，包括较少 W 的布局。未穷尽候选时只报告 `best_found`。

实现按计算网络的带符号输出自同构群复用视图规范化，并在单次调用内缓存精确结果。大目标将同一布局下的整组转置视图一起融合，再组合互不冲突的布局，避免只替换部分视图而增加计算网络数。搜索保留最多 32 个大目标状态、每轮最多 8 个批次；这些限制会使结果标记为 `best_found`。局部图规范化和单次批次评分是原子操作，秒数预算在这些操作之间检查，因此运行时间还可能略超搜索预算。

即使完成搜索，也不证明任意多-propagator block、跨方案线性组合、额外 Fierz/颜色恒等式、投影或 contraction tree 的全局最优。最终网络调用数也不等于 FLOPs：构造 W 有成本，某种网络的 contraction path 可能更昂贵。计数改进与实测时间应分别报告。

## 保持通用性的速度和内存优化

当前 `block_reduce` 的首要目标是**不同非零完整网络数**，不是最短运行时间。规划器对规范图、自由输出视图、切口和已访问状态使用有限缓存；默认单 W 路径先按传播子身份缩小候选，多个 W 类型仍走通用搜索。`CompactPairingTerms` 不预存全部原始 Wick 网络，按交换轨道直接生成融合代表；但与目标三表的精确映射和最终证书仍需遍历完整表达式，ppnn 规模下这部分可能耗时。

`evaluate_plan` 按 `network_id` 分组，只保留当前网络输出和目标累加器；W 按定义构造一次，收缩路径按形状与拓扑在 256 条容量内缓存。额外数组内存更接近“结果 + 当前网络 + 收缩临时量 + W 缓存”。四分量 ppnn 八开放轴的一个 `complex128` 结果为 `4**8 * 16 = 1 MiB`；每种稠密 W 是 `12**4 * 16 = 324 KiB`。这两个数字不是整个计算的峰值上界。局部基准使用 24 个八轴网络和 144 个转置视图，`tracemalloc` 峰值从约 170 MiB 降至 3.3 MiB；该基准不代表 ppnn 的真实收缩成本。复现：`.venv/bin/python -B -m verify_code.benchmark_block_execution`。

收缩路径、转置访问和真实峰值仍需在目标硬件上分别测量。组合 Burnside 数只指导布局搜索顺序，不能代替完整网络的带符号评分与证书。

## 实际 normal-wick-block ppn 验收

正式入口从 normal 算符自动构造紧凑原始配对族，并在 `block_reduce` 内先证明它与 219 个基础收缩、2760 个视图的三表目标完全相等，再搜索 W 布局。无需调用方提前提供 pairing family。ppn 的验收结果是 **2880→180→48**：原始配对项 2880，选定 W 布局的融合项 180，最终不同非零完整网络 48；三表基线仍为 219。两个独立布局均得到 48，完整六轴数值核对误差约为 `1e-15`。详见[全局搜索报告](../../report/validation/global_block_validation_zh.md)。

```sh
.venv/bin/python -B -m verify_code.audit_global_block --directory /private/tmp/global_block_audit \
  --seeds 20261013 20261014 --json /private/tmp/global_block_audit/result.json
```

正式入口可直接验收：

```sh
.venv/bin/python -B verify_code/normal_block_validation.py --example ppn \
  --search-seconds 5 --search-budget 40 \
  --seeds 20261013 20261014 20261015
```

独立测试覆盖完整四分量反对称恒等式、残余项、带符号的非对合 views、同一 W 类型多次出现、类型预算、显式别名、端口顺序、抵消、缺少交换伙伴的回退和 distillation/上两分量拒绝。运行：

```sh
.venv/bin/python -B -m unittest tests.test_block_canonical tests.test_block_execution tests.test_block_normal_integration tests.test_block_global_integration
```
