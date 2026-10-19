# normal 模式的通用 block 接入计划

> normal block 的设计说明；现行入口、输入和验证方式见 [`docs/reference/block-api.md`](../../docs/reference/block-api.md)。

状态：设计计划，尚未实现生产 block。双模式输入及当前文档已保存在 Git 检查点 `5ecf2a8`。本计划首先优化数值执行，不改变现有 Wick 枚举、transporter 搜索、置换符号、sink Dirac 判断和 source Gram 约化。

## 1. 建议与接入时机

采用“输入可声明候选，约化后自动规划”的组合，以输出侧规划为主。

```text
算符输入（normal / distillation、Gamma、槽位、端点 p）
        ↓
现有 Wick 枚举与线性约化
        ├── 最终基底、复系数、开放轴转置
        └── 原始 pairing / sign → 基底视图的稀疏映射
                    ↓
结构化收缩表达式
                    ↓
normal block 规划：合法交换组 → 成本选择 → 缓存与执行 DAG
                    ↓
Python 输出：block 定义、块间收缩、残余项、展开映射
                    ↓
check：原始项、基底重构、block 结果三方核对
```

输入时可以声明某些槽允许构造 block，但此时尚不知道具体的反夸克端点、传播方向、外部连接和复用次数。因此声明只提供候选，不提前删除或替换 Wick 项。

约化之后已有实际表达式和成本信息，适合自动决定是否生成 block。这个模块只依赖算符描述、收缩表达式与约化映射，与 reducer 的搜索和 Gram 实现解耦。

| 时机 | 能做什么 | 主要限制 |
| --- | --- | --- |
| 算符输入时 | 声明允许交换的槽、偏好的 block 类型；自动识别同味槽 | 不能仅凭输入确定完整外部端口、合法交换组及缓存收益 |
| 最终独立代表之后 | 利用实际基底、轴映射及成本选择执行方式 | 独立代表不一定含有交换伙伴，必须读取或重建原始项来源 |
| 生成 Python 字符串之前 | 在结构化张量网络上规划、生成共享中间数组 | 需要从当前表达式生成器抽出结构化接口 |

不建议通过解析已经生成的 einsum/Python 字符串识别 block。

## 2. 为什么不能只扫描最终独立图

当前输出的一项由“独立 Wick ID、开放自旋轴转置、复系数”共同定义。source Gram 可能把一个交换伙伴表示成多个独立代表的线性组合；相同 Wick ID 的不同转置也可能是不同的张量视图。

对当前 normal 示例的只读检查得到：

| 示例 | 原始 Wick 数 | 独立代表数 | sink 内互不重叠同味槽对形成的交换组 | 成员全部在独立代表中的组 |
| --- | ---: | ---: | ---: | ---: |
| `pn` | 36 | 18 | 9 组，每组 4 项 | 2 组 |
| `delta`，选择一对同味槽 | 6 | 2 | 3 组，每组 2 项 | 1 组 |

检查方法：将内置算符切换为 normal，调用现有 reducer；在每个 sink 重子的精确 flavor 中选同味槽对，在 pairing 的 quark 端交换其目标，以交换闭包划分原始项，再统计闭包是否完全包含于 independent pairing 集合。

所以规划器同时接收最终基底视图及原始 Wick 映射。9 组 block 计算可以覆盖 `pn` 的特定总和，但不意味着当前 reducer 的独立代表数从 18 降为 9。

现有 `_relation_term_callback(relation, state, basis, raw_term)` 可收集依赖项信息；独立代表补充恒等映射。流式 `check` 已使用 SQLite 保存逐图到基底视图的关系，新规划器可复用此机制，并增加 pairing/sign 数据，避免开启 `detailed` 后把全部图常驻内存。也可以按需重新枚举 pairing，但必须保留与当前 Wick 编号、截断范围相同的顺序。

## 3. 首期支持的 block

首期选择 sink quark 重子的互不重叠同味槽对。识别依据是有序 flavor、Gamma 槽、开放槽和张量连接，不根据 `p`、`n`、`ppn` 等算符名称判断。

针对当前 `(u,u,d)`、开放槽 0、Gamma 槽 `(1,2)`，定义

\[
F_{\alpha;IJK}=
\sum_{abc,\beta\gamma}\epsilon_{abc}\Gamma_{\beta\gamma}
S^u_{\alpha a,I}S^u_{\beta b,J}S^d_{\gamma c,K},
\qquad
B_{\alpha;IJK}=F_{\alpha;IJK}-F_{\alpha;JIK}.
\]

这里的交换表示重新分配两条 u 线的反夸克端点；存在不同端点标签时，每个分配都重新解析正确的 peram 变量。各源端口的类型及标签随端口保留，不能盲目把一个数组做转置。

三夸克 block 先收缩本地 Gamma、epsilon 和三条传播子，保留核子的开放自旋及三个反夸克边界端口。端口记录自旋/颜色顺序、所在算符/槽位、端点 `p` 和方向，随后通过源侧 Gamma、epsilon 或其他因子与其余 block 收缩。

另一个候选是两传播子反对称积：

\[
W_{AB;IJ}=S^u_{A,I}S^u_{B,J}-S^u_{A,J}S^u_{B,I}.
\]

规划器可根据共享范围，在缓存 `W` 与缓存完整 `B` 之间选择。normal 下每个联合颜色–Dirac 指标维数为 12；complex128 的完整 `W` 为 `12^4` 元素，约 324 KiB；一个开放自旋的 `B` 为 `4×12^3` 元素，约 108 KiB。这是每个固定标签/核组合的大小，不包括额外空间坐标轴和多个缓存实例。

若有 r 个互不重叠的 sink 同味槽对，交换对 bijective pairing 的作用形成 `2^r` 个不同原始项。各次交换的相对 Wick 符号为负，多个 block 的乘积准确表达其完整有符号和。不能用某一个代表乘 `2^r` 替代这个和。

三同味槽的完整 `S_3` 六项反对称和、source 侧 block、跨算符 block 作为后续扩展。初版的三同味重子可以选其中一对槽。两侧同时交换可能出现重复成员或稳定子，需单独的闭包及计数逻辑。介子和其余因子保留常规收缩，并可使用通用子表达式缓存。

## 4. 接口与模块划分

新增独立的输出策略 `BlockOptions`，不把执行选项加入 `ReductionOptions` 或算符的物理颜色。

建议的接口草案（当前尚不可调用）：

```python
write_output(
    correlator,
    "normal_output.py",
    mode="independent",
    blocks=BlockOptions(
        policy="auto",       # off / auto / explicit；默认 off
        side="sink",
        memory_budget_bytes=128 * 1024**2,
    ),
)
```

`explicit` 接收通用 `BlockRule`，例如以 operator label 选择候选，指定 `slots=(0,1)` 和 kernel 类型。label 只用于选择/报告，不进入可复用核的语义键。规则也可由 `make_operator()` 产生的对象及其槽位自动生成，调用者不必重新写物理算符。

- 仅当显式启用 blocks 且所有算符均为 normal 时允许规划；distillation 请求 `auto/explicit` 明确报错。默认 `off` 保持现有输出。
- 现有算符构造 `mode`、reducer 的 `coarse/independent/detailed`、输出的 `independent/check` 保持各自含义。block 策略单独命名为 `policy`。
- 首期仅针对输出/执行；不向基础 `reduce_operators()` 引入 block 求解。

建议新增两个生产模块，并对现有输出做有限重构：

| 模块 | 职责 |
| --- | --- |
| `kernels/wick_reducer/contraction_ir.py` | `TensorFactor`、有序张量端口、`ContractionTerm`、系数、输出轴、原始项来源；把当前 einsum 构造先变成结构化数据 |
| `kernels/wick_reducer/block_planner.py` | `BlockOptions`、`BlockRule`、`BlockDefinition`、`BlockPlan`；候选识别、合法性验证、展开证书、成本选择及执行 DAG |
| `kernels/wick_reducer/expression_output.py` | `blocks` 参数、模式验证、来源收集、调用 planner；同一结构化接口渲染普通及流式输出 |
| `kernels/wick_reducer/__init__.py` | 导出公共 block 配置和独立规划接口 |
| `tests/` 与数值验证工具 | 结构展开、实际数组收缩、回归及缓存收益验证 |

normal block 需要同时验证符号分组、张量可因式分解性和数值重构；生产模块以 `block_reduce` 的证书为准。

## 5. 融合合法性与缓存

融合之前验证：

1. 两槽的原始物理 flavor 相同。`isospin_degenerate=True` 可以复用 u/d 数值数组，但不产生新的 flavor Wick 配对。
2. 完整交换闭包存在于实际目标项集，相对系数符合置换符号；闭包之外的因子与外部连接相同。
3. Gamma 槽次序、epsilon 次序、开放轴映射和边界端口顺序一致；必要的转置显式记录。
4. 每个交换成员按自己的 pairing 重新解析 peram 类别及变量，覆盖端点 `p` 和传播方向。
5. 每个原始项只计入一次。缺失伙伴、截断项集、重叠候选或不同相对系数时，不强行融合；保留残余项。

source `p` 不同不绝对禁止融合：完整的两种 assignment 均存在时仍可形成反对称和，但不同标签组合必须使用不同核或不同有类型端口，不能错误共用一个数值数组。

缓存键规范化局部网络中的临时指标名，并保留：

- kernel 类型、实际 Gamma 输入的标识、Gamma 槽及开放槽；
- 原始 flavor 和数值 flavor class；
- 每条线的方向、有序 quark/antiquark 端点描述及 p；
- 边界端口类型、顺序、允许的置换及符号；
- 会影响实际核的元数据。可证明为共同标量的 phase/clebsch 可提至外部并只乘一次，其余按当前语义保留。

缓存默认限定在一次关联函数求值中。输入传播子或 Gamma 数组改变之后不能复用旧值。计划报告唯一核数、引用次数、数组尺寸、估算成本和回退原因；遵守内存预算，可只缓存频繁使用的核或采用分批执行。

## 6. 如何接入最终约化结果与继续合并

block 规划器构造原始有符号项的候选组，再把每个组通过现有关系展开为基底视图的稀疏系数向量。向量的键必须同时包含 Wick ID 和开放轴转置。

以这些向量和未融合项表示目标系数向量，验证覆盖无重复、无缺失，然后比较已有独立项执行与 block 执行的估算成本。可以采用两者混合的计划；block 化不保证每个示例更快。

结构层的置换奇性和展开项精确记录；Gram 得到的数值复系数沿用现有容差，不能声称这些浮点关系是新的精确符号证明。

下一阶段在 block 向量上做恒等、零和线性组合合并，以展开证书证明最终结果一致。这个步骤使用现有约化关系，不改变 reducer。若进一步寻找颜色×Dirac 有限空间产生的新关系，需要独立的 block 层数学判据和数值验证。

优化指标明确为预计运行成本和峰值内存，辅以收缩调用数、唯一核数、缓存复用率。不能仅凭 block 数最少宣称取得全局最简单表达式。原始 Wick 数、独立图数与 block 执行节点数分别报告。

## 7. check 与流式输出

`B=F-F_swap` 不提供单独 F 的数值，完整展开映射只提供重新计算组件的方法。因此生产求值可以只算融合块，逐图 `check` 则仍保留未融合参考收缩。

核验分为三个层次：

1. 每个原始 Wick 与当前基底重构一致，继续沿用现有 check。
2. 每个 block 数组与其所有带符号组件的直接收缩一致。
3. 全部 blocks 加残余项与原始 signed Wick 总和、当前 reduced result 一致。

普通和流式路径共享块定义与展开规则。流式路径把稀疏映射、组成员和缓存计划写入临时 SQLite，按组处理，不强制生成全部 detailed graph 记录。关闭 blocks 时不支付这些额外成本。

## 8. 实现顺序与验收

1. **结构化表达式与配置。**抽出 contraction IR，新增 normal-only 验证和 BlockOptions；关闭 block 时保持现有命名、轴、系数和输出。运行现有相关测试及默认输出对照。
2. **单 block。**实现任意同味槽对的反对称传播子积及完整三夸克核，支持不同 Gamma/开放槽/槽次序；验证准确符号、端点 p 和数组布局。
3. **多 block 执行规划。**形成交换闭包和展开证书，用来源映射覆盖最终目标；加入唯一核缓存、残余项及内存成本选择。先验证 `pn`，再验证三个/四个 block 的 `2^3/2^4` 展开。
4. **双输出与数值验收。**接入普通及流式 check，共用 IR/planner；随机复传播子、实际 epsilon/delta 核验。覆盖非默认 p、四种传播方向（含未融合因子）、isospin 开关、phase/clebsch、重子与介子混合、零系数、缺失伙伴、重叠候选和内存回退。容差沿用 `1e-10`。
5. **性能与文档。**测量真实 contract 调用数、运行时间、缓存大小和峰值内存。重复核只计算一次应通过实际执行计数验证。完整多核子大例子单独作为可选基准，避免常规测试枚举巨大项集。
6. **块层进一步合并。**基础执行验证通过后，再加入基底视图向量的线性合并；完整 `S_3`、source 侧和更一般的块作为后续扩展。

先完成阶段 1–5，得到可实际复用、可展开检查的通用 normal block；阶段 6 独立评估收益。
