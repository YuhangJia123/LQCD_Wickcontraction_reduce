# 轨道约化与 Wick Reducer：等价性、性能和集成计划

## 结论摘要

当前群轨道脚本与 `wick_reducer` 不在同一等价关系上计数，因此 93/1068 不能直接替代 reducer 报告中的 698/31,092。

- 轨道脚本约化的是离散的味道配对/标签分配：`ppn: 2880 -> 360 -> 93`，`ppnn: 518400 -> 32400 -> 1068`。
- `wick_reducer` 约化的是给定算符下的 peram incidence、合法 transporter 和 sink/source Dirac 收缩张量。记录的 contract 数为 `ppn: 2880 -> 698`、历史 `ppnn: 518400 -> 31,092`。
- 轨道法适合作为候选代表和缓存索引。只有为轨道成员证明完整收缩等式（包括符号、系数和自由轴置换）后，才可以据此少算代表 contract。
- 架构上可输出和现有相同风格的 `contract(...)` Python 文件；目前轨道脚本没有生成输出所需的关系见证，尚不能直接生成正确文件。

因此，研究目标应表述为：验证哪些群轨道关系是特定关联函数的精确张量对称，并把已验证关系接到当前 reducer 的输出接口。目标不是强行令两个现有“独立数”相同。

原 reducer 与 improve 因子图路径的并列流程见
[original_vs_improved_flowchart.md](original_vs_improved_flowchart.md)。

## 1. 两种等价关系

### 离散群轨道

`verify_code/ppnn_group_action.py` 将每个汇端核子内部一对同味夸克的交换先取商。对 `ppnn`，这给出 32,400 个状态；随后加入源/汇端两个质子、两个中子块的交换，以及整体 `p <-> n, u <-> d` 翻转。这个 32 元群把状态分成 1,068 个轨道。对 `ppn`，同一类构造得到 93 个轨道。脚本仅操作标签分配，不含每个成员的 Grassmann 相位、VVV epsilon 相位、Gamma 变换或开放轴转置。

### 收缩张量约化

生产 reducer 从按精确味道产生的 Wick 配对开始，再按 peram profile/incidence 分桶，搜索允许的算符和线 transporter，计算 Wick 与 VVV 符号，对 sink Dirac 张量作比例检查，并对 source/完整 Dirac payload 作 Gram 投影。`isospin_degenerate=True` 让 `u,d` 进入同一个数值 flavor class；在 improve 的显式 p/n 无关模式中，raw Wick 已生成后还会把 p/n 名称从因子图 key 中删除。当前夹具的 VVV 和 Gamma 与重子无关，所以该模式的同位旋特有数值条件就是 `peram_u` 与 `peram_d` 相同；不同 momentum、投影、`external_tag`、Clebsch 系数或额外相位仍会保留在 key 中。算法始终保留输出自由 spin 轴的排列。

所以二者的等价条件不同：一个群轨道最多先提供“这两个图值得尝试映射”的候选关系；reducer 的 Gram 关系则是具体 Gamma、VVV、peram、自由指标定义下的线性张量关系。群关系可能导出带符号/转置的完全相等，也可能因算符标签或动量不对称而不成立。反过来，Gram 线性约化也可能合并不同轨道的线性组合。不存在不附加条件的集合包含关系。

交换重子块还会置换其自由 spin 轴。因此，若把每个外部 spin 标签视为固定有序坐标，轨道成员一般不是代表张量逐元素相同，而可能是代表的带符号转置视图。比较时必须将以下三类数量分开记录：原始 Wick 项数、实际 `contract(...)` 次数、带轴置换的输出 basis view 数。

## 2. 现有基线与速度判断

仓库报告中的基线为：

| 系统/路径 | 原始 Wick | 独立 contract 或轨道 | 时间 | 峰值 RSS |
| --- | ---: | ---: | ---: | ---: |
| `ppn`，生产 reducer 历史单次 | 2,880 | 698 | 0.917 s | 未记录 |
| `ppn`，生产 reducer 三次中位报告 | 2,880 | 698 | 1.500 s | 未记录 |
| `ppnn`，生产 reducer 历史完整运行 | 518,400 | 31,092 | 539.4 s | 1.21 GiB |
| `ppn`，本分支轨道生成+枚举（单独测量） | 360 个商状态 | 93 | 0.008 s | 约 13.3 MiB 进程峰值 RSS |
| `ppnn`，本分支直接群脚本 `analyze_ppnn()` | 32,400 个商状态 | 1,068 | 20.517 s | 约 22.0 MiB |

生产 reducer 的 `ppnn` 数字来自仓库中的历史运行记录；本轮没有重新运行 539.4 秒的完整 reducer。轨道脚本的 ppn 0.008 秒/约 13.3 MiB 和 ppnn 20.517 秒/约 22.0 MiB，分别是独立 Python 进程中执行状态生成与轨道遍历（ppnn 还计算 Burnside 固定点直方图）的实测时间和进程峰值 RSS。脚本没有构造 518,400 个原始 Wick 对应关系，没有运行 Dirac/peram 认证，也没有生成最终 `contract` 输出。因此这些测量说明群计数层很轻，不能当作端到端加速比。

对 `ppn`，当前 reducer 的 2.289 秒实测包括具体算符下的收缩关系约化；93 轨道计数几乎只涉及 360 个商状态。不能据此声称轨道路径已更快地产生等价结果。

如果全部 orbit witness 最终通过认证，并且每个轨道只需一个代表 contract，则理论 contract 调用数可从 698 降至 93（约 7.5 倍），`ppnn` 可从 31,092 降到 1,068（约 29.1 倍）。这些是待验证的潜在收益，不是当前已实现或已证明的收益。带轴置换的 basis views、关系表、代码生成和数值 contract 成本不会按同一比例消失。

## 3. 轨道能否把当前算法约化到 93/1068

组合状态计数已经给出 93 和 1068；完整收缩目前没有。能否把实际 contract 调用数降到这些数，取决于是否能对每个允许生成元证明

`signed_contract(g.w) = coefficient(g,w) * transpose(contract(w), axis_permutation(g,w))`。

`coefficient` 必须包含 Wick 配对符号、VVV 腿换位奇偶、同位旋/算符相位，以及必要的 Gamma 变换系数。`axis_permutation` 必须精确对应开放 sink/source spin 轴的重排。该映射还要保持 peram profile/incidence，或者给出合法的 u/d 线替换规则。improve 的 factor-graph key 只负责前置分桶，不能替代这一步。

目前不能无条件宣称任意算符输入都能达到 93/1068，理由有三：

1. 只有在 VVV、Gamma、momentum、projection 和外部标签满足协变条件，并且 `peram_u == peram_d` 时，p/n 名称才可从因子图 key 删除。
2. 轨道算法仍需 raw-Wick 到代表的双射/规范化、成员到代表的符号、Gamma 和自由轴 witness。
3. 轨道个数本身不证明轨道代表是完整张量空间的线性基底；还需 Dirac/张量级认证。

如果所用 source/sink 插值场、Gamma、动量标签与元素张量具有要求的交换和同位旋不变性，则存在把每轨道压成一个昂贵 contract 加若干符号/转置视图的可能。若有任一算符选择打破对称性，对应生成元应禁用；所得代表数会高于 93/1068。保持 peram 集合不变、同时使 sink 等价的剩余指标对称性，已经由原有 reducer 的线映射、裸轴 transpose、VVV parity、sink projective 检查和 source Gram 层处理；improve 继续复用这一层。

## 4. 最终 contract 输出

可以复用当前输出形式。现有 writer 已将每个 basis 的 `contract(...)`、Wick 符号、裸轴 `.transpose(...)`、合并系数和最终结果分开输出；check 模式还保留逐原始 Wick 的回代关系。

轨道适配层需把每个成员转为类似现有 `dependent_relations` 的数据，至少包含：

- `wick_index` 与 `representative_wick_id`；
- 轨道 action word / transporter witness；
- 原始 Wick sign、sink/source VVV signs、同位旋及算符 phase 的乘积系数；
- 输出裸轴置换；
- 可选的 Gamma/payload 投影系数与数值残差。

writer 可以对代表调用 `contraction_expression()`/`contract(...)`，再从 witness 构造带系数的转置视图。系数不能简单设成 orbit size：轨道内可能有稳定子、不同符号和不同输出轴排列，且累加必须对每个 Wick 项恰好计数一次。`check` 输出应保留全部 raw mapping；它的文件大小和运行量仍随原始 Wick 数增长。

## 5. 时间与空间复杂度

令 `A` 为重子数、`N=prod_f N_f!` 为原始味道 Wick 数，`M=N/2^A` 为内部同味交换后的状态数，`G` 为群阶，`R` 为轨道数，`k` 为每个状态要重标记的夸克标签数（本例至多 12）。`T` 表示 transporter 数，`D` 表示每次 Dirac/Gamma 判断的成本。

### 轨道层

- 状态构造：`O(M k)` 时间。
- 枚举群作用或逐态规范化：`O(G M k)` 时间，`G=4`（ppn）或 `32`（ppnn）为常数。
- 若先从每个原始 Wick 在线映射到状态并累计系数：增加 `O(N k)` 时间。
- 只保存规范代表与每轨道稀疏系数：`O(R k)` 至 `O(R V)` 空间，`V` 是代表涉及的不同轴视图/关系数。
- 若保存全部商状态和未访问集合（当前测试脚本做法）：`O(M k)` 空间；若保存所有 raw-to-representative 映射或启用 check 数据：再增加 `O(N)` 空间。

### 当前张量 reducer

Wick 枚举和 profile 构造至少线性于 `N`。之后的成本取决于 peram bucket 大小、平均候选 transporter 数 `T`、 sink/source payload 和 Gram 基大小。保守地说，当前实现的昂贵阶段随每个 bucket 中需比较的图数乘 transporter/Dirac 判断增长；极端同质 bucket 可接近二次图比较，而真实成本由预筛和短路决定。不能仅从 Wick 数推导精确阶数，宜用 profile bucket 直方图和计时器测量。

### 集成后的额外成本

若轨道映射在 Wick 枚举时流式生成，群 canonicalization 的额外成本为 `O(N k + G M k)`；固定群阶下关于 `N` 仍为线性。理论上可将昂贵 Dirac/peram 认证从每个候选 Wick 移到每个轨道代表/少数 witness，目标成本接近

`O(N k + R T D + N k)`，

其中首尾两个线性项分别用于轨道归类和聚合/写映射。这里尚未包含 reducer 自身跨代表进行 Gram 比较的代价，也不保证每个轨道只需一次认证。若为保证严谨而检查每个轨道成员，则验证阶段仍有 `O(N T D)` 的成本；可通过先逐生成元认证、证明相位表示满足群关系并对代表做抽查，将认证和生产运行分开，但需要形式化/测试证据支撑。

## 6. 实施计划与验收门槛

### 阶段一：固定输入与对称性描述

为 `ppn`、`ppnn` 定义明确的 source/sink 算符 fixture：有序夸克槽、Gamma 名称和作用槽、开放槽、source 共轭约定、VVV/VDV 元素、动量标签。逐个生成元声明是否允许，并给出对 quark fields、baryon blocks、Gamma、VVV、peram 及自由轴的变换。

**验收：** 如果 p/n 插值场、Gamma 或 momentum 不满足对称性，对应生成元会被关闭，不会误合并。

### 阶段二：构造并验证 raw-Wick 到状态映射

实现从精确 `WickTerm.pairing` 到内部商状态的 canonicalization，并保存反向编号。处理 source/sink 标签约定，证明内部 16 重商在 `ppnn` 下无遗漏/重复。

**验收：** `ppn` 的 2,880 个 raw Wick 恰落在 360 个商状态；`ppnn` 的 518,400 个 raw Wick 恰落在 32,400 个商状态；每个商状态有预期重数且 raw ID 可复原。

### 阶段三：输出 orbit representative 和 witness

对每个 raw Wick 计算 canonical representative，同时保存将其映射到代表的 group word、orbit ID 和稳定子信息。不要只保留一个代表编号。

**验收：** group relation 与 `ppn=93`、`ppnn=1068` 组合回归通过；所有 raw Wick 恰分配到一个 orbit。

### 阶段四：收缩级认证

把 witness 转成当前 reducer 使用的数据：`operator_map`、`line_map`、line order、Wick sign、sink/source VVV parity、Gamma payload 比例/残差、peram 对应及 bare-axis map。检查关系属于标量相等、带符号相等、转置相等，还是失败。

**验收：** `ppn` 全 2,880 张图均通过精确符号/Dirac 检查，并用多组随机复 peram 与 VVV 数值回放完整六轴张量；轨道关系与现有 698-basis 输出一致。任何失败成员退回常规 reducer，不丢项。

### 阶段五：加入 `orbit` reduction mode

保留当前 `independent/detailed/check` 行为。新增显式选择的 `orbit` 路径：先按已批准的 symmetry descriptor 分轨道，再以现有 sink/source Gram reducer 认证代表或未认证 orbit witness。失败时保留成员，成功时向现有 reduced-term schema 写入 relation。

**验收：** `ppn` 最终 result 与原始 2,880 项直接和在随机张量回放下相符；能生成与现有结构一致的 `basis_contract`、`basis`、`coefficients`、`result` 和 raw-to-basis check 文件。

### 阶段六：推广到 `ppnn` 与性能评估

加入四核子 fixture；先确认对指定 operator 允许完整 32 元作用，再做 witness 认证和随机数值回放。分别测量 raw enumeration、state mapping、candidate certification、Gram reduction、Python 输出生成时间及 peak RSS。

**验收：** 与完整原始 Wick 求和一致；报告 raw Wick 数、contract calls、basis views、最终系数和残差。若验证后代表数仍大于 1,068，应报告实际物理算符对称子群/线性关系数，而不是强行沿用离散轨道数。

## 7. 论文依据

- Yamazaki、Kuramashi、Ukawa（PACS-CS），[arXiv:0912.1383v1](https://arxiv.org/abs/0912.1383v1)，第 2 页报告 He 与 `^3He` 分别保留 1,107 和 93 个 contractions，并称计算机筛选后的结果在一个构型上与完整结果一致。论文描述利用质子/中子置换及 He 的同位旋对称性，但没有给出可直接复现 1,107 的完整生成元和成员映射清单。
- Doi、Endres，[arXiv:1205.0585](https://arxiv.org/abs/1205.0585)，提出同时处理 Wick 置换和 color/spinor 收缩的统一索引算法；摘要给出不假设同位旋时的降幅，并指出在存在相应对称性时还可利用同位旋及汇端重子互换。它支持“必须把组合关系和张量收缩一起处理”的判断，但不能单独证明本分支的 1,068 轨道就是论文中的 1,107 缩并数。
- 本分支显式定义的 ppnn 标签群得到 1,068 个轨道，与上述论文报告的 1,107 不相同；`ppn=93` 只能作为计数校准，不能替代 ppnn 收缩级复现。

## 参考实现

- 群作用枚举：[ppnn_group_action.py](ppnn_group_action.py)
- 组合轨道说明：[ppnn_group_action.md](ppnn_group_action.md)
- 当前 reducer 算法：[Wick reducer 实现](../docs/implementation/wick.md)
- contract 文件格式：[Wick 输出格式](../docs/reference/wick-output.md)
