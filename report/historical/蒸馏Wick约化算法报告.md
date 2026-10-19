# 蒸馏 LQCD Wick 约化算法报告

## 0. 阅读说明

本文把“要得到什么结果的数学算法”和“当前代码怎样完成这件事的工程实现”严格分开。

- **算法**回答：哪些 Wick 收缩可以视为同一个约化对象，哪些必须保留为独立对象，以及系数怎样累加。
- **例子**帮助读者把抽象对象对应到具体指标。
- **当前实现**回答：代码使用哪些数据结构、键、缓存和筛选顺序。
- **实现原因**只解释速度、内存和可维护性，不改变算法定义。

本文讨论有裸 Dirac 指标的蒸馏 LQCD 关联函数。输入是一个有序二元组：

~~~text
[O1_sink, O2_source]
~~~

其中 O1 是 sink 算符，O2 是 source 算符。默认同位旋模式开启时，u、d 的 peram 可以共用等价类；s 等其他味道仍然只和自身匹配。Wick 枚举阶段仍然严格区分实际味道。

---

## 第一部分：算法

### 1. 总体目标

#### 算法

给定一个 sink/source 算符对，算法完成以下工作：

1. 为算符中的每个夸克、VVV/VDV 张量和 Gamma 矩阵分配可追踪的槽位。
2. 按费米场收缩规则枚举所有允许的 Wick 配对，并保留每个图的费米符号。
3. 把每个 Wick 图写成：
   - Gamma 矩阵；
   - sink/source 的 VVV 或 VDV 张量；
   - 带方向、味道和裸指标的 peram。
4. 先按 peram 的粗分类分组。
5. 在每个粗分类中，把可以通过内部指标重命名联系起来的图规范到同一代表。
6. 对规范化后的图先比较 sink VVV/VDV 和 sink Dirac 结构，再比较 source 指标。
7. 用 source incidence 和 source Dirac 内积判断独立性。重复图的系数相加；线性相关图可以按细分模式保留。
8. 输出独立 basis、每个原始 Wick 图到 basis 的换算关系，以及每个 basis 的系数。

约化的核心顺序是：

~~~text
Wick 枚举
  -> peram 粗分类
  -> peram 重命名与必要的裸指标转制
  -> sink VVV/VDV 与 sink Dirac 判定
  -> source incidence 与 source Dirac 判定
  -> basis、系数和原始图映射
~~~

#### 例子

如果两个图只差一个哑指标名字，它们属于同一个候选等价类。若两个图的左侧 peram 完全相同、右侧裸指标相差一个交换，则左侧可以共用一次计算，但右侧必须通过转制记录差异。

#### 当前实现

主流程通过 `reduce_operators(operators, label=..., options=...)` 调用；内部实现位于 `kernels/wick_reducer/reducer.py`。算符和 Wick 原语位于 `operator_model.py` 与 `kernels/correlator_wick_normalizer.py`，输出位于 `expression_output.py`。

#### 实现原因

按层次筛选能让大多数不可能等价的图尽早退出，避免对所有图都直接构造完整的 sink/source Dirac 比较。不同粗分类之间互不影响，因此算法天然适合并行。

---

### 2. 初始条件与输入

#### 算法

输入必须明确以下信息：

1. 有序算符对 [O1_sink, O2_source]。
2. 每个算符的场列表：夸克、反夸克、味道以及场所在端点。
3. 每个 VVV/VDV 张量的腿槽位。
4. 每个 Gamma 的两个槽位及其具体矩阵。
5. 裸指标和哑指标的角色。
6. 是否开启同位旋等价：
   - 开启：u、d 的 peram 可进入同一等价类；
   - 关闭：u、d 也必须分别匹配。
7. 输出模式：
   - independent：只输出独立 basis；
   - check：同时输出原始 Wick、basis 和换算关系；
   - 细分模式：保留线性相关但不独立的图。

同一个算符中出现两个相同粒子并不意味着它们可以共用槽位；它们必须有不同的场、VVV 和 Gamma 槽位编号，之后算法才可以判断它们是否通过指标对称性重新联系。

#### 例子

ppn 关联函数可写成：

~~~text
[ppn_sink, ppn_source]
~~~

其中每个 p、n 都是独立的 VVV 槽位。每条传播子可以带有类似：

~~~text
peram_u_sin_sou[uaiA]
~~~

的结构。u、a 是 VVV/哑指标，i、A 是裸指标；具体字母只是展示，等价判定不依赖某个固定字母名字。

Gamma 可以直接输入具体的 4×4 矩阵。例如 C_g5、C_g5_dagger 或用户给出的 γμγ5 数值矩阵都可以作为 Gamma 数据。

#### 当前实现

operator_model.py 提供算符构造、字段收集、味道类别和内置算符库。kernels/correlator_wick_normalizer.py 将场、Gamma、VVV/VDV 和裸指标统一成 Wick 数据结构。source 算符是否为 dagger 或共轭由输入决定，代码不会自动替换用户指定的槽位顺序。

#### 实现原因

把“算符如何写”与“如何约化”解耦，可以支持任意可被成功 Wick 收缩的介子和重子组合，也避免内置粒子名称限制后续扩展。

---

### 3. Wick 收缩枚举

#### 算法

对 sink 和 source 的费米场分别建立有向端点。只允许满足味道、夸克/反夸克方向和端点规则的配对。每个完整配对产生一个 Wick 图，并计算费米置换符号。

枚举结果必须保留：

- 每条传播子的方向；
- 每条传播子的味道；
- 对应的 sink/source 场槽位；
- 裸指标连接；
- 哑指标连接；
- VVV/VDV 槽位连接；
- Gamma 槽位连接；
- Wick 符号。

不同味道在枚举时始终不同。只有在之后进行 peram 等价判定时，才根据同位旋设置允许 u/d 共用类别。

#### 例子

若 sink 中有两个 u、一个 d，source 中也有对应的反夸克，则每个合法的全配对都产生一个图。交换两个相同味道的配对通常会引入费米符号；交换不同味道不能因为同位旋模式而在枚举阶段合并。

#### 当前实现

enumerate_wicks 逐个生成配对，并通过 wick_sign 计算费米置换符号。生成过程使用字段槽位而不是只使用粒子名称，因此两个 p 的场不会被错误地当成同一个槽位。

#### 实现原因

Wick 图数在重子多粒子关联函数中可能快速增长。使用槽位化的流式枚举可以避免提前保存不需要的中间张量；费米符号在图生成时确定，后续约化只处理结构信息。

---

### 4. Peram 分类

#### 算法

把每条 peram 按以下信息分类：

1. 味道类别；
2. 方向类型：
   - sin_sin
   - sin_sou
   - sou_sin
   - sou_sou
3. 两端裸指标的角色：
   - 两端都是裸指标；
   - 只有 sink 端是裸指标；
   - 只有 source 端是裸指标；
   - 两端都是哑指标；
4. 传播子的 quark/antiquark 方向；
5. sink/source 场槽位连接；
6. VVV/VDV 和 Gamma 所属端点。

分类只产生候选桶，不直接宣布等价。进入同一桶只表示它们有可能通过允许的指标重命名联系起来。

#### 例子

~~~text
peram_u_sin_sou[uaiA]
~~~

可以读成：

- 味道：u；
- 方向：sin_sou；
- sink 端：u、a；
- source 端：i、A；
- 裸/哑角色：由具体指标的角色表确定。

把哑指标 a 改名为 b，不会改变桶：

~~~text
peram_u_sin_sou[ubiA]
~~~

但把 sin_sou 改成 sou_sin，或把 source 裸指标移到 sink 端，必须进入不同候选桶。两个图的 peram 顺序和端点连接也必须兼容。

#### 当前实现

peram_incidence.py 产生带颜色的有向 incidence key。颜色包含味道类别、方向、裸/哑边界和端点信息；VVV 的连接关系用图的边和节点编码。reducer.py 先按 profile、incidence key 和裸指标角色进行分桶。

同位旋模式下，flavor_class 把 u、d 归到同一个等价类别；其他味道保留自身类别。因此：

~~~text
u -> light
d -> light
s -> s
~~~

#### 实现原因

候选桶可以在最早阶段排除方向和端点明显不匹配的图。把 peram 的关系表示为带颜色有向图，可以在不枚举所有字母置换的情况下快速测试 incidence 是否一致；具体指标名字不进入第一阶段 key，所以不会冻结裸指标字母。

---

### 5. Peram 规范化与 transporter

#### 算法

对每个候选桶选择第一个图作为基准。后续图尝试通过合法 transporter 重命名指标，使不同尽可能放到 source 端。

允许的操作包括：

1. 重命名哑 VVV 指标；
2. 在保持传播子端点和味道类别的前提下重命名裸指标；
3. 在裸指标被交换后记录对应的转制；
4. 交换具有相同 VVV/VDV 类型的腿，但不改变它们绑定的 Gamma 槽位；
5. 在同位旋模式下允许 u/d 作为同一 peram 味道类别。

规范化目标不是强行固定某些字母，而是把两个图变成同一个有向 peram 关系；若无法把差异全部放到 source，就保留导致 sink 变化的 transporter，交给下一层 sink 判定。

#### 例子

基准图：

~~~text
uAiD, vBjE, wAkF, xCld, ybme, zcnf --- ABCDEF
~~~

另一个图：

~~~text
uAjE, vBiD, wAkF, xCld, ybme, zcnf --- ABCDEF
~~~

通过交换 D、E 相关的绑定关系，可以把第二个图规范到：

~~~text
uAiD, vBjE, wAkF, xCld, ybme, zcnf --- ABCEDF
~~~

左侧传播子形式可以相同，但右侧裸指标顺序不同，所以输出时必须保留转制：

~~~text
B = A.transpose(0, 1, 2, 4, 3, 5)
~~~

另一个例子：

~~~text
uAiD, vBjE, wAkF, xCme, ybld, zcnf --- ABCDEF
~~~

只需要交换 source 端的 me 和 ld，即可与基准使用同一个 peram 代表。

#### 当前实现

reducer.py 使用 _operator_maps 生成合法的算符槽位映射，_line_candidates 先做味道和端点候选筛选，_iter_line_maps 逐步生成 transporter。_bare_axis_map 记录裸轴映射，并在发生裸指标交换时产生 transpose 信息。_transport_state 应用完整映射，生成规范化的 peram 集合、VVV 连接和 sink/source 指标。

当前代码把 u/d 的兼容性放在候选匹配中检查，并且把味道类别放进 profile/incidence key。这是当前等价关系的一部分；它不是只针对 ppn 的特殊分支。

#### 实现原因

逐条生成 transporter 并在每一项不同处提前退出，避免为每个候选图无条件构造完整 payload。裸轴映射单独保存，既能支持广义等价，又能保证输出中不丢失转制关系。

---

### 6. Sink 指标规范化与多 transporter

#### 算法

peram 规范化后，先比较 sink 端，因为 sink 不同的图不能仅靠 source 重命名变成同一个图。

sink 判定包含：

1. VVV/VDV 的类型和动量标签；
2. 每个 VVV/VDV 的腿连接顺序；
3. VVV/VDV 与 Gamma 槽位的绑定；
4. Gamma 的具体矩阵；
5. sink 裸指标和哑指标的有序结构；
6. 允许的腿置换及其符号；
7. transporter 作用后得到的 sink Dirac payload。

VVV 结构相同只是候选条件。最终是否相同必须检查与 Gamma 绑定后的有序 Dirac 结构。若一个候选图存在多个允许 transporter，不能只取第一个；必须把所有可能的 sink 解作为候选，再判断它们与基准的 Dirac 关系。

#### 例子

基准 sink VVV 和 Gamma 槽位为：

~~~text
[uvw][xyz], Gamma = [ga, gb]
~~~

某个规范化图得到：

~~~text
[vuw][xyz]
~~~

如果这只是同一组三条腿的置换，它在 VVV 拓扑上可能与基准相容；但要继续检查 Gamma 绑定的 Dirac 内积。

例如两个可能解的 sink Dirac 部分分别为：

~~~text
D1 = contract(ab, cd, ba, cd, ga, gb, ga*, gb*)
D2 = contract(ab, cd, dc, ab, ga, gb, ga*, gb*)
~~~

判定规则：

- D1、D2 都为零：两个解都不能与基准相同；
- 只有一个非零：只有对应 transporter 可以继续进入 source 判定；
- 两个都非零：两个 transporter 都是允许解，不能因为顺序不同就直接拆开；它们可能通过指标重命名把 source 结构联系起来；
- 多于两个解时，先找 sink Dirac 的独立方向，再与基准比较，而不是逐个把所有 source 结构展开。

ga == gb 不是充分条件，ga != gb 也不是充分条件。真正的判定是具体矩阵产生的 Dirac payload 或内积。即使两个 Gamma 相同，某个 VVV 置换引起的转制也可能使内积为零；即使 Gamma 不同，也可能存在另一个排列使内积非零。

#### 当前实现

_sink_orbit_diagnostics 计算当前 sink 类的参考结构，_projective_sink_match 测试候选 payload 与参考 payload 的比例关系。_payload_key 为具体 payload 建立稳定键，_cached_dirac_inner 缓存已经计算过的 Dirac 内积。_reduce_bucket 会让候选图依次尝试已有 sink class；没有匹配时才创建新的 sink class。

当前实现遵循短路顺序：

~~~text
VVV/VDV 类型
  -> 腿和端点连接
  -> Gamma 槽位
  -> 具体 sink Dirac payload
  -> source
~~~

未来若需要完全一般的多 transporter sink Dirac 判定，应将一个候选图的所有 transporter 组成有序 payload 集合，先做集合级的 rank/投影比较，再把可行 transporter交给 source 层。

#### 实现原因

sink payload 是后续最昂贵比较之一，因此必须在 VVV 和 Gamma 槽位不一致时立即丢弃。缓存只保存已经遇到的 payload/内积，避免重复做同一个具体矩阵计算。多 transporter 的处理放在 sink 层，可以避免把不可能的候选送入 source Gram 矩阵。

---

### 7. Source 指标验证与独立性

#### 算法

只有在 peram 类别和 sink 类别都相同后，才判断 source 是否独立。source 由两部分组成：

1. source VVV/VDV 的 incidence 结构；
2. source Gamma/裸 Dirac 指标的数值结构。

先比较 source incidence。incidence 不同表示哑指标图结构不同，不能直接合并。incidence 相同后，把 source Dirac 结构视为向量，使用内积判断是否已有线性关系。

设已有 basis 向量为 b1,...,bk，候选向量为 x：

- 若 x 对 incidence 或已有 Dirac 子空间不匹配，保留为新的 basis；
- 若 x 对已有子空间的残差为零，则它是依赖项，系数并入已有 basis；
- 若残差非零，则新增一个独立 basis 方向。

Gram 矩阵和 Gram-Schmidt 是两种等价的实现方式：

- Gram：一次生成所有内积，适合小组；
- Gram-Schmidt：逐个投影并保留残差，适合流式处理和大组。

sink 多 transporter 造成的“看似不同 source、实际可由同一 transporter联系起来”的情况，也必须在 source 独立性之前合并。只有在 sink transporter 和 Dirac 条件都允许后，source 才能参与 Gram 判定。

#### 例子

两个图的 source VVV incidence 相同，但 source Dirac 顺序不同。若它们的内积为零，则它们在当前表示中是独立的；若某个 sink transporter 可以同时保持 peram 集合和 sink 集合不变，并把一个 source 顺序变成另一个，则这两个图仍应视为同一个物理约化对象。

#### 当前实现

_streaming_gram_projection 使用流式正交投影维护 orthonormal_rows，在新候选到达时计算残差。_gram_projection 用于需要一次性诊断一组 sink 方向的场景。_state_inner 负责 source Dirac 与 incidence 结构的内积/兼容性计算。_reduce_bucket 保存每个 basis 的系数和 witness transporter。

#### 实现原因

流式投影不需要保存整个组的完整数值矩阵，能降低大 N 情况下的峰值内存。先做 incidence 再做 Dirac 内积，可以把大量零内积和结构不匹配的候选排除在昂贵计算之外。

---

### 8. 指标对称性与裸指标转制

#### 算法

指标对称性允许在不改变 peram 集合的情况下，对同类 VVV/VDV 腿和绑定的 Gamma 槽位同时重命名。判断对称性时必须同时考虑：

1. VVV/VDV 腿的排列；
2. 绑定的 Gamma 槽位；
3. peram 集合是否保持；
4. sink VVV/VDV 是否保持；
5. 具体 Dirac 矩阵内积是否非零；
6. source incidence 和 source Dirac 是否因此变成同一个对象。

裸指标不是普通哑指标。若一个 transporter 交换了裸轴，必须保留该交换对最终关联函数的转制。左侧 peram 文本相同并不意味着完整关联函数相同。

#### 例子

若两个结果的左侧相同，但右侧为：

~~~text
ABCDEF
ABCEDF
~~~

则它们可以共享左侧 peram 计算，但必须输出：

~~~text
B = A.transpose(0, 1, 2, 4, 3, 5)
~~~

若两个相同动量的 VVV 结构只差腿置换，则是否合并取决于 Gamma 绑定和 Dirac 内积，而不是只看 VVV 名称。

#### 当前实现

_bare_axis_map 记录裸轴置换并传给输出层。sink/source payload 的 key 中保留有序槽位信息，因此同一个 VVV 名称不会掩盖不同的 Gamma 绑定。expression_output.py 把 transpose 信息写入 independent/check 文本。

#### 实现原因

把裸轴交换单独记录可以同时满足两点：减少重复的 peram 左侧计算，并保持完整关联函数的数学正确性。把 Gamma 槽位纳入 sink key 可避免把仅仅因为 ga == gb 而产生的错误合并。

---

### 9. 算法输出

#### 算法

independent 模式输出：

1. 原始 Wick 编号；
2. 独立 basis contract；
3. 每个 basis 的约化系数；
4. 裸指标交换对应的 transpose；
5. 需要时的 source/sink VVV、VDV 和 Gamma 标识。

check 模式额外输出：

1. 每个原始 Wick 的完整表达式；
2. 它对应的 basis 编号；
3. 从 basis 得到原始 Wick 的系数和 transpose；
4. basis contract 与实际 basis 张量的关系。

换算关系应能直接读成：

~~~text
original_wick[0] = coefficient * basis[2]
original_wick[1] = coefficient * basis[2].transpose(...)
~~~

同一左侧但不同裸指标右侧的图，不能只输出同一个字符串；必须明确转制。

#### 例子

可以把输出组织为：

~~~text
ppn_basis_contract[0] = contract(...)
ppn_basis[0] = ppn_real_basis[0]

original_wick[0] = 1 * ppn_basis[0]
original_wick[1] = 1 * ppn_basis[0].transpose(0, 1, 2, 4, 3, 5)
~~~

如果多个原始 Wick 完全合并到同一个 basis，则 basis 的系数是它们系数的代数和。

#### 当前实现

expression_output.py 负责生成 compact/independent/check 文本。代码将原始图、basis contract、basis 系数和 transpose 关系分开写出，便于后续 GPU 计算时只计算一次公共左侧，再按关系恢复每个原始图。

#### 实现原因

输出格式同时服务于两种用途：

- independent：减少实际需要计算的 contract 数量；
- check：用具体 peram 数值验证原始 Wick 与 basis 的等式。

---

### 10. 算法小结

#### 算法

对于任意可完成 Wick 收缩的算符对，约化判定可以概括为：

1. 精确枚举味道分辨的 Wick 图；
2. 按 peram 的味道、方向、裸/哑边界和端点连接建立候选桶；
3. 以第一个图为基准生成合法 transporter；
4. 尽可能把差异移到 source，同时保存裸轴转制；
5. 按 VVV/VDV、Gamma 绑定和具体 sink Dirac payload 判定 sink；
6. 对 sink 兼容的图比较 source incidence；
7. 用 Gram 或 Gram-Schmidt 判断 source Dirac 线性独立性；
8. 输出 basis、系数、transpose 和原始图映射。

#### 例子

如果某个候选在 peram 桶阶段不匹配，它不需要进入 sink Gamma；如果 sink payload 与所有已有类都正交，它不需要进入 source Gram；只有通过前面所有层的图才会竞争同一个 basis。

#### 当前实现

当前代码将每一层拆成可替换模块，并保持 `reduce_operators` 作为统一公开入口。`operator_model.py`、`kernels/correlator_wick_normalizer.py`、`peram_incidence.py`、`reducer.py` 和 `expression_output.py` 分别承担输入、枚举、分类、约化和输出职责。

#### 实现原因

模块边界使任意新算符只需提供字段、VVV/VDV、Gamma 和指标连接，而不需要重写约化逻辑。

---

## 第二部分：当前代码实现

### 11. 算符、场和 Gamma 数据结构

#### 算法

算符是带槽位的场和张量集合。Gamma 是作用于两个 Dirac 槽位的具体矩阵；Gamma 的名称只是标签，矩阵数值才决定 Dirac 内积。

#### 例子

~~~text
Gamma(name="C_g5", matrix=4x4)
Field(flavor="u", endpoint="sink", role="quark", slot=0)
VVV(name="p", momentum=0, legs=(u,v,w))
~~~

#### 当前实现

operator_model.py 的 make_operator、fields_for_operators 和 operator_library 建立算符。kernels/correlator_wick_normalizer.py 定义 Operator、Field、Gamma、VVV/VDV 和 Wick 相关数据结构。Gamma payload 只接受两个 Gamma 槽位；γμγ5 这类对象应先作为具体 4×4 矩阵输入。

#### 实现原因

固定的两个槽位表示覆盖常用 CΓ 结构，同时保留未来扩展为多端口 spin tensor 的边界。

---

### 12. Wick 与 peram 结构

#### 算法

每条传播子必须保存它的起点、终点、味道、方向以及四类指标关系。指标的角色表区分裸指标和哑指标，不能通过字母大小写猜测物理角色。

#### 例子

~~~text
peram_u_sin_sou[uaiA]
~~~

由结构字段解释，而不是由字符串字面量解释：

~~~text
flavor=u
kind=sin_sou
sink_axes=(u,a)
source_axes=(i,A)
~~~

#### 当前实现

Wick 图保存 lines、gamma_payload、vvv_payload、bare_axes、sign 和端点槽位。peram profile 用结构化元组表示，输出层再把它格式化成文本。

#### 实现原因

结构化表示避免在大规模枚举中反复解析字符串，也能保证不同字母但相同角色的指标进入同一候选桶。

---

### 13. Peram 图论分桶的实现

#### 算法

把每条 peram 看成带颜色的有向边：

- 节点表示算符端点、VVV/VDV 腿或 Gamma 槽位；
- 边表示传播子；
- 边颜色包含味道类别、方向和裸/哑角色；
- 有向连接表示 sink/source 边界。

两组边只有在带颜色的有向 incidence 同构时，才进入同一后续候选比较。

#### 例子

~~~text
peram_u_sin_sou[uaiA]
peram_d_sin_sou[vbjB]
~~~

在同位旋模式开启时，u/d 的颜色都可标为 light；s 仍标为 s。改名 a -> c 只改变标签，不改变图结构。

#### 当前实现

peram_incidence.py 生成 incidence key；reducer 根据 profile 和 key 建立桶。具体图同构使用算符映射和端点约束，而不是把全部指标置换暴力枚举。

#### 实现原因

图论 key 将大量名字置换压缩为一个短结构 key，降低候选数量和内存占用。由于颜色中保留方向和端点，图论分桶不会偷偷交换 sink/source。

---

### 14. Transporter 搜索的实现

#### 算法

对候选图，按字段优先级逐步检查：

1. 味道类别；
2. peram 方向；
3. sink/source 端点；
4. VVV/VDV 连接；
5. 裸/哑角色；
6. 合法的指标映射。

任意一步不一致就停止该候选。通过的候选才生成完整 TransportState。

#### 例子

两个图只有 source 哑指标名字不同，前五项都相同，最后生成一个 dummy rename；若第二个图把 sink 裸轴与 source 裸轴交换，角色检查失败，不能仅靠改名合并。

#### 当前实现

_line_candidates 生成每条线的候选，_iter_line_maps 组合合法映射，_transport_state 才物化完整状态。_operator_maps 提供 VVV/VDV 腿和 Gamma 槽位的允许映射。

#### 实现原因

这是“逐项比较、不同即丢弃”的实现位置，避免每条 Wick 图无条件执行完整的六类 canonicalization。

---

### 15. Sink payload、缓存和短路

#### 算法

sink 的最终等价判定使用有序 payload，而不是只使用 VVV 名称或 ga == gb。payload 中应含 VVV/VDV 腿顺序、Gamma 槽位、具体矩阵作用和裸轴排列。

#### 例子

~~~text
[uvw][xyz] + [ga, gb]
[vuw][xyz] + [ga, gb]
~~~

两者先作为同一 VVV 轨道的候选，随后由具体 Dirac payload 决定是否与基准相同或成比例。

#### 当前实现

_payload_key 保存 payload 元数据，_cached_dirac_inner 缓存重复内积；_projective_sink_match 处理比例关系。当前代码在 _transport_state 中按需构造 sink payload，并缓存已经遇到的结果；“预先为所有 Wick 生成 sink payload”仍属于未来优化，不是当前行为。

#### 实现原因

按需构造能控制峰值内存；缓存避免相同 Gamma/指标结构被重复计算。未来可以在每个 peram 桶内预计算 sink payload，但必须保持短路顺序和内存上限。

---

### 16. Source Gram/Gram-Schmidt 的实现

#### 算法

source incidence 相同后，使用具体 Dirac 矩阵计算内积。给定候选向量 x 和当前正交基 q：

~~~text
projection = <q, x>
residual = x - projection * q
~~~

残差低于容差则依赖，否则加入新基。

#### 例子

若两个 source Dirac 向量内积为零，它们在当前 sink/peram 类中是独立方向；若经过 sink transporter 后两个 source payload 完全一致，则它们应使用同一个方向和同一个系数。

#### 当前实现

_streaming_gram_projection 按候选到达顺序维护正交行；_gram_projection 支持组内批量诊断；_state_inner 计算具体内积；_reduce_bucket 保存 basis witness、系数和原始图映射。

#### 实现原因

流式 Gram-Schmidt 不要求保存整个组的密集 Gram 矩阵，适用于 ppn、ppnn 等大 Wick 数目的情况。若某个 sink 组很小，批量 Gram 可以减少 Python 层循环。

---

### 17. 输出文件的实现

#### 算法

输出必须能同时表达“公共 basis”和“原始图如何恢复”。因此 basis contract、系数、transpose 和原始图映射是四个不同概念。

#### 例子

~~~text
ppn_basis_contract[0] = contract(...)
ppn_basis[0] = ppn_real_basis[0]

original_wick[0] = 1 * ppn_basis[0]
original_wick[1] = 1 * ppn_basis[0].transpose(0, 1, 2, 4, 3, 5)
~~~

#### 当前实现

expression_output.py 生成 independent 和 check 两种文档。check 模式先写原始 Wick，再写 basis 和关系；独立模式只写 basis 及约化系数。长数组形式的系数仍保留，便于 GPU 端直接读取。

#### 实现原因

文档格式既能人工检查指标，又能作为后续数值计算的输入说明。把 transpose 显式写出，可以验证裸指标互换没有被错误地当成普通哑指标改名。

---

### 18. 复杂度、内存与并行

#### 算法

算法的高层复杂度由三部分决定：

1. Wick 枚举数量；
2. 同一 peram/sink 类中的 transporter 候选数量；
3. source Dirac 独立方向数量。

不同粗分类之间没有相互依赖，source Gram 只在同一 sink/peram 类内进行。

#### 例子

如果一个关联函数产生四个互不相同的 peram profile，则四个 profile 可以分别完成规范化和 source 判定，最后只需拼接输出。

#### 当前实现

当前代码在桶内流式处理，缓存 payload 和内积，使用短路候选筛选。README 中记录的未来方向包括：

- 并行处理不同 peram/sink 桶；
- 预计算并缓存重复 sink payload；
- 对极大桶使用稀疏或分块 Gram；
- 将纯结构 key 与数值 Dirac 计算分离。

#### 实现原因

桶之间互不写共享 basis，因此可以使用进程并行或任务并行。需要注意每个 worker 的缓存和输出合并，避免复制过大的 Wick 列表。

---

### 19. 验证与边界

#### 算法

验证必须分别检查：

1. 原始 Wick 的费米符号；
2. peram profile 与 incidence 分类；
3. transporter 后的裸轴映射；
4. sink payload 的非零/零关系；
5. source Gram 的独立性；
6. original-to-basis 等式；
7. 同位旋开关前后的 flavor 等价变化。

#### 例子

ppn 的 check 输出可以使用随机 peram 数组和反对称 VVV 测试：

~~~text
raw_wick = sum(coeff_i * basis_i)
~~~

再比较每个数组元素或整体数值误差。如果原始图只差裸轴转制，数值关系应在应用对应 transpose 后成立。

#### 当前实现

项目测试位于 `tests/`，测试命令为：

~~~text
python3 -B -m unittest discover -s tests -t . -p 'test_*.py'
~~~

ppn 等价性报告位于 ppn等价性初步报告.md，README 中保留了七个示例及 ppnn/ppnnpi 的历史计时信息。

#### 实现原因

结构测试保证约化规则没有被输出格式改变；数值 check 则验证实际 GPU/CPU peram 代入后的等式。两者分开可以快速定位是指标结构错误还是矩阵数值错误。

---

### 20. 未来的一般化方向

#### 算法

更一般的 sink Dirac 多 transporter 判定需要把“一个 Wick 图可能有多个 sink 解”作为一个整体处理：

1. 对候选图收集所有合法 transporter；
2. 为每个 transporter 生成有序 sink Dirac payload；
3. 计算这些 payload 之间的独立方向；
4. 先与当前 sink basis 做投影；
5. 只把非零且 sink 兼容的 transporter 送入 source 层；
6. 对保持 peram 集合和 sink 集合不变的 transporter 建立 witness，合并由它联系起来的 source 结构。

#### 例子

ga == gb 时两个 Gamma 标签可能不可区分，但这并不能直接合并所有 VVV 排列；某个排列引起的 Gamma/Dirac 转制仍可能使内积为零。ga != gb 时也不能直接判定不同，因为具体矩阵可能产生非零比例关系。

#### 当前实现

当前代码已经使用具体矩阵的 payload/内积，并且不会把 ga == gb 作为唯一判据。README 的“未来的修正方向”记录了更一般的多 transporter 轨道级 rank/投影算法；该算法尚未替换当前实现。

#### 实现原因

常用的 C_g 矩阵通常具有良好的正交性、对称性或反对称性，因此目前很多情形中检查具体矩阵相等或非零关系已经足够。若未来使用更一般的 spin tensor、非标准 Gamma 组合或大量多解 transporter，应使用上面的轨道级 rank 方法。不同类别之间天然独立，也适合并行计算。

---

### 21. 结论

本报告的算法定义是：

~~~text
味道分辨的 Wick 枚举
  -> peram 候选分类
  -> transporter 规范化
  -> sink VVV/VDV 与具体 Dirac 判定
  -> source incidence 与 Gram 独立性
  -> basis、系数、transpose 和原始图映射
~~~

peram_u_sin_sou[uaiA]、图论 incidence 分桶、payload 缓存、zero-overlap 短路和流式 Gram-Schmidt 都是实现手段；它们不改变裸指标、VVV/VDV、Gamma、同位旋和多 transporter 的等价判据。
