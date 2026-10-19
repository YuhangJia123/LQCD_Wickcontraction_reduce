# Wick 收缩的群作用与当前 Reducer 对照

本文记录在当前 `ppn` 算符定义下，对 $^{3}\mathrm{He}$ 的 Wick 配对、拟议的 93 个标签轨道，以及项目 reducer 输出的逐项核对。另列出同一群作用在 `ppnn` 上的计数，避免把组合轨道数和具体张量收缩的独立数混为一谈。

## 1. 组合状态与轨道

对于 $^{3}\mathrm{He}=ppn$，共有 5 个 $u$ 夸克和 4 个 $d$ 夸克。因此按 flavor 分组的 raw Wick 配对数为

\[
N_{\mathrm{Wick}}=5!\,4!=2880.
\]

在离散标签层面，将每个质子汇端的两个 $u$ 配对视为无序对，并将中子汇端的两个 $d$ 配对视为无序对，得到 $2^3$ 个排列一类的商：

\[
2880/2^3=360.
\]

这 360 个状态再受两个质子块交换生成元作用：交换汇端的两个质子，以及交换源端的两个质子。生成群为

\[
G\cong C_2\times C_2,\qquad |G|=4.
\]

直接枚举得到 93 个轨道，其中 6 个大小为 2，87 个大小为 4：

\[
6\cdot2+87\cdot4=360.
\]

所以“93”首先是指定标签空间和这两个质子块置换下的组合轨道数。每个轨道选择一个 Wick 配对，可以产生 93 条候选 `contract(...)` 调用；轨道关系本身尚未给出张量恒等式的系数、符号和自由轴置换。

## 2. 与生产 Reducer 的 698 项比较

当前项目对 ppn 的详细 reducer 报告把 2,880 项分成 698 个独立 basis 项和 2,182 个依赖项。把 698 个 basis Wick ID 按上述 93 个轨道归类，得到：

| 每轨道的 reducer basis 数 | 轨道数 |
| ---: | ---: |
| 2 | 1 |
| 4 | 2 |
| 6 | 14 |
| 7 | 4 |
| 8 | 72 |
| **总数** | **93** |

加权总数为 698。每个轨道至少含有一个生产 basis 元素，因此可以从每轨道选出一个“真实且独立”的 reducer Wick 作为候选代表；这只证明 93 个候选自身都在当前 698 项 basis 中，不证明原 2,880 项总和能由这 93 项重构。

候选列表在 [ppn_orbit_candidate_contracts.py](ppn_orbit_candidate_contracts.py)，交叉表在 [ppn_orbit_reducer_crosswalk.md](ppn_orbit_reducer_crosswalk.md)。生成器保留 Wick ID、轨道成员和 Wick 符号；候选文件刻意不写入未经证明的轨道权重或总和。

## 3. 何谓已验证的轨道代表

令 $C_w$ 表示 Wick 配对 $w$ 对应的完整六开放自旋指标收缩张量，$\epsilon_w$ 为该项 Wick 符号。若要把轨道代表 $r$ 用作完整计算的代表，必须对轨道内每个 $w$ 给出并验证

\[
\epsilon_w C_w=\kappa_w\,P_w(\epsilon_r C_r),
\]

其中 \(\kappa_w\) 必须包含 Grassmann/Wick、color epsilon、Gamma 与算符约定所产生的系数或相位，\(P_w\) 是自由 spin 轴置换。若关系不是单一张量的带符号置换，而是线性组合，也必须显式记录其系数。轴置换后的张量通常不与代表逐元素相同，因此也不能只把代表乘以轨道大小。

当前 ppn fixture 中，两端的两个质子算符分别相同，具有相同 flavor、Gamma、open slot 和默认外部标签。因此质子块的整块交换在这个 fixture 中是候选精确协变；更换算符定义或动量标签后须重新检查。三个内部同 flavor 槽位换序所产生的 $2^3$ 商还涉及固定的 Gamma 槽位和开放轴，不能仅凭无序标签证明收缩恒等式。

## 4. 数值和指标结构审计

每项 `contract(...)` 返回完整张量，shape 为 `(4,4,4,4,4,4)`，输出轴为 `ABCDEF`。数值审计直接计算带 Wick 符号的 2,880 项总和，并与生产 reducer 的带系数、带轴置换重构比较；同时检验完整总和是否落入 93 个候选张量的张成空间。随机测试是对当前 fixture 的反例检验，不单独构成任意算符下的代数证明。

数值审计报告见 [ppn_orbit_numeric_audit.md](ppn_orbit_numeric_audit.md)。它在三组独立随机复 peram 与 $N_c=3$ epsilon VVV 输入上直接求完整 2,880 项总和；候选张成空间残差分别为 `0.8563`、`0.8202`、`0.8284`，而 production reducer 重构误差均约为 $1.4\times10^{-15}$。93 个候选张量在每组样本的数值秩都是 93。因此当前 93 条候选调用不能重构本项目完整六轴 ppn 输出。

对固定状态的 6 个 basis ID `1, 3, 27, 121, 123, 147`，结构脚本把 dummy 名称 alpha-canonicalize，保留 `ABCDEF` 和 tensor 因子的端口顺序，并允许四种汇/源质子开放轴置换；更强的因子关联图比较还允许同名、同秩 operand 因子互换。两种形式比较都得到 6 个不同类。四组输入的具体张量测试也都得到 6 个不同类，六个 basis 张量的秩为 6，8 个 raw 张量秩为 8。详情和 canonical signature hashes 在 [ppn_orbit_crosswalk_audit.md](ppn_orbit_crosswalk_audit.md)。

整块质子互换确实给出可验证关系。例如 Wick ID 1 和 781 仅相差汇端两个质子块的交换。未乘 Wick 符号的网络满足

\[
C_{781}=P_{A\leftrightarrow B} C_1,
\]

其中 Wick 符号分别为 `+1` 和 `-1`，故带符号项满足

\[
(+1)C_{781}=-P_{A\leftrightarrow B}\bigl((-1)C_1\bigr).
\]

此结构恒等式成立的前提是交换的两个质子使用同名、同定义的算符与 VVV 因子；具体配对、einsum 式与因子换序见 [ppn_orbit_crosswalk_audit.md](ppn_orbit_crosswalk_audit.md)。它验证的是整块 proton exchange，不代表内部 $2^3$ 槽位商中所有收缩都能由一个现有格式的代表及其开放轴置换覆盖。

## 5. `ppnn` 计数与论文 1,107

对 $^{4}\mathrm{He}=ppnn$，raw flavor 配对数为

\[
6!\,6!=518400.
\]

将四个汇端重子内的同 flavor 对作无序商后，有 $518400/2^4=32400$ 个标签状态。当前 `ppnn_group_action.py` 测试的作用还包含汇/源 p、n 块交换及整体 $p\leftrightarrow n,\;u\leftrightarrow d$ 翻转，直接枚举得到 1,068 个轨道，而不是 1,107。因此，这个明确写出的群作用并未复现文献所述的 ppnn/$^{4}\mathrm{He}$ 筛选数；不能将该差异解读为论文错误，因为论文计数还可能依赖该显式作用之外的收缩恒等或筛选条件。

arXiv:1205.0585 说明其 unified contraction algorithm 同时处理 Wick 排列与 color/spinor 收缩，并指出特定核素的成本缩减；它也提到等同位旋或 sink baryon interchange 可带来额外缩减。该文回顾先前计算曾把 \(^{3}\mathrm{He}\) 的排列数降为 93、把 \(^{4}\mathrm{He}\) 降为 1,107。arXiv:0912.1383v1 的摘要将相关减少归因于质子/中子置换对称和其他等价收缩。文献的数值是针对其相关子和算符约定的结果，不自动等于本项目保留完整六轴 `contract(...)` 张量时的独立项数。

## 6. 什么是等价 Wick 项：群作用与轨道

一个具体 Wick 项可以用一个离散标签 `w` 表示。标签记录了夸克到源端、汇端槽位的配对，以及由费米交换产生的符号。设 `G` 是在当前问题中确实保持不变的合法对称变换集合，例如相同质子块的交换，或者一个核子内部两个相同味道夸克槽位的交换。对 `g in G`，记其作用为

\[
w \longmapsto g\cdot w.
\]

从 `w` 出发的轨道定义为

\[
\mathcal O(w)=\{g\cdot w\mid g\in G\}.
\]

因此，两个 Wick 项属于同一个轨道，当且仅当存在一个合法的 `g` 使得

\[
w_2=g\cdot w_1.
\]

这里的“等价”不是说两个 Wick 字符串完全相同，也不是说它们未经处理就具有相同数值。变换还可能改变费米符号、颜色 epsilon 的符号、自由 Dirac 轴的排列或投影系数。实际可合并的关系必须写成

\[
\epsilon_{w_2} C_{w_2}
 = \kappa(g)\,P_g\bigl(\epsilon_{w_1}C_{w_1}\bigr),
\]

其中 `C_w` 是该 Wick 项的开放指标张量，`epsilon_w` 是 Wick 符号，`P_g` 是自由指标置换，`kappa(g)` 包含颜色、费米和算符约定带来的符号或系数。若关系是多个张量的线性组合，则必须记录全部线性系数，不能只乘轨道大小。

### 6.1 `ppn` 中的 8 项内部轨道

对 `ppn`，三个核子分别含有两个相同味道的夸克槽位。可以考虑三个内部交换：

\[
s_1:u_1\leftrightarrow u_2,
\qquad
s_2:u_3\leftrightarrow u_4,
\qquad
s_3:d_3\leftrightarrow d_4.
\]

它们生成

\[
G_{\mathrm{int}}\simeq C_2\times C_2\times C_2,
\qquad |G_{\mathrm{int}}|=2^3=8.
\]

从一个原始 Wick 项 `w_0` 出发，内部轨道最多包含

\[
\mathcal O_{\mathrm{int}}(w_0)
 =\{s_1^{e_1}s_2^{e_2}s_3^{e_3}\cdot w_0
   \mid e_i\in\{0,1\}\}
\]

中的 8 个成员。当前投影审计中的一个 state block 正是这 8 项的带符号和：

\[
b_s=\sum_{w:\,s(w)=s}\operatorname{sgn}(w)
       \langle P,C_w\rangle .
\]

例如，形式上可以写成

\[
b_s=W_0-W_1+W_2-W_3+W_4-W_5+W_6-W_7,
\]

其中具体正负号由 Wick 排列、颜色反对称性、Gamma 槽位和投影约定共同决定。这个公式表达的是“一个块包含 8 个相关 Wick 项”，不是假定所有问题中的符号都恰好按 `+ - + - + - + -` 排列。

### 6.2 从 360 个 state 到 93 个代表块

对 `ppn`，2880 个原始 flavor 配对先按上述三个内部同味交换整理为

\[
2880/2^3=360
\]

个 state labels。之后再考虑汇端两个质子块和源端两个质子块的交换：

\[
G_{pp}=C_2^{\mathrm{sink}}\times C_2^{\mathrm{source}}.
\]

在当前固定的 `ppn` 自旋投影下，两个质子块交换产生的块符号与质子自旋 singlet 投影产生的符号相互抵消，所以同一 `G_{pp}` 轨道中的 state block 具有相同的投影值。该作用产生 93 个轨道：6 个轨道大小为 2，87 个轨道大小为 4，满足

\[
6\times2+87\times4=360.
\]

因此当前审计的约化公式是

\[
F_{ppn}
 =\sum_{s\in S}b_s
 =\sum_{O\in S/G_{pp}}|O|\,b_{s_O},
\]

其中 `s_O` 是轨道 `O` 选出的代表。这里的 93 是 93 个 **投影后的 state-block 代表**：每个代表块已经包含 8 个带符号的原始 Wick 项；其轨道权重是 2 或 4。

### 6.3 块不是单独的物理观测量

单个 `b_s` 或单个轨道代表 `b_{s_O}` 通常没有独立的物理意义。最终相关函数是所有代表块按轨道权重相加：

\[
C_{ppn}=\sum_{O=1}^{93}|O|\,b_{s_O}.
\]

因此，93 不是 93 个原始 Wick 项，也不是必然的 93 次普通 `contract(...)` 调用。若每个块的 8 项逐项执行，调用数是

\[
93\times8=744
\]

次标量收缩；若实现共享传播子、颜色和 Dirac 中间结果的融合块内核，则 8 项可以作为一个复合计算单元处理。这正是重子 `B_3` 块方法的计算意义：块是可缓存、可复用、可并行的中间对象，最后再按投影和轨道权重组合成物理相关函数。

### 6.4 合法对称性是有条件的

只有当变换保持当前输入时，才可以用它形成轨道。例如交换两个质子通常要求它们具有相同的 flavor、Gamma 结构、源汇定义、smearing 和动量标签。若

\[
\mathbf p_1\ne\mathbf p_2,
\]

交换会把输入变成 `(p_2,p_1)`，一般不再是同一个固定问题中的等价变换。类似地，若自旋—同位旋投影没有在交换下保持协变，就不能仅凭“都是质子”合并对应块。

这也是为什么“先找轨道再乘轨道大小”不是普适规则。必须先证明算符、传播子、颜色结构、投影和开放指标在该变换下的具体关系；证明成立后，才能把轨道内的项安全地收进同一个带权块。

## 7. 条件性调用数收益

如果将来能对每个离散轨道内的每一项都证明完整张量关系，并且验证后确实每个轨道只需计算一个代表，则按当前报告中的 basis 数，调用数上限估算为

| 系统 | 当前记录的 contract/basis 数 | 条件性每轨道一个代表 | 调用数缩减因子 |
| --- | ---: | ---: | ---: |
| `ppn` | 698 | 93 | 约 7.5 倍 |
| `ppnn` | 31,092 | 1,068 | 约 29.1 倍 |

这只是“昂贵的代表收缩调用数”的潜在收益估算，仍需生成成员到代表的系数/轴映射，并承担映射、视图、累加和验证成本。`ppnn` 的 31,092 来自项目中较早的完整 reducer 运行记录；这一分支没有重跑该 539 秒任务。更关键的是，目前没有证明 93/1,068 个离散轨道就足以重构本项目的完整相关子；ppn 的三组全和数值审计已显示当前 93 候选张量不张成完整结果，因此不能把上表说成已实现的缩减或端到端加速。

## 8. 复现

在项目根目录运行：

```sh
python3 -B verify_code/ppn_orbit_contracts.py
python3 -B verify_code/ppn_orbit_crosswalk_audit.py
python3 -B verify_code/ppn_orbit_reducer_crosswalk.py
python3 -B -m unittest tests.test_ppnn_group_action tests.test_wick_reducer
```

多组随机 `peram`/VVV 的完整数值审计见 `verify_code/ppn_orbit_numeric_audit.py`；结构/单轨道审计见 `verify_code/ppn_orbit_crosswalk_audit.py`。安装项目的可选数值依赖后，可直接从项目根目录运行这些脚本；依赖安装方式见根目录 README。

## 参考

- T. Doi and M. G. Endres, [Unified contraction algorithm for multi-baryon correlators on the lattice, arXiv:1205.0585](https://arxiv.org/abs/1205.0585).
- T. Yamazaki, Y. Kuramashi and A. Ukawa, [Helium Nuclei in Quenched Lattice QCD, arXiv:0912.1383v1](https://arxiv.org/abs/0912.1383v1).
