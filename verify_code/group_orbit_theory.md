# Wick 轨道约化的数学规范

> Orbit 群作用的补充说明。当前 normal 完整张量 W 约化统一使用 [`block_reduce`](../docs/reference/block-api.md)。本文用于解释验证中的群作用和计数。

本文档规定 `verify_code/` 中 ppn 和 ppnn 轨道研究实现的对象、商映射、输出和验证边界。它不修改 `wick_reducer` 的生产算法，也不把轨道数解释为开放 Dirac 张量的秩。

## 1. 四个不同的对象

设 \(\Omega_{\rm raw}\) 是按精确 flavor 枚举的 Wick 配对集合。每个
\(\omega\in\Omega_{\rm raw}\) 带有一个费米符号
\(\varepsilon(\omega)\)，以及由 Gamma、peram、VVV/VDV 和开放轴组成的收缩张量 \(C_\omega\)。

内部同味槽位交换群 \(K\) 只改变一个核子块内部的槽位顺序。状态商空间是

\[
S=\Omega_{\rm raw}/K.
\]

对状态 \(s\in S\)，定义状态块

\[
B_s=\sum_{\omega\in q^{-1}(s)}
\varepsilon(\omega)\,\langle P,C_\omega\rangle,
\]

其中 \(q\) 是 raw-to-state 映射，\(P\) 是可选的 spin-isospin projector。状态块仍然包含一个状态内部的全部 8 项（ppn）或 16 项（ppnn）。

外部群 \(G\) 作用在状态空间上，轨道是

\[
O=G\cdot s,\qquad S/G=\{O\}.
\]

轨道代表块是一个代表状态的 \(B_s\)，而不是单个裸 Wick 项。若投影后的收缩满足

\[
B_{g\cdot s}=\chi(g,s)B_s,
\]

则轨道贡献必须保留字符 \(\chi\)。只有在 \(\chi=1\) 的已认证情形下，才可以写成

\[
C_P=\sum_{O\in S/G}|O|B_{s_O}.
\]

开放 Dirac basis 是生产 reducer 对未投影张量作线性约化后得到的对象。它与 \(|S/G|\) 没有一般的相等关系；轴置换和 Gram 线性组合必须单独保存。

## 2. ppn 与 ppnn 计数

ppn 的原始数目和内部群为

\[
|\Omega_{\rm raw}|=5!\,4!=2880,\qquad
K_{\rm ppn}\cong(C_2)^3,
\]

所以

\[
|S_{\rm ppn}|=2880/8=360.
\]

两个外部质子交换生成元构成

\[
G_{\rm ppn}\cong(C_2)^2,
\qquad |S_{\rm ppn}/G_{\rm ppn}|=93.
\]

状态轨道直方图是 6 个大小 2 的轨道和 87 个大小 4 的轨道。扩展群
\(\widetilde G_{\rm ppn}\) 的阶为 \(8\cdot4=32\)，直接作用 raw Wick 图时，raw 轨道大小是 16（6 个）和 32（87 个）。

ppnn 的原始数目和内部群为

\[
|\Omega_{\rm raw}|=6!\,6!=518400,\qquad
K_{\rm ppnn}\cong(C_2)^4,
\]

所以

\[
|S_{\rm ppnn}|=518400/16=32400.
\]

外部生成元是 sink/source 的 proton/neutron 交换和整体
\(p\leftrightarrow n,\ u\leftrightarrow d\) 翻转。它们给出

\[
G_{\rm ppnn}\cong(C_2)^4\rtimes C_2,
\qquad |G_{\rm ppnn}|=32.
\]

当前明确群作用的 state 轨道直方图为

\[
\begin{array}{c|rrrr}
|O|&4&8&16&32\\\hline
\#O&4&14&83&967
\end{array}
\]

因而代表数为 1068。扩展群阶为 \(16\cdot32=512\)，raw 轨道直方图为

\[
\begin{array}{c|rrrr}
|\widetilde O|&64&128&256&512\\\hline
\#\widetilde O&4&14&83&967.
\end{array}
\]

两层轨道满足

\[
\Omega_{\rm raw}/\widetilde G
\simeq
(\Omega_{\rm raw}/K)/G
\]

的前提是 \(K\) 是内部作用、\(G\) 归一化 \(K\)，并且商映射满足

\[
q(\widetilde g\cdot\omega)=g\cdot q(\omega).
\]

研究实现对 raw pairing 直接验证这个等变性，而不只比较轨道数量。

## 3. Burnside 与显式枚举

对有限群 \(G\) 的有限集合 \(X\)，Burnside 公式是

\[
|X/G|=\frac{1}{|G|}\sum_{g\in G}|\operatorname{Fix}_X(g)|.
\]

每个研究实现必须同时提供：固定点直方图、显式 BFS/DFS 轨道、轨道大小直方图、raw-to-state、state-to-orbit、覆盖和无重复检查。Burnside 只检查轨道数；显式枚举检查代表和成员本身。

## 4. 可读 independent/check 输出

`independent` 只输出代表块和最终加权结果，但每个代表必须保留：

- `orbit_id`、`representative_wick_id`；
- `orbit_members`、`orbit_size`、`block_weight`；
- `raw_wick_ids`、`raw_wick_signs`、pairing；
- internal/external action word；
- contract equation、axis permutation 和 projection metadata。

普通块必须展开为可读代码：

```python
a0 = contract(...)
a1 = contract(...)
block_0 = a0 - a1
orbit_result = weight_0 * block_0
```

`check` 必须输出或保存 raw mapping，并验证

\[
\text{orbit blocks}
\xrightarrow{\text{weight and expansion}}
\text{full raw Wick sum}.
\]

每个 raw Wick ID 必须恰好属于一个 state、一个 orbit；符号必须保留，block 展开必须恢复该 block 的全部 raw 项。此反向展开不是把 raw 项错误地当作某一个 block 的常数倍。

`verify_code/generate_orbit_output.py` 提供两个显式入口：`--mode independent` 生成代表
state block 和权重结果，`--mode check` 生成完整 raw contract 映射、逐轨道展开以及
两种 residual。`full_from_blocks` 遍历每个 `block_expansion` 中的 raw ID，
`full_from_weighted_blocks` 才是代表 block 的权重求和；两者都与 `full_raw` 保留在输出文件中。

## 5. 块内反对称融合

一个内部交换对的两个带符号项可写成

\[
D(a,b;X,Y)=S(a;X)S(b;Y)-S(a;Y)S(b;X)
\]

或写成二阶行列式。ppn 有三组内部交换，因此块展开为 \(2^3=8\) 项；ppnn 有四组，因此展开为 \(2^4=16\) 项。融合只有在 Gamma、VVV、开放轴、momentum 和公共收缩骨架满足相应交换关系时才合法。否则必须使用显式展开。

最终导出仍必须是普通 `contract(...)` 赋值和清晰的线性组合；内部可以用 wedge/fusion 数据结构，但不能导出不透明的 `contract_block(...)`。

## 6. 随机验证与适用边界

随机 peram/VVV 验证应比较

\[
\text{direct\_raw}
=\text{orbit\_explicit}
=\text{check\_expanded}
=\text{fused\_result}
\]

并记录 seed、raw Wick 数、block 数、contract 调用数、融合调用数和最大残差。相对残差目标为小于 \(10^{-12}\)，理想为 \(10^{-14}\) 量级。随机验证不能替代群作用和指标变换的结构证明。

`verify_code/numeric_validation.py` 的三种子审计使用形状为 `(4,3,3,4)` 的复 peram
和颜色完全反对称的 `(3,3,3)` VVV。完整 raw 和 orbit 重构使用同一组 peram/VVV
样本，通过共同输入因子进入两条路径；为检验商空间代数，它再把带符号 raw contribution
构造成同一 state orbit 内相等。因此验证的是
`direct_raw = orbit_explicit = check_expanded = weighted representative` 的展开恒等式，
而不是宣称任意未变换随机 peram 都具有标签交换协变性。实际输入应通过
`verify_code/operator_input.py` 载入，并为其保持完整算符的生成元另行提供见证。

输入层支持现有 `Operator`、`[sink_operators, source_operators]`、JSON specification，
并保留有序 flavor、`gamma_slots`、`open_slot` 和 `external_tag`。它只负责 raw Wick 输入
归一化；没有对任意重子/介子自动猜测群作用，无法证明的生成元必须禁用。

任意动量、Gamma、VVV、source/sink 标签和投影都必须重新认证。不同动量通常只保留动量配置的稳定子；p/n peram 简并并不自动等于核子插值算符可交换。若某个生成元不能保持完整输入，禁用该生成元或回退到生产 reducer。

因此 93 和 1068 是指定输入与指定群作用下的代表 block 数，不是任意算符、任意动量或任意投影的普适常数，也不是 698 或其他开放 Dirac basis 数。

## 7. 论文数字

本研究明确群作用得到 ppnn=1068。文献报告 1107，但没有公开足以逐项复现的 survivor 映射；因此当前结果能证明本项目固定 fixture 下的 1068 合法，而不能仅凭差异断言论文算术错误。
