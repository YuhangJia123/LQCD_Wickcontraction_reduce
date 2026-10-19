# ppn 93 轨道与 698 个 reducer basis 的交叉核对

此审计把项目实际 `ppn` Wick 配对映射到 `ppnn_group_action.py` 中的 360 个 `He3State`，再将 reducer 返回的独立 Wick ID 按轨道分组。脚本还用固定随机输入计算一个轨道内的实际六轴收缩张量，检查六个 basis contract 是否能由一个代表经群允许的自由轴交换得到。

## 复现

在项目根目录运行：

```sh
python3 -B verify_code/ppn_orbit_crosswalk_audit.py
```

脚本只读取现有模型、Wick 枚举器、reducer 和输出方程生成器，不修改生产算法。数值部分直接用 NumPy `einsum` 执行 `expression_output.contraction_expression` 返回的方程，因此不要求安装输出文件运行时使用的 `opt_einsum`。

## 组合交叉表

Wick 标签约定与研究脚本一致：汇端两个质子的两个 `u` 配对作为无序对，中子 `u` 为单标签；两个质子的 `d` 为有序单标签，中子的两个 `d` 为无序对。结果如下：

| 项目 | 结果 |
| --- | ---: |
| 实际 raw Wick 项 | 2,880 |
| 离散商状态 | 360 |
| 每个商状态对应 raw Wick 数 | 8 |
| 离散群轨道 | 93 |
| 轨道大小 | 87 个大小 4，6 个大小 2 |
| reducer 独立 basis Wick / contract | 698 |
| 有 reducer basis 的轨道 | 93 / 93 |

每个轨道内的 reducer basis Wick 数量：

| 离散轨道大小 | basis 数 | 轨道数 |
| ---: | ---: | ---: |
| 2 | 2 | 1 |
| 2 | 4 | 2 |
| 2 | 6 | 3 |
| 4 | 6 | 11 |
| 4 | 7 | 4 |
| 4 | 8 | 72 |

因此 93 个轨道覆盖了全部 698 个 basis Wick ID，但不是一轨道一个 basis。71 个轨道只有一个商状态中出现 reducer basis，22 个轨道有两个商状态中出现 reducer basis。占据的商状态共 115 个；其 reducer basis 数直方图为：10 个状态各有 2 个、30 个各有 4 个、19 个各有 6 个、4 个各有 7 个、52 个各有 8 个。

## 六轴张量核验

固定状态

```text
He3State(up_protons=((1, 2), (3, 4)), up_neutron=5,
         down_protons=(1, 2), down_neutron=(3, 4))
```

包含 raw Wick ID `1, 3, 25, 27, 121, 123, 145, 147`，其中 reducer 保留的六个 basis contract 是 `1, 3, 27, 121, 123, 147`。每种输入都给这些 Wick 使用相同的 peram/VVV 数组，以直接比较它们的张量；不同样本则由不同随机种子独立生成输入。使用项目的 `Cgamma5` / `Cgamma5_dagger`，每个结果都包含对应 Wick 符号，输出轴顺序为 `ABCDEF`。

测试了三组 3 维 epsilon 输入，每组 peram 的形状为 `(4,3,3,4)`；VVV 为标准三维 epsilon 乘随机复数比例因子，source 和 sink 各自生成比例因子。另测试一组颜色维度为 6 的输入，peram 形状为 `(4,6,6,4)`，sink/source VVV 分别由 seed `20261009` 生成独立复数 rank-3 张量，再完全反对称化。矩阵秩使用相对奇异值阈值 `1e-10`。结果为：

| 样本 | VVV 类型 | 六个 basis 秩 | 八个 raw 秩 | 到 Wick 1 的最小相对残差（按 ID `1,3,27,121,123,147`） |
| --- | --- | ---: | ---: | --- |
| seed `721` | 3D epsilon | 6 | 8 | `0, 0.6279, 0.8009, 0.6279, 0.8009, 0.8896` |
| seed `20261007` | 3D epsilon | 6 | 8 | `0, 0.8280, 0.9849, 0.8280, 0.9849, 1.0226` |
| seed `20261008` | 3D epsilon | 6 | 8 | `0, 0.6547, 0.8387, 0.6547, 0.8387, 0.9244` |
| seed `20261009` | random fully antisymmetric 6D | 6 | 8 | `0, 0.7336, 0.8961, 0.7336, 0.8961, 0.9589` |

相对残差对所有样本都允许汇端两个质子的开放轴 `A <-> B`、源端两个质子的开放轴 `D <-> E`、两者乘积及整体正负号。每个样本中六个保留张量都形成 6 个不同的轴换位等价类；最小非零残差远大于浮点误差。因此，对这些输入和当前项目的 contract 定义，不能把该轨道的六个 reducer basis contract 全部替换成一个原始代表 contract 加上现有群允许的符号/轴换位。

## Contract 索引结构

脚本还把 `contraction_expression` 输出拆为有序 operand 名和各 operand 的索引槽。它固定开放轴 `ABCDEF`，按首次出现顺序为 dummy 指标重新编号，再对每一对 basis Wick 尝试四种开放轴置换。对更强的对比，另构造带 operand 名着色的因子关联图：dummy 指标作为连接因子端口的边，开放指标带固定标签；允许同名、同秩 tensor operand 因子的节点互换，但保留每个 tensor 内的端口顺序。

| 对比层 | 允许的等价变换 | 六个 basis contract 的等价类数 |
| --- | --- | ---: |
| 有序 operand 列表 | dummy alpha-renaming 和四种开放轴置换 | 6 |
| 因子关联图 | 上述变换，加同名 operand 因子重排 | 6 |

有序 dummy canonical signature 的 SHA-256 前 12 位如下，方便复核脚本所比较的具体 contract 结构：

| Wick ID | Canonical signature hash |
| ---: | --- |
| 1 | `a724f12e1073` |
| 3 | `92ef75617dc1` |
| 27 | `f81351293701` |
| 121 | `9941ea3701a0` |
| 123 | `61f4f76ee18a` |
| 147 | `dab0cb16f0a1` |

这说明差异不只是 dummy 指标字母命名，也不只是项目轨道生成元允许的开放轴转置；即使把相同 operand 的因子视为可交换节点，六个结构仍不同。结构检查不会自动证明或排除所有 Gamma、epsilon 或其他张量恒等式；数值回放用于核对这里的具体张量，而不是推导论文完整的 93 项求和系数。

整块质子交换可由一个不依赖具体矩阵分量的 tensor-network 同构验证：Wick ID 1 与 781 的 pairing 分别为 `(0,2,1,3,5,4,6,8,7)`、`(3,5,4,0,2,1,6,8,7)`，Wick 符号分别为 `-1`、`+1`。把 Wick 1 的自由轴 `A`、`B` 互换，再一致重命名 dummy 指标，并交换相同名称的汇端质子 Gamma、peram 与 VVV 因子后，图同构检查成立。因此未乘 Wick 符号的张量满足 `C[781] = transpose(C[1], (1,0,2,3,4,5))`，带符号项则满足 `(+C[781]) = -transpose((-C[1]), (1,0,2,3,4,5))`。此关系依赖两个汇端质子算符和 elemental 使用相同定义；审计脚本现在包含该同构的断言。

这里的随机回放针对项目当前 `ppn` 算符 fixture 和列出的 peram/VVV 输入。它是对“一个现有格式代表加群允许转置/符号即可覆盖该轨道六个 basis”这一主张的反例检查，不是对任意外部算符、动量标签或张量定义的完整分类。组合轨道仍可用于组织候选关系；尚未推导或验证论文所报 93 项结果对应的完整系数及求和重建。
