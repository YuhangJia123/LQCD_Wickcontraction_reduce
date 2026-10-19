# 通用 check 验证摘要

本文所有输出均使用同一条通用 `reduce_operators` 路径；species 和 factor-graph 选项不属于软件包 API。

数值比较为：

\[
  s_w\,C_w^{\rm direct}
  \;\stackrel{?}{=}
  \sum_{b,\pi} a_{w b\pi}\,s_b\,
  \operatorname{transpose}_{\pi}(C_b^{\rm basis\_contract}),
\]

其中 `basis_contract` 按设计不含符号，`s` 是精确的 Wick 符号，`a` 和 `pi` 是 reducer 记录的系数及开放轴置换。随机 perambulator 数组的形状为 `(4,N,N,4)`；重子 VVV 数组是完全反对称的 `(N,N,N)` 张量；介子 VDV 数组的形状为 `(N,N)`。七个小型夹具均在 `N=3` 和 `N=6` 下完成检查。

| 夹具 | raw Wick 数 | basis_contract 数 | dependent 数 | 最大相对残差 |
|---|---:|---:|---:|---:|
| `pipi` | 24 | 10 | 14 | `6.10e-16` |
| `pn` | 36 | 18 | 18 | `3.63e-16` |
| `ppn` | 2880 | 219 | 2661 | `4.56e-16` |
| `delta` | 6 | 2 | 4 | `0` |
| `delta++delta-` | 36 | 3 | 33 | `1.32e-16` |
| `pip` | 24 | 24 | 0 | `0` |
| `pip-p` | 6 | 6 | 0 | `0` |

完整的流式 ppnn check 文件包含：

| 对象 | 数量 |
|---|---:|
| `ppnn_origen_wick_raw` | 518400 |
| `ppnn_basis_contract` | 4298 |
| `ppnn_origen_wick` | 518400 |
| `ppnn_origen_wick_to_independent` | 518400 |
| dependent 关系 | 514102 |

因此 `518400 = 4298 + 514102`。ppnn 文件采用流式写入，大小约为 463 MB。从实际关系映射中均匀抽取 130 行，使用新的随机输入在两个颜色维度下进行评估：

| 颜色维度 | 抽样行数 | 最大绝对残差 | 最大相对残差 |
|---:|---:|---:|---:|
| 3 | 130 | `5.33e-7` | `9.52e-16` |
| 6 | 130 | `1.88e-1` | `1.97e-15` |

ppnn 抽样检查使用完整 check 文件生成的精确系数和转置关系。它不表示已经完成全部 518400 次高开销的直接 einsum 评估；完整覆盖的是 check 数据和 reducer 关系，而 ppnn 数值检查采用抽样方式。

运行相应检查后，生成文件会保存在 `verify_output/check/`（该目录由 Git 忽略，不随仓库分发）：

- `seven_summary.json` 和七个小型夹具的 `*_check.py` 文件；
- `ppnn_check.py` 与 `ppnn_sample_summary.json`。
