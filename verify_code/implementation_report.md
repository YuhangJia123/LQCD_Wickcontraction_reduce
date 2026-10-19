# 研究实现验收报告

> 补充验收记录：本文分析 orbit 表示与当前 W 约化之间的关系。当前 W 约化统一使用 [`block_reduce`](../docs/reference/block-api.md)。

本分支为 `codex/ppnn-1107-verify_code`。生产 `wick_reducer` 文件、生产
`independent/detailed/check` 模式保持不变；公开入口统一为
`reduce_operators(operators, label=..., options=...)`，内部 `reduce_case`
不再接受 species 策略开关。

## 结果

| 系统 | raw Wick | state quotient | state orbits | state orbit histogram | raw orbit histogram |
|---|---:|---:|---:|---|---|
| ppn | 2,880 | 360 | 93 | `{2: 6, 4: 87}` | `{16: 6, 32: 87}` |
| ppnn | 518,400 | 32,400 | 1,068 | `{4: 4, 8: 14, 16: 83, 32: 967}` | `{64: 4, 128: 14, 256: 83, 512: 967}` |

ppn 的 state 群阶为 4，raw 扩展群阶为 32；ppnn 的 state 群阶为 32，raw 扩展群阶为
512。ppnn Burnside 固定点直方图为
`{32400: 1, 576: 1, 180: 4, 96: 4, 24: 4, 0: 18}`，除以 32 得到 1068。
每个 raw Wick 恰好进入一个 state，每个 state 恰好进入一个 state orbit，raw orbit
覆盖完整 raw universe 且没有重复。

## 输出与调用量

`verify_code/generate_orbit_output.py` 生成普通 Python `contract(...)` 赋值。
在未融合的代表 state-block 输出中：

| 模式 | ppn | ppnn |
|---|---:|---:|
| independent 代表 block 数 | 93 | 1,068 |
| independent 代表 state 内普通 contract 调用 | 744 (=93x8) | 17,088 (=1,068x16) |
| check 完整 raw contract 调用 | 2,880 | 518,400 |
| check 覆盖 raw IDs | 2,880 | 518,400 |

已生成并做 Python 3.9 AST/compile 检查的样例文件为
`verify_code/ppn_orbit_independent.py`（744 个代表 state contract）和
`verify_code/ppn_orbit_check.py`（2,880 个完整 raw contract）。ppnn 使用同一 CLI
按需生成，避免把 518,400 行 raw 输出默认放进仓库。

`full_from_blocks` 是通过 `block_expansion` 遍历所有 raw ID 的显式展开；
`full_from_weighted_blocks` 是代表 state-block 的轨道权重结果；`full_raw` 是完整带符号
raw 和。check 文件同时保存 raw-to-state、state-to-orbit、raw signs、block members、
block signs、equation、axis/projector metadata 和 residual 表达式。

## 块融合

生产 `normal-wick-block` writer 从 normal 算符自动生成完整 pairing family，调用
`kernels.block_reduction.block_reduce` 建立三表计划，并验证原始配对到输出网络的精确
映射。ppn 和 ppnn 的当前验收值分别为 48 和 288 个输出网络；不能满足交换协变时，
证书会保留显式 contract，而不会把它隐藏成不可读的 `contract_block`。

## 数值粗审计

`verify_code/numeric_validation.py` 对三个固定 seed 生成复 peram `(4,3,3,4)` 和颜色
完全反对称 VVV `(3,3,3)`。完整 raw 和 orbit 重构使用同一组 peram/VVV 样本，通过共同
输入因子进入两条路径，再在带符号 raw contribution 的 state orbit 内使用等变随机标量。
ppn 的 seeds `(1,2,3)` 最大相对残差为 `1.03e-14`（低于 `1e-12`），调用数为 raw 2,880、
代表块 93；ppnn 的 seeds `(20261009,20261010,20261011)` 在 518,400 raw 和 1,068 blocks
上的相对残差均为 0（使用实部/虚部稳定求和）。该审计验证 direct/raw、显式 orbit、check
展开和加权代表块在声明协变前提下相等；它不宣称任意未经变换的随机 peram 自动满足
核子交换、同位旋、动量或 Gamma 协变。

## 与生产 basis 和论文数字的关系

生产 ppn reducer 的开放 Dirac basis 是 698；历史 ppnn 生产报告为 31,092。93 和 1068
是指定 flavor-label 商空间中的 projected/state-block orbit 数，不是开放 Dirac 张量
空间的 basis rank，也不意味着生产 698/31,092 项可无条件删成 93/1068 个普通张量
`contract`。只有在生成元保持完整算符、Gamma、VVV、动量、开放轴和 projector 时，代表
block 才能按轨道权重替代成员。

本实现得到当前明确 ppnn 群作用的 1068。论文 arXiv:0912.1383v1 报告 1107，但没有
公开可逐项复现的 survivor 映射和全部输入约定；因此差异不能仅凭本分支计数断言论文
算术错误。

## Checkpoint

| checkpoint | commit | 内容 |
|---|---|---|
| orbit core | `5159a33` | OrbitSystem、raw/state/orbit 映射、Burnside、ppn/ppnn 全量测试 |
| readable output | `afc832c` | independent/check contract 输出和覆盖检查 |
| block fusion | `08492af` | 8/16 项显式融合原型及测试 |
| final verify_code | `ace3d41` | 输入归一化、三 seed 粗审计、CLI、数学文档和既有研究记录 |
| generated outputs | `d4f3eff` | ppn independent/check 可读 Python 文件 |
