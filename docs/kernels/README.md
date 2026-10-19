# 三个计算内核

三个生产计算包现在统一放在 `kernels/` 下；它们通过 `kernels.*` 导入，统一命令入口是 `python -m kernels`：

| 内核 | Python 包 | 职责 |
| --- | --- | --- |
| 通用 Wick 内核 | [`kernels/wick_reducer`](../../kernels/wick_reducer) | 枚举 Wick 配对，执行 peram/Dirac 约化，写出 `independent`、`check` 和 normal `block`。 |
| incidence/S3 颜色融合内核 | [`kernels/wick_colour_orbit`](../../kernels/wick_colour_orbit) | normal-special 的 `cross_time` 与 `complete` 两条融合内核、共享 incidence/spin/S3 原语和 JSON v3 输出。 |
| normal-wick-block 内核 | [`kernels/block_reduction`](../../kernels/block_reduction) | 从 normal Wick 配对族验证并搜索 W 结构，生成证书和 block 输出。 |

数学推导、实现边界和验证数据分别见 [算法数学文档](../../report/algorithms/mathematics)、[实现文档](../../report/algorithms/implementation) 和 [验证文档](../../report/validation)。
