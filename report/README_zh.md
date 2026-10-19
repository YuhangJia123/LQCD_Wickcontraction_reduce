# 报告与研究资料

本目录保存项目的最终报告、算法推导、实现记录、验证结果、性能基准和实验数据。根目录 README 负责项目总览，Quickstart 负责运行步骤；本目录负责解释算法、记录证据和归档研究结果。

## 入口

| 目的 | 入口 |
| --- | --- |
| 阅读当前最终报告 | [`final/wick_diss_reduce_algorithm_report.md`](final/wick_diss_reduce_algorithm_report.md) |
| 查看 PDF 和 TeX | [`final/wick_diss_reduce_algorithm_beamer.pdf`](final/wick_diss_reduce_algorithm_beamer.pdf)、[`final/wick_diss_reduce_algorithm_beamer.tex`](final/wick_diss_reduce_algorithm_beamer.tex) |
| 理解数学思路 | [`algorithms/mathematics/`](algorithms/mathematics) |
| 理解代码实现 | [`algorithms/implementation/`](algorithms/implementation) |
| 查看验证结果 | [`validation/`](validation) |
| 查看性能基准 | [`benchmarks/`](benchmarks) |
| 查看历史方案 | [`historical/`](historical) |
| 查看原始数据 | [`data/`](data) |

## 文档职责

- `algorithms/mathematics/` 只描述 Wick 等价关系、群作用、incidence/S3、W 基底和约化条件。
- `algorithms/implementation/` 只描述数据结构、搜索、缓存、输出和工程实现边界。
- `validation/` 保存可复现的正确性、数值残差、block 证书和专项审计结果。
- `benchmarks/` 保存时间、内存、contract/network 数量以及运行口径。
- `historical/` 保存已被当前实现取代的方案和早期报告，不作为生产接口说明。
- `data/` 保存报告使用的 JSON、输入夹具和固定实验记录。

验证脚本和命令统一见 [`verify_code/README_zh.md`](../verify_code/README_zh.md)。三个生产计算内核的导航见 [`docs/kernels/README.md`](../docs/kernels/README.md)。
