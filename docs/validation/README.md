# 验证与基准

`verify_code/` 是独立的验证目录，负责：

- 直接 Wick 与 basis/check 回代的正确性比较；
- incidence/S3、orbit 和 block 的专项审计；
- 时间、峰值内存和 contract/network 数量基准。

生产入口不导入 `verify_code/`。生产入口不依赖该目录，因此 `python -m kernels` 和 `kernels.block_reduction` 可以独立运行。

可执行验证命令见 [`verify_code/README_zh.md`](../../verify_code/README_zh.md)。研究结果和固定数据按主题分布在：

- [数学算法](../../report/algorithms/mathematics)
- [实现说明](../../report/algorithms/implementation)
- [验证结果](../../report/validation)
- [性能基准](../../report/benchmarks)
- [最终与历史报告](../../report/README_zh.md)
