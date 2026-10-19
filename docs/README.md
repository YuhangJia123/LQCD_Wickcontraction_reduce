# 文档结构

项目文档按使用任务分层：

- 根目录 `README.md` 说明项目用途、统一接口、模块边界和限制。
- 根目录 `QUICKSTART.md` 说明环境准备、首次运行、输入配置、结果目录和错误处理。
- `reference/` 保存稳定的接口参考，避免把完整字段说明重复写进 Quickstart。
- `kernels/` 介绍三个生产计算内核的数学职责和实现入口。
- `validation/` 说明正确性验证、性能测试和研究报告之间的关系。
- `../report/` 保存最终报告、数学分析、实现记录、验证结果和实验数据。

先读 [项目总览](../README.md)，再按任务进入 [Quickstart](../QUICKSTART.md)、[统一接口参考](reference/input-mode-output.md) 或 [报告目录](../report/README_zh.md)。
