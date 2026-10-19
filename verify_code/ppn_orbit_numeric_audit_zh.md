# ppn 轨道数值审计

> 这是 [ppn orbit numerical audit](ppn_orbit_numeric_audit.md) 的中文伴随版本。

## 文档定位

本页为同一材料提供中文入口，便于在项目文档中检索和导航。完整公式、表格、文件名、命令记录和测量值仍以链接的原文为准；本页统一说明当前项目的接口术语。

## 当前项目约定

- 公共选择器为 `input`、`mode` 和 `output`。
- 输入类型为 `distillation` 与 `normal`；模式为 `wick` 与 `special`（后者只用于 `normal`）。
- Wick 侧输出包括 `independent`、`check`、`block`；`count` 只表示 `normal-special` 的计数输出。
- 当前生产 block 路径统一称为 `normal-wick-block`，由写出层自动构造原始 pairing family；验收值为 ppn = 48、ppnn = 288。
- 结果写入 `result/<input>_<mode>_<output>/`，仓库中的轻量示例位于 `examples/example_result/`。

## 使用范围

需要完整推导、审计数据、命令输出或历史限定时，请打开链接的原文。本页只是对应语言的导航与术语说明，不新增另一套实现，也不改写原始测量结果。

## 原文

[打开维护中的原语言文档](ppn_orbit_numeric_audit.md)。
