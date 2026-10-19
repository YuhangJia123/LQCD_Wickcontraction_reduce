# 统一 input/mode/output 接口参考

> 这是 [Unified input/mode/output reference](input-mode-output.md) 的中文伴随版本。

## 文档定位

本页为同一材料提供中文入口，便于在项目文档中检索和导航。完整公式、表格、文件名、命令记录和测量值仍以链接的原文为准；本页统一说明当前项目的接口术语。

## 当前项目约定

- 公共选择器为 `input`、`mode` 和 `output`。
- 输入类型为 `distillation` 与 `normal`；模式为 `wick` 与 `special`（后者只用于 `normal`）。
- `normal-special` 的 `wick_scope` 为 `complete` 或 `cross_time`，缺省为 `complete`；前者走完整图通用融合，后者走紧凑跨时间融合。
- Wick 侧输出包括 `independent`、`check`、`block`；`count` 只表示 `normal-special` 的计数输出。
- 当前生产 block 路径统一称为 `normal-wick-block`，由写出层自动构造原始 pairing family；验收值为 ppn = 48、ppnn = 288。
- 结果写入 `result/<input>_<mode>_<output>/`，仓库中的轻量示例位于 `examples/example_result/`。

两条 special 内核共享 incidence、spin 和全局颜色 S3 融合原语。`check` 在相同图构造下关闭 S3 轨道合并，再与生产结果逐项比较。

## 使用范围

需要完整推导、审计数据、命令输出或历史限定时，请打开链接的原文。本页只是对应语言的导航与术语说明，不新增另一套实现，也不改写原始测量结果。

## 原文

[打开维护中的原语言文档](input-mode-output.md)。
