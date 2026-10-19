# 群轨道与生产版 reducer 基准比较

本文记录 `verify_code` 中的轨道表示与现有生产版 `wick_reducer` 的本地基准。两条路径约化的对象不同，因此应结合[群轨道理论的数学范围](../../verify_code/group_orbit_theory.md)理解这些数字：轨道路径约化固定的 ppn/ppnn 标签状态作用；生产版 reducer 则保留开放 Dirac 张量及其 transporter 关系。

以下测量于 2026-10-09 在本仓库检出版本中使用系统 Python 3.9 完成。各命令分别在独立 Python 进程中运行，因此 `resource.getrusage(...).ru_maxrss` 是该进程的峰值内存。在 macOS 上，该值以字节报告。

## 构造与约化测量

| 路径 | raw | 状态 | 轨道 / basis | 墙钟时间（秒） | 峰值 RSS | 表示的 contract 调用数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| verify_code ppn 轨道 | 2,880 | 360 | 93 | 0.4898 | 26,361,856 B (25.1 MiB) | 744 次状态块调用；若融合并通过认证则为 93 次 |
| 生产版 ppn independent | 2,880 | - | 698 basis | 2.9853 | 28,884,992 B (27.5 MiB) | 698 次 basis 调用 |
| verify_code ppnn 轨道 | 518,400 | 32,400 | 1,068 | 90.5676 | 649,199,616 B (619.2 MiB) | 17,088 次状态块调用；若融合并通过认证则为 1,068 次 |
| 生产版 ppnn independent | 518,400 | - | 31,092 basis | 539.4（历史数据） | 约 1.21 GiB（历史数据） | 31,092 次 basis 调用 |

生产版 ppn 时间来自直接运行 `reduce_case("ppn", mode="independent")`。verify_code 测量会构建并验证从 raw 到状态、再从状态到轨道的完整映射，包括分区覆盖和等变性；它不只是最终轨道数。生产版 ppnn 行采用 `docs/reference/wick-api.md` 中已有的历史测量；2026-10-09 轨道基准测试没有重新运行完整的 518,400 项生产版约化。同一检出版本中通用 reducer 的后续比较见[优化基准](../benchmarks/optimization_benchmark.md)；其 4,298 个开放 Dirac basis 与 1,068 个轨道状态不是同一对象。

产生三项新测量的直接命令如下：

```text
python3 -B -c 'import resource,time; t=time.perf_counter(); from verify_code.orbit_ppn import ppn_orbit_system; s=ppn_orbit_system(); print({"counts":(s.raw_wick_count,s.state_count,s.orbit_count),"wall":time.perf_counter()-t,"rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})'
python3 -B -c 'import resource,time; t=time.perf_counter(); from kernels.wick_reducer.reducer import reduce_case; x=reduce_case("ppn",mode="independent"); print({"counts":(x["wick_count"],x["independent_count"]),"wall":time.perf_counter()-t,"rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"timing":x["timing_seconds"]})'
python3 -B -c 'import resource,time; t=time.perf_counter(); from verify_code.orbit_ppnn import ppnn_orbit_system; s=ppnn_orbit_system(); print({"counts":(s.raw_wick_count,s.state_count,s.orbit_count),"wall":time.perf_counter()-t,"rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})'
```

可复用的驱动程序是 [`benchmark_orbit_vs_reducer.py`](../../verify_code/benchmark_orbit_vs_reducer.py)。它默认将 ppnn 生产模式设为 `historical`，因此普通运行不会意外启动需要 GiB 级内存的生产任务。只有明确需要新的生产测量时，才使用 `--production-ppnn-mode coarse` 或 `independent`。驱动程序还可通过 `--contract-sample N` 对真实的 `opt_einsum.contract` 调用进行小样本计时；这些样本可用于估算 744、17,088、93 或 1,068 次调用的开销，但不能证明不透明的融合张量恒等式成立。

## 解释与限制

对于 ppn，测得的 verify_code 构造时间短于生产版 independent 约化（约 0.49 秒对 2.99 秒），但每个含 8 项的状态块展开后仍表示 744 次普通收缩。要采用潜在的 93 次调用路径，必须对块融合另行认证。

对于 ppnn，verify_code 构造所表示的 basis 调用数远少于生产版历史结果（1,068 对 31,092），但 verify_code 的完整审计本身耗时约 90.6 秒、占用 619 MiB。这并非端到端加速声明：生产版时间是历史数据，verify_code 时间则包含完整映射构造和验证。要公平比较生产版与 verify_code 的墙钟时间，必须在同一机器、Python 运行环境、输入夹具和验证范围下重新运行两者。

普通状态块工作量为每个 ppn 轨道 8 次调用、每个 ppnn 轨道 16 次调用。因此，未融合的轨道路径分别需要 744 和 17,088 次调用。只有在所选夹具下已经证明 Gamma、VVV、开放轴、动量及投影算符的变换关系时，融合路径才能按 93 和 1,068 次高层调用计数；否则应保留显式的块展开作为正确回退路径。
