# 因子图优化结果

> 本文为英文原文 [optimization_benchmark.md](optimization_benchmark.md) 的中文翻译。

本报告比较原始代码快照（`a3b76f8`）和优化后的工作树。每个小型算例都在独立进程中运行。峰值 RSS 使用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 测量；时间为单次运行结果，用于展示大致量级，并非稳定的微基准测试精度。

两组测量都使用 macOS 14.8.4 arm64 上的 Python 3.9.6 和 NumPy 2.0.2。优化环境还安装了 `opt_einsum` 3.4.0。原始基线没有安装 `opt_einsum`，但只运行 reducer 的基准测试既不导入也不调用它；数值审计会用到该库。

## 优化内容

1. 规范化现在会在第一次遍历时返回并保存算符映射见证。分桶约化器不再对每个原始项重复进行第二次规范化。ppnn 见证缓冲区额外占用约 4 MiB。
2. 每次约化使用一个上下文，预先计算算符组、端点路径和边颜色；置换表会缓存并复用。
3. 直接从共享缓冲区读取紧凑配对和见证，并使用缓存的 `Struct.unpack_from`，避免为每条记录创建临时 `bytes` 切片。
4. 对于二色因子图，如果精确颜色细化能为每个顶点分配唯一不变量颜色，就直接使用该结果。仍存在未决对称性时，则沿用已有的穷举规范化路径。

## 小型算例

| 算例 | 原始项 -> 基项 / 依赖项 | 基线时间（秒） | 优化后时间（秒） | 时间变化 | 基线 -> 优化后峰值 RSS（MiB） |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pipi` | 24 -> 10 / 14 | 0.011745 | 0.006747 | -42.6% | 22.59 -> 20.50 |
| `pn` | 36 -> 18 / 18 | 0.018669 | 0.013393 | -28.3% | 21.14 -> 20.17 |
| `ppn` | 2,880 -> 219 / 2,661 | 1.653858 | 1.265962 | -23.4% | 28.75 -> 27.63 |
| `delta` | 6 -> 2 / 4 | 0.002432 | 0.001777 | -26.9% | 25.25 -> 21.84 |
| `delta++delta-` | 36 -> 3 / 33 | 0.018650 | 0.014868 | -20.3% | 21.84 -> 21.28 |
| `pip` | 24 -> 24 / 0 | 0.007863 | 0.005935 | -24.5% | 25.00 -> 22.42 |
| `pip-p` | 6 -> 6 / 0 | 0.001796 | 0.001656 | -7.8% | 25.88 -> 21.20 |
| `pn_omega_pi+` | 3,456 -> 908 / 2,548 | 2.342970 | 1.798358 | -23.2% | 40.27 -> 39.73 |

## ppnn

| 指标 | 原始版本 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| 原始 Wick 项 | 518,400 | 518,400 | 不变 |
| 因子桶 / 基 `contract` 数 | 4,298 | 4,298 | 不变 |
| 依赖项 | 514,102 | 514,102 | 不变 |
| Peram 分桶阶段 | 140.655 秒 | 65.164 秒 | -53.7% |
| 传递器、sink 和 source 阶段 | 322.928 秒 | 223.523 秒 | -30.8% |
| reducer 总时间 | 463.584 秒 | 288.688 秒 | -37.7%（快 1.61 倍） |
| 峰值 RSS | 406.25 MiB | 410.23 MiB | +3.98 MiB（+0.98%） |

优化运行保持相同的约化计数，ppnn reducer 的实测时间减少了 174.9 秒。峰值 RSS 上升约 1%，与每个原始项保存一个紧凑算符见证相符。

提供的历史聊天还记录了一次 915.216 秒、约 407.4 MiB 的运行，计数为 518,400 -> 4,298/514,102。该运行的确切源代码修订版和测量环境未注明，因此保留为历史观察值，不用于计算直接加速比。本报告中的可复现比较，是已提交的原始快照 `a3b76f8`（463.584 秒）与当前优化工作树（288.688 秒）的比较。

使用以下命令重新运行原始快照基准：

```sh
mkdir -p /tmp/wick-reduce-original
git archive --format=tar a3b76f8 | tar -xf - -C /tmp/wick-reduce-original
cd /tmp/wick-reduce-original
python3 verify_code/run_improved_ppnn.py
```

## 正确性检查

使用 `opt_einsum.contract`、随机复数 peram、随机反对称 VVV 张量和随机稠密 VDV 数组，对八个小型算例在颜色维度 3 和 6 下进行检查。将每个带符号的原始 Wick 收缩与生成的基项重构进行比较，并对开放自旋轴使用共享的随机投影。16 组算例/维度检查全部通过；最大逐 Wick 相对残差为 C=6 时 `ppn` 的 `4.9873e-13`。每个 Wick 项的相对残差定义为 `|direct - reconstructed| / max(|direct|, |reconstructed|, 1e-30)`，与下方完整 ppnn 检查的定义一致。

ppnn 检查源文件包含全部 518,400 个原始收缩，以及全部 518,400 条 `origen_wick_to_independent` 映射。完整随机 peram 审计在 C=3 和 C=6 下评估全部直接原始收缩，并将生成的每条 check 映射与其基项视图比较。两个维度下的全部 518,400 个映射槽位均通过检查。每个八轴开放自旋恒等式都用固定随机投影计算，避免为每个原始项构造稠密 `4**8` 输出张量。

| 颜色维度 | 原始收缩数 | 已检查映射行 | 最大相对残差 | 总相对残差 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 518,400 | 518,400 | `1.1053e-12` | `3.6964e-14` |
| 6 | 518,400 | 518,400 | `6.5752e-13` | `2.8591e-14` |

C=6 时最大绝对残差为 `1.8971e-4`，对应的收缩值约为 `1e12`；其相对残差为 `6.5752e-13`。两个直接求和结果与基项重构求和结果，也分别在上表所示的总相对残差范围内一致。两个维度的完整审计总耗时为 6,121.8 秒。

独立规范化审计穷举测试了所有 19,683 个三态 3×3 矩阵，并检查了 708,588 种行/列重标记下的结果；规范签名均保持不变。对所有 36 个 `pn` 项和 2,880 个 `ppn` 项，均检查了所存算符见证是否为双射，并能否重构规范矩阵。

完整单元测试套件的 70 项测试全部通过。分发用 wheel 也成功构建，文件名为 `wick_diss_reduce-0.1.0-py3-none-any.whl`。

## 复现

安装可选数值依赖，并运行结构测试：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install '.[numeric]'
.venv/bin/python -B -m unittest discover -s tests -t . -p 'test_*.py'
```

运行八个算例的数值审计并生成小型 check 文件：

```sh
.venv/bin/python verify_code/check_basis_vs_original.py \
  --without-ppnn --include-eighth --dimensions 3,6 \
  --output-dir verify_output/optimized-check \
  --json verify_output/optimized-check/summary.json
```

生成并完整审计 ppnn check 数据：

```sh
.venv/bin/python verify_code/ppnn_check_stream.py \
  --output verify_output/optimized-check/ppnn_check.py
.venv/bin/python verify_code/ppnn_random_end_to_end.py \
  --check verify_output/optimized-check/ppnn_check.py \
  --color-dimension 3 --color-dimension 6
```

运行仅测 reducer 的 ppnn 基准：

```sh
.venv/bin/python verify_code/run_improved_ppnn.py
```
