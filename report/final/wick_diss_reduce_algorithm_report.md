# 蒸馏 / normal Wick 约化项目报告

本报告是原蒸馏 Wick 约化报告的扩展版。正文固定使用三层结构：前人工作、算法、不同输入/模式/输出及对比。算法页保留原有 VVV/VDV、peram、incidence、Gamma、裸轴转置和 source Gram 的定义；本版把普通颜色的 incidence + 全局 S3 融合接入同一项目接口，并把输出统一路由到 `result/{input}_{mode}_{output}/`。

## 1. 前人工作

- PACS-CS（用户描述中的 PCAS）报告了特定投影和核子 block 口径下的 `^3He: 93`、`^4He: 1,107`。本地声明群作用复现为 `93/1068`；一个 ppn block 展开 8 个内部 Wick 项，一个 ppnn block 展开 16 个。93/1068/1107 是高层 block 计数，不自动等于开放 Dirac 张量的独立 `contract` 数；推导放在 PDF 附录。
- UCA 和递归法分别把颜色–Dirac 系数整理为统一列表、或逐重子递归构造并复用反对称中间分量。普通颜色 3H 同输出重跑数据见第 3.2 节。
- Redstar 只作为图收缩方法背景介绍；本报告不把其论文硬件和输出口径与本项目的 contract 字典混排。

## 2. 算法

输入是有序算符对 `[O_sink, O_source]`。顶层 `input` 有两类：

- `distillation`：VVV/VDV、peram、裸指标和具体 Gamma；
- `normal`：普通颜色 delta/epsilon、传播子 incidence 和具体 Gamma。

`mode` 有 `wick` 和 `special`。`wick` 对所有合法味道配对逐图处理；`special` 只对 normal 路径启用 incidence + S3 融合及其专用统计。主 `output` 为：

- `independent`：独立 contract/basis 与系数；
- `check`：原始 Wick 到 basis 的逐图回代关系；
- `block`：normal-wick 的 W/block 计划、展开证书和重构结果；
- `count`：normal-special 合并后非零项的计数。

组合约束为 `distillation-wick-*`、`normal-wick-*` 和 `normal-special-*`；`block` 只属于 normal-wick，`count` 只属于 normal-special。输出表达式沿用两个原项目的格式，结果目录使用输入、模式、输出三元组命名。

约化顺序保持不变：Wick 枚举 → peram/incidence 分类 → 哑指标规范化和裸轴转置 → sink VVV/VDV 与 Gamma 筛选 → source incidence/Gram 线性独立性 → basis、系数和回代关系。normal-special 的颜色 S3 只在张量变换律和开放轴条件经过检查时启用；否则回退到逐图合并。

内积比较可推广到 Dirac 直积空间。令 `A_ab=(C gamma5)_ab`、
`B_ab=(C gamma5)_ba`，项目矩阵给出 `tr(A†A)=4`、`tr(A†B)=-4`。
因此归一化内积为 `tr(A†B)/4=-1`，且 `B=-A`。
欧式 Gamma 满足 `tr(gamma_mu† gamma_nu)/4=delta_mu_nu`；
`I, gamma5, gamma_mu, gamma5 gamma_mu, sigma_mu_nu (mu<nu)` 构成 16 个正交基，
这里 `sigma_mu_nu=[gamma_mu,gamma_nu]/2`。`gamma5 sigma_mu_nu` 与 sigma 对偶，
不增加独立方向。n 个 Dirac 因子的归一化是 `4^(-n)`；固定因子不影响 Gram 的
线性相关判定。源端结构对齐后，也可用零投影残差识别 Fierz 重排形成的线性组合。

## 3. 不同模式与对比

### 3.1 功能、输出与验证

现有 distillation-wick-independent 的参考计数仍为：`ppn` 为 2,880 raw Wick、219 basis contract、2,661 dependent；`ppnn` 为 518,400 raw Wick、4,298 basis contract、514,102 dependent。随机 peram/VVV 回代在 `N_c=3,6` 下通过；ppnn 的 518,400 条关系使用流式 check 文件和固定随机投影核验。

2026-10-14 的 normal-special 验收（macOS arm64、Python 3.11.17、独立进程）包含 7 个可完整 materialize 的小例子：`pipi` 264 项、`pn` 9,432 项、`ppn` 3,775,680 项、`delta` 54 项、`delta++delta-` 2,916 项、`pip` 1,260 项；对应 fused 中位时间分别为 0.000630、0.004905、2.707331、0.000455、0.002387、0.001106 s，峰值 RSS 最大值分别为 20.2、22.0、833.2、20.3、20.9、20.2 MiB，结果摘要逐项一致。`pip-p` 不满足完整 fused 路径的场数约束，exact independent 输出为 107,136 项，1.535478 s，317.9 MiB。第八个 `pn_omega_pi+` 完成 3,456 个 raw Wick 枚举，但完整 fused 字典超过 100,000,000 项安全上限，因此只记录 bounded failure，不伪造完整输出规模。

`ppnn` 的 normal-special-count 重新运行得到 518,400 Wick maps、`Nlist=1,407,974,400`、`Ncontr=87,998,400`、峰值 RSS 193.5 MiB，总计时 4.407 s。它是 exact count-only 输出，没有生成完整坐标字典，不能代替 normal-special-independent 的完整输出基准。

### 3.2 3H 同输出对比：UCA、递归和 incidence + S3

以下四行使用同一个 3H（p+n+n）完整颜色–Dirac 系数字典，source spins 为 `(0,0,1)`，
输出均为 3,775,680 项。2026-10-15 在 Python 3.11.17、macOS arm64 上，
各运行三个独立进程；时间取中位数，峰值 RSS 取三轮最大值。
每个候选字典在计时外与首轮 UCA 参考逐键、逐整数系数比较，缺失、多余、差异均为零。
UCA 首轮建立参考，后两轮与该参考逐项相等；其余三种方法各三轮全部相等。
时间不含进程启动、导入、输入读取、SQLite 序列化和核对。
递归实现是本地稀疏反对称积复现，不等同于论文依赖 sink 的 Lambda 列表。

| 算法 / 来源 | 中位时间 (s) | 峰值 RSS | 输出项 | 一致性 |
| --- | ---: | ---: | ---: | --- |
| UCA-P1 direct 本地复现 | 18.6046 | 1,007.86 MiB | 3,775,680 | 参考字典，另两轮逐项相等 |
| qcd colour-first 本地快照 | 14.5085 | 1,146.70 MiB | 3,775,680 | 三轮逐项相等 |
| 本项目 normal-special-independent | 5.1505 | 1,103.75 MiB | 3,775,680 | 三轮逐项相等 |
| 递归反对称积 + 完整展开 | 13.1612 | 620.50 MiB | 3,775,680 | 三轮逐项相等 |

### 3.3 PACS-CS、群论 block 与本项目输出

| 输入/方法 | 原始 Wick | 结果/块 | contract 数目（融合/展开） | 全流程时间 | 峰值内存 |
| --- | ---: | --- | ---: | ---: | ---: |
| ³He / ppn，PACS-CS 论文口径 | 2,880 | 93 个构造更大的传播子 W 的 block | 93 / 744 | 论文未报告 | 论文未报告 |
| ³He / ppn，本地群论复现 | 2,880 | 93 blocks | 93 / 744 | 0.4898 s | 25.1 MiB |
| ³He / ppn，normal-wick-block | 2,880 | 48 networks | 48 / -- | 15.8508 s | 176.66 MiB |
| ³He / ppn，distillation-wick-independent | 2,880 | 219 basis | -- / 219 | 1.265962 s | 27.63 MiB |
| ⁴He / ppnn，PACS-CS 论文口径 | 518,400 | 1,107 blocks | 1,107 / 17,712 | 论文未报告 | 论文未报告 |
| ⁴He / ppnn，本地群论复现 | 518,400 | 1,068 blocks | 1,068 / 17,088 | 90.5676 s | 619.2 MiB |
| ⁴He / ppnn，normal-wick-block | 518,400 | 288 networks | 288 / -- | 211.2 s（数值执行） | 12.25 GiB |
| ⁴He / ppnn，distillation-wick-independent | 518,400 | 4,298 basis | -- / 4,298 | 288.688 s | 410.23 MiB |

这里的构造对象为反对称两传播子：

`W_{ab;ij} = S_{ai} S_{bj} - S_{aj} S_{bi}`，其中 `a,b` 是 sink 颜色--Dirac
复合指标，`i,j` 是 source 端复合指标。48-network 的新测量采用 Python 3.11.17，
时间包括三表生成、适配、规划与内置符号证明、执行表落盘；峰值 RSS 在数值验证前采样。
这与表中 ppnn 的 211.2 s 数值执行时间范围不同。

48/288-network 验证使用随机 normal peram 与固定 epsilon，不随机生成 VVV。
完整张量的最大绝对差除以两结果的最大幅值，ppn 新测为 `4.1613e-15`，
ppnn 为 `4.5147e-14`；与逐 Wick 的相对误差定义分开统计。

### 3.4 本项目优势与展望

本项目支持任意动量/坐标、任意味道的介子--重子混合，并把输入、Wick、special、block、count 统一到可审计输出目录。PACS-CS 和局部群论 block 需要提前声明并证明特定交换、同位旋和投影结构；本项目可在结构规范化后再检查 transporter、Dirac payload 和轴置换。完整颜色--Dirac 字典的同一 3H 三轮中位时间为 UCA 18.605 s、本项目 normal-special-independent 5.151 s，且三轮均与参考逐项相等。

机器可读数据位于：

- `../data/benchmark_20261015/gamma_inner_product.json`
- `../data/benchmark_20261015/ppn_block.json`
- `../data/benchmark_20261015/special_three_runs.json`
- `../data/benchmark_20261014/uca_3h_rerun_20261014.json`
- `../data/benchmark_20261014/recursive_3h_rerun_20261014.json`
- `../data/benchmark_20261014/seven_examples_rerun_20261014.json`
- `../data/benchmark_20261014/eighth_example_pn_omega_pi_plus_20261014.json`
- `../data/benchmark_20261014/ppnn_count_rerun_20261014.json`
- `../validation/ppnn_fixed_288_audit_zh.md`
- `../data/benchmark_20261014/ppnn_fixed_288_progress_20261014.json`
- `../data/benchmark_20261014/pnn_special_input_20261014.json`
- `../validation/global_block_validation.json`

PDF 源文件为 [`wick_diss_reduce_algorithm_beamer.tex`](wick_diss_reduce_algorithm_beamer.tex)，编译输出为 [`wick_diss_reduce_algorithm_beamer.pdf`](wick_diss_reduce_algorithm_beamer.pdf)。
