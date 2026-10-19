# 多 transporter 的蒸馏 Wick 约化

本项目按每个 peram incidence 类的首张 Wick 图建立固定基准。实现位于 `kernels/wick_reducer/`，入口位于 `kernels/wick_reducer/run_examples.py`。Wick 枚举和 4x4 Dirac 矩阵收缩原语由项目根目录的 `kernels/correlator_wick_normalizer.py` 提供；peram incidence 预筛代码位于 `kernels/wick_reducer/peram_incidence.py`。项目不依赖其他 qct/qcd 目录。

## 约定

- 析构算符的夸克组成沿用 qct 的标准表：`p=(u,u,d)`、`n=(d,d,u)`、`Delta++=(u,u,u)`、`Delta-=(d,d,d)`，介子沿用 `pi0_uubar` 分量。Wick 配对始终按精确味道生成；同位旋模式只作用于后续 peram 等价比较，默认开启，并且只将 `u`、`d` 映射到同一个数值味道类，`s` 等其他味道仍各自独立。可用 `--no-isospin-symmetry` 关闭 u/d 等价。
- `operator_model.make_operator()` 接受有序夸克组成。重子 flavors 是三个夸克味道；介子 flavors 是 `(反夸克味道, 夸克味道)`，因此 pi+ sink 为 `("d", "u")`，其 source 共轭为 `("u", "d")`。一个元素的介子输入保留为同味道 `qbar q` 的简写。字段生成保留每个端点的精确味道，因而任意异味介子也能进行精确 Wick 配对。
- 每个算符可带一个作用于两个槽位的 Dirac 矩阵。标准矩阵使用内置名称；自定义矩阵通过 `gamma_matrices={name: matrix}` 传入具体数值 4x4 矩阵，算符的 `gamma=name` 只用于查找该矩阵。诸如 `gamma1 * gamma5` 的乘积由调用方先算成 4x4 矩阵再传入；本实现不解析 Gamma 字符串表达式。比如：

  ```python
  concrete_matrix = (
      (1, 0, 0, 0),
      (0, 2, 0, 0),
      (0, 0, 3, 0),
      (0, 0, 0, 4),
  )  # put the actual product matrix entries here
  operators = (
      make_operator("sink", "meson", "M", "sink_M", ("u", "u"), "G_custom"),
      make_operator("source", "meson", "M", "source_M", ("u", "u"), "G_custom"),
  )
  result = reduce_operators(
      operators, label="custom", gamma_matrices={"G_custom": concrete_matrix}
  )
  ```
- peram 记录含 q/bar 方向、`sin_sin/sin_sou/sou_sin/sou_sou`、`bb/b0/0b/00`、端点算符和场类型。哑指标和裸指标的具体字母不进第一阶段 key；bare/dummy 身份、sink/source 边界和 q/bar 方向会保留。因而 `A/D` 或 `i/j` 可重命名，但裸指标不会与哑指标互换，也不会把 sink 端和 source 端暗中互换。
- 第一阶段使用紧凑的 peram incidence key 建桶。它是候选预筛；最终的线映射还必须逐条保持 peram 类型、裸/哑角色、方向和算符端点连接。唯一的 generic factor-graph key 不把 meson/baryon species 名字作为顶点颜色，只保留 side、VVV/VDV 类型、Gamma/槽位、开放槽、外部/动量标签和连接边。同位旋简并时只把 `u`、`d` 线按同一数值 flavor 类别匹配；`s` 等其它 flavor 仍保留。
- 公开约化入口不依赖 `ppn` 或 `ppnn` 名称：`reduce_operators(operators, label=..., options=ReductionOptions(...))` 可直接处理任意 `Operator` 序列，`label` 只写入输出元数据。`reduce_case` 仅是内置夹具和输出模块使用的内部实现帮手；它不再提供另一种 species 策略。
- 对当前夹具，VVV 和 Gamma 与 hadron 名字无关，因此 generic 模式的同位旋数值条件就是 `peram_u` 与 `peram_d` 相同；不同 momentum、投影、`external_tag`、Clebsch 系数或额外相位会继续留在 key 中，不能被自动合并。
- species-blind 因子图只负责提前识别可能相同的连接图。原有的 peram 线映射、哑指标重命名、VVV parity、bare-axis transpose、sink projective 检查和 source Dirac Gram 约化仍然执行；这些步骤处理保持 peram 集合不变、同时使 sink 等价的剩余指标对称性。
- VVV 的腿没有固定槽位顺序，合法 transporter 可以交换腿并产生 epsilon parity。VDV 两腿保持 `bar -> q` 的有向槽位关系。Gamma 矩阵随它所属的算符字段一起移动，而不是从 VVV incidence 中提前剥离。

## 分阶段比较

1. **peram 候选类。** 只让同一 profile 和有向 peram incidence key 的图比较。每类第一张图为基准，不对它做约化。
2. **枚举 transporter。** 对重复算符块映射和逐条 peram 线完美匹配做受限回溯。每个解给出算符置换、候选线到基准线的映射、逆 line order、sink/source VVV parity，以及 bare axis map。bare 轴映射非恒等时设置 `transpose_required`，供数值后端恢复轴顺序。
3. **sink Dirac 门控。** 每个 transporter 先构造 sink Gamma payload；source payload 延迟到 sink 命中已有 sink 类后才生成。对每个 transporter 计算候选 sink Gamma 张量与本 peram 类基准 sink Gamma 张量的内积。如果所有 overlap 都为零，直接认定这些 transporter 不可能与该基准相连；不计算 transporter 之间的 sink 秩，也不构造 source payload 或 source Gram。
4. **sink 多解秩。** 只要存在非零 overlap，先去重相同的 sink Dirac payload，再以 Gram 投影增量求 transporter sink 子空间的秩。对每个独立方向报告它与基准的归一化 overlap、投影系数和残差。只有 sink 张量与某个已知 sink 类成比例时，才进入该类的 source 检查；仅有非零 overlap 不会被误当成相同张量。
5. **source 约化。** sink 成比例后，将该比例吸收到系数中，并在同一 sink 类内对 sink/source 张量积执行 Dirac Gram 投影。依赖关系保存 basis Wick、复系数、残差，以及产生该关系的 transporter。多个 transporter 只作为同一个 Wick 的备选见证，不会重复累加 Wick 系数；若多解都导出依赖关系，会保留 transporter tie 数。
6. **输出。** `independent` 返回独立原始 Wick 图和约化系数；`detailed` 还返回每个依赖关系的 transporter 轨道与 sink overlap/秩报告；`coarse` 只流式统计 Wick 和 profile。

对于任意数目的解，sink rank 的投影系数是复数，算法没有将 overlap 限定为 0/1。若没有一个 sink 方向与基准成比例，该图留在独立 sink 类，不会调用 source Gram。若有若干不同 transporter 方向，其中只有部分方向与现存 sink 类成比例，则只对这些方向继续 source 检验。

## 性能和资源

Wick 图逐项生成。factor-graph 颜色在每个 reduction 开始时分配确定性的整数 ID，规范图签名以带长度的 `bytes` 保存。bucket 使用结构化数组保存 Wick ID、符号、定宽 pairing 和定宽 canonical operator map；ppnn 的映射额外占用约 4 MiB。保存这个紧凑 witness 可以避免 bucket reduction 中对每项 Wick 再做一次完整图规范化。pairing 与映射通过缓存的 `struct.Struct.unpack_from` 直接从连续 buffer 解码，不创建中间 pairing `bytes` 对象。不会为每张 Wick 图构造完整 TensorNetwork。

重复算符分组、每条可能 peram 线的 edge color，以及受限排列表在一次 reduction 中预计算并复用。二分图 canonicalizer 先进行保持同构不变的颜色细化；只有 sink 和 source 两侧的所有顶点都被唯一标记时，才按该标记顺序生成完整 edge-color 矩阵 key。否则仍执行有界的精确排列搜索，因此对称图不会依赖 refinement 近似分类。线匹配用最少候选优先的回溯，并在 16384 个 transporter 时显式报错，避免静默截断后错误合并。Dirac inner 只访问 4x4 因子，不展开全部 spin tensor 元素。

独立模式保留每个 bucket 的当前 sink/source Gram basis；详细模式另保留依赖关系与 transporter witness，因此内存会随输出详情增加。`write_output(mode="check")` 使用独立约化和 relation callback，把依赖映射暂存到 SQLite，并重新枚举 raw Wick 输出标准的 `basis_contract`、`origen_wick_raw`、`origen_wick` 和 `origen_wick_to_independent`，不要求 reducer 返回全量 `graph_records`。Wick 数和最坏 transporter 数仍可能随算符数阶乘增长；遇到超出 transporter 上限的高对称输入时，本实现会明确失败，而不会声称得到了完整约化。大型体系应优先用 `coarse` 探测 profile，再对目标相关函数做 `independent` 或 streamed `check`。

## 八个回归例子

内置例子的 flavor-compatible Wick 数必须为：

| 例子 | Wick 数 |
| --- | ---: |
| `pipi` | 24 |
| `pn` | 36 |
| `ppn` | 2880 |
| `delta` | 6 |
| `delta++delta-` | 36 |
| `pip` | 24 |
| `pip-p` | 6 |
| `pn_omega_pi+` | 3456 |

`pn_omega_pi+` uses sink `p+n+Omega-+pi+` and the conjugate source. The
Omega- interpolator uses the existing spin-3/2 `Cgamma3` / dagger convention;
the charged pion is `dbar u` at the sink and `ubar d` at the source.

## 补充本机实测

下表记录早期 species-aware 约化路径的运行数据。当前通用路径的维护中
计数见项目根目录 `README.md`；本表只保留作历史对照。

下面是在当前工作区同一次 benchmark 中测得的秒数。`independent` 输出代表图与合并系数；`detailed` 额外保留依赖关系和 transporter witness。两种模式执行相同的判别流程，计时差主要来自详细结果组装。

| 例子 | Wick 数 | 独立图 | 依赖图 | coarse | independent | detailed | transporter 解最大数 | sink 秩检查数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pipi` | 24 | 10 | 14 | 0.000679 | 0.007532 | 0.007761 | 2 | 14 |
| `pn` | 36 | 35 | 1 | 0.001311 | 0.009142 | 0.010067 | 1 | 1 |
| `ppn` | 2880 | 698 | 2182 | 0.127208 | 1.917658 | 1.948887 | 16 | 2182 |
| `delta` | 6 | 2 | 4 | 0.003189 | 0.001927 | 0.001997 | 2 | 4 |
| `delta++delta-` | 36 | 4 | 32 | 0.001292 | 0.016006 | 0.017025 | 4 | 32 |
| `pip` | 24 | 24 | 0 | 0.000976 | 0.006287 | 0.006717 | 0 | 0 |
| `pip-p` | 6 | 6 | 0 | 0.000240 | 0.001670 | 0.001611 | 0 | 0 |
| `pn_omega_pi+` | 3456 | 1144 | 2312 | 0.249241 | 2.581081 | 2.566801 | 8 | 2312 |

计时单位为秒；该表为一次本机重跑，受系统负载影响，不应把小数末位当作稳定基准。七个例子的 Wick 数和独立/依赖图数与前次记录一致；新增 `pn_omega_pi+` 有 3456 张 Wick 图。味道类和算符置换已按相关函数预计算，且 sink 未通过时不构造 source payload。与上一轮单次测量相比，ppn independent 为 1.92 秒、`pn_omega_pi+` 为 2.58 秒；这组计时包含系统负载波动，适合看量级，不宜据此计算精确加速比。

`delta` 的 Wick 2 有两条 sink transporter 解；它们的 sink Dirac 张量秩为 1，二者均与基准成比例，source Gram 将该图合并，因此首个独立代表的约化系数为 2。

运行全模式基准：

```sh
python3 kernels/wick_reducer/run_examples.py --mode benchmark
```

单例详细检查和 JSON 导出：

```sh
python3 kernels/wick_reducer/run_examples.py --example delta --mode detailed --json
python3 kernels/wick_reducer/run_examples.py --example ppn --mode independent --output verify_output/ppn.json
```

可用 `python3 -B -m unittest discover -s tests -t . -p 'test_*.py'` 执行回归测试。
