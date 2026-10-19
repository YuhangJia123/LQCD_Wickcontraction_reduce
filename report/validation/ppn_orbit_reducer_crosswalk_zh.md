# ppn 轨道代表与 reducer basis 交叉表

本表将 93 个确定性的标签作用轨道代表与当前生产 reducer 的 `mode='detailed'` 结果逐项比较。`C[w]` 表示 Wick ID `w` 对应的 raw `contract(...)` 张量；生产 basis 视图包含该 basis Wick 项的 `wick_sign`。带符号 Wick 表达式的映射约定为：

`wick_sign[w] * C[w] = coefficient * transpose(wick_sign[b] * C[b], axes)`。

恒等 transpose 为 `(0, 1, 2, 3, 4, 5)`。独立候选项映射到自身。reducer 的依赖关系在当前数值 Dirac 投影下残差为零；这是当前夹具上的证据，并非对任意张量的证明。

reducer 计数：2,880 个 raw Wick、698 个 basis Wick、2,182 个依赖 Wick。轨道代表共 93 个，其中已属于依赖项的代表为 0 个。

| 轨道 | 候选 Wick | Reducer 状态 | 到生产 basis 的映射 | 残差范数平方 |
| ---: | ---: | --- | --- | ---: |
| 0 | 1 | basis | `-1*C[1]`（basis 代表） | 0 |
| 1 | 2 | basis | `1*C[2]`（basis 代表） | 0 |
| 2 | 4 | basis | `-1*C[4]`（basis 代表） | 0 |
| 3 | 7 | basis | `1*C[7]`（basis 代表） | 0 |
| 4 | 8 | basis | `-1*C[8]`（basis 代表） | 0 |
| 5 | 10 | basis | `1*C[10]`（basis 代表） | 0 |
| 6 | 13 | basis | `-1*C[13]`（basis 代表） | 0 |
| 7 | 14 | basis | `1*C[14]`（basis 代表） | 0 |
| 8 | 16 | basis | `-1*C[16]`（basis 代表） | 0 |
| 9 | 20 | basis | `-1*C[20]`（basis 代表） | 0 |
| 10 | 31 | basis | `-1*C[31]`（basis 代表） | 0 |
| 11 | 32 | basis | `1*C[32]`（basis 代表） | 0 |
| 12 | 34 | basis | `-1*C[34]`（basis 代表） | 0 |
| 13 | 38 | basis | `-1*C[38]`（basis 代表） | 0 |
| 14 | 62 | basis | `1*C[62]`（basis 代表） | 0 |
| 15 | 63 | basis | `1*C[63]`（basis 代表） | 0 |
| 16 | 64 | basis | `-1*C[64]`（basis 代表） | 0 |
| 17 | 241 | basis | `1*C[241]`（basis 代表） | 0 |
| 18 | 242 | basis | `-1*C[242]`（basis 代表） | 0 |
| 19 | 244 | basis | `1*C[244]`（basis 代表） | 0 |
| 20 | 247 | basis | `-1*C[247]`（basis 代表） | 0 |
| 21 | 248 | basis | `1*C[248]`（basis 代表） | 0 |
| 22 | 250 | basis | `-1*C[250]`（basis 代表） | 0 |
| 23 | 253 | basis | `1*C[253]`（basis 代表） | 0 |
| 24 | 254 | basis | `-1*C[254]`（basis 代表） | 0 |
| 25 | 256 | basis | `1*C[256]`（basis 代表） | 0 |
| 26 | 259 | basis | `-1*C[259]`（basis 代表） | 0 |
| 27 | 260 | basis | `1*C[260]`（basis 代表） | 0 |
| 28 | 262 | basis | `-1*C[262]`（basis 代表） | 0 |
| 29 | 271 | basis | `1*C[271]`（basis 代表） | 0 |
| 30 | 272 | basis | `-1*C[272]`（basis 代表） | 0 |
| 31 | 274 | basis | `1*C[274]`（basis 代表） | 0 |
| 32 | 277 | basis | `-1*C[277]`（basis 代表） | 0 |
| 33 | 278 | basis | `1*C[278]`（basis 代表） | 0 |
| 34 | 280 | basis | `-1*C[280]`（basis 代表） | 0 |
| 35 | 283 | basis | `1*C[283]`（basis 代表） | 0 |
| 36 | 284 | basis | `-1*C[284]`（basis 代表） | 0 |
| 37 | 286 | basis | `1*C[286]`（basis 代表） | 0 |
| 38 | 301 | basis | `1*C[301]`（basis 代表） | 0 |
| 39 | 302 | basis | `-1*C[302]`（basis 代表） | 0 |
| 40 | 304 | basis | `1*C[304]`（basis 代表） | 0 |
| 41 | 307 | basis | `-1*C[307]`（basis 代表） | 0 |
| 42 | 308 | basis | `1*C[308]`（basis 代表） | 0 |
| 43 | 310 | basis | `-1*C[310]`（basis 代表） | 0 |
| 44 | 331 | basis | `1*C[331]`（basis 代表） | 0 |
| 45 | 332 | basis | `-1*C[332]`（basis 代表） | 0 |
| 46 | 334 | basis | `1*C[334]`（basis 代表） | 0 |
| 47 | 481 | basis | `-1*C[481]`（basis 代表） | 0 |
| 48 | 482 | basis | `1*C[482]`（basis 代表） | 0 |
| 49 | 484 | basis | `-1*C[484]`（basis 代表） | 0 |
| 50 | 487 | basis | `1*C[487]`（basis 代表） | 0 |
| 51 | 488 | basis | `-1*C[488]`（basis 代表） | 0 |
| 52 | 490 | basis | `1*C[490]`（basis 代表） | 0 |
| 53 | 493 | basis | `-1*C[493]`（basis 代表） | 0 |
| 54 | 494 | basis | `1*C[494]`（basis 代表） | 0 |
| 55 | 496 | basis | `-1*C[496]`（basis 代表） | 0 |
| 56 | 499 | basis | `1*C[499]`（basis 代表） | 0 |
| 57 | 500 | basis | `-1*C[500]`（basis 代表） | 0 |
| 58 | 502 | basis | `1*C[502]`（basis 代表） | 0 |
| 59 | 511 | basis | `-1*C[511]`（basis 代表） | 0 |
| 60 | 512 | basis | `1*C[512]`（basis 代表） | 0 |
| 61 | 514 | basis | `-1*C[514]`（basis 代表） | 0 |
| 62 | 517 | basis | `1*C[517]`（basis 代表） | 0 |
| 63 | 518 | basis | `-1*C[518]`（basis 代表） | 0 |
| 64 | 520 | basis | `1*C[520]`（basis 代表） | 0 |
| 65 | 523 | basis | `-1*C[523]`（basis 代表） | 0 |
| 66 | 524 | basis | `1*C[524]`（basis 代表） | 0 |
| 67 | 526 | basis | `-1*C[526]`（basis 代表） | 0 |
| 68 | 541 | basis | `-1*C[541]`（basis 代表） | 0 |
| 69 | 542 | basis | `1*C[542]`（basis 代表） | 0 |
| 70 | 544 | basis | `-1*C[544]`（basis 代表） | 0 |
| 71 | 548 | basis | `-1*C[548]`（basis 代表） | 0 |
| 72 | 549 | basis | `-1*C[549]`（basis 代表） | 0 |
| 73 | 550 | basis | `1*C[550]`（basis 代表） | 0 |
| 74 | 572 | basis | `1*C[572]`（basis 代表） | 0 |
| 75 | 573 | basis | `1*C[573]`（basis 代表） | 0 |
| 76 | 574 | basis | `-1*C[574]`（basis 代表） | 0 |
| 77 | 1922 | basis | `1*C[1922]`（basis 代表） | 0 |
| 78 | 1923 | basis | `1*C[1923]`（basis 代表） | 0 |
| 79 | 1924 | basis | `-1*C[1924]`（basis 代表） | 0 |
| 80 | 1928 | basis | `-1*C[1928]`（basis 代表） | 0 |
| 81 | 1929 | basis | `-1*C[1929]`（basis 代表） | 0 |
| 82 | 1933 | basis | `-1*C[1933]`（basis 代表） | 0 |
| 83 | 1934 | basis | `1*C[1934]`（basis 代表） | 0 |
| 84 | 1936 | basis | `-1*C[1936]`（basis 代表） | 0 |
| 85 | 1939 | basis | `1*C[1939]`（basis 代表） | 0 |
| 86 | 1940 | basis | `-1*C[1940]`（basis 代表） | 0 |
| 87 | 1942 | basis | `1*C[1942]`（basis 代表） | 0 |
| 88 | 1957 | basis | `1*C[1957]`（basis 代表） | 0 |
| 89 | 1958 | basis | `-1*C[1958]`（basis 代表） | 0 |
| 90 | 1963 | basis | `-1*C[1963]`（basis 代表） | 0 |
| 91 | 1964 | basis | `1*C[1964]`（basis 代表） | 0 |
| 92 | 1966 | basis | `-1*C[1966]`（basis 代表） | 0 |

生成器会在每个拟议标签轨道的全部 raw Wick 成员中（包括每个商状态内的 8 种重子槽位变体），选择生产 reducer 判定为独立的最低 Wick ID。因而所选的 93 个调用全部是生产 basis 元素，各自映射到自身，系数为 1，transpose 为恒等变换。成员资格本身不能证明完整 raw Wick 总和落在这 93 项的张成空间中；随机张量张成空间审计对此另行检查。

复现命令：

```sh
python3 -B verify_code/ppn_orbit_reducer_crosswalk.py
```

逐轨道完整 ID 表和映射数据也可与原始报告 [ppn_orbit_reducer_crosswalk.md](../../verify_code/ppn_orbit_reducer_crosswalk.md) 对照。
