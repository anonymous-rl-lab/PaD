# 补充：方向分量重算与逐记录分解

数据来自 `scc_predictions/` 的逐条账本，**没有重跑任何实验、没有改任何设置**。
这些量是事后的描述性计算，**不是预注册的**，不替换 `scc_readings.json` 里的
J2 / X3 / C3 落点。

## 1. 为什么会有两个数字

Jonikas 是非有向面板，论文冻结的统计量是 `AUROC(score)`。每条记录的分数可以拆成

```
score = support + direction ,   direction = (score - reverse_score)/2
```

`direction` 是端点交换下的反对称部分，也就是模型对"A→B 还是 B→A"的判断；
`support` 是对称部分。用 `direction` 打分等于只看方向、丢掉对称部分。

## 2. 两种打分下的同一批模型（Jonikas，168 条）

| 模型 | AUROC(score) | AUROC(direction) | 21 个标注正例中方向为正 |
|---|---|---|---|
| PaD | 0.881114 | **0.844833** | 20 / 21 |
| `A_match` | 0.827664 | 0.714286 | 16 / 21 |
| `A_rbf` | 0.810982 | 0.678167 | 17 / 21 |
| `D_only` | 0.834467 | 0.713638 | 18 / 21 |
| `P_only` | 0.676709 | 0.563654 | 11 / 21 |
| `Dplus_only` | 0.746032 | 0.622935 | 13 / 21 |

换成方向分量后，所有模型都下降，但**PaD 下降得最少**（−0.036），
`D_only` 下降 −0.121，`A_match` 下降 −0.113。

## 3. J2 的增量，两种打分

```
用 score      +0.046647    95% [-0.0157, +0.1116]   93.5% 的重抽为正
用 direction  +0.131195    95% [+0.0355, +0.2412]   99.7% 的重抽为正
```

**用方向分量算，区间不含零。** 配对自助，4000 次，重抽单位是 84 个无序对。

这值得报，但必须连同下一节一起报，并且标明是事后量。把预注册的 J2 落点换成
这个数字是事后换统计量，不做。

## 4. 逐记录分解：方向正确性（Jonikas 168 条）

| | `D_only` 对 | `D_only` 错 |
|---|---|---|
| **PaD 对** | 79 | **24** |
| **PaD 错** | **18** | 47 |

净 **+6 / 168**。仅看 21 个标注正例：PaD 独有 2 条，`D_only` 独有 0 条。

所以逐记录看，PaD 相对 `D_only` 的优势是**温和的**——24 比 18，不是
AUROC(direction) 的 +0.131 在直觉上暗示的那么大。两个数字都对，它们量的是
不同的东西：AUROC 是排序质量，24/18 是硬判决的翻转计数。**论文两个都要给。**

## 5. 逐记录分解：Costanzo 657 对

| | `D_only` 对 | `D_only` 错 |
|---|---|---|
| **`P_only` 对** | 456 | 141 |
| **`P_only` 错** | 45 | 15 |

正确数：`P_only` 597，`D_only` 501，PaD 611，`A_match` 613，`A_rbf` 606。

两条从这张表直接读出来的：

```
两个单通道都错、PaD 对的      0 / 657
至少一个单通道对、PaD 错的   31 / 657
```

**PaD 在 Costanzo 上没有救回任何一个两个单通道都失败的对。** 它的 611 完全落在
`P_only ∪ D_only` 能做对的 642 对之内，并且在其中 31 对上反而做错了。

这比 C3 的点估计更硬：C3 说"Costanzo 是 P 驱动的"，这张表说**联合项在
Costanzo 上没有产生任何单通道达不到的正确判断**。论文里 Costanzo 的等价读法
因此要进一步收紧——不是"加法够用"，而是"联合项在这块面板上没有新增任何东西"。

## 6. 账本清单

`scc_predictions/` 现在有 11 份，格式与现有 ledger 相同
（`a,b,pair,target,score,reverse_score,direction,support`）：

- SCC 五个预注册 cell：`P_only_*`、`D_only_*`、`Dplus_only_jonikas_Dplus`
- 论文现有学习器六份：`PaD_*`、`A_match_*`、`A_rbf_*`

Jonikas 的三份 PaD/A_match/A_rbf 是重新拟合的（AUROC 复现 0.881114 /
0.827664 / 0.810982）；Costanzo 的三份从 `reference/costanzo/internal_controls.csv`
转换而来（611 / 613 / 606），`reverse_score` 由 `score − 2·direction` 还原。

之所以需要重新产出，是因为归档的 `reference/jonikas/pair_predictions.csv`
**只有 score 列，没有方向分量**，本文件第 2 到 4 节的任何一项都算不出来。
