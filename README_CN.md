# PaD GitHub 复现包 v4

论文冻结为 **v08**。本次只整理复现工程：学习结构、原 P 系数、数据划分、候选池、种子和论文结果均不变。

## 最快核验

Python 3.12，Linux，CPU：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python reproduce.py verify
```

该命令复算全部 120 个机制学习器的测试指标、B1 九个分类认证、B2 配对等效检验，以及 Costanzo、Jonikas 和背景削减结果；同时核查文件完整性和原始测量编码。**复算不等于重新训练。**

重新训练前安装完整依赖，再做小规模执行检查：

```bash
python -m pip install -r requirements.txt
python reproduce.py smoke
```

## 三组正式实验

| 实验 | 正式入口 | 保留的证据 |
|---|---|---|
| 机制学习 | `python reproduce.py mechanism` | 30 个机制条件/种子块 × 4 个学习器，全部 4,096 条测试记录、配置、认证与失败项 |
| Costanzo 2016 | `python reproduce.py yeast --dataset costanzo`；`python reproduce.py external --dataset costanzo`；`python reproduce.py red --dataset costanzo` | PaD、内部归因对照及外部监督方法/Réd 的全部预定种子 |
| Jonikas 2009 | `python reproduce.py yeast --dataset jonikas`；`python reproduce.py external --dataset jonikas`；`python reproduce.py red --dataset jonikas` | 168 条有序参照记录、外层选择、排名与已知方向结果 |

`python reproduce.py calibrate` 重建独立选档与确认；`python reproduce.py background` 重建 100%、50%、0% 背景可用性核对。50% 是一半**无向候选对**保留原有完整背景，不是每对删除一半伙伴。

仅检查一次完整训练块，可运行：

```bash
python reproduce.py mechanism --scenario B1 --seed 17 --level ideal
```

所有生成结果放在 `runs/`。可通过 `--output` 指定其他目录，不会覆盖 `reference/` 中的正式交付结果。详细说明见 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 清理原则

删除重复模块、旧架构探索、先导/调试日志、重复选参缓存、模型 pickle、重复 Réd 因子、重复清单及无关子面板准备文件。保留复现实验必需的输入、原始 JSON、完整预测与数据来源；大 CSV 采用无损压缩，没有删样本或降低浮点精度。

Costanzo 内部匹配加法的 613/657、排序认证未通过、50% 背景配置的下降等结果仍然可查。这些是科学证据，不作为“调试历史”删除。

真实数据 PaD 的可复现起点为冻结的 P/D 特征；不会将这称为已经重建原 P 全部历史训练过程。机制实验则包含原始表达/表型生成、完整编码、训练和独立测试的全过程。

上传 GitHub 时将本目录内容作为仓库根目录。默认工作流执行核验，不自动启动昂贵的全量外部训练。

训练后使用相同输出根目录运行 `python reproduce.py report`，生成本次结果汇总和完成度清单；未运行项目不会用历史结果填充。
