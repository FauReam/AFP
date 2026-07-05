---
name: experiment-triage
description: AFP experiment triage — pre/post evaluation of every experiment. Prevents GPU time waste. Use when user wants to run any experiment, training, IVN, grid search, or asks about experiment results. Also invocable via /tri or /triage.
allowed-tools:
  - Read
  - Write
  - Bash
  - Edit
---

# Experiment Triage — AFP 实验评估

在跑任何实验前后强制三问评估。防止 GPU 时间浪费。

## 命令

| 命令 | 作用 |
|------|------|
| `/tri pre <描述>` | 跑实验前的 pre-triage |
| `/tri post` | 跑完实验后的 post-triage，读取最新日志自动评估 |
| `/tri list` | 列出所有历史实验评估（从 EXPERIMENT_PLAN.md 提取） |
| `/tri drift` | 显示所有标记为 DRIFTING 的实验和教训 |
| `/tri` | 等价于 `/tri pre`，要求用户描述想做的实验 |

## 自动触发

当用户说这些话时自动调此 skill：
- "跑一下"、"测一下"、"试一下"、"跑个实验"
- "train"、"训练"、"IVN"、"grid search"、"队列"
- "结果怎么样"、"跑完了"、"实验进度"

## Pre-triage 流程

读取 `docs/internal/EXPERIMENT_PLAN.md` 了解当前论文主线和已知 DRIFTING 模式。

向用户输出四行评估：

```
假设:    [一句话：这个实验预期验证什么？]
论文价值: [HIGH / MED / LOW / NONE]
方向判定: [ON TRACK / DRIFTING RISK]
预计耗时: [分钟]
```

决策规则：
- **NONE**（纯工程）→ 直接跑
- **LOW** → 提醒但不阻止
- **MED/HIGH + DRIFTING RISK** → **阻止，要求明确确认**
- **>60min + LOW** → **阻止**

已知 DRIFTING 模式（直接阻止）：
- 不对称差异（一方 1ep + 一方 3/5ep）
- 模型质量不达标就 IVN（B→code < 0.76）
- 改 importance 不改 τ

## Post-triage 流程

1. 读取最新的实验结果（`experiments/phase0_ivn/results/` 或训练日志）
2. 在 `EXPERIMENT_PLAN.md` 的评估表中追加一行
3. 更新有效/浪费 GPU 时间计数器
4. 如果 DRIFTING：写教训总结
5. 输出下一优先级的实验建议

## 批量实验

用户要排队跑 N 个实验时：
1. 只跑第 1 个
2. Post-triage 判定：ON TRACK → 跑下一个；DRIFTING → 终止
3. 禁止一口气全跑
