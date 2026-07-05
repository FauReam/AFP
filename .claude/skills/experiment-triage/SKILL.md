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
| `/tri review` | 验收模式：统一评估所有已完成实验，更新有效/浪费计数器 |

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
- **NONE**（纯工程）→ 直接跑，不输出 triage
- 其他所有 → 输出四行评估作为**信息标签**，不阻止执行
- 进程安全（nohup, nohup, nohup）永远是第一优先级
- DRIFTING 标记留到验收时统一讨论

## Post-triage 流程

1. 读取最新的实验结果（`experiments/phase0_ivn/results/` 或训练日志）
2. 在 `EXPERIMENT_PLAN.md` 的评估表中追加一行
3. 更新有效/浪费 GPU 时间计数器
4. 如果 DRIFTING：写教训总结
5. 输出下一优先级的实验建议

## 批量实验（自动序列）

用户排队跑 N 个实验时，**全部自动串行跑完**。进程安全（nohup, 时序日志, 崩溃恢复）优先于实验评估。

流程：
1. Pre-triage 快速扫描，标记已知 DRIFTING 但**不阻止执行**
2. 写入一个自动序列脚本，每个实验继承前一个的退出码和环境
3. 所有 nohup + &，日志带时间戳，PID 记录
4. 全程不等待用户确认

验收时统一评估：
1. 用户说"验收"/"结果"时，`/tri post` 逐项评估
2. 更新 EXPERIMENT_PLAN.md 的有效/浪费时间计数器
3. 标记 DRIFTING 的写教训
