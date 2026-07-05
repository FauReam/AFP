# AFP — Claude Code 项目上下文

> **论文**: [docs/internal/PAPER.md](docs/internal/PAPER.md)
> **实验**: [docs/internal/EXPERIMENT_PLAN.md](docs/internal/EXPERIMENT_PLAN.md)
> **Bug 清单**: [docs/internal/ENGINEERING.md](docs/internal/ENGINEERING.md)

## 入口

```bash
# 训练
bash scripts/train_and_run_phase0.sh

# LMC 扫描
python -u scripts/lmc_barrier_scan.py

# 看状态
bash scripts/monitor.sh
```

## 关键文档

| 文件 | 内容 |
|------|------|
| `docs/internal/EXPERIMENT_PLAN.md` | 实验方案 + 全部数据 + 结论 |
| `docs/internal/ENGINEERING.md` | 21 条已知 Bug |
| `docs/internal/ROOT_CAUSE_IMPORTANCE.md` | 权重差异分析 |
| `experiments/RESTART_PROMPT.md` | 新会话重启指南 |

## 工作约定

- 提交信息用英文，`feat(scope): description`
- 训练用 `nohup ... &`，日志写 `experiments/`
- Python 加 `-u`（无缓冲）
- HF 调用加 `local_files_only=True`
- 跑任何实验前先读 `EXPERIMENT_PLAN.md` 的当前状态
