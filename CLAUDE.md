# AFP — Claude Code 项目上下文

> **新会话必读**：[docs/internal/EXPERIMENT_PLAN.md](docs/internal/EXPERIMENT_PLAN.md)

## 研究问题

**Domain-specific fine-tuning 产生的模型权重差异有多大？这些差异是否足以突破 LMC (Linear Mode Connectivity) 的线性连通盆地？**

两个从 Pythia-1.4B 出发、分别在 code 和 medical 上 full-FT 的模型——它们在权重空间中有多远？线性插值路径上是否存在 loss barrier？

## 实测数据（2026-07-05）

| 模型 | mean ΔW | max ΔW | 训练 |
|------|---------|--------|------|
| code_e1 | 0.26% | 6.2% | 1 epoch |
| medical_e1 | 1.91% | 13.5% | 1 epoch |
| medical_e5 | 1.95% | 14.5% | 5 epoch |

### Per-block 差异模式

code 和 medical 的权重偏移集中在不同的 block 区间：
- code: 前层变化更大（embedding → early attention）
- medical: 中层变化更大（mid-MLP blocks）
- 两个模型的 importance cosine > 0.94（结构相似性主导）

### 当前实验：LMC barrier 扫描

测量 `L((1-α)W_code + αW_medical)` 在 α∈[0,1] 的 loss 曲线。
如果 barrier ≈ 0，说明两个模型处于同一线性盆地——即使 domain 不同、训练量不同。

## 环境

| 项 | 值 |
|----|-----|
| 模型 | EleutherAI/pythia-1.4b |
| 数据 | VersaPRM code + medical |
| 训练 | full-FT, bf16, L=256, batch=128, 1ep, LR=1e-4→3e-6 |
| 硬件 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| 速度 | 训练 ~40 min/domain, LMC scan ~20 min |

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
