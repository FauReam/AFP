# AFP — Claude Code 项目上下文

> **论文**: [docs/internal/PAPER.md](docs/internal/PAPER.md)

## 研究问题

**Domain fine-tuning 产生多大权重差异？不同差异度下模型是否仍在同一 LMC 盆地？**

Pythia-1.4B 分别在 code 和 medical 上 full-FT。测量权重偏移量级和线性插值 loss barrier。

## 核心数据（3 seeds）

| Divergence | ΔW | code_bar(F) | med_bar(F) |
|-----------|-----|-------------|------------|
| Standard | 1.2-1.8% | 0.053 ± 0.011 | 0.051 ± 0.013 |
| High | 7.3-9.0% | 0.118 ± 0.031 | 0.228 ± 0.102 |

## 环境

| 项 | 值 |
|----|-----|
| 模型 | EleutherAI/pythia-1.4b |
| 训练 | full-FT, bf16, L=256, batch=128, 1ep |
| 硬件 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| 速度 | ~42min/训练, ~15min/LMC |

## 入口

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

# 训练
bash scripts/train_and_run_phase0.sh

# LMC 扫描
bash scripts/lmc_6scans.sh

# 状态
bash scripts/monitor.sh
```

## 工作约定

- 提交用英文 `feat/fix/docs(scope): description`
- 训练用 `nohup ... &`，日志写 `experiments/`
- Python 加 `-u`，HF 加 `local_files_only=True`
- 模型保存到独立目录：`{domain}_lr{lr}_s{seed}/`，禁用 symlink
