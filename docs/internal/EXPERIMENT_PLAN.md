# Phase 0 实验数据与待做

> 2026-07-12 | 论文 v2

## 全部实验数据

### LMC Barrier (Frankle definition, 3 seeds)

| 条件 | Code barrier | Medical barrier |
|------|-------------|----------------|
| Standard (跨域) | 0.053 ± 0.011 | 0.051 ± 0.013 |
| High divergence (跨域) | 0.118 ± 0.031 | 0.228 ± 0.102 |
| 域内 code | 0.048 ± 0.000 | — |
| 域内 medical | 0.147 ± 0.027 | — |
| 噪声地板 (identical) | ~0.000 | — |
| 噪声上界 (random init) | 0.033 | 0.222 |
| Pythia-160M | 实验中 | — |

### 权重差异

| 条件 | Code ΔW | Medical ΔW | Code↔Med Cross |
|------|---------|------------|----------------|
| Standard | 1.4 ± 0.0% | 1.5 ± 0.1% | 2.0 ± 0.1% |
| High | 8.0 ± 0.3% | 8.5 ± 0.2% | 11.6 ± 0.3% |

## 待完成

| 优先级 | 任务 | 时间 |
|--------|------|------|
| ✅ | 噪声地板校准 | 已完成 |
| 🔄 | Pythia-160M 复制 | 运行中 |
| ⬜ | Related Work 章节 | 30min |
| ⬜ | Bootstrap CI on barriers | 15min |
| ⬜ | 论文润色最终 pass | 30min |

## 环境

| 项 | 值 |
|----|-----|
| 模型 | EleutherAI/pythia-1.4b |
| 训练 | full-FT, bf16, L=256, batch=128, 1ep |
| 硬件 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |
| Python | `/home/jiayu/AFP/venv/bin/python3` |
