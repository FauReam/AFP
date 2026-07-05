# AFP — Claude Code 项目上下文

## 当前研究方向（2026-07-05）

**Domain-specific fine-tuning 产生的权重差异有多大？LMC (Linear Mode Connectivity) 何时破裂？**

两个从 Pythia-1.4B 出发、分别在 code 和 medical 上 full-FT 的模型——测量它们的权重空间距离和线性插值 loss barrier。

## 核心发现

| 模型 | mean ΔW | LMC barrier | 关键发现 |
|------|---------|-------------|---------|
| code_e1 (1ep, lr=1e-4) | 1.79% | — | code 训练对 medical 有正外部性 |
| medical_e1 (1ep, lr=1e-4) | 1.77% | — | 1ep 已经接近饱和 |
| c1m1 (code+med_e1) | — | 0.071 | U形：α=0.2 改善 medical 8.3% |
| c1m3 (code+med_e3) | — | 0.076 | U形减弱：6.8% |
| c1m5 (code+med_e5) | — | 0.076 | U形消失：med_e5 自域最优 |
| **假设** | | | **低 LR (1e-4) 是顽固性的根因** |

## 论文

[docs/internal/PAPER.md](docs/internal/PAPER.md) — 完整草稿

## 下一步

1. 🔴 验证 LR 假设：code lr=5e-4 训练中 → 测 ΔW
2. 🔴 medical lr=5e-4 → LMC scan
3. 🟡 lr=1e-3 极端测试
4. 🟡 3-seed 统计显著性

## 环境

| 项 | 值 |
|----|-----|
| 模型 | EleutherAI/pythia-1.4b |
| 训练 | full-FT, bf16, L=256, batch=128, 1ep |
| LR | Cosine 1e-4→3e-6 (默认) / 5e-4 (测试中) |
| 硬件 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| 速度 | ~5.5s/step, ~42min/epoch, LMC scan ~15min |

## 入口

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

# 训练
bash scripts/train_and_run_phase0.sh

# LMC 扫描
python -u scripts/lmc_barrier_scan.py [--med-e N]

# 状态
bash scripts/monitor.sh

# 模型验证（训练后必须跑！）
python -c "
import torch
c = torch.load('experiments/trained_models/code_e1/W_code_final.pt', map_location='cpu', weights_only=True)
b = torch.load('experiments/trained_models/code/W_code_final.pt', map_location='cpu', weights_only=True)
d = sum((c[k]-b[k]).float().norm().item()**2 for k in c if k in b)**0.5
n = sum(b[k].float().norm().item()**2 for k in b if k in c)**0.5
print(f'TRAINED: Δ={(d/(n+1e-8))*100:.2f}%' if d/n > 0.001 else 'BUG: model is base!')
"
```

## 已知 Bug（影响新会话的）

| # | 描述 | 状态 |
|---|------|------|
| 22 | code_e1 曾是 base 模型（7/5 发现，已重训） | ✅ |
| 18 | code epoch 2+ 退化 | ⬜ |
| 21 | 训练后模型可能是 base（加验证步骤） | ✅ 已加 |

详见 [docs/internal/ENGINEERING.md](docs/internal/ENGINEERING.md)（Bug 1-22 完整清单）

## 已废弃方向（不要看）

- `docs/internal/VISION.md` — AFP/IVN/gate 协议（历史存档）
- `docs/internal/IMPORTANCE_ANALYSIS.md` — importance 指标分析（历史存档）
- `scripts/run_ivn_phase0.py` — IVN 实验（已被 LMC scan 替代）
- `scripts/queue_experiments.sh` — 批量 IVN（已被 LMC scan 替代）

## 工作约定

- 提交用英文 `feat/fix/docs(scope): description`
- 训练用 `nohup ... &`，日志写 `experiments/`
- Python 加 `-u`，HF 加 `local_files_only=True`
- 训练完成后必须验证 ΔW > 0.1%（见上方命令）
- 跑实验前先读 `docs/internal/EXPERIMENT_PLAN.md`
