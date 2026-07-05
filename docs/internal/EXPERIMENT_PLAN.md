# Phase 0 实验执行文件

> **论文**: [PAPER.md](PAPER.md) | **Bug 清单**: [ENGINEERING.md](ENGINEERING.md)

## 研究问题

Domain fine-tuning 产生多大权重差异？LMC 何时破裂？LR 是否是主控因素？

## 已完成实验

| ID | 内容 | 核心结果 |
|----|------|---------|
| LMC-c1m1 | code_e1(1.8%Δ) vs med_e1(1.8%Δ) | barrier=0.071, U形(α=0.2, 8.3%) |
| LMC-c1m3 | code_e1 vs med_e3(1.8%Δ) | barrier=0.076, U形减弱(6.8%) |
| LMC-c1m5 | code_e1 vs med_e5(1.8%Δ) | barrier=0.076, U形消失(0%) |

## 当前结论

1. lr=1e-4 下 Pythia-1.4B 权重偏移 <2%，LMC 成立
2. Code→medical 有不对称正迁移，随 medical 训练增强而衰减
3. Medical→code 始终单调损害
4. **假设**: 低 LR 是顽固性的根因

## 下一步：验证 LR 假设

| 优先级 | 实验 | 预测 | 时间 |
|--------|------|------|------|
| 🔴1 | code, lr=5e-4, 1ep | ΔW > 5% | 42min |
| 🔴2 | medical, lr=5e-4, 1ep | ΔW > 5% | 42min |
| 🔴3 | LMC scan lr=5e-4 pair | barrier > 0.15 | 15min |
| 🟡4 | code, lr=1e-3, 1ep | ΔW > 10%? | 42min |
| 🟡5 | 3-seed 重复 lr=1e-4 | CI estimation | 4h |

## 环境

| 项 | 值 |
|----|-----|
| 模型 | EleutherAI/pythia-1.4b |
| 训练 | full-FT, bf16, L=256, batch=128 |
| 硬件 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |
| 速度 | ~5.5s/step, ~42min/epoch |

## Bug 状态

| # | 描述 | 状态 |
|---|------|------|
| 22 | code_e1 曾是 base 模型（7/5 发现） | ✅ 已修复重训 |
| 18 | code epoch 2+ 退化 | ⬜ |
| 17,19-21 | importance, rm, venv, base覆盖 | ✅ |
