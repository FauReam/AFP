# Phase 0 实验执行文件

> 2026-07-06 | 当前阶段: 离线 12h 训练队列

## 全部实测数据

### LMC Barrier

| 实验 | 模型对 | ΔW | code_bar(abs) | code_bar(Frankle) | med_bar(abs) | med_bar(Frankle) |
|------|--------|-----|---------------|--------------------|---------------|--------------------|
| lr1e4 | code_lr1e4 × med_e3 | 7.3/9.0% | 0.135 | 0.068 | 0.282 | 0.141 |
| lr5e4 | code_lr5e4 × lr5e4 | 7.4/7.3% | 0.290 | 0.162 | 0.630 | 0.315 |
| driveputt | code_dp × med_dp | 7.3/9.0% | 0.339 | 0.192 | 0.919 | 0.488 |

### 发现

1. barrier 随 ΔW 单调上升，但不成比例（ΔW 差 1.2×，barrier 差 2.5×）
2. Frankle 正确定义的 barrier 只有绝对定义的一半
3. 驱动-推杆 vs 纯余弦：ΔW 无显著差异（7.3% vs 7.4%），推杆阶段把分歧拉回去了
4. 纯高 LR（5e-4 cosine）模型未收敛（med loss=1.15）——barrier 不可信

## 12h 离线队列

`bash scripts/offline_12h.sh`

| Phase | 内容 | 时间 |
|-------|------|------|
| 1 | 4 LR × 3 seeds × 2 domains = 24 次训练 | ~10h |
| 2 | ΔW 验证 | ~10min |
| 3 | 12 LMC 扫描 | ~2h |

输出：`experiments/trained_models/{domain}_lr{lr}_s{seed}/` + `results/lmc_lr{lr}_s{seed}.json`

## 下一步

1. 运行 12h 队列
2. 基于结果更新论文（修复 barrier 定义 + Table 3.1 数据）
3. 3-seed error bars + LR 谱图

## 环境

| 项 | 值 |
|----|-----|
| 模型 | EleutherAI/pythia-1.4b |
| 训练 | full-FT, bf16, L=256, batch=128, 1ep |
| 调度 | 驱动-推杆 (70% flat + 30% cosine) |
| 硬件 | DGX Spark GB10, 121GB |
| 速度 | ~42min/训练, ~15min/LMC |
