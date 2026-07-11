# Phase 0 实验执行文件

> 2026-07-11 | 论文 v2 - 3-seed 数据完成

## 最终数据

### LMC Barrier (Frankle 定义, 3 seeds)

```
                         code_bar(F)          med_bar(F)
Standard divergence      0.053 ± 0.011       0.051 ± 0.013
High divergence          0.118 ± 0.031       0.228 ± 0.102
```

### 模型清单

`experiments/trained_models/`
- Standard: `code_lr1e-4_s{0,1,2}/` + `medical_lr1e-4_s{0,1,2}/`
- High: `code_lr5e-4_s{0,1,2}/` + `medical_lr5e-4_s{0,1,2}/`

### 结果文件

`experiments/phase0_ivn/results/lmc_lr{1,5}e-4_s{0,1,2}.json`

## 结论

1. 标准 fine-tune 后，LMC 成立（barrier ≈ 0.05）。
2. 增大权重差异后，barrier 升高 2-5× 但仍适中（≤0.23）。
3. 即使 7-9% 的权重偏移，域模型仍在同一盆地。
4. Code→medical 有不显著的正外部性（不对称）。
