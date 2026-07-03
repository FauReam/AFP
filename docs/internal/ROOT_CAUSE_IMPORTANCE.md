# Importance 失效的根因分析

> 2026-07-03 | AFP Phase 0 | 与 IMPORTANCE_ANALYSIS.md 互补

## 问题

importance cosine ≈ 0.996 (MAS) / 0.991 (L2 rel) / 0.994 (STA)，所有方法都接近 1.0。即使改了 importance 指标，也无法降到 0.8 以下。

## 根因：共同模式主导

### 1. Backbone 的语言能力占比太大

Pythia-1.4B 在 The Pile 上预训练，语料包含 code、medical、math、web text 等。模型的核心能力——embedding、early attention、mid MLP——被通用语言理解占据。

1 epoch full-FT on 55k domain samples：
- 权重偏移仅 1-4% per block
- 模型 = `0.99 × base_Pythia + 0.01 × domain_specialization`
- 两个 domain 特化模型的差异部分只有 ~1%，共享部分 ~99%

**任何在完整权重/重要性向量上计算的 cosine 都会被这 99% 的共同模式淹没。**

### 2. 开源模型经过广泛常识训练

Pythia 预训练数据包含：
- GitHub (code)
- PubMed Abstracts (medical)
- arXiv (math)
- Wikipedia (general knowledge)

→ "code 特化"和"medical 特化"之间本质的区别，早在预训练阶段就已经被学到过。fine-tuning 只是"轻轻唤起"已有知识，不是"从零构建"。

### 3. Full-FT 不遗忘

1 epoch Cosine LR 训练无法让模型"遗忘"无关领域的能力：
- Medical 特化模型：仍然保留 code 能力（cross-acc 0.514）
- Code 特化模型：仍然保留 medical 能力（cross-acc 0.745）

→ 两个模型本质上是同一个基础模型的微小扰动版本。

## 类比：人类基因组

```
人类 vs 人类： 基因 cosine ≈ 0.999
人类 vs 猩猩： 基因 cosine ≈ 0.988
```

但 0.999 和 0.988 的区别微不足道——我们仍然能区分两个人类个体。

**区分个体的信息在剩余的 0.1% 里，不在 99.9% 的共同部分里。**

同理，区分 code agent 和 medical agent 的信息在 `Δ_code - Δ_med`（差异部分），不在 `Δ_code` 和 `Δ_med` 各自与 base 的距离里。

## 为什么"模型种类权重 > 模型功能"是错的

当前 importance 方法的隐含假设：
```
importance(model) = f(model, base)  ← 测量模型与 base 的距离
```

这天然包含 backbone 结构（embedding 层 > attention 层 > MLP 层）的信号。Pythia 的 embedding 层 gradient 永远 > LayerNorm gradient——这不是因为 embedding 层对领域更重要，而是因为它有 50304×2048 个参数，梯度幅值天然大。

**正确的方法应该测量模型之间的差异**：

```
importance_diff(model_A, model_B) = f(model_A, model_B)  ← 测量两个模型互相差异的部分
```

只关注 A 和 B 分歧最大的参数子空间，忽略两者都偏离 base 但方向相同的部分。

## 解决方案

### 核心思路：过滤共同模式，只留差异化信号

1. **训练高度特化模型**：
   - 超多数据、更大 LR (如 5e-4)、更多 epoch
   - 让模型在领域数据上训练到"遗忘"部分通用语言能力
   - 迫使 backbone 产生真正的结构性差异

2. **差异化 importance 度量**：
   - `importance_diff[j] = ||ΔW_code[j] - ΔW_med[j]||` —— 两个模型分歧最大的 block
   - 或者：`importance[j] = cosine_distance(ΔW_code[j], ΔW_med[j])` —— 变化方向不同的 block
   - 这类似于 ICA/PCA 去除共模后的残差分析

3. **按比例过滤**：
   - 计算 per-block 的"共享度"：`shared[j] = min(||ΔW_code[j]||, ||ΔW_med[j]||) / max(...)`
   - 只对 shared[j] < 0.5 的 block 做 AFP gating
   - shared[j] > 0.9 的 block 直接用 FedAvg（两者几乎相同）

## 相关文档

- [IMPORTANCE_ANALYSIS.md](IMPORTANCE_ANALYSIS.md) — importance 指标数学推导与对比
- [REFERENCES.md](REFERENCES.md) — **Linear Mode Connectivity** 文献：Frankle 2020, Entezari 2022, Ainsworth 2023 (Git Re-Basin)
- [ENGINEERING.md](ENGINEERING.md) — 工程手册
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) — Phase 0 实验方案

## LMC 视角下的 Phase 0 失败

Linear Mode Connectivity (Frankle et al., ICML 2020) 提供了最清晰的理论解释：

1. Pythia-1.4B code 和 medical 两个 full-FT 模型，权重距离仅 ~1.2%（相对于 base）
2. LMC 理论预言：从同一预训练模型出发微调的模型，通常处于同一个线性连通盆地
3. 在同一个盆地内，**任何线性插值 `(1-α)θ_code + αθ_medical` 的 loss 都相似**
4. AFP gate 的本质也是线性插值（只是 per-block 的 α 不同）
5. → **gate 无法产生区分 → AFP = FedAvg**

**正向理解**：AFP 不是"失败了"——它正确地识别了"这两个模型没有需要门控的差异"。LMC 提供了一个前置检验：先在两个模型之间做线性插值的 loss barrier 测试。如果 barrier ≈ 0（LMC 成立），AFP 就不可能优于 FedAvg。如果 barrier > 0，AFP 才有发挥空间。
