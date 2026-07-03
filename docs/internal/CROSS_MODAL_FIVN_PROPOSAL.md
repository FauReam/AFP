# 跨模态 F-IVN：语言模型 × 视觉模型融合实验

> 2026-07-04 | AFP Phase 1 提案

## 1. 动机：从 Phase 0 失败到新方向

### Phase 0 做了什么

Pythia-1.4B full-FT on VersaPRM code + medical → 同一架构两领域特化模型 → AFP/IVN 门控整合。

### Phase 0 发现了什么

importance cosine 在所有度量下都 ≈ 1.0：

| 方法 | Cosine | 原理 |
|------|--------|------|
| MAS E[|g|] | 0.996 | 梯度衰减是架构属性 |
| STA |g·ΔW|| 0.994 | ΔW ∝ g，退化到 Fisher |
| L2 rel ||ΔW||/||W_base|| | 0.991 | 仅 0.009 区分度 |

**AFP/IVN 在所有实验中 = FedAvg。IVN 1 轮收敛，门控无任何效果。**

### 为什么 cosine ≈ 1.0 是结构必然

1. **梯度幅值随层深单调衰减**：`∂L/∂θ_0 ∝ J_{N-1}·...·J_0`，`∂L/∂θ_N ∝ I`。Layer 0 的梯度天然 > Layer N，与领域无关。
2. **全参数训练下 ΔW ∝ g**：任何 `g` 和 `ΔW` 的线性/二次组合都继承该衰减模式。
3. **Pythia 预训练覆盖广**：The Pile 已包含 code + medical → fine-tuning 是"轻推"，不创造新能力。
4. **Backbone 占比过大**：1 epoch full-FT 偏移仅 1-4% → 99% 信号是共享的语言 backbone。

**类比**：人类基因 cosine ≈ 0.999，但不妨碍区分个体。区分个体的信息在剩余的 0.1% 里，不在 99.9% 的共同部分里。

### 关键洞察

问题不在 importance 度量——在**训练设定本身产生不了有意义的领域分化**。要打破 cosine ≈ 1.0 的困局，需要：
- 两个模型的结构天然不同（不同架构）
- 两个模型的输入模态不同（不同数据空间）
- 测量的是模型的**差异部分**，不是各自和 base 的距离

## 2. 新方向：跨模态 F-IVN

### 核心思路

融合语言模型和视觉模型，测试 AFP 是否能让语言模型获得"原生视觉理解能力"。

**为什么这个方向天然解决 cosine 问题：**

```
Pythia-160M:    token embedding → causal attention → language logits
ViT-Base:       patch embedding → bidirectional attention → vision logits

共同模式占比: ≈ 0%（两种完全不同的参数空间、架构、输入模态）
→ importance cosine 天然 << 0.5
→ AFP gate 真正能发挥作用
```

**为什么用函数空间（F-IVN）而非权重空间（IVN）：**
- 架构不同 → 权重空间不对齐 → 无法直接权重融合
- 函数空间：两个模型在共享任务（图文匹配）上输出同维度 logit → 可谈判共识

### 模型选择

| | 语言 | 视觉 |
|---|------|------|
| 模型 | EleutherAI/pythia-160m | google/vit-base-patch16-224 |
| 参数 | 160M | 86M |
| 架构 | GPT-NeoX causal | ViT bidirectional |
| 层数 | 12 | 12 |
| 隐层 | 768 | 768 ✓ 天然对齐 |
| 总参数 | ~250M（GB10 121GB 完全够） |

选择理由：
1. 相同 hidden dim (768) → projection head 简单
2. 相同层数 (12) → 结构对比清晰
3. 架构完全不同（causal vs bidirectional）→ 天然低 cosine
4. 不用 CLIP（已对齐图文，无法测"从零创建跨模态理解"）

### 数据集

**COCO 2017 + Flickr30k 混合**：图文匹配二分类

```
正样本: (image, matched_caption) → label=1
负样本: (image, random_caption)  → label=0, 1:1 平衡

Train: COCO train2017 (118K) + Flickr30k train (29K) ≈ 147K 图文对
       + 负采样 → ~294K 训练样本
Eval:  COCO val2017 (5K)
```

选择理由：
- COCO 最广泛使用的图文基准，baseline 好找
- 二分类任务简单明确，两个模型都有有意义输入
- 标签 {0,1} 统一，BCE loss 直接可用
- `datasets` 库一键加载，无需手动下载图片

## 3. 实验路线

```
Phase A: Head 训练（需要标记数据）
  - 语言模型 + projection(768→768→1) → text matching head
  - 视觉模型 + projection(768→768→1) → image matching head
  - 训练：COCO + Flickr30k 图文匹配
  - 目标：建立单模态匹配基线

Phase B: F-IVN 谈判（无标记数据）
  - 同一 (image, text) 对 → 两模型各自预测 P_text, P_image
  - 函数空间谈判 → 共识标签 V ← F-IVN negotiate_fspace()
  - Gate 控制：importance = ||ΔW||/||W_base|| per block
  - 蒸馏 V 回各自模型

Phase C: 跨模态评估（需标记数据）
  - 语言模型仅凭 text → 图像匹配准确率 (baseline vs after)
  - 视觉模型仅凭 image → 文本匹配准确率
  - 单模态退化检查：融合后的语言能力是否下降？
  - 跨模态增益检查：融合后的跨模态能力是否提升？
```

## 4. 预期结果与学术意义

### 预期

- **H1**：F-IVN 融合后，语言模型能够仅凭 text 判断图像匹配（从零获取视觉理解）
- **H2**：视觉模型融合后，单模态性能适度下降（合理的"遗忘→重学" trade-off）
- **H3**：AFP gate 在跨模态设定中发挥实际作用（importance cosine << 0.5）

### 学术意义

1. **首次展示 AFP 在真实跨模态场景中有效**：之前所有同架构实验都是 null result
2. **证明 AFP 的核心价值在于"天然正交的模型"之间**：架构相同 → cosine ≈ 1 → AFP 无用；架构不同 → cosine << 1 → AFP 有区分
3. **"从零跨模态理解"**：不是 CLIP 式的联合训练，而是模型**自主选择**学什么
4. **门控协议的必要性**：跨模态融合需要"选择性吸收"——视觉模型不应污染语言核心能力

## 5. 与 Phase 0 的对比

| | Phase 0 | Phase 1 (本提案) |
|---|---------|-----------------|
| 模型关系 | 同架构同模态 | **异架构异模态** |
| importance cosine | 0.991-0.997 | **预估 <0.3** |
| AFP 效果 | = FedAvg (null) | **预测: AFP > FedAvg** |
| 融合空间 | 权重空间 (IVN) | **函数空间 (F-IVN)** |
| 学术贡献 | 证明同架构 AFP 无效 | **证明跨模态 AFP 有效** |

## 6. 文献搜索关键发现

### 模型合并 (Model Merging)

- **TIES-Merging** (Yadav et al., NeurIPS 2023): 权重幅度 + 符号共识 → 解决冲突
- **DARE** (Yu et al., ICML 2024): 随机丢弃 90-99% delta 参数 → 幅度 ≠ 重要性
- **Task Arithmetic** (Ilharco et al., 2023): θ = θ_base + Σ λ_i τ_i
- **STA** (Tian et al., arXiv 2411.16139): 一阶 Taylor 展开 → 损失敏感的 per-param 重要性
- **APL** (Kong et al., arXiv 2408.09485): 因果干预 → 激活参数定位
- **DF-Merge** (Lee et al., NAACL 2025): 动态 Fisher 加权 + 贝叶斯优化

### 重要性度量

- **MAS** (Aljundi et al., ECCV 2018): E[|∂F²/∂θ|] → 梯度幅值 → 架构衰减
- **EWC** (Kirkpatrick et al., PNAS 2017): 对角 Fisher → E[(∂L/∂θ)²]
- **Fishers for Free** (ICML 2025): Adam v 作为免费 Fisher → 零额外成本

### 跨模态融合

- **CLIP** (Radford et al., 2021): 对比学习 → 联合图文空间
- **Flamingo** (Alayrac et al., 2022): 门控交叉注意力 → 冻结 LLM + 视觉编码器
- **LLaVA** (Liu et al., 2023): 视觉编码器 → 投影 → LLM → 指令跟随
- **F-IVN** (本研究): P2P 函数空间谈判 → 无需联合训练、各自保留架构

### 关键洞察：为什么现有跨模态方法不同于 AFP

所有现有跨模态方法都需要**联合训练**或**架构修改**。F-IVN 是唯一让两个**完全独立训练的模型**在**不访问对方数据/参数**的情况下通过**函数空间谈判**进行知识交换的方法。这是真正的 P2P 互学习。

## 7. 参考文献

1. Aljundi et al., "Memory Aware Synapses: Learning what (not) to forget", ECCV 2018
2. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017
3. Ilharco et al., "Editing Models with Task Arithmetic", ICLR 2023
4. Yadav et al., "TIES-Merging: Resolving Interference When Merging Models", NeurIPS 2023
5. Yu et al., "Language Models are Super Mario: Absorbing Abilities from Homologous Models", ICML 2024
6. Tian et al., "Beyond Task Vectors: Selective Task Arithmetic", arXiv 2411.16139, 2024
7. Kong et al., "Activated Parameter Locating via Causal Intervention for Model Merging", arXiv 2408.09485, 2024
8. Lee et al., "Dynamic Fisher-weighted Model Merging via Bayesian Optimization", NAACL 2025
9. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
10. Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022
11. Liu et al., "Visual Instruction Tuning", NeurIPS 2023

---

> **状态**：等待 ViT-Base 下载完成后开始 Phase A Head 训练。
> **已有**：Pythia-160M ✓ | torch CUDA 13.0 ✓ | transformers 5.12.1 ✓ | datasets 5.0.0 ✓
> **待下载**：ViT-Base | COCO 2017
