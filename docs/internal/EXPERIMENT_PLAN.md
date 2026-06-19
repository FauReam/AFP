# AFP Phase 0 实验方案（敲定版）

> **目标**：用最小可行实验同时验证 AFP 的三个核心主张
> **设备**：DGX Spark GB10 · 121GB · ARM64
> **数据**：VersaPRM（math 200k / code 296k / medical 124k / general 188k steps）

---

## 实验结构总览

```
Experiment A — 同构：证明 AFP > FedAvg（验证门控机制的价值）
Experiment B — 异构：证明 AFP 能跨架构运行（FedAvg 数学上跑不了）
Experiment C — 鲁棒：证明 AFP 抵抗恶意权重注入（FedAvg 无防御）
```

---

## 共同设定

### 模型初始化

```
Experiment A:
  Agent A: Pythia-1.4B pretrained + 随机 PRM 分类头
  Agent B: Pythia-1.4B pretrained + 随机 PRM 分类头

Experiment B:
  Agent A: Pythia-1.4B pretrained + 随机 PRM 分类头
  Agent B: TinyLlama-1.1B pretrained + 随机 PRM 分类头

Experiment C:
  同 Experiment A 的设定，但 Agent B 是恶意方
```

### 数据切分

```
每个领域独立切分:
  本地训练集: 70%
  验证集:     15%（用于调 α、τ 等超参数）
  测试集:     15%（只看一次，最终报告）

Agent A 使用 math 数据
Agent B 使用 code 数据
```

### 本地训练（产生专业化）

```
目标: 每个 agent 在自己的领域上产生显著专业化
      验证标准: A 在 math_val 上的 acc 显著高于 B 在 math_val 上的 acc

训练量: 在各自训练集上做 1 epoch（约 50-74K steps）
        batch=128, max_length=384, lr=1e-4
        frozen backbone, 只训 PRM 分类头

产出:
  W_A^math: Pythia-1.4B backbone + math-specialized head
  W_B^code: Pythia-1.4B backbone + code-specialized head (Expt A/C)
             或 Pythia-160M backbone + code-specialized head (Expt B)
```

### 交换内容

```
只交换 backbone 权重（所有 transformer block 的参数）
PRM 分类头各自保留，不交换
理由: 分类头是任务-specific 的（输出维度可能不同），交换 backbone
      验证了"通用知识可迁移、专用知识各自保留"的核心假设
```

---

## Experiment A：同构对比（AFP vs FedAvg vs 不交换）

### A.1 参与方

```
Agent A: Pythia-1.4B, 专精 math
Agent B: Pythia-1.4B, 专精 code
```

### A.2 方法

#### A.2.1 Baseline 1 — 不交换（下界）

```
W_A' = W_A
W_B' = W_B
什么都不做。衡量"不交流"的性能。
```

#### A.2.2 Baseline 2 — FedAvg 线性插值

```
W_A' = (1-α)·W_A + α·W_B
W_B' = (1-α)·W_B + α·W_A

α 在验证集上 grid search: {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
选最优 α 报告（给 FedAvg 最大优势）

关键: 这是"1 个自由度"的方法——一个标量 α 统一施加到所有 24 个 block
```

#### A.2.3 AFP — 逐 block 重要性门控

```
对 Agent A 的 24 个 transformer block:

Step 1 — 计算 per-block 重要性:
  importance_A[j] = mean(|W_A[j] - W_init[j]|)
  归一化到 [0,1]（除以 max(importance_A)）

Step 2 — 计算门控值:
  M_A[j] = clamp(1 - importance_A[j] / τ, 0, 1)
  其中 τ 是温度参数（唯一超参数）

  直觉:
    importance_A[j] 高 → 这个 block 对 math 很重要 → M_A[j] 低 → 保护
    importance_A[j] 低 → 这个 block 对 math 不重要 → M_A[j] 高 → 开放学习

Step 3 — 选择性更新:
  W_A'[j] = W_A[j] + M_A[j] · (W_B[j] - W_A[j])

  对 B 做对称操作（用 importance_B 控制 B 的门控）

关键: 这是"24 个自由度"的方法——每个 block 有自己的门控值
```

### A.3 自由参数对比

```
FedAvg: 1 个标量 α → 1 个自由度
AFP:    1 个温度 τ 控制 24 个门控值 → 24 个有效自由度

核心问题: 这额外的 23 个自由度是否带来显著的性能收益？
```

### A.4 评估

对 Agent A（同理对 B）：

| 指标 | 测试数据 | 计算方式 |
|------|---------|---------|
| **自域保留** | math_test | A' 在 math 上的 acc − A 在 math 上的 acc（希望 ≥0） |
| **跨域泛化** | code_test | A' 在 code 上的 acc − A 在 code 上的 acc（希望 >0） |
| **净收益** | — | 自域保留 + 跨域泛化（希望 >0） |

### A.5 预期结果（如果 AFP 有效）

```
Pareto 前沿图:

跨域泛化 ↑
    │
    │        ● AFP
    │       ╱
    │      ╱  ● FedAvg (最优 α)
    │     ╱
    │    ● 不交换
    └──────────────────→ 自域保留

AFP 在 Pareto 意义上严格优于 FedAvg:
  同等的自域保留 → AFP 提供更高的跨域泛化
  同等的跨域泛化 → AFP 保留更多自域能力

每个 block 的门控模式可解释:
  math agent: 某些 block M≈0（被保护），某些 block M≈1（开放）
  code agent: 保护的 block 与 math agent 不同
  → 说明两个领域确实依赖 backbone 的不同部分
  → 这本身是一个发现
```

---

## Experiment B：异构跨架构（AFP vs 不交换）

### B.1 参与方

```
Agent A: Pythia-1.4B (24 blocks, GPT-NeoX, GELU, LayerNorm, learned pos)
Agent B: TinyLlama-1.1B (22 blocks, LLaMA, SwiGLU, RMSNorm, RoPE)

关键: 这不是同一家族的缩放，而是真正不同的架构
      → 不同的 normalization、不同的 activation、不同的 position encoding
      → 如果 AFP 能在这里传输知识，它证明了架构无关性
```

### B.2 为什么 FedAvg 不能跑

```
维度差异:
  Pythia 1.4B:  24 layers, hidden=2048, intermediate=8192
  TinyLlama 1.1B: 22 layers, hidden=2048, intermediate=5632

虽然 hidden dim 碰巧相同，但:
  - Layer 数量不同 (24 ≠ 22)
  - 内部权重形状不同 (intermediate 8192 ≠ 5632)
  - Norm 类型不同 (LayerNorm ≠ RMSNorm)
  - 参数的语义含义不同

FedAvg 要求 W_A[i] + W_B[i] 语义对齐 → 不同架构下即使形状相同，加法也无意义
AFP 不要求逐元素对齐 → 通过功能层面的知识映射桥接架构差异
```

### B.3 跨架构知识传输策略

由于是真正不同的架构，不能假设层间对应。Phase 0 采用**功能层面的知识映射**：

```
Step 1 — B 的权重在 A 的私有数据上"展示能力":
  A 加载 W_B 到 TinyLlama-1.1B 模型
  → 在自己的私有数据 D_A (math) 上跑推理
  → 记录每个样本: A 预测对/错 vs B 预测对/错
  → 得到 "B 比 A 强的样本集合" S_B_better

Step 2 — 识别 A 的"可塑区域":
  A 计算 per-block importance（同 Experiment A）
  → 低 importance 的 block = A 的"可塑区域"
  → 这些 block 对 math 不那么关键，可以接受外部知识

Step 3 — 对可塑区域做选择性 fine-tuning:
  对 S_B_better 中的样本:
    → A 以 B 的预测作为 soft target
    → 但只在低 importance block 上反向传播
    → 高 importance block 的梯度被置零（保护核心能力）

Step 4 — 评估 trust:
  trust(B) = |S_B_better| / |D_A|
  B 能在多大比例的 A 的私有数据上比 A 做得好？
  → 高 → B 的知识对 A 有用 → trust 高 → 后续交互中加大学习权重
```

### B.4 为什么用功能映射而非直接权重映射

```
Phase 0 的核心目标是证明"跨架构知识传输可行且有用"。
功能映射（用 B 在 A 数据上的表现指导 A 的选择性 fine-tuning）
是最直接的证明方式——它不依赖任何架构假设。

直接权重映射（学习从 TinyLlama weight space 到 Pythia weight space 的 projection）
是一个更难的独立研究问题——留给 Phase 1。
```

### B.5 评估

```
与 Experiment A 相同的指标体系
但 baseline 只有"不交换"（FedAvg 跑不了）

额外指标:
  知识传输效率 = Δacc_cross / ||W_A' - W_A||
  → 度量"每单位权重变化带来的跨域提升"
```

---

## Experiment C：鲁棒性（AFP vs FedAvg 面对恶意注入）

### C.1 参与方

```
Agent A: Pythia-1.4B, 专精 math（诚实方）
Agent B: Pythia-1.4B, 专精 code（恶意方——注入噪声权重）
```

### C.2 恶意注入方式

```
B 正常训练得到 W_B_code
B 向 A 发送: W_B_malicious = W_B_code + ε·noise

其中 noise ~ N(0, σ²)，在 backbone 所有参数上添加
噪声水平: ε ∈ {0.1, 0.3, 0.5, 1.0, 2.0}（以 W_B 的标准差为单位）
```

### C.3 方法对比

```
FedAvg:
  W_A' = (1-α)·W_A + α·W_B_malicious
  → α 选验证集最优值
  → 噪音被直接平均进去，无法防御

AFP:
  Step 1: 分析 W_B_malicious 和 W_init 的距离
          noise_magnitude = ||W_B_malicious - W_init|| / ||W_A - W_init||
          如果 noise_magnitude 异常大 → 降低 trust

  Step 2: trust(B) = exp(-λ · noise_magnitude)
          正常 B → noise_magnitude ≈ 1 → trust ≈ exp(-λ)
          恶意 B → noise_magnitude >> 1 → trust → 0

  Step 3: 门控 M[j] = (1 - importance_A[j]) × trust(B)
          trust → 0 → 所有 M[j] → 0 → 不学习
```

### C.4 评估

```
对每种噪声水平，对比:
  AFP 的自域保留（应该接近 0 退化）
  FedAvg 的自域保留（应该随 ε 增加而退化）

关键图: 噪声水平 ε vs 自域退化
         FedAvg: 退化随 ε 单调递增
         AFP:    退化始终接近 0（门控关掉了）
```

---

## 实施顺序与计算预算

### 顺序

```
第一步: Experiment A（4-6 小时）
  → 验证 AFP > FedAvg 在同构场景下成立
  → 如果 A 的结果是 AFP ≈ FedAvg → 先理解为什么，再决定是否继续

第二步: Experiment C（2-3 小时，基于 A 的模型）
  → 验证 AFP 抵抗噪声注入
  → 对 A 已有的 W_A 和 W_B，注入不同水平噪声，跑对比

第三步: Experiment B（8-12 小时）
  → 验证 AFP 跨架构运行
  → 需要额外训练 Pythia-160M + 实现跨架构映射
```

### 总计算预算

```
本地训练:
  Expt A: 2 × Pythia-1.4B head-only 训练 ≈ 2.5h
  Expt B: Pythia-1.4B + TinyLlama-1.1B head-only 训练 ≈ 3h
  Expt C: 复用 A 的模型，只做推理

集成评估（每种方法）:
  AFP 集成 ≈ 0.5h
  FedAvg 集成 ≈ 0.3h
  跨架构功能映射 ≈ 1.5h（需加载两个模型 + 推理 + 选择性 fine-tuning）

总计: 约 14-20 小时（1-2 天，含调试）
```

---

## 风险预案

| 如果... | 则... |
|---------|-------|
| **A 的结果 AFP ≈ FedAvg** | 检查 per-block importance 是否真的有差异。如果所有 block 的 importance 差不多 → 说明对于 Pythia-1.4B + VersaPRM，backbone 使用模式是均匀的 → 这本身是一个发现（解释了为什么旧项目 CKA=1.0）。转向解释"为什么均匀"并测试其他模型/数据组合 |
| **A 的结果 AFP < FedAvg** | 门控机制引入了过度正则化。降低 τ 让更多 block 开放。如果仍然差 → 说明"选择性"在 head-only 训练下不成立（head 没怎么动 backbone）。需要更重的本地训练（partial-FT 而非 head-only） |
| **B 的跨架构映射不 work** | 检查 B 在 A 数据上是否真的比 A 好——如果 TinyLlama 在 A 的 math 数据上表现不如 Pythia-1.4B，说明两模型间不存在知识差 → 换领域对（math-medical）或增大本地训练量增加专业化差异 |
| **C 的噪声检测失败** | 如果 trust 函数对噪声不敏感，尝试其他统计量：权重方向一致性、梯度角度、layer-wise 的异常检测 |

---

## 评估矩阵（最终汇总）

| 实验 | 方法 | 自域保留 | 跨域泛化 | 净收益 | 抗噪声 |
|------|------|---------|---------|--------|--------|
| A | 不交换 | 0（基准） | 0（基准） | 0 | — |
| A | FedAvg | 测 | 测 | 测 | — |
| A | AFP | 测 | 测 | 测 | — |
| B | 不交换 | 0（基准） | 0（基准） | 0 | — |
| B | AFP | 测 | 测 | 测 | — |
| C | FedAvg + noise | 测（退化） | — | — | ❌ |
| C | AFP + noise | 测（~0） | — | — | ✅ |
