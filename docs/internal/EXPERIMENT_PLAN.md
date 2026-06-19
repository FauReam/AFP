# AFP Phase 0 实验方案（v2 — 2026-06-19 修订）

> **v2 核心变更**：从 head-only → full-FT，从 math+code → code+medical，从 FedAvg 对比 → 先诊断后对比
> **目标**：用最小可行实验同时验证 AFP 的三个核心主张
> **设备**：DGX Spark GB10 · 121GB · ARM64
> **数据**：VersaPRM（code 296K / medical 124K steps）

---

## v2 修订理由

### 为什么弃用 head-only

head-only 训练在实验 A 中存在结构性矛盾：

```
head-only → backbone frozen → W_A_backbone == W_B_backbone == W_init
→ importance[j] = 0 ∀j → M[j] = 1 ∀j
→ AFP 退化 = FedAvg = 不交换
→ 实验无效
```

FCL-PRM 旧项目的经验也表明：head-only 下 CKA ≡ 1.0，backbone 表示空间完全不可撼动。门控机制没有可操作的空间。

### 为什么换用 full-FT

- 全量训练让 backbone 产生有意义的 per-block 专业化差异
- 1 epoch 在完整 VersaPRM 数据上做 full-FT：~340 batches @ batch=1024
- per-block importance 有真正的结构 → AFP 门控有信息可用

### 为什么换用 code + medical

```
原方案 math vs code:
  两者共享大量推理结构（逻辑、序列、形式化语法）
  → backbone 激活区间高度重叠
  → importance 模式趋同 → 测不出差异

新方案 code vs medical:
  code:     语法解析、结构化生成、算法抽象
  medical:  领域术语、诊断推理、事实性知识检索
  → backbone 激活区间重叠小
  → importance 模式更可能呈现正交结构
  → 对 AFP 是更干净的测试
```

---

## 实验结构总览

```
Phase 0 分两步：

Step 1 — 诊断（5-6h）：
  ├─ Agent A: Pythia-1.4B full-FT on medical, 1 epoch
  ├─ Agent B: Pythia-1.4B full-FT on code, 1 epoch
  └─ 分析 per-block importance 是否存在领域特异的、结构化的模式

Step 2 — 如果诊断有信号 → AFP vs FedAvg 对比：
  Experiment A — 同构：证明 AFP > FedAvg（验证门控机制的价值）
  Experiment B — 异构：证明 AFP 能跨架构运行（FedAvg 数学上跑不了）
  Experiment C — 鲁棒：证明 AFP 抵抗恶意权重注入（FedAvg 无防御）
```

---

## Step 1：诊断训练

### 训练设定

```
模型:       Pythia-1.4B
训练方式:   full-FT（所有 backbone + PRM 分类头）
Batch:      1024（极限值，利用 121GB 统一内存）
Epoch:      1
LR schedule: Cosine decay: 1e-4 → 3e-6（先大步后小步）
             → 大 lr 前期快速推到 domain-specific 区域
             → 小 lr 后期精细收敛
Optimizer:  AdamW (β=0.9, 0.999), weight_decay=0.01
Precision:  bf16 autocast + fp32 head
Max length: 384 tokens
```

### 数据切分

```
每个 domain 从 VersaPRM 提取：

code:     296K steps × 70% = 207K 训练样本
medical:  124K steps × 70% = 86.8K 训练样本

验证集: 15%（调超参用，Step 2 启用）
测试集: 15%（只看一次，Step 2 启用）

过滤: tokenize 后 >384 token 的 step 丢弃（不截断）
```

### 两次并行运行

```
Run 1: lr_peak=1e-4（激进） — 更大的步长，测试能否暴露 domain 差异
Run 2: lr_peak=3e-5（保守） — 如果 Run 1 有噪声，Run 2 提供对照

并行启动，一晚跑完。
```

### 诊断指标

```
对每个 agent 计算 per-block importance:

  importance[j] = mean(|W[j] - W_init[j]|)  for j = 1..24
  归一化到 [0, 1]

核心问题:
  ① 是否有跨 block 的显著差异（不是所有 block 差不多）？
  ② code 和 medical 的 importance 模式是否不同？
  ③ 不同模式是否正交（medical 重要的 block code 不重要，反之亦然）？

量化指标:
  - Cosine similarity of importance vectors（1.0 = 完全相同, 0.0 = 正交）
  - Rank correlation of block rankings
  - Gate-ability score: 领域特异 block 数 / 总 block 数
```

### 判断标准

```
Cosine similarity < 0.5  → STRONG DIVERGENCE  → 推进 Step 2
Cosine similarity < 0.8  → MODERATE DIVERGENCE → 推进 Step 2，但预期效果适中
Cosine similarity < 0.95 → WEAK DIVERGENCE     → 多 epoch 或更大 lr 重试
Cosine similarity ≥ 0.95 → NO DIVERGENCE        → 重新审视假设
```

---

## Step 2：实验 A — 同构对比（AFP vs FedAvg vs 不交换）

> **前置条件**：Step 1 诊断通过（cosine similarity < 0.8）

### A.1 参与方

```
Agent A: Pythia-1.4B, 专精 medical (full-FT on medical)
Agent B: Pythia-1.4B, 专精 code (full-FT on code)
```

### A.2 交换内容

```
只交换 backbone 权重（所有 transformer block 的参数）
PRM 分类头各自保留，不交换
```

### A.3 方法

#### Baseline 1 — 不交换（下界）

```
W_A' = W_A
W_B' = W_B
```

#### Baseline 2 — FedAvg 线性插值

```
W_A' = (1-α)·W_A + α·W_B
W_B' = (1-α)·W_B + α·W_A

α 在验证集上 grid search: {0.1, 0.2, 0.3, 0.5, 0.7, 0.9}
选最优 α 报告（给 FedAvg 最大优势）

1 个自由度 — 一个标量 α 统一施加到所有 24 个 block
```

#### AFP — 逐 block 重要性门控

```
对 Agent A 的 24 个 transformer block:

Step 1 — 计算 per-block 重要性:
  importance_A[j] = mean(|W_A[j] - W_init[j]|)
  归一化到 [0,1]

Step 2 — 计算门控值:
  M_A[j] = clamp(1 - importance_A[j] / τ, 0, 1)
  τ 是温度参数（唯一超参数，在验证集上调）

  直觉:
    importance_A[j] 高 → 这个 block 对 medical 很重要 → M_A[j] 低 → 保护
    importance_A[j] 低 → 这个 block 对 medical 不重要 → M_A[j] 高 → 开放学习

Step 3 — 选择性更新:
  W_A'[j] = W_A[j] + M_A[j] · (W_B[j] - W_A[j])

24 个自由度 — 每个 block 有自己的门控值
```

### A.4 评估

对 Agent A（同理对 B）：

| 指标 | 测试数据 | 计算方式 |
|------|---------|---------|
| **自域保留** | medical_test | A' 在 medical 上的 acc − A 在 medical 上的 acc（希望 ≥0） |
| **跨域泛化** | code_test | A' 在 code 上的 acc − A 在 code 上的 acc（希望 >0） |
| **净收益** | — | 自域保留 + 跨域泛化（希望 >0） |

### A.5 预期结果

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
  medical agent: 某些 block M≈0（被保护），某些 block M≈1（开放）
  code agent: 保护的 block 与 medical agent 不同
  → 说明两个领域确实依赖 backbone 的不同部分
```

---

## Step 2：实验 B — 异构跨架构

### B.1 参与方

```
Agent A: Pythia-1.4B (24 blocks, GPT-NeoX, GELU, LayerNorm, learned pos)
Agent B: TinyLlama-1.1B (22 blocks, LLaMA, SwiGLU, RMSNorm, RoPE)
```

### B.2 为什么 FedAvg 不能跑

```
维度差异:
  Pythia 1.4B:   24 layers, hidden=2048, intermediate=8192
  TinyLlama 1.1B: 22 layers, hidden=2048, intermediate=5632

Layer 数量不同 (24 ≠ 22)
内部权重形状不同 (intermediate 8192 ≠ 5632)
Norm 类型不同 (LayerNorm ≠ RMSNorm)
参数的语义含义不同

FedAvg 要求 W_A[i] + W_B[i] 语义对齐 → 不同架构下加法无意义
AFP 不要求逐元素对齐 → 通过功能层面的知识映射桥接架构差异
```

### B.3 跨架构知识传输策略（功能映射）

```
Step 1 — B 的权重在 A 的私有数据上"展示能力":
  A 加载 W_B 到 TinyLlama-1.1B 模型
  → 在自己的私有数据 D_A (medical) 上跑推理
  → 得到 "B 比 A 强的样本集合" S_B_better

Step 2 — 识别 A 的"可塑区域":
  A 计算 per-block importance
  → 低 importance 的 block = A 的"可塑区域"

Step 3 — 对可塑区域做选择性 fine-tuning:
  对 S_B_better 中的样本:
    → A 以 B 的预测作为 soft target
    → 但只在低 importance block 上反向传播
    → 高 importance block 的梯度被置零（保护核心能力）

Step 4 — 评估 trust:
  trust(B) = |S_B_better| / |D_A|
```

---

## Step 2：实验 C — 鲁棒性

### C.1 参与方

```
Agent A: Pythia-1.4B, 专精 medical（诚实方）
Agent B: Pythia-1.4B, 专精 code（恶意方——注入噪声权重）
```

### C.2 恶意注入方式

```
B 正常训练得到 W_B_code
B 向 A 发送: W_B_malicious = W_B_code + ε·noise

noise ~ N(0, σ²)，在 backbone 所有参数上添加
噪声水平: ε ∈ {0.1, 0.3, 0.5, 1.0, 2.0}（以 W_B 的标准差为单位）
```

### C.3 AFP 防御

```
Step 1: 分析 W_B_malicious 和 W_init 的距离
        noise_magnitude = ||W_B_malicious - W_init|| / ||W_A - W_init||

Step 2: trust(B) = exp(-λ · noise_magnitude)
        正常 B → noise_magnitude ≈ 1 → trust ≈ exp(-λ)
        恶意 B → noise_magnitude >> 1 → trust → 0

Step 3: 门控 M[j] = (1 - importance_A[j]) × trust(B)
        trust → 0 → 所有 M[j] → 0 → 不学习
```

---

## 训练参数速查

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-1.4B |
| 训练方式 | full-FT（所有参数可训练） |
| Batch size | 1024 |
| LR schedule | Cosine: 1e-4 → 3e-6 |
| Optimizer | AdamW (β=0.9, 0.999) |
| Weight decay | 0.01 |
| Precision | bf16 autocast + fp32 head |
| Max length | 384 |
| Epochs | 1 |
| PRM head | 2048 → 256 → 1 (ReLU) |

## 预估计算量

```
Agent A (medical): 86.8K / 1024 = 85 batches × ~55s ≈ 1.3h
Agent B (code):    207K  / 1024 = 203 batches × ~55s ≈ 3.1h
─────────────────────────────────────────────────────────
串行总计:                                        ≈ 4.4h
双 lr 并行:                                      ≈ 9h（一晚）
```

---

## 实施顺序

```
1. 数据准备：python scripts/prepare_data.py（~30min，含 HF 下载）
2. 诊断训练：bash scripts/run_diagnostic.sh（~9h，双 lr）
3. 分析：python scripts/analyze_importance.py
   → 如果有结构性偏移 → 4
   → 如果没有 → 讨论原因，决定是否 multi-epoch
4. Step 2：AFP vs FedAvg 对比实验
```

## 风险预案

| 如果... | 则... |
|---------|-------|
| **诊断无结构性偏移** | 增加 epoch 数（3-5 epoch）重试；或增加 lr 到 3e-4 |
| **训练 OOM @ batch=1024** | 梯度累积：物理 batch=512, accumulation=2 |
| **loss spike / NaN** | 降低 lr_peak 到 3e-5，加 gradient clipping |
| **code 和 medical importance 模式高度重叠** | 换 math vs medical（更极端的领域差异） |
| **full-FT 后模型遗忘预训练能力** | 加入 EWC 正则项约束预训练知识 |
