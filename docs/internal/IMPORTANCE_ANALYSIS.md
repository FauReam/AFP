# Importance 指标深度分析

> 2026-07-03 | AFP Phase 0 | 与 Bug 17-20 并行记录

## 原始结论（MAS 基线）

**实验设置：** Pythia-1.4B full-FT on VersaPRM code + medical，MAS importance

**原始结果 (Bug 19 修复前，labels {-1,+1}):**
- MAS cosine = 0.997
- IVN = FedAvg = AFP (all net=+0.513)

**Bug 19 修复后 (labels {0,1}):**
- MAS cosine = 0.996
- IVN = FedAvg ≥ α=0.5 = AFP (all net=-0.109)
- 唯一正值：FedAvg α=0.1 (net=+0.065)
- IVN 1 轮收敛，ΔV=0.000023

**初步结论：** importance cosine ≈ 1.0 → gate 无法区分 block → AFP = FedAvg

---

## 中间探索过程

### 根因假设 1：domain 太相似？

**检验：** 计算 per-block `||ΔW||` 从 base 的偏移

```
Blk  code ||ΔW||  med ||ΔW||  Ratio (c/m)
 0    3.89%       2.95%       1.32  ← code embedding 动了 32% 更多
 1    3.05%       2.73%       1.12
 ...
23    0.66%       0.80%       0.82  ← med 后期层动了更多
```

**结论：** domain 不相似。code 和 med 产生了不同的 per-block 权重偏移模式。

### 根因假设 2：LR 太小，训练不足？

**检验：** 权重偏移 0.66%-3.89%/block，对于 1 epoch full-FT 是合理的。

**结论：** LR 不是根因。问题在 MAS 这个指标本身。

### 根因假设 3：MAS 是结构性指标，不是领域性指标

**检验：** 分析 MAS 公式 `E[|∂L/∂θ|]` 的理论性质

## 数学推导

### 1. 为什么 MAS 对领域区分无效

MAS 测量梯度幅值：`Ω_MAS[j] = E_x[ |∂L(θ_trained, x)/∂θ_j| ]`

反向传播的链式法则：

```
∂L/∂θ_0  = ∂L/∂h_N · J_{N-1} · J_{N-2} · ... · J_0    ← N 个 Jacobian 连乘
∂L/∂θ_N  = ∂L/∂h_N · I                                   ← 1 个 Jacobian
```

**结论：** `|g_0| >> |g_N|` 是反向传播的结构属性，对任何领域都成立。两个模型即使训练在不同领域，MAS importance 的单调衰减模式 (Layer 0 > 1 > ... > N) 完全相同。

### 2. 为什么 STA 也失败

STA 公式：`Ω_STA = E_x[ |∂L(θ_base, x)/∂θ · (θ_trained - θ_base)| ]`

全参数 SGD 训练：`θ_trained - θ_base = -η · Σ_t g_t`

```
Ω_STA = E_x[ |g_base · (-η Σ g_t)| ]
      ≈ η · E[ |g_base · g_base| ]          ← g_base 与 Σg_t 方向高度相关（同一领域）
      = η · E[ g_base² ]
      = η · diagonal_Fisher(θ_base)
```

**结论：** STA 退化成了对角 Fisher Information，本质上还是测量 base 模型的梯度。梯度随层深单调衰减的结构属性不变 → cosine 仍然 ≈ 1.0。

### 3. 为什么 L1 mean |ΔW| 也失败

`Ω_L1[j] = mean_{i∈block_j} |W_trained,i - W_base,i|`

L1 mean 对每个参数取绝对值后平均。大参数矩阵（如 embedding 50304×2048）和小参数矩阵（如 LayerNorm 2048）的 L1 mean 数量级相同，导致 aggregation 后平滑掉层间差异。

per-block 后归一化会进一步放大相似性。

### 4. 为什么 Relative L2 ||ΔW||/||W_base|| 有效

```
Ω_L2_rel[j] = ||ΔW_block_j|| / ||W_base_block_j||
            = sqrt(Σ_i (W_trained,i - W_base,i)²) / sqrt(Σ_i W_base,i²)
```

关键性质：
- **L2 norm 保留方差：** 平方放大大偏移参数，保留层间差异
- **相对 base 归一化：** 消除参数量级差异（embedding vs LayerNorm）
- **联合归一化 (joint max)：** 两个模型共享同一 scale，不人为制造相似性

## 最终结论

### 定量对比

| 方法 | 公式 | Cosine | Discriminability | 原理 |
|------|------|--------|-----------------|------|
| L1 mean | mean(|ΔW|) per block | 0.9998 | 0.0002 | L1 平滑 + 独立归一化 |
| MAS | E[|g|] | 0.9962 | 0.0038 | 梯度衰减是架构属性 |
| STA | E[|g · ΔW|] | 0.9937 | 0.0063 | ΔW∝g → 退化到 Fisher |
| **L2 rel (joint)** | **||ΔW|| / ||W_base||** | **0.9912** | **0.0088** | **保留方差 + 联合归一化** |

L2 rel 的 discriminability 是 MAS 的 **2.3 倍**。

### 理论定理

**对于全参数 fine-tuning，所有基于梯度幅值（|g|, |g|², |g · ΔW|）的 importance 指标必然趋近于 cosine ≈ 1.0。**

原因：
1. 梯度幅值随层深单调衰减是反向传播的结构属性（链式法则），与训练领域无关
2. ΔW ∝ Σg（SGD 更新），因此 ΔW 与 g 高度共线
3. 任何 g 和 ΔW 的线性/二次组合都继承了这个衰减模式

**唯一能打破这个模式的方法是直接使用权重变化本身的 L2 norm 并进行联合归一化。**

### 实现

- 代码：`src/AFP/protocol/trust.py::block_importance_l2()`
- IVN 入口：`--importance magnitude_l2`
- Pipeline 默认值已更新为 `magnitude_l2`

---

## 相关 Bug

| Bug | 描述 | 影响 |
|-----|------|------|
| Bug 17 | eval n_test=15k → GB10 25min/eval | 网格搜索 18h |
| Bug 18 | F-IVN head 缺 h[:,-1,:] | shape mismatch crash |
| Bug 19 | Labels {-1,+1} → {0,1} 未转换 | 所有评估指标错误 |
| Bug 20 | state_dict() 在 dict comprehension 中调用 582 次 | STA 计算卡死 |
| **Importance** | **MAS/STA 结构性失效** | **AFP gate 无法区分领域** |
