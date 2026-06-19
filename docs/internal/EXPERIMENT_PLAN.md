# AFP+IVN Phase 0 实验方案（v4 — 2026-06-19）

> **v4 核心变更**：不训练 — 直接用 Qwen2.5 系列预训练特化模型。
> **核心问题**：IVN 多轮虚拟谈判是否优于 AFP 单次门控更新？

---

## 实验设定

```
Teacher: Qwen/Qwen2.5-Coder-1.5B-Instruct     (代码特化)
Student: Qwen/Qwen2.5-Math-1.5B-Instruct      (数学特化)

基座:    Qwen/Qwen2.5-1.5B                    (计算 importance 的基线)
架构:    24 层，同 hidden dim，同参数结构 → 重量空间 IVN 直接跑
数据:    VersaPRM code + math (仅用于评估，不训练)
设备:    DGX Spark GB10 (121GB)

不训练，不准备数据，直接下载 HuggingFace 模型。
```

## 为什么是 code vs math

```
code:  语法解析、结构化生成、算法抽象 → 激活 backbone 中前层 + attention 区
math:  逻辑链推理、符号操纵、多步验证 → 激活 backbone 中后层 + MLP 区

两个领域的 backbone 使用模式大概率正交 → importance cosine 低 → 门控有信息
Qwen 系列在同一基座上做 domain-specific continued pretraining → 干净的对比
```

## 为什么 IVN 需要多轮谈判

```
AFP one-shot:
  W_math' = W_math + M_math ⊙ (W_code - W_math)    ← 一次通信，一次更新

IVN:
  V_0 = W_init
  每轮: A 用 math 数据在 V_t 上算梯度 → 提案 d_A
        B 用 code 数据在 V_t 上算梯度 → 提案 d_B
        V_{t+1} = V_t + M_math ⊙ d_code + M_code ⊙ d_math
        直到 V 收敛

  W_math' = W_math + M_math ⊙ (V_T - W_init)
  W_code' = W_code + M_code ⊙ (V_T - W_init)
```

核心假设: 多轮来回 → V 收敛到一个双方力量平衡的不动点 → 每个人从这个不动点中提取的知识比一次性交换更优。

---

## 对比方法

| 方法 | 描述 |
|------|------|
| **No Exchange** | 各自独立，不交流 |
| **Noise Control** | 注入高斯噪声（幅度 = peer delta std），经门控后更新。**控制条件**：排除"任何扰动都提升 zero-shot 精度" |
| **FedAvg** | W' = (1-α)W_math + α·W_code, grid search α |
| **AFP one-shot** | W' = W + M⊙(W_other - W), grid search τ |
| **IVN** | 多轮虚拟谈判 |

### 评估效度说明

VersaPRM step-level correctness 是 code/math 能力的间接代理度量——backbone 模型未在 PRM 上训练。但：

1. **相对比较有效**：度量噪声均等污染所有方法。`Δ_IVN - Δ_FedAvg` 中噪声抵消。
2. **Noise control 排除扰动假说**：如果噪声交换 Δ ≈ 0 而 IVN Δ > 0，则信号来自知识迁移而非随机扰动。
3. **无系统性偏向**：没有理由认为噪声系统性地偏向 IVN 而不偏向 FedAvg。

---

## 评估指标

| 指标 | 数据 | 含义 |
|------|------|------|
| **自域精度** | math_test / code_test | 自己的专长是否保留 |
| **跨域精度** | code_test / math_test | 是否从对方学到了知识 |
| **净收益** | — | Δ自域 + Δ跨域 |
| **谈判轮次** | — | V 收敛需要的轮次 |
| **收敛曲线** | — | ΔV_t vs t |

---

## 运行

```bash
# 默认: Qwen Coder ↔ Qwen Math, MAS importance, with noise control
python scripts/run_ivn_phase0.py

# 自定义模型 + 消融选项
python scripts/run_ivn_phase0.py \
  --teacher Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --student Qwen/Qwen2.5-Math-1.5B-Instruct \
  --importance mas        # 或 magnitude (消融)
```

## 预估

```
模型下载: ~15min (每个 3GB)
importance 计算: ~1min
IVN 谈判: ~5-10min (取决于收敛轮次)
FedAvg grid search: ~10min (6 alpha × 评估)
AFP grid search: ~12min (7 tau × 评估)
总计: ~40min
```

---

## 判断标准

```
IVN net > AFP net  → 多轮谈判优于单次更新 ✓
IVN net ≈ AFP net  → 多轮无增益 → 检查 V 是否收敛到平凡解
importance cosine < 0.8 → 领域互补 ✓
importance cosine > 0.95 → code 和 math 不够正交 → 换 domain 对
```
