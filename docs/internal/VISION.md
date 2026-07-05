# AFP (Agentic Federated Protocol) — 核心愿景

> **AFP 是一种去中心化的 P2P 互学习协议。模型不再被动接受聚合，而是像学者一样——通过"权重论文"交流知识，自主决定学什么、学多少、信谁。**

---

## 一、现行联邦学习的根本局限：初等运算

现行 FL 的核心运算是 FedAvg：

```
W_global = Σ (n_i / N) · W_i
```

这是**凸组合**——两个数的加权平均。不管你跑多少轮，每一轮的"学习"就是一次线性插值。

### FedAvg 的三个隐含假定（全都不成立）

| 假定 | 现实 |
|------|------|
| W_A 和 W_B 的每个参数位有相同的"单位" | 不同数据分布训练出的权重，其参数空间结构完全不同 |
| 噪声在平均中自然抵消 | 恶意客户端可以系统性注入偏差，平均不能消除 |
| 所有客户端同等可信 | 客户端的数据质量、计算能力、诚实度天然不同 |

### 为什么这些假定在真实 FL 场景下致命？

真实 FL 的定义性特征是**服务器不知道数据长什么样**。当你看不到数据的时候：
- 你无法验证哪个客户端的更新是"好的"
- 你无法区分"因为数据不同导致的正常差异"和"恶意注入"
- 你能做的只有平均——而这恰好是最容易被攻击的操作

**现行 FL 假设了一个可信的全知服务器——但这恰恰是隐私场景下最不该假设的东西。**

---

## 二、AFP 的核心思想：从"优化问题"到"学习问题"

### 范式对比

| | 现行 FL (FedAvg 范式) | AFP (Agentic Federated Protocol) |
|---|---|---|
| **数学模型** | 分布式优化（凸组合） | 元学习 + 博弈 + 信息传输 |
| **Agent 角色** | 被动执行者（服从服务器指令） | **主动学习者**（自主决策） |
| **聚合方式** | 固定公式（加权平均） | **学出来的策略**（每个 agent 有自己的学习策略） |
| **信任** | 预设（假设所有客户端诚实或服务器能识别恶意） | **交互中建立**（trust 是学出来的） |
| **鲁棒性** | 依赖服务器检测 | **每个 agent 自主防御** |
| **拓扑** | 星型（服务器在中心） | **P2P 网状**（无中心，或中心被弱化） |
| **知识传输** | 线性插值 | **选择性子空间投影 + 对比约束** |

### 核心类比：学术界

学术界的知识传播不是"所有论文平均"：

```
一个研究生（经历少的模型）读了一篇资深教授的论文（经历多的模型）
→ 不是把论文和自己已有的知识做加权平均
→ 而是：
  1. 评估这位教授是否可信（查引用、看履历） → trust 评估
  2. 提取论文的核心贡献（不是全部内容） → 知识提取
  3. 与自己的研究结合（选择性整合） → 门控更新
  4. 决定在自己的论文中引用多少 → 学习权重
  5. 如果教授的方法在自己的数据上不 work → 降低 trust，减少引用
```

**AFP 就是把学术界的知识传播机制形式化为一个数学协议。**

---

## 三、AFP 协议的形式化

### 设定

- N 个 Agent，每个 Agent i 拥有：
  - 私有数据集 D_i（不可共享）
  - 模型权重 W_i
  - 学习策略模块 L_i（自主决策）
  - 信任模型 T_i（对每个 peer j 的信任度 T_i(j) ∈ [0,1]）
  - 知识提取器 E_i（从对方权重中提取可用知识）

### 一次交互回合（Agent A 收到 Agent B 的权重 W_B）

```
Step 1 — 分析（Analysis）
  Input:  W_A, W_B, history_A_B
  Output: d = subspace_analysis(W_A, W_B)
          K_B = extract_knowledge_subspace(W_B, d)
          → "B 的知识主要体现在哪些参数子空间？"

Step 2 — 信任评估（Trust）
  Input:  history_A_B, d, reputation_B
  Output: trust_A(B) ∈ [0,1]
          → "B 值得信任吗？"
          → high trust: B 的历史表现好、知识互补、不是恶意注入
          → low trust: B 是未知 agent、历史表现差、检测到异常

Step 3 — 自主决策（Decision）
  Input:  trust_A(B), K_B, W_A, loss_on_D_A
  Output: learn? ∈ {yes, no}
          what?  = K_B ∩ plastic_subspace(A)
          how_much? α = trust_A(B) × complementarity × LR_multiplier
          → 不是服务器替我决定，是我自己决定

Step 4 — 选择性整合（Integration）
  Input:  W_A, W_B, α, what
  Output: W_A' = W_A + M_A ⊙ project_what(W_B - W_A)
          其中 M_A 是 A 自主生成的门控掩码（0-1 向量）
          → 不是 (1-α)W_A + α·W_B（线性插值）
          → 而是选择性的、结构化的、子空间约束的更新

Step 5 — 反馈（Feedback，形成"对话"）
  Input:  integration_result
  Output: feedback_to_B = {
            accepted_layers: [attention, embedding],
            rejected_layers: [mlp],
            trust_update: +0.03,
            anomaly_detected: false
          }
          → B 据此调整对 A 的教学策略
```

### 关键创新点

1. **门控掩码 M_A 替代了标量 α**：不是用一个数来加权，而是对每个参数维度做一个独立的接受/拒绝决策
2. **子空间投影替代了全量平均**：只在对方知识的"有效子空间"内更新，避免引入噪声
3. **trust 是学出来的**：不是超参数，而是交互历史的函数
4. **反馈通道**：形成双向对话，而非单向的"客户端上传→服务器下发"

---

## 四、为什么这比 FedAvg 更好（理论直觉）

### 4.1 选择性更新 → 天然抗噪声

```
FedAvg:    W' = (1-α)W_A + α·W_B
           如果 W_B 包含 50% 噪声，那么 W' 包含 α·50% 的噪声

AFP:      W' = W_A + M ⊙ (W_B - W_A)
           如果 M 在噪声维度上 = 0（被门控掩码关掉），噪声不入
```

### 4.2 trust 机制 → 天然抗恶意注入

```
FedAvg:    服务器不知道哪个客户端是恶意的
           → 恶意更新被平均进去

AFP:       Agent A 在交互中自己评估 trust
           → 低 trust → 低 α → 恶意更新基本不影响 A
           → 即使 B 是恶意的，A 不会被污染
```

### 4.3 回应导师的五连问

| 导师的问题 | FedAvg 的答案 | AFP 的答案 |
|-----------|-------------|-----------|
| "如果是噪声你怎么办？" | 寄希望于平均消除 | **门控掩码在检测阶段就拒绝噪声** |
| "噪声的原因是什么？" | 不知道 | **trust 函数天然追溯噪声源** |
| "为什么必然产生噪声？" | — | **低质量 agent 的权重结构可被检测** |
| "噪声在不同水平的影响？" | 无法量化 | **trust ∈ [0,1] 天然量化** |
| "不去噪有什么影响？" | 不知道 | **消融实验：关掉 M → 对比性能下降** |

---

## 五、IVN：Multiple-Gradient Descent with Mutual Gating

AFP one-shot 的核心局限：`W_A' = W_A + M_A ⊙ (W_B - W_A)` 只做了一次通信。双方没有来回，没有"谈判"。

**IVN (Iterative Virtual-Negotiation Protocol)** 引入一个虚拟权重 V，A 和 B 在 V 上多轮梯度谈判，直到 V 收敛：

```
V_0 = W_init (中性起点)

每轮 t:
  d_A = -η · ∇L_A(V_t)           ← A 的提案："我想往这走"
  d_B = -η · ∇L_B(V_t)           ← B 的提案："我想往这走"
  V_{t+1} = V_t + M_A⊙d_B + M_B⊙d_A   ← 双方审核后更新 V
  即: V_{t+1} = V_t - η·(M_A⊙∇L_B + M_B⊙∇L_A)

收敛后:
  W_A' = W_A + M_A ⊙ (V_T - W_init)   ← A 从谈判轨迹中学习
  W_B' = W_B + M_B ⊙ (V_T - W_init)   ← B 从谈判轨迹中学习
```

### 数学本质：多梯度下降

IVN 不是神秘的不动点迭代——它是 **multiple-gradient descent** on the combined objective：

> **Φ(V) = Σⱼ mbⱼ · L_A(V) + Σⱼ maⱼ · L_B(V)**

每个参数块的更新方向是两个梯度的门控加权组合。收敛点满足：

> **M_A ⊙ ∇L_B(V^\*) + M_B ⊙ ∇L_A(V^\*) = 0**

即 **Pareto stationarity**：两个 agent 在 V^\* 处都不再有改善空间。

### 理论支撑

| 理论基础 | 来源 | 与 IVN 的关系 |
|---------|------|:---:|
| **Descent lemma** | Nesterov (2004) | Φ L-smooth + η ≤ 1/L → 单调收敛到平稳点 |
| **MGDA** | Désidéri (2012) | IVN 是 MGDA 的轻量变体：importance gate 替代每轮 QP 求解 |
| **MGDA-UB** | Sener & Koltun (NeurIPS 2018) | MGDA 用于深度网络的推广；IVN 的 per-block gate 是 Frank-Wolfe 的替代 |
| **GD → minimizers** | Lee et al. (COLT 2016) | 随机初始化 + GD 几乎必然避开严格鞍点 |
| **D-PSGD** | Lian et al. (NeurIPS 2017) | M=1 时 IVN 退化为此（2节点去中心化 SGD），收敛性已被证明 |

详见 [REFERENCES.md](REFERENCES.md)。

**两种变体的对比：**

| | FedAvg | AFP one-shot | **IVN** |
|---|---|---|---|
| **操作** | W' = avg(W_A, W_B) | W' = W + M⊙(W_other-W) | **V_{t+1} = V_t - η(M_A⊙∇L_B + M_B⊙∇L_A)** |
| **数学结构** | 凸组合 | elementwise gated linear update | **multiple-gradient descent** |
| **交互轮次** | 1 | 1 | **多轮直到收敛** |
| **提案内容** | 整个W | 整个W | **梯度方向（比原始权重更少信息泄漏）** |

### 门控掩码 M 的理论基础：参数重要性

M 的语义是"这个 block 对我有多重要"——越重要 → 门越低 → 越保护。M 的定义直接决定 AFP 的效用。Phase 0 v5 提供两种度量：

#### 1. Magnitude-based（TIES-Merging 风格，消融用）

```
imp[j] ∝ mean(|W_trained[j] - W_init[j]|)
```

- **优点**：零数据依赖，零计算开销，与 TIES-Merging (Yadav, NeurIPS 2023) 对标
- **缺陷**：度量"训练改动了多少"，不等价于"对模型行为有多重要"

#### 2. MAS-based（Memory Aware Synapses — Phase 0 v5 默认）

```
Ω[j] ∝ E_x[ mean_{p∈block_j} |∂[F(x)²]/∂θ_p| ]
```

- **来源**：Aljundi et al. (ECCV 2018)
- **优点**：直接度量功能敏感性，无需标签，一次前向+反向即可
- **解决反例**：embedding 层变化小但输出高度敏感 → MAS 给高重要性 → 正确保护

#### 3. 门控函数（独立于 importance 度量）

```
Gate "rational" (EWC 推导, 默认):
  M[j] = τ / (τ + imp[j])

Gate "linear" (消融):
  M[j] = clamp(1 - imp[j]/τ, 0, 1)
```

- **rational 推导**：EWC Lagrangian dual (Kirkpatrick et al., PNAS 2017) → B 的梯度被 `1/(1+λ·F)` 放缩。令 τ=1/λ, Ω=F → `M=τ/(τ+Ω)`。处处光滑，无硬截断。
- **linear**：`--gate=linear` 消融对比。

#### 为什么剪枝文献（SNIP/GraSP/SynFlow）不可用

Frankle et al. (ICLR 2021) 证明：这些 at-initialization 方法的 per-weight 重要性是假的——层内打乱掩码不影响性能。它们只捕获 per-layer 比率。AFP 工作在**训练后**，需要后训练重要性（EWC/MAS/SI 家族）。

详见 [REFERENCES.md](REFERENCES.md)。

---

## 六、F-IVN：跨架构的函数空间谈判

重量空间 IVN 的前提是同架构。当 A 是 Pythia(24层)、B 是 TinyLlama(22层) 时，`W_A + W_B` 无定义。

**F-IVN 把谈判从参数空间搬到预测空间：**

```
V ∈ [0,1]^{N}     — 虚拟预测（不是虚拟权重）
d_A = -η·(V - P_A) — 提案 = "V 应该向我靠拢"
M_A[i] = 1 - |P_A[i]-0.5|/τ — 门控 = "我确定的样本我坚持"

V_{t+1} = V_t + M_A ⊙ d_B + M_B ⊙ d_A
```

V 是 N 维向量，与架构无关。Pythia 和 TinyLlama 各自用自己的方式预测、用自己的方式逼近 V_T（通过 KL 蒸馏）。

| | 重量空间 IVN | **F-IVN** |
|---|---|---|
| **谈判空间** | 参数空间 | **预测空间** |
| **V 维度** | 1.4B | **500** |
| **架构要求** | 必须相同 | **任意** |

---

## 七、研究路线图

### Phase 0：同构 IVN — LMC 边界探测（重量空间）

**核心问题**：模型差异度如何影响损失景观连通性？AFP/IVN 在什么临界点上开始优于 FedAvg？

**理论框架**：Linear Mode Connectivity (Frankle et al., ICML 2020)
- 从同一预训练出发 fine-tune 的模型通常处于同一线性连通盆地
- 盆地内：任何线性组合（FedAvg, AFP, IVN）等价 → **选择性集成无增益**
- 盆地间：线性插值有 loss barrier → **AFP/IVN 可能找到更好的非线路径**

**实验**：
1. Pythia-1.4B full-FT on code+medical
2. LMC pretest：测线性插值 loss barrier
3. 差异度光谱：逐步增加训练强度，直到 LMC 破裂
4. 在每个差异度级别对比：IVN vs AFP vs FedAvg vs No Exchange

**预期**：
- 低差异：AFP ≈ FedAvg（验证 LMC 框架正确性）
- 高差异：IVN > AFP > FedAvg（核心 claim）

**入口**：`bash scripts/train_and_run_phase0.sh`

### Phase 1：异构 F-IVN（函数空间）
- Pythia-1.4B vs TinyLlama-1.1B
- 谈判在预测空间，蒸馏回到各自参数空间
- 入口：`python scripts/run_fivn_phase0.py`

### Phase 2：trust 学习 + 恶意防御

### Phase 1：trust 的学习
- trust 不再是固定的，而是交互历史的函数
- Agent 学习"谁可信"——通过验证对方权重在自己私有数据上的表现
- **核心指标**：trust 能否收敛到真实的可靠性排序？

### Phase 2：扩展到 N 个 agent
- 去中心化拓扑：agent 之间可以自由选择交互对象
- Agent 需要同时决策：和谁交流？学什么？
- **核心指标**：系统是否收敛？收敛速度与中心化 FL 对比？

### Phase 3：对抗鲁棒性
- 注入恶意 agent（随机噪声 / 定向攻击 / 后门）
- **核心指标**：AFP 的鲁棒性是否显著优于 FedAvg？

### Phase 4：理论分析
- AFP 的收敛性证明
- trust 函数的统计性质
- 信息论视角：权重中的知识传输效率

---

## 六、与旧项目 (FCL-PRM) 的关系

- FCL-PRM 的**工程积累**可复用：训练脚本、模型定义、联邦模拟器
- FCL-PRM 的**科学教训**已吸取：不要在没有理论支撑的情况下做"诊断工具"
- FCL-PRM 的**实验数据**可作为 AFP 的 baseline 对比

**AFP 不是 FCL-PRM 的延续——它是一个全新的范式。FCL-PRM 试图诊断一个已有系统中的问题，AFP 试图重新定义系统本身。**
