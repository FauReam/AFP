# AFP 审稿问题清单（2026-06-19）

> 严格审稿意见中提出的所有问题，按类别记录。不包含推荐解决方案——仅记录问题本身。

---

## 一、理论问题

### 1.1 "高阶"的宣称无法成立 ~~— 已解决 (2026-06-19)~~
- FedAvg → "一阶"（线性插值）✓
- AFP → "二阶"不准确：没有使用二阶导数或曲率信息，只是 elementwise gated linear update
- IVN → "不动点迭代"形式上成立，但**没有证明不动点的存在性、唯一性或收敛性**
- 收敛判据 `ΔV < 1e-4` 是纯粹的数值 heuristics，无理论保证
- ✅ **已解决（框架层面）**：重新定义为 multiple-gradient descent on combined objective Φ(V)。收敛性引用 descent lemma (Nesterov) + D-PSGD (Lian et al. 2017) + MGDA (Désidéri 2012)。VISION.md §5 已更新。
- ⚠️ **注意**：代码没改，改的是说法。没有独立证明 IVN 的收敛性——只是论证它可以归约到已有理论。审稿人若要求独立证明，仍需附录中的 descent lemma 推导。

### 1.2 Importance 度量的信息论基础缺失 ~~— 部分解决 (2026-06-19)~~
- `importance[j] = mean(|W_trained[j] - W_init[j]|)` 存在三个问题：
  - **(a) 量级不等价于重要性**：变化小的 block 可能包含通用知识（应被保护），但当前逻辑会对其开放
  - **(b) 方向信息完全丢失**：相同 L1 偏差可编码完全不同的变换
  - **(c) 归一化问题**：`imp[i] / max(imp)` 使最活跃 block 始终为 1.0，在所有权重变化都小时产生虚假差异
- ✅ (a) **已解决**：实现 MAS (Memory Aware Synapses, Aljundi ECCV 2018) 作为默认。`Ω_j ∝ E[|∂F²/∂θ_j|]` 直接度量功能敏感性。Magnitude 保留为消融。
- ✅ (b) **对 MAS 不是真问题**：`|∂F²/∂θ|` 的绝对值聚合是正确的——重要性 = 敏感度幅度，天然是标量。类比：测梁的承重只需幅度不需方向。只在 magnitude-based 下方向信息丢失才是问题（两个模型相反 ΔW 被赋予相同 imp）。Magnitude 已降级为消融。
- 🟡 (c) **半解决**：max 归一化使门控在相对值上操作。当 block 间有真实敏感度差异时（MAS Ω 的 max/min > 10x），归一化只是正常 scaling。当所有 block 几乎同等敏感时（< 2x），归一化放大噪声。**验证脚本**：`python scripts/check_importance_quality.py` 直接报告 Ω 的 block 间方差。

### 1.3 门控函数缺乏原则性推导 ~~— 已解决 (2026-06-19)~~
- `M[j] = clamp(1 - imp[j]/τ, 0, 1)` 为什么是线性衰减？
- 为什么不是 softmax / sigmoid / 其他形式？
- 为什么 τ 是全局标量而非 per-block？
- 无推导、无消融对比、无灵敏度分析
- ✅ **已解决**：
  1. **原则性形式**：`M[j] = τ/(τ + Ω[j])` — 从 EWC Lagrangian dual 直接推导（Kirkpatrick et al., PNAS 2017）。不是随便选的。
  2. **消融可用**：`--gate linear` 回退旧形式做对比实验。
  3. **实现**：`trust.py` 提供 `gate_rational()` (默认) 和 `gate_linear()` (消融)，integrator + negotiate + noise ctl 全部统一调用。
- ⚠️ **未解决**：τ 仍为全局标量。per-block τ 是可行的扩展方向但需要更多实验支撑。

---

## 二、实验设计问题

### 2.1 Phase 0 设定存在根本性混淆 ~~— 已解决 (2026-06-19)~~
- Qwen2.5-Coder 和 Qwen2.5-Math 的 specialization 来自 Qwen 团队的 continued pretraining + SFT，不是可控实验
- 测试数据 VersaPRM（步骤级正确性标注）与 code/math 能力间接相关但不等价
- 两个模型从未在 VersaPRM 上训练过——AFP/IVN 更新后的提升可能是权重插值的随机扰动而非真知识迁移
- ✅ **已解决**：三个防御：
  1. **相对比较**：度量噪声均等污染所有方法，Δ_IVN - Δ_FedAvg 中噪声抵消
  2. **Noise control**：注入同幅度高斯噪声经门控更新——若 noise Δ ≈ 0 而 IVN Δ > 0，则信号来自知识迁移
  3. **论文中明确承认**评估的不完美性，将 claim 限定为"relative improvement"而非"absolute capability"
- 未解决：自训练数据（路A）是更干净的长期方案，但当前不划算

### 2.2 Baseline 选择严重不足
只比较了 FedAvg。缺失：
- Model Soup / TIES-Merging / DARE
- Task Arithmetic
- FedProx / SCAFFOLD
- Random Gate（消融）
- Identity Gate M=1（消融）

### 2.3 核心判据循环论证
- IVN 和 AFP 使用不同的 τ grid search 策略（AFP 搜 7 个值，IVN 用固定 `args.tau`）
- 没有验证 IVN 的最优 τ 是否等于 AFP 的最优 τ
- `IVN net > AFP net` 和 `importance cosine < 0.8` 互相依赖

### 2.4 可复现性问题
- `torch.randperm` 无固定随机种子
- 谈判过程有随机性（不同 mini-batch 梯度方向）
- 没有多次运行的均值和方差报告

### 2.5 "联邦"二字名不副实
Phase 0 只有 2 个 agent，不涉及：
- 隐私保护（无 DP、无安全聚合）
- 通信约束
- 客户端采样
- 数据异构性的系统性研究
- 大规模 agent 网络的收敛行为

### 2.6 缺失消融实验
未回答以下问题：
- 关掉 M（即 M=1 全局接受），IVN 还工作吗？
- 只用 1 轮（等价于 AFP one-shot with V），还工作吗？
- 门控机制 vs 多轮迭代 vs V_T 中间表示的贡献各是多少？

---

## 三、工程问题

### 3.1 代码重复和设计不一致
- `AFPAgent`（src/afp/protocol/agent.py）和 `FAgent`（run_fivn_phase0.py）是两个独立实现的类
- 共享大量逻辑但互不兼容
- tokenizer 创建方式不统一
- 无统一的 ModelRegistry 或 AgentFactory

### 3.2 BCEWithLogitsLoss 语义问题
- 在 `evaluate` 中对每个 mini-batch 取 `mean`，然后按 batch size 加权
- 只在所有 mini-batch 大小完全相等时严格正确

### 3.3 梯度泄漏风险
- `_compute_proposal` 中每次调用 `load_state_dict` + `train_mode()` 重建参数图
- 多次调用后 `agent.backbone` 参数可能持有残留 `.grad`
- 当前没有 optimizer 所以侥幸没问题，但设计不够安全

---

## 四、文献定位问题

### 4.1 "联邦 PRM 完全真空"的宣称需要限定
- 联邦 RLHF 中 step-level reward 和 outcome-level reward 边界模糊
- 关键词搜索可能遗漏：federated reward model、federated verifier、decentralized reward modeling
- 至少应提及 federated preference learning（PluralLLM 等）的最新进展

### 4.2 Model Merging 社区的轻视
- 将 Model Merging 归类为"离线合并"是动机上的区分，不是机制上的
- TIES-Merging 的 trim-elect-sign 本质上是一种选择性参数更新
- 应做直接实验对比而非用一句话 dismiss

---

## 五、最根本的问题

### 5.1 机制 ≠ 好处
- 即使 AFP/IVN 比 FedAvg 好，不知道是哪个机制在起作用
- 门控掩码的选择性？多轮迭代的隐式正则化？V_T 收敛到中间表示？
- 没有消融来区分

### 5.2 从 2 agent 到 N agent 的推广未经检验
- 所有代码和实验只有 2 个 agent
- 没有拓扑设计（ring / fully connected / random）
- 没有通信轮次调度
- 没有 N-agent 收敛性质讨论
- "P2P 去中心化"是志向，不是当前系统属性

### 5.3 Trust 学习的完全缺失
- `trust = 1.0` 硬编码在整个 Phase 0 中
- 没有 trust 更新机制
- 没有恶意 agent 防御实验
- `compute_trust()` 函数虽已实现但从未被调用
- 作为 AFP 核心创新的声明，无实验支撑

---

## 六、评分维度（汇总）

| 维度 | 评分 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 技术深度 | 2 |
| 实验充分性 | 2 |
| 表述清晰度 | 3 |
| 重要性 | 3 |

**总体结论**：方向有趣，但当前状态距离可发表水平有较大差距。核心缺陷：(a) 声称与证据不匹配，(b) baseline 不足，(c) 缺乏消融区分机制。
