# AFP — 研究方向与文献地图

> **状态**：Phase 0 v2 诊断进行中 — full-FT code+medical，验证 per-block importance 结构性偏移

---

## 一、文献全景

### 1.1 联邦 PRM（过程奖励模型）— 完全真空

经过 25+ 次跨数据库搜索（arXiv, Semantic Scholar, OpenAlex）：

| 搜索组合 | 结果 |
|----------|------|
| "federated process reward model" | **0 篇** |
| "federated PRM" + LLM reasoning | **0 篇** |
| "federated step-level reward" | **0 篇** |
| "federated verifier" + LLM | **0 篇** |

### 1.2 邻近但从未交叉的两条线

**A 线 — 联邦奖励建模（只有 outcome-level，标准 FedAvg 范式）**

| 论文 | 做了什么 | 局限 |
|------|---------|------|
| PluralLLM (2025.03) | 联邦偏好预测器，FedAvg 聚合 | Outcome-level，标准 FedAvg |
| MoR (2026.05) | 各客户端独立训 reward model，服务器融合 | Outcome-level，中心化融合 |
| FedMOA (2026.01) | 联邦 GRPO 训 policy model | 训的是 actor 不是 verifier |
| FedVPA-GP (2026.05, ICML 2026) | 联邦变分偏好对齐 | Outcome-level |
| PWFF (2024.04) | 个性化联邦 LLM 微调，reward model 作为辅助 | Reward model 不是主要训练目标 |

**B 线 — 领域专用 PRM（全是集中式训练）**

| 论文 | 领域 | 关键发现 |
|------|------|---------|
| Fin-PRM (2025) | 金融推理 | **明确声明**"通用 PRM 在金融推理上失败" |
| Clinical PRM (2024, EMNLP) | 临床笔记验证 | 需要域内医学数据训练 |
| VisualPRM (2025) | 多模态推理 | 400K 样本, 8B 参数 |
| DreamPRM (2025) | 多模态 | 用 bi-level optimization 处理领域不均衡——但仍集中式 |
| FG-PRM (2024, EMNLP) | 幻觉检测 | 细粒度 step-level 评估 |
| uPRM (2026) | 通用 | 无需人工标注的无监督 PRM 训练 |

**A 和 B 从未交汇。联邦 × PRM 是真空。**

### 1.3 AFP 相关领域

**去中心化/P2P FL**

| 论文 | 做了什么 | 与 AFP 的关系 |
|------|---------|:---:|
| Gossip Learning (经典) | P2P 拓扑，gossip averaging | AFP 的拓扑基础（但 Gossip 仍用平均） |
| BrainTorrent (2018) | P2P FL for medical imaging | 早期 P2P FL，但使用简单平均 |
| Swarm Learning (2021, Nature) | 去中心化 FL with blockchain | 拓扑去中心化，但聚合仍是平均 |

**选择性/质量感知聚合**

| 论文 | 做了什么 | 与 AFP 的关系 |
|------|---------|:---:|
| FedStrategist (2025.07) | 元学习自适应聚合 | 有"学习如何聚合"的意识，但决策在服务器端 |
| Attention-based Aggregation | 用 attention 加权客户端 | 比等权平均好，但仍是线性组合 |
| Robust Aggregation (Krum, Trimmed Mean, etc.) | 统计方法剔除异常值 | 被动防御，非主动选择 |

**模型合并 / 权重空间分析**

| 论文 | 做了什么 | 与 AFP 的关系 |
|------|---------|:---:|
| Model Soups / Model Merging | 多个 fine-tuned 模型的线性合并 | 关注"什么时候合并有效"，但仍是线性 |
| Task Arithmetic | 权重空间中做加/减法来组合任务能力 | AFP 的子空间操作可借鉴 |
| Git Re-Basin | 通过 permutation symmetry 对齐权重空间 | AFP 的知识提取器可借鉴 |

**多目标梯度下降 / MGDA（IVN 的数学基础 — 2026-06-19 新增）**

| 论文 | 做了什么 | 与 AFP 的关系 |
|------|---------|:---:|
| **Désidéri (2012)** MGDA | 多目标最速下降：求 convex hull 中 min-norm 方向。证明收敛到 Pareto 平稳点 | IVN 是 MGDA 的轻量变体：importance gate 替代每轮 QP 求解 |
| **Sener & Koltun (NeurIPS 2018)** MGDA-UB | MGDA 用于深度多任务学习。Frank-Wolfe 求解器 + 上界近似 | Per-block gate 是 Frank-Wolfe optimization 的替代 |
| **Lian et al. (NeurIPS 2017)** D-PSGD | 去中心化 SGD 的首个收敛分析：通信 O(Deg) vs 中心化 O(n) | M=1 时 IVN 退化为此（2 节点情形），收敛性已证明 |
| **Lee et al. (COLT 2016)** GD → Minimizers | 随机初始化 + GD 几乎必然避开严格鞍点，收敛到局部极小 | 保证 V 谈判不会卡在鞍点 |

**损失景观几何 / Linear Mode Connectivity（Phase 0 v5 的理论基础 — 2026-07-03 新增）**

| 论文 | 做了什么 | 与 AFP 的关系 |
|------|---------|:---:|
| **Frankle et al. (ICML 2020)** LMC | 从同一预训练出发 fine-tune 的模型处于线性连通盆地。Barrier 测试：`max_α L((1-α)θ_A + αθ_B)` | **Phase 0 v5 前置检验**：barrier≈0 → AFP=FedAvg；barrier>0 → AFP 有发挥空间 |
| **Entezari et al. (ICLR 2022)** | LMC 在 permutation symmetry 下几乎普遍成立。随机初始化已被对齐消除 | 解释为什么同架构模型 importance cosine 接近 1.0 |
| **Ainsworth et al. (ICLR 2023)** Git Re-Basin | 通过 permutation matching 将独立训练的模型对齐到同一盆地 | AFP 的知识提取器可以借鉴 permutation alignment 技术 |
| **Garipov et al. (NeurIPS 2018)** | 神经网络 loss surface 上存在低损连接曲线（Bezier curves） | 选择性集成可能在 loss surface 上找到比线性插值更优的路径 |

**参数重要性估计（AFP gate 的度量基础 — 2026-06-19 新增）**

| 论文 | 做了什么 | 与 AFP 的关系 |
|------|---------|:---:|
| **EWC** (Kirkpatrick, PNAS 2017) | Fisher 对角 = 参数重要性。贝叶斯推导 | 比 magnitude 更 principled，需标签 |
| **MAS** (Aljundi, ECCV 2018) | 输出敏感度 = 参数重要性。无标签 | **Phase 0 v5 默认**：直接度量功能重要性 |
| **SI** (Zenke, ICML 2017) | 路径积分 = 参数重要性。训练中在线 | 需训练轨迹，Phase 1 可用 |
| **TIES-Merging** (Yadav, NeurIPS 2023) | `\|τ\|` magnitude 做 trim-elect-sign | AFP magnitude 方法对标 TIES 的硬门控 |
| **SNIP** (Lee, ICLR 2019) / **GraSP** (Wang, ICLR 2020) | 初始化时剪枝重要性 | ❌ 不适用（at-init，非训练后） |
| **Frankle et al.** (ICLR 2021) | 证明 SNIP/GraSP per-weight 重要性是假的 | 教训：初始化重要性不可靠，后训练才可靠 |

**数据无关知识蒸馏**

| 论文 | 做了什么 | 与 AFP 的关系 |
|------|---------|:---:|
| Data-Free KD (经典) | 不用原始数据，教师模型蒸馏到学生模型 | 单向、不可见数据 → AFP 的双向 + 选择性 |

### 1.4 引用验证（来自 FCL-PRM 旧项目）

以下论文从旧项目 DIRECTION.md 继承，已逐篇验证真实存在：

| 论文 | 正式信息 | 相关性 |
|------|---------|:---:|
| FedAPM (KDD 2025) | `2506.04672` — partial model personalization 加剧 client drift | 中等 |
| FedDecorr (TPAMI 2023) | `2210.00226` — FL 数据异构驱动维度坍缩 | 中等 |
| FedBRICK (AAAI 2026) | `10.1609/aaai.v40i34.40083` — 冻结层越多 structural bias 越大 | 高 |
| SCAFFOLD (ICML 2020) | `1910.06378` — 控制变量修正 client drift（注：drift 来自数据异构，非参数维度） | 中等 |
| FedRCL (CVPR 2024) | `2401.04928` — 正式标题 "Relaxed Contrastive Learning for FL" | 低 |
| Momentum Benefits Non-IID FL (ICLR 2024) | `2306.16504` — 动量消除异构性假设 | 低 |

---

## 二、AFP 的文献定位

### 空白确认

AFP 的六大组件各自在文献中有零件，但**完整系统无先例**：

| 组件 | 文献现状 | 空白 |
|------|---------|------|
| P2P 拓扑 | Gossip learning, Swarm Learning | 仍用平均聚合 |
| 选择性聚合 | Attention-based, quality-aware | 决策在服务器端 |
| 信任/声誉 | Trust-aware FL, blockchain FL | 用于"选客户端"，不是"学多少" |
| 子空间操作 | Task arithmetic, model merging | 不涉及隐私约束 |
| Data-free 蒸馏 | Data-free KD | 单向、压缩场景 |
| Agent 自主决策 | **无先例** | **AFP 的核心创新** |

### 理论工具缺口

AFP 需要的数学工具：

- **子空间分析**：如何从权重中提取"有效知识子空间"？
- **信任的统计理论**：trust 作为交互历史的函数，其统计性质？
- **P2P 互学习的收敛性**：当每个 agent 的更新规则不同、学习率不同、选择对象不同时，系统收敛吗？
  - ✅ **IVN 2-agent 收敛性已可证明**：直接应用 descent lemma (Nesterov) 到组合目标 Φ(V)。退化情形 (M=1) 收敛性来自 D-PSGD (Lian et al. 2017)。
  - ❌ **N-agent 收敛性未解决**：需要去中心化 Gossip + 异构更新规则的联合分析。
- **信息论视角**：权重作为通信媒介，信息传输率是多少？瓶颈在哪？
- **MGDA 对比**：IVN 作为 MGDA 的轻量变体，在什么条件下二者的 Pareto-front 质量可比较？

---

## 三、不再考虑的旧方向

以下方向来自 FCL-PRM 旧项目，已评估并关闭：

| 方向 | 关闭原因 |
|------|---------|
| CD-SPI 诊断框架 | 实验证伪（全配置 sym ≈ 0.001，无区分度） |
| 容量-发散性质因果理论 | 依赖"去噪有用"的前提（未验证），且问题设定不如 AFP 根本 |
| Null Result 叙事 | 可发表但学术价值有限 |
| 纯理论（子空间约束 → 噪声发散） | 仍在现有 FL 范式内，不触及范式本身的问题 |
