# AMP — 研究方向与文献地图

> **状态**：完全开放探索 → 方向已收敛至 AMP 协议

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

### 1.3 AMP 相关领域

**去中心化/P2P FL**

| 论文 | 做了什么 | 与 AMP 的关系 |
|------|---------|:---:|
| Gossip Learning (经典) | P2P 拓扑，gossip averaging | AMP 的拓扑基础（但 Gossip 仍用平均） |
| BrainTorrent (2018) | P2P FL for medical imaging | 早期 P2P FL，但使用简单平均 |
| Swarm Learning (2021, Nature) | 去中心化 FL with blockchain | 拓扑去中心化，但聚合仍是平均 |

**选择性/质量感知聚合**

| 论文 | 做了什么 | 与 AMP 的关系 |
|------|---------|:---:|
| FedStrategist (2025.07) | 元学习自适应聚合 | 有"学习如何聚合"的意识，但决策在服务器端 |
| Attention-based Aggregation | 用 attention 加权客户端 | 比等权平均好，但仍是线性组合 |
| Robust Aggregation (Krum, Trimmed Mean, etc.) | 统计方法剔除异常值 | 被动防御，非主动选择 |

**模型合并 / 权重空间分析**

| 论文 | 做了什么 | 与 AMP 的关系 |
|------|---------|:---:|
| Model Soups / Model Merging | 多个 fine-tuned 模型的线性合并 | 关注"什么时候合并有效"，但仍是线性 |
| Task Arithmetic | 权重空间中做加/减法来组合任务能力 | AMP 的子空间操作可借鉴 |
| Git Re-Basin | 通过 permutation symmetry 对齐权重空间 | AMP 的知识提取器可借鉴 |

**数据无关知识蒸馏**

| 论文 | 做了什么 | 与 AMP 的关系 |
|------|---------|:---:|
| Data-Free KD (经典) | 不用原始数据，教师模型蒸馏到学生模型 | 单向、不可见数据 → AMP 的双向 + 选择性 |

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

## 二、AMP 的文献定位

### 空白确认

AMP 的六大组件各自在文献中有零件，但**完整系统无先例**：

| 组件 | 文献现状 | 空白 |
|------|---------|------|
| P2P 拓扑 | Gossip learning, Swarm Learning | 仍用平均聚合 |
| 选择性聚合 | Attention-based, quality-aware | 决策在服务器端 |
| 信任/声誉 | Trust-aware FL, blockchain FL | 用于"选客户端"，不是"学多少" |
| 子空间操作 | Task arithmetic, model merging | 不涉及隐私约束 |
| Data-free 蒸馏 | Data-free KD | 单向、压缩场景 |
| Agent 自主决策 | **无先例** | **AMP 的核心创新** |

### 理论工具缺口

AMP 需要的数学工具：

- **子空间分析**：如何从权重中提取"有效知识子空间"？
- **信任的统计理论**：trust 作为交互历史的函数，其统计性质？
- **P2P 互学习的收敛性**：当每个 agent 的更新规则不同、学习率不同、选择对象不同时，系统收敛吗？
- **信息论视角**：权重作为通信媒介，信息传输率是多少？瓶颈在哪？

---

## 三、不再考虑的旧方向

以下方向来自 FCL-PRM 旧项目，已评估并关闭：

| 方向 | 关闭原因 |
|------|---------|
| CD-SPI 诊断框架 | 实验证伪（全配置 sym ≈ 0.001，无区分度） |
| 容量-发散性质因果理论 | 依赖"去噪有用"的前提（未验证），且问题设定不如 AMP 根本 |
| Null Result 叙事 | 可发表但学术价值有限 |
| 纯理论（子空间约束 → 噪声发散） | 仍在现有 FL 范式内，不触及范式本身的问题 |
