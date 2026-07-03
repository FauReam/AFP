# AFP 参考文献

> 最后更新：2026-06-19
> PDF 本地副本：`papers/` 目录（9 篇，共 ~15MB）

---

## 核心理论支撑

### Linear Mode Connectivity（AFP 失效的根本原因）

- **Frankle, J., Dziugaite, G.K., Roy, D.M., & Carbin, M. (2020).** "Linear Mode Connectivity and the Lottery Ticket Hypothesis." *ICML 2020*.
  - arXiv: [1912.05671](https://arxiv.org/abs/1912.05671)
  - **核心发现**：从同一初始化出发、不同 SGD 噪声训练的神经网络，沿线性路径 `(1-α)θ_A + αθ_B` 的 loss 保持在低水平——即它们在**同一个损失盆地**内线性连通。
  - **与 AFP 的关系**：AFP 的 IVN/FedAvg 本质上就是线性插值 `V_new = V_A + M⊙(V_B - V_A)`。如果两个模型在同一个线性连通的盆地内（cosine ≈ 1.0），则**任何线性插值都是等价的** → gate 无法产生区分 → AFP = FedAvg。这正是我们 Phase 0 观察到的现象。

- **Entezari, R., Sedghi, H., Saukh, O., & Neyshabur, B. (2022).** "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks." *ICLR 2022*.
  - arXiv: [2110.06296](https://arxiv.org/abs/2110.06296)
  - 核心贡献：独立训练的模型在**正确排列神经元**后满足 LMC。SGD 解之间存在隐式的排列对称性。
  - 与 AFP 的关系：Pythia-1.4B 两个 domain 特化模型的权重距离仅 ~1.2%（相对于 base），几乎在同一个 basin 里。排列不变性意味着它们本质上是同一个模型——gate 无法区分。

- **Ainsworth, S., Hayase, J., & Srinivasa, S. (2023).** "Git Re-Basin: Merging Models modulo Permutation Symmetries." *NeurIPS 2023*.
  - arXiv: [2209.04836](https://arxiv.org/abs/2209.04836)
  - 核心贡献：实际算法——通过求解线性分配问题找到神经元排列，使两个独立训练的模型进入同一个 basin。
  - 与 AFP 的关系：Git Re-Basin 的成功暗示：**在 AFP 能发挥作用之前，我们首先需要两个模型确实在不同 basin 里**。Phase 0 失败的根本原因不是 gate 不好，而是**两个模型之间没有需要 gate 来区分的差异**。

- **Juneja, J., Bansal, R., Cho, K., Sedoc, J., & Saphra, N. (2023).** "Linear Connectivity Reveals Generalization Strategies." *ICLR 2023*.
  - arXiv: [2205.12411](https://arxiv.org/abs/2205.12411)
  - 核心贡献：LMC 不仅关于收敛——不同数据/正则化训练的模型可能落在**不同的线性连通分量**中。LMC 的缺失指示了 qualitative 的功能差异。
  - 与 AFP 的关系：**AFP 的价值应该出现在 LMC 不成立的模型对之间**。如果我们测量 `(1-α)θ_code + αθ_medical` 的 loss 曲线是一条平坦线（LMC 成立），那 AFP 就没用。如果 loss 在中间位置出现 barrier（LMC 不成立），则 AFP 的 gate 可以找到各自最佳的"接收区间"。**这应该是 AFP 实验的前置检验。**

### 多目标梯度下降（IVN 的数学基础）

- **Désidéri, J.-A. (2012).** "Multiple-gradient descent algorithm (MGDA) for multiobjective optimization." *Comptes Rendus Mathematique*, 350(5–6), 313–318.
  - DOI: [10.1016/j.crma.2012.03.014](https://doi.org/10.1016/j.crma.2012.03.014)
  - 核心贡献：将最速下降推广到多目标情形，证明收敛到 Pareto 平稳点。在凸组合中求最小范数方向作为共同下降方向。
  - 与 IVN 的关系：IVN 是 MGDA 的简化变体——用预计算的 per-block importance gate 替代每轮 QP 求解。

- **Désidéri, J.-A. (2012).** "MGDA II: A direct method for calculating a descent direction common to several criteria." INRIA Research Report RR-7922.
  - 扩展：Gram-Schmidt 正交化计算下降方向。

- **Sener, O. & Koltun, V. (2018).** "Multi-Task Learning as Multi-Objective Optimization." *NeurIPS 2018*.
  - URL: [papers.nips.cc/paper/7334](http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf)
  - 核心贡献：将 MGDA 应用于深度神经网络多任务训练（MGDA-UB），证明可以找到 Pareto 最优解。Frank-Wolfe 求解器避免每轮 QP 开销。
  - 与 IVN 的关系：IVN 的 per-block gating 是 MGDA-UB 的替代方案——用 importance 一次性确定权重，无需每轮求解优化问题。更轻量，但非 Pareto 最优。

### 梯度下降收敛理论

- **Lee, J.D., Simchowitz, M., Jordan, M.I., & Recht, B. (2016).** "Gradient Descent Conquerges to Minimizers." *COLT 2016*.
  - arXiv: [1602.04915](https://arxiv.org/abs/1602.04915)
  - 核心贡献：随机初始化 + GD 几乎必然收敛到局部极小值，避开严格鞍点。
  - 与 IVN 的关系：保证 V 谈判过程不会卡在鞍点。

- **Nesterov, Y. (2004).** *Introductory Lectures on Convex Optimization*. Springer.
  - 核心贡献：Descent lemma — 对 L-smooth 函数，η ≤ 1/L 保证单调下降。
  - 与 IVN 的关系：IVN 收敛性的最简证明路径——直接应用 descent lemma 到组合目标 Φ(V)。

### 去中心化 SGD

- **Lian, X., Zhang, C., Zhang, H., Hsieh, C.-J., Zhang, W., & Liu, J. (2017).** "Can Decentralized Algorithms Outperform Centralized Algorithms? A Case Study for Decentralized Parallel Stochastic Gradient Descent." *NeurIPS 2017*.
  - arXiv: [1705.09056](https://arxiv.org/abs/1705.09056)
  - 核心贡献：去中心化 SGD 的首个理论分析——通信复杂度 O(Degree) vs 中心化 O(n)，总计算复杂度等价。
  - 与 IVN 的关系：M=1 时 IVN 退化为 2 节点 D-PSGD，收敛性已有充分证明。

---

## 参数重要性估计（AFP gate 的理论基础）

### 后训练重要性（训练后度量哪些参数对模型行为关键）

- **Kirkpatrick, J. et al. (2017).** "Overcoming catastrophic forgetting in neural networks." *PNAS*, 114(13), 3521–3526.
  - 提出 **EWC (Elastic Weight Consolidation)**：用 Fisher 信息矩阵对角元 `F_ii = E[(∂log p/∂θ_i)²]` 度量参数重要性。贝叶斯推导（Laplace 近似）。
  - 与 AFP 的关系：Fisher 重要性比 magnitude 更 principled，但需要标签 + 额外前向/反向 pass。

- **Aljundi, R. et al. (2018).** "Memory Aware Synapses: Learning what (not) to forget." *ECCV 2018*.
  - arXiv: [1711.09601](https://arxiv.org/abs/1711.09601)
  - 提出 **MAS**：`Ω_p = E_x[‖∂[‖F(x)‖²]/∂θ_p‖]` — 度量输出对参数的敏感度。**无需标签**、在线可算。
  - **AFP Phase 0 v5 默认采用 MAS**。直接回答："改动这个参数会多大程度改变模型行为？"

- **Zenke, F., Poole, B., & Ganguli, S. (2017).** "Continual Learning Through Synaptic Intelligence." *ICML 2017*.
  - arXiv: [1703.04200](https://arxiv.org/abs/1703.04200)
  - 提出 **SI (Synaptic Intelligence)**：`ω = -∫ g(t)·θ'(t) dt` — 沿训练轨迹的路径积分。训练中在线累积，零额外开销。
  - 与 AFP 的关系：需要训练时记录，不适合 Phase 0（使用预训练模型无训练轨迹）。Phase 1 训练时可用。

### 模型合并中的重要性

- **Yadav, P. et al. (2023).** "TIES-Merging: Resolving Interference When Merging Models." *NeurIPS 2023*.
  - arXiv: [2306.01708](https://arxiv.org/abs/2306.01708)
  - Trim-Elect-Sign: 用 `|τ| = |θ_trained - θ_init|` 做 top-k% 选择。**AFP magnitude 方法直接对标 TIES**（改硬门控为软门控）。
  - 与 AFP 的关系：TIES 的 magnitude trimming 就是当前 `block_importance` 的理论祖先。

### 剪枝 at initialization（不适用于 AFP，但提供反面教训）

- **Lee, N. et al. (2019).** "SNIP: Single-shot Network Pruning based on Connection Sensitivity." *ICLR 2019*.
  - `s_q = |w_q · ∂L/∂w_q|` — 连接敏感度。在初始化时判断重要性。

- **Wang, C. et al. (2020).** "GraSP: Picking Winning Tickets Before Training by Preserving Gradient Flow." *ICLR 2020*.
  - `S(-θ) = -θ ⊙ Hg` — 梯度流保留。二阶（HVP）。

- **Frankle, J. et al. (2021).** "Pruning Neural Networks at Initialization: Why Are We Missing the Mark?" *ICLR 2021*.
  - arXiv: [2009.08576](https://arxiv.org/abs/2009.08576)
  - **关键发现**：SNIP/GraSP/SynFlow 的 per-weight 重要性是假的——层内 shuffle 掩码不影响精度。只捕获了 per-layer 比率。
  - 与 AFP 的关系：**AFP 不是 pruning-at-init**（工作在训练后），但这些方法的失败说明 magnitude-based 重要性在初始化时不可靠。后训练重要性（EWC/MAS/SI）才可靠。

---

## 联邦学习

- **McMahan, B. et al. (2017).** "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017*.
  - FedAvg 的原始论文。arXiv: [1602.05629](https://arxiv.org/abs/1602.05629)

- **Li, T. et al. (2020).** "Federated Optimization in Heterogeneous Networks." *MLSys 2020*.
  - FedProx — proximal term 处理异构性。arXiv: [1812.06127](https://arxiv.org/abs/1812.06127)

- **Karimireddy, S.P. et al. (2020).** "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." *ICML 2020*.
  - Control variate 修正 client drift。arXiv: [1910.06378](https://arxiv.org/abs/1910.06378)

---

## 模型合并 / 权重空间操作

- **Yadav, P. et al. (2023).** "TIES-Merging: Resolving Interference When Merging Models." *NeurIPS 2023*.
  - Trim-Elect-Sign: 选择性参数更新（离线场景）。arXiv: [2306.01708](https://arxiv.org/abs/2306.01708)

- **Yu, L. et al. (2023).** "DARE: Language Model Merge via Data Augmented REgularization." arXiv: [2310.03073](https://arxiv.org/abs/2310.03073)
  - 随机 drop + rescale 合并方法。

- **Ilharco, G. et al. (2022).** "Editing Models with Task Arithmetic." *ICLR 2023*.
  - 权重空间中做加减法组合任务能力。arXiv: [2212.04089](https://arxiv.org/abs/2212.04089)

---

## 联邦奖励建模

- **PluralLLM (2025).** Federated preference predictor, FedAvg aggregation. 仅 outcome-level.

- **MoR (2026.05).** Federated reward model with centralized fusion. Outcome-level.

---

## 去中心化 FL / P2P

- **Warnat-Herresthal, S. et al. (2021).** "Swarm Learning for decentralized and confidential clinical machine learning." *Nature*, 594, 265–270.
  - 去中心化 FL + blockchain，但聚合仍是简单平均。

- **BrainTorrent (2018).** P2P FL for medical imaging. 早期 P2P FL，简单平均聚合。
