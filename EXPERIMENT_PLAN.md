# AMP — 实验计划

> Phase 0 是最简原型，目标：**证明 AMP 的选择性子空间更新优于线性平均。**

---

## Phase 0：2-Agent 最简原型

### 目标

在两个 agent、固定 trust（预设双方可信）的最简设定下：
1. 证明 AMP 的选择性子空间更新（W'=W_A+M⊙(W_B-W_A)）优于 FedAvg 的线性平均（W'=(1-α)W_A+α·W_B）
2. 建立评估框架，为后续 Phase 提供 baseline

### 实验设定

```
Agent A: 私有数据 D_A（数学推理步骤）
Agent B: 私有数据 D_B（代码推理步骤）

初始化：两个 agent 从相同的 pretrained backbone 出发（e.g., Pythia-1.4B）
训练：各自在私有数据上做若干 epoch 的本地训练 → 得到 W_A, W_B
交互：交换权重，分别用 AMP 和 FedAvg 做知识整合
评估：各自在对方的私有数据上的表现（作为泛化能力的 proxy）
```

### 核心对比

| 方法 | 更新规则 | 自由参数 |
|------|---------|---------|
| **FedAvg** | W_A' = (1-α)W_A + α·W_B | 标量 α |
| **AMP v0** | W_A' = W_A + M_A⊙(W_B - W_A) | 门控掩码 M_A（per-parameter） |

### Phase 0 评估指标

1. **跨域泛化**：整合 B 的知识后，A 在 D_B 上的性能提升
2. **自域保留**：整合 B 的知识后，A 在 D_A 上的性能变化（不应变差）
3. **知识传输效率**：每单位"权重变化量"带来的性能提升
4. **噪声抗性**：当 B 注入不同程度噪声时，AMP vs FedAvg 的性能差异

### Phase 0 预期结果

- AMP 在跨域泛化上优于 FedAvg（因为选择性更新保留了 A 的原有知识，同时只在相关子空间接受 B 的知识）
- AMP 在自域保留上优于 FedAvg（门控掩码保护了 A 的核心能力）
- AMP 在噪声抗性上显著优于 FedAvg（门控掩码在噪声维度上关闭）

### 旧项目可复用组件

| 组件 | 路径 | 用途 |
|------|------|------|
| StepRewardModel | `FCL-PRM-cdspi/src/fclprm/models/base_wrapper.py` | PRM 模型结构 |
| 联邦模拟器 | `FCL-PRM-cdspi/src/fclprm/federated/simulator.py` | 多客户端训练调度 |
| 训练脚本 | `FCL-PRM-cdspi/scripts/train_centralized_prm.py` | 本地训练逻辑 |
| 数据分区 | `FCL-PRM-cdspi/src/fclprm/data/heterogeneity.py` | Dirichlet 分区 |
| VersaPRM 数据 | `FCL-PRM-cdspi/data/versaprm/` | 多领域 step-level 数据 |

---

## 后续 Phase（概要）

### Phase 1：trust 的学习

- trust 从"预设"变成"交互历史的函数"
- Agent 通过验证对方权重在自己私有数据上的表现来更新 trust
- 对比：固定 trust vs 学习 trust 的性能差异

### Phase 2：N-agent 去中心化

- 3+ agent，各自选择与谁交互
- 研究：系统是否收敛？收敛速度？
- 研究：agent 的交互策略（谁最值得交流？）

### Phase 3：对抗鲁棒性

- 注入恶意 agent（随机噪声 / 定向攻击 / 后门）
- 对比 AMP vs FedAvg vs Robust Aggregation 的鲁棒性

### Phase 4：理论分析

- AMP 的收敛性证明
- trust 函数的统计性质
- 信息论视角：权重中的知识传输率
