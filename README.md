# AMP (Agentic Mutual Protocol)

**去中心化的 P2P 互学习协议**

---

AMP 是一种全新的联邦学习范式。模型不再被动接受服务器的平均聚合，而是像学者一样——通过权重论文交流知识，自主决定学什么、学多少、信谁。

- 🔗 **核心愿景**：[VISION.md](VISION.md) (HTML: [docs/reports/vision.html](docs/reports/vision.html))
- 📚 **文献地图**：[DIRECTION.md](DIRECTION.md)
- 🔬 **实验计划**：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- 📊 **HTML 报告**：[docs/reports/](docs/reports/) — 新颖性审查、竞争格局、Phase 0 方案

## 一句话

> 把联邦学习从"分布式优化问题"变成"多 agent 互学习问题"。

## 核心创新

| 现行 FL (FedAvg) | AMP |
|---|---|
| W' = (1-α)W_A + α·W_B | W' = W_A + M⊙(W_B - W_A) |
| 线性插值（初等运算） | 选择性子空间更新（高等运算） |
| 服务器替你决定学多少 | **你自己决定**学不学、学什么、学多少 |
| 信任是预设的 | **信任是交互中学出来的** |
| 星型拓扑（中心服务器） | **P2P 网状拓扑** |
| 被动执行者 | **主动学习者** |

## 当前状态

- Phase 0：2-agent 最简原型设计阶段
- 文献定位：AMP 的完整形态在文献中无先例；零件散落在 P2P FL、data-free KD、trust-aware FL 等子领域
- 旧项目 FCL-PRM 的工程积累可参考，科学教训已吸取
