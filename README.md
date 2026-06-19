# AFP (Agentic Federated Protocol)

**去中心化的 P2P 互学习协议**

---

AFP 是一种全新的联邦学习范式。模型不再被动接受服务器的平均聚合，而是像学者一样——通过权重论文交流知识，自主决定学什么、学多少、信谁。

- [docs/reports/](docs/reports/) — 所有 HTML 报告（愿景、新颖性审查、竞争格局、Phase 0 方案）

## 一句话

> 把联邦学习从"分布式优化问题"变成"多 agent 互学习问题"。

## 核心创新

| 现行 FL (FedAvg) | AFP |
|---|---|
| W' = (1-α)W_A + α·W_B | W' = W_A + M⊙(W_B - W_A) |
| 线性插值（初等运算） | 选择性子空间更新（高等运算） |
| 服务器替你决定学多少 | **你自己决定**学不学、学什么、学多少 |
| 信任是预设的 | **信任是交互中学出来的** |
| 星型拓扑（中心服务器） | **P2P 网状拓扑** |

## 当前状态

- Phase 0 v2：实验方案已修订（2026-06-19）
  - 训练方式：**full-FT**（非 head-only）
  - Domain 对：**code + medical**（非 math + code）
  - Batch：**1024**（121GB 极限）
  - LR：**余弦 1e-4 → 3e-6**（先大步后小步）
- 先诊断 per-block importance 是否有结构性偏移，再跑 AFP vs FedAvg 对比
- 跨架构恶意注入防御：完全真空（0 篇论文）
- 旧项目 FCL-PRM 的工程积累已迁移至 `docs/internal/ENGINEERING.md`
