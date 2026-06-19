# AFP (Agentic Federated Protocol)

**去中心化的 P2P 互学习协议**

---

AFP 是一种全新的联邦学习范式。模型不再被动接受服务器的平均聚合，而是像学者一样——通过权重论文交流知识，自主决定学什么、学多少、信谁。

- [docs/reports/](docs/reports/) — 所有 HTML 报告（愿景、新颖性审查、竞争格局、Phase 0 方案）

## 一句话

> 把联邦学习从"分布式优化问题"变成"多 agent 互学习问题"。

## 核心创新

| 现行 FL (FedAvg) | AFP one-shot | **AFP+IVN** |
|---|---|---|
| W' = (1-α)W_A + α·W_B | W' = W_A + M⊙(W_B - W_A) | **V_{t+1} = V_t + M_A⊙d_B + M_B⊙d_A** |
| 线性插值（一阶） | 选择性更新（二阶） | **不动点迭代（高阶）** |
| 1次广播 | 1次交换 | **多轮谈判直到收敛** |

## 当前状态

- Phase 0 v4：IVN 多轮虚拟谈判（2026-06-19）
  - Teacher: Qwen2.5-Coder-1.5B（代码特化）
  - Student: Qwen2.5-Math-1.5B（数学特化）
  - 同架构，不训练，直接下载
  - 对比：IVN vs AFP one-shot vs FedAvg
  - 入口：`python scripts/run_ivn_phase0.py`
- 设备：NVIDIA DGX Spark GB10（121GB，ARM64）
