# AFP — Claude Code 项目上下文

> **新会话必读**：[docs/internal/VISION.md](docs/internal/VISION.md) — AFP 核心愿景与协议形式化

## 项目简介

**AFP (Agentic Federated Protocol)** — 去中心化的 P2P 互学习协议。模型不再被动接受聚合，而是自主决定学什么、学多少、信谁。

核心文档（均在 `docs/internal/`）：
- [VISION.md](docs/internal/VISION.md) — 协议形式化定义
- [DIRECTION.md](docs/internal/DIRECTION.md) — 文献地图与研究空白
- [EXPERIMENT_PLAN.md](docs/internal/EXPERIMENT_PLAN.md) — Phase 0 实验方案
- [ENGINEERING.md](docs/internal/ENGINEERING.md) — 工程手册（必读）
- [contribution-assessment.md](docs/internal/contribution-assessment.md) — 投稿级别评估

用户 HTML 报告（均在 `docs/reports/`）：
- [vision.html](docs/reports/vision.html)
- [novelty-review-2026-06-19.html](docs/reports/novelty-review-2026-06-19.html)
- [heterogeneous-fl-review-2026-06-19.html](docs/reports/heterogeneous-fl-review-2026-06-19.html)
- [phase0-summary-2026-06-19.html](docs/reports/phase0-summary-2026-06-19.html)

## 项目结构

```
AFP/
├── README.md
├── CLAUDE.md
├── .gitignore
├── src/afp/
│   ├── protocol/          # AFP 交互协议核心
│   ├── models/
│   ├── data/
│   └── experiments/
├── experiments/
├── configs/
└── docs/
    ├── internal/          # MD（AI agent 用）
    └── reports/           # HTML（用户用）
```

## 设备

NVIDIA DGX Spark GB10（121GB 统一内存，ARM64）

## 工程手册

**[docs/internal/ENGINEERING.md](docs/internal/ENGINEERING.md)** — 从旧项目迁移的完整工程知识。新 agent 在写任何训练代码前必须先读。

## 旧项目参考

- 路径：`/Users/Apple/project/FCL-PRM-cdspi`
- 已吸取教训：CD-SPI 诊断框架被实验证伪

## Phase 0 关键路径

1. 定义 `protocol/agent.py` — 自主决策
2. 定义 `protocol/integrator.py` — 选择性子空间整合
3. 定义 `protocol/trust.py` — trust 模块
4. 实验：2-agent 原型，AFP vs FedAvg vs 不交换

## 工作约定

- 提交信息用英文，`feat(scope): description`
- 训练 >1h 用 nohup + &
- 日志写入 `experiments/`
