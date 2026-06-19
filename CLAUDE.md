# AFP — Claude Code 项目上下文

> **新会话必读**：[docs/internal/VISION.md](docs/internal/VISION.md) — AFP 核心愿景与协议形式化

## 项目简介

**AFP (Agentic Federated Protocol)** — 去中心化的 P2P 互学习协议。模型不再被动接受聚合，而是自主决定学什么、学多少、信谁。

核心文档（均在 `docs/internal/`）：
- [VISION.md](docs/internal/VISION.md) — 协议形式化定义
- [DIRECTION.md](docs/internal/DIRECTION.md) — 文献地图与研究空白
- [EXPERIMENT_PLAN.md](docs/internal/EXPERIMENT_PLAN.md) — Phase 0 实验方案（v2）
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
├── scripts/
│   ├── run_experiment_a.py      # 实验 A：训练 + AFP vs FedAvg vs 不交换
│   └── generate_reports.py      # MD → HTML 报告生成
├── src/afp/
│   ├── protocol/                # AFP 交互协议核心
│   ├── models/
│   ├── data/
│   └── experiments/
├── experiments/
│   └── phase0_diagnostic/       # Step 1 诊断输出
├── configs/
└── docs/
    ├── internal/                # MD（AI agent 用）
    └── reports/                 # HTML（用户用）
```

## 设备

NVIDIA DGX Spark GB10（121GB 统一内存，ARM64）

## Phase 0 训练设定（v2 — 2026-06-19）

| 参数 | 值 |
|------|-----|
| 模型 | Pythia-1.4B |
| 训练方式 | **full-FT**（所有 backbone + PRM 分类头） |
| 数据 | VersaPRM code + medical |
| Batch size | **1024**（121GB 极限） |
| LR schedule | **Cosine: 1e-4 → 3e-6**（先大步后小步） |
| Optimizer | AdamW (β=0.9, 0.999), weight_decay=0.01 |
| Precision | bf16 autocast + fp32 head |
| Max length | 384 tokens |
| Epochs | 1 |
| PRM head | 2048 → 256 → 1 (ReLU) |

### 为什么是 full-FT 而非 head-only

head-only 下 backbone 不变 → importance[j] = 0 ∀j → 门控无意义 → 实验无效。
full-FT 让 backbone 产生真正的 per-block 专业化差异。

### 为什么是 code + medical 而非 math + code

code 和 medical 激活的 backbone 区间重叠更小 → importance 模式更正交 → 对 AFP 是更干净的测试。

## Phase 0 关键路径（两步）

**Step 1 — 诊断（~5-6h）：**
1. `python scripts/run_experiment_a.py --skip-train` — 如果已训练，直接跑对比
2. `python scripts/run_experiment_a.py` — 完整流程（训练 + 对比）

**Step 2 — 如果诊断有信号 → 推进实验 B/C：**
- 实验 B：跨架构（Pythia vs TinyLlama）
- 实验 C：鲁棒性（恶意注入）

## 工程手册

**[docs/internal/ENGINEERING.md](docs/internal/ENGINEERING.md)** — 从旧项目迁移的完整工程知识。新 agent 在写任何训练代码前必须先读。

## 旧项目参考

- 路径：`/Users/Apple/project/FCL-PRM-cdspi`
- 已吸取教训：CD-SPI 诊断框架被实验证伪；head-only 下发散不可测；CKA ≡ 1.0 因为 Pythia 1.4B backbone 过于顽固

## 工作约定

- 提交信息用英文，`feat(scope): description`
- 训练 >1h 用 nohup + &
- 日志写入 `experiments/`
