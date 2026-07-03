# AFP — Claude Code 项目上下文

> **新会话必读**：[docs/internal/VISION.md](docs/internal/VISION.md) — AFP 核心愿景与协议形式化

## 项目简介

**AFP (Agentic Federated Protocol)** — 去中心化的 P2P 互学习协议。模型不再被动接受聚合，而是自主决定学什么、学多少、信谁。

核心文档（均在 `docs/internal/`）：
- [VISION.md](docs/internal/VISION.md) — 协议形式化定义
- [DIRECTION.md](docs/internal/DIRECTION.md) — 文献地图与研究空白
- [REFERENCES.md](docs/internal/REFERENCES.md) — 引用文献（MGDA, D-PSGD, GD收敛）
- [EXPERIMENT_PLAN.md](docs/internal/EXPERIMENT_PLAN.md) — Phase 0 实验方案（v4）
- [ENGINEERING.md](docs/internal/ENGINEERING.md) — 工程手册（必读）
- [contribution-assessment.md](docs/internal/contribution-assessment.md) — 投稿级别评估
- [REVIEW_ISSUES.md](docs/internal/REVIEW_ISSUES.md) — 审稿问题清单
- [IMPORTANCE_ANALYSIS.md](docs/internal/IMPORTANCE_ANALYSIS.md) — importance 指标数学推导（为什么梯度方法结构失效）
- [ROOT_CAUSE_IMPORTANCE.md](docs/internal/ROOT_CAUSE_IMPORTANCE.md) — importance 失效根因（共同模式问题）

用户 HTML 报告（均在 `docs/reports/`）：
- [vision.html](docs/reports/vision.html) — 核心愿景
- [EXPERIMENT_PLAN.html](docs/reports/EXPERIMENT_PLAN.html) — Phase 0 实验方案
- [phase0-summary-*.html](docs/reports/phase0-summary-2026-06-19.html) — 当前状态

## 项目结构

```
AFP/
├── README.md
├── CLAUDE.md
├── .gitignore
├── scripts/
│   ├── run_experiment_a.py      # 实验 A：训练 + AFP vs FedAvg vs 不交换
│   └── generate_reports.py      # MD → HTML 报告生成
├── src/AFP/
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

## 实验入口

```bash
# 默认: Qwen2.5-Coder ⇄ Qwen2.5-Math (同架构，不训练)
python scripts/run_ivn_phase0.py

# 异构 F-IVN (函数空间)
python scripts/run_fivn_phase0.py
```

## Phase 0 关键路径

1. `python scripts/run_ivn_phase0.py` — IVN vs AFP vs FedAvg (~40min)
2. 确认 importance cosine < 0.8
3. IVN net > AFP net → 多轮谈判优于单次更新

## 工程手册

**[docs/internal/ENGINEERING.md](docs/internal/ENGINEERING.md)** — 从旧项目迁移的完整工程知识。新 agent 在写任何训练代码前必须先读。

## 旧项目参考

- 路径：`/Users/Apple/project/FCL-PRM-cdspi`
- 已吸取教训：CD-SPI 诊断框架被实验证伪；head-only 下发散不可测；CKA ≡ 1.0 因为 Pythia 1.4B backbone 过于顽固

## 工作约定

- 提交信息用英文，`feat(scope): description`
- 训练 >1h 用 nohup + &
- 日志写入 `experiments/`
