# AFP — Claude Code 项目上下文

> **新会话必读**：[docs/internal/ENGINEERING.md](docs/internal/ENGINEERING.md) — 已知 Bug 清单
> **实验数据**：[docs/internal/EXPERIMENT_PLAN.md](docs/internal/EXPERIMENT_PLAN.md) — 实测结果

## 项目简介

**AFP (Agentic Federated Protocol)** — 去中心化 P2P 互学习协议。与 FedAvg 的核心区别：不是 `(1-α)W_A + α·W_B`（盲平均），而是 `W + M⊙(W_peer - W)`（per-block 选择性门控）。

Phase 0 实验：Pythia-1.4B full-FT on code + medical，用 IVN/AFP/FedAvg 三种方式交换知识。

## 设备与速度

| 项 | 值 |
|----|-----|
| GPU | NVIDIA GB10 (121GB 统一内存, ARM64 CUDA 13.0) |
| 训练速度 | ~5.5s/step, 433 steps/epoch, ~40 min/domain |
| IVN 实验 | ~10 min/run |

## 关键实测结论（2026-07-05）

1. **只有 FedAvg α=0.1 有正收益**（c1m1: net=+0.014）。所有其他方法/参数都破坏模型。
2. **门控范围太窄**：gate = τ/(τ+imp)，τ=0.5 时 gate ∈ [0.33, 0.83]。每个 block 至少混 33% peer 权重，远超可接受的 10%。
3. **IVN 从不谈判**：每次都在第 1 轮收敛（ΔV < 0.001）。
4. **importance cosine > 0.94**：所有重要性指标在两模型间高度相关。不是指标选错了，是两个模型确实太像。
5. **code 训练有 bug**：epoch 2+ 退化，e3 甚至保存了 base 模型。

## Phase 0 入口

```bash
# 训练 + IVN 全流程（唯一入口）
bash scripts/train_and_run_phase0.sh

# 排队多个实验
bash scripts/queue_experiments.sh
```

## 常用命令

```bash
# 状态检查
bash scripts/monitor.sh

# 环境变量（必须）
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

# Python
/home/jiayu/AFP/venv/bin/python3
```

## 项目结构

```
AFP/
├── scripts/
│   ├── train_agent.py              # 训练 full-FT
│   ├── train_and_run_phase0.sh     # 全流程 pipeline
│   ├── queue_experiments.sh        # 批量 IVN 实验
│   ├── run_ivn_phase0.py           # IVN 实验
│   └── run_fivn_phase0.py          # F-IVN 函数空间
├── src/AFP/protocol/               # 协议核心
├── experiments/
│   ├── trained_models/             # 训练权重（_eN 后缀）
│   └── phase0_ivn/results/         # IVN 结果 JSON
├── docs/internal/                  # 设计文档
└── data/versaprm/                  # 预处理数据
```

## 工作约定

- 提交信息用英文，`feat(scope): description`
- 训练用 `nohup ... &`，日志写 `experiments/`
- Python 加 `-u`（无缓冲）
- 所有 HF API 调用加 `local_files_only=True`
- 修改参数后删除旧缓存（缓存 key 包含 MAX_LEN）
