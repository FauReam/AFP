# AFP Phase 0 — 新会话重启

> 粘贴给新 Claude Code 会话。读完即可操作。

---

```markdown
# AFP Phase 0

环境、入口、决策树。读完后直接执行。

---

## 设置

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3
```

## ⚠️ 跑实验前必须读 docs/internal/EXPERIMENT_PLAN.md

每个实验必须评估：假设 → 论文价值(高/中/低) → 方向判定(ON TRACK/DRIFTING)。偏离方向 = 浪费 GPU。

## 决策树

```
bash scripts/monitor.sh
        │
        ├─ 有 Python 进程在跑 (train_agent / run_ivn / queue)
        │   → 不要动。等待即可。
        │
        ├─ 没进程，但 experiments/trained_models/code_e1/ 存在
        │   → 训练已完成。直接跑实验队列：
        │     nohup bash scripts/queue_experiments.sh > experiments/queue.log 2>&1 &
        │
        └─ 没进程，也没 trained_models/code_e1/
            → 从头训练：
              nohup bash scripts/train_and_run_phase0.sh > /dev/null 2>&1 &
              echo $! > experiments/pipeline_pid.txt
```

## 三个命令

```bash
# 1. 看状态
bash scripts/monitor.sh

# 2. 全流程训练 + IVN（~50 min: 40 训练 + 10 IVN）
nohup bash scripts/train_and_run_phase0.sh > /dev/null 2>&1 &

# 3. 批量 IVN 实验（~13 min/实验，使用已有 checkpoints）
nohup bash scripts/queue_experiments.sh > experiments/queue.log 2>&1 &
```

## 环境

| 项 | 值 |
|----|-----|
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| HF mirror | `https://hf-mirror.com`（无 VPN，必须） |
| 设备 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |

## 项目路径

```
experiments/trained_models/    ← 训练权重（code_e1/, medical_e1-e5/）
experiments/phase0_ivn/        ← IVN 结果（results/*.json）
experiments/phase0_training/   ← 训练日志
docs/internal/ENGINEERING.md   ← 完整 Bug 清单（Bug 1-21）
```

## 关键 Bug（可能影响新会话）

1. **code 训练退化** — 只信任 code_e1。e2-e5 是坏的。新训练加 `--epochs 1`。

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | EleutherAI/pythia-1.4b |
| 训练 | full-FT, bf16, batch=128, L=256 |
| 领域 | code ⇄ medical (VersaPRM) |
| LR | Cosine 1e-4 → 3e-6, 1 epoch |
| 速度 | ~5.5s/step, 433 steps, ~40 min/domain |
```
