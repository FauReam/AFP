# AFP LMC — 实验恢复指南

> 如果你断联后重新连接，读这个文件。
> 最后更新: 2026-07-22 (paper v19)

## 快速检查

```bash
cd /home/jiayu/AFP
bash scripts/monitor.sh
```

## 环境

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3
```

## 当前实验架构

项目已从 AFP/IVN 协议完全转向 LMC barrier 测量。没有 master/watchdog/queue 架构。

### 检查是否有训练/扫描在跑

```bash
ps aux | grep -E 'train_agent|lmc_barrier' | grep -v grep
```

### 查看最近的实验日志

```bash
ls -lt experiments/phase0_ivn/logs/ 2>/dev/null | head -5
ls -lt experiments/*.log 2>/dev/null | head -5
```

## 如果需要重启 Claude Code

```bash
tmux new -s afp
cd /home/jiayu/AFP
claude
```

断联后重新连接:
```bash
ssh <host>
tmux attach -t afp
```

## 手动启动实验

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3

# 训练单个模型（~42 min）
nohup $VENV -u scripts/train_agent.py --domain code --lr 1e-4 \
    --model-id EleutherAI/pythia-1.4b \
    >> experiments/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# LMC 扫描
nohup $VENV -u scripts/lmc_barrier_scan.py \
    >> experiments/lmc_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 批量实验（详见 ICLR_SPRINT_PLAN.md）
nohup bash scripts/phase1_batch.sh \
    >> experiments/phase1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 结果文件

- LMC barriers: `experiments/phase0_ivn/results/lmc_*.json` (122 files)
- 训练模型: `experiments/trained_models/{domain}_lr{lr}_s{seed}/`
- 合并基准: `experiments/phase0_ivn/results/merge_benchmark.json`
- Bootstrap CI: `experiments/phase0_ivn/results/bootstrap_ci.json`

## 已知问题

- 详见 `docs/internal/ENGINEERING.md`（23 条 Bug + 共享内存警告）
- **绝对不能清除 CPU-GPU 共享内存** — 会导致 Tailscale 永久断连
- Bug 22/23 已修复，所有当前数据已验证
