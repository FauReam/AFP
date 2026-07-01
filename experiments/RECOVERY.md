# AFP Phase 0 — 实验恢复指南

> 如果你断联后重新连接，读这个文件。

## 当前状态

生成时间: 2026-07-01 00:55 HKT

| 组件 | PID | 状态 |
|------|-----|------|
| 主控脚本 | `cat experiments/master_pid.txt` | nohup 保护 ✅ |
| IVN Python | `ps aux \| grep run_ivn` | nohup 保护 ✅ |
| F-IVN Python | 等待 IVN 完成 | — |
| Watchdog | `cat experiments/watchdog_pid.txt` | nohup 保护 ✅ |

## 快速检查

```bash
cd /home/jiayu/AFP
bash scripts/monitor.sh
```

## 实验日志

- IVN: `experiments/phase0_ivn/logs/run_*.log`
- F-IVN: `experiments/phase0_fivn/logs/run_*.log`
- Master: `experiments/master_*.log`
- Watchdog: `experiments/watchdog.log`
- 状态: `experiments/phase0_status.txt`

## 结果文件

- IVN: `experiments/phase0_ivn/ivn_results.json`
- F-IVN: `experiments/phase0_fivn/fivn_results.json`
- Crash: `experiments/phase0_*/crashes/crash_*.json`

## 如果需要重启 Claude Code

```bash
# 使用 tmux 避免断联
tmux new -s afp
cd /home/jiayu/AFP
claude
```

断联后重新连接:
```bash
ssh <host>
tmux attach -t afp
```

## 如果需要手动重启实验

```bash
cd /home/jiayu/AFP
# 设置 HF mirror
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

# 使用 FCL-PRM 的 venv (有 torch/transformers)
VENV=/home/jiayu/FCL-PRM-cdspi/venv/bin/python3

# IVN (weight-space)
nohup $VENV scripts/run_ivn_phase0.py \
    --importance mas --gate rational --tau 0.5 --max-rounds 30 \
    >> experiments/phase0_ivn/logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# F-IVN (function-space)
nohup $VENV scripts/run_fivn_phase0.py \
    --tau 0.5 --max-rounds 30 \
    >> experiments/phase0_fivn/logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 尾注

- FCL-PRM venv 路径: `/home/jiayu/FCL-PRM-cdspi/venv/bin/python3`
- 设备: DGX Spark GB10, 121GB, CUDA 13.0
- HF mirror: `https://hf-mirror.com`
