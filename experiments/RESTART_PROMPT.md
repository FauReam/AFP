# AFP Phase 0 — 新会话重启

> 粘贴给新 Claude Code 会话。读完即可操作。

---

```markdown
# AFP Phase 0

## 研究问题

Domain fine-tuning 产生的模型权重差异有多大？LMC 线性连通性是否成立？

## 设置

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3
```

## 决策树

```
bash scripts/monitor.sh
    ├─ 有进程在跑 → 不要动
    ├─ 没进程，有 trained_models/code_e1/ → 跑 LMC 扫描：
    │     nohup $VENV -u scripts/lmc_barrier_scan.py > experiments/phase0_ivn/logs/lmc_scan_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    └─ 没进程，没模型 → 训练：
          nohup bash scripts/train_and_run_phase0.sh > /dev/null 2>&1 &
```

## 三个命令

```bash
bash scripts/monitor.sh                                          # 状态
nohup bash scripts/train_and_run_phase0.sh > /dev/null 2>&1 &   # 训练 (~40min)
nohup $VENV -u scripts/lmc_barrier_scan.py > ...log 2>&1 &      # LMC 扫描 (~20min)
```

## 环境

| 项 | 值 |
|----|-----|
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| HF mirror | `https://hf-mirror.com`（无 VPN 必须） |
| 设备 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |

## 关键状态检查

```bash
ls -d experiments/trained_models/*/                              # 训练权重
ls experiments/phase0_ivn/results/                                # LMC 结果
tail -20 experiments/phase0_ivn/logs/lmc_scan_*.log              # 最新 LMC 日志
```

## 已测数据

| 模型 | mean ΔW | 训练 |
|------|---------|------|
| code_e1 | 0.26% | 1 epoch |
| medical_e1 | 1.91% | 1 epoch |
| medical_e5 | 1.95% | 5 epoch |

Pythia-1.4B 极度顽固——1 epoch full-FT 只改变 <2% 权重。
LMC 扫描验证中：线性插值路径上是否存在 loss barrier？

## 已知 Bug

| # | 描述 |
|---|------|
| 18 | code epoch 2+ 退化，只用 e1 |
| 17,19-21 | 已修 |

详见 `docs/internal/ENGINEERING.md`。
```
