# AFP Phase 0 — 新会话重启

> 粘贴给新 Claude 会话。读完即可操作。

---

```markdown
# AFP Phase 0

## 研究问题

Domain fine-tuning 产生的权重差异有多大？LMC 是否成立？**假设：低 LR 是 Pythia 顽固性的根因。**

## 设置

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3
```

## 决策树

```
bash scripts/monitor.sh
    ├─ 有进程 → 别动
    ├─ 没进程，有 trained_models/code_e1/ → 验证模型 + LMC 扫描：
    │     python -c "验证 ΔW > 0.1%"
    │     python -u scripts/lmc_barrier_scan.py
    └─ 没模型 → 训练（用 --lr 5e-4）：
          nohup $VENV -u scripts/train_agent.py --domain code --lr 5e-4 >> ...log 2>&1 &
```

## 当前结论

| 模型对 | LMC barrier | 关键发现 |
|--------|------------|---------|
| code_e1 + med_e1 | 0.071 | U形：code 改善 medical 8.3% |
| code_e1 + med_e3 | 0.076 | U形减弱：6.8% |
| code_e1 + med_e5 | 0.076 | U形消失：自域最优 |
| **假说** | | **lr=5e-4 会产生 >5% ΔW，可能打破 LMC** |

## 当前训练

lr=5e-4 验证进行中。如果 ΔW > 5%，立即训 medical lr=5e-4 → LMC scan。

## 环境

| 项 | 值 |
|----|-----|
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| HF mirror | `https://hf-mirror.com`（无 VPN，必须） |
| 设备 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |
| 训练 | ~42min/epoch, lr=5e-4 (测试中) |
| LMC scan | ~15min |

## ⚠️ 训练后必做验证

```bash
python -c "
import torch
c = torch.load('experiments/trained_models/code_e1/W_code_final.pt', map_location='cpu', weights_only=True)
b = torch.load('experiments/trained_models/code/W_code_final.pt', map_location='cpu', weights_only=True)
d = sum((c[k]-b[k]).float().norm().item()**2 for k in c if k in b)**0.5
n = sum(b[k].float().norm().item()**2 for k in b if k in c)**0.5
print(f'ΔW={(d/(n+1e-8))*100:.2f}%' if d/n > 0.001 else 'BUG: model is base!')
"
```

## 已知 Bug

| # | 描述 |
|---|------|
| 22 | code_e1 曾是 base 模型（所有 E2-E7 无效） |
| 18 | code epoch 2+ 退化 |
| 21 | 训练后模型可能是 base（做验证！） |

## 已废弃（不要碰）

- `scripts/run_ivn_phase0.py` — IVN 实验
- `scripts/queue_experiments.sh` — 批量 IVN
- `docs/internal/VISION.md` — AFP/gate/IVN 协议（历史）
- `docs/internal/IMPORTANCE_ANALYSIS.md` — importance 分析（历史）
```
