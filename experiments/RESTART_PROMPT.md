# AFP Phase 0 — 新会话重启

> 粘贴给新 Claude 会话。读完即可操作。

---

```markdown
# AFP Phase 0

## 你是谁

`claude-code`，运行在 DGX Spark GB10 (ARM64, 121GB, CUDA 13.0)。
项目路径: `/home/jiayu/AFP`

## 要做什么

**测量 domain fine-tune 后模型的权重差异和 LMC barrier。**

1. 在 code 和 medical 上分别 full-FT Pythia-1.4B。
2. 测量 weight divergence (从 base 偏移多少)。
3. 在模型间做 11 点线性插值，测 LMC barrier。
4. 域内基线：同域不同 seed 的 barrier。
5. 噪声地板：identical copy 和 random init 的 barrier。

## 不要做什么

- **不要把 LR 写进结论或标题** — LR 只是制造差异的手段，不是贡献。
- **不要提 AFP/IVN/gate** — 已废弃。`docs/internal/VISION.md` 是历史存档。
- **不要追 "LMC 破裂"** — 实验证明它没破。barrier 升高但仍然适中。
- **不要用 symlink 管理模型** — 模型保存到独立目录 `{domain}_lr{lr}_s{seed}/`。
- **训练完必须验证 ΔW > 0.1%** — 曾多次发生 base 模型被当作训练结果。

## 🛡️ 批量实验强制规则（启动任何 >1h 任务前逐条确认）

| # | 规则 |
|---|------|
| R1 | 必须 `nohup ... &`，只返回 PID |
| R2 | stdout/stderr → `experiments/` 下时间戳日志 |
| R3 | 零 stdin 读取 |
| R4 | `HF_DATASETS_OFFLINE=1` 全程零网络 |
| R5 | **禁止 `set -e`** — 单步失败不杀 batch |
| R6 | 失败用 `return 1`，不得 `exit 1` |
| R7 | 每步 `[ -f "$out" ] && continue` 幂等 skip |
| R8 | 训练后验证 ΔW > 0.1% |
| R9 | `sleep 3 && ps -p $PID` 验证存活 |
| R10 | 失败日志写入 `crashes/` |

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
    ├─ 没进程 → 看要做什么：
    │   ├─ 训练: $VENV -u scripts/train_agent.py --domain <code|medical> --lr <1e-4|5e-4>
    │   ├─ LMC扫描: bash scripts/phase1_batch.sh  (标准批量)
    │   ├─ 合并基准: $VENV scripts/merge_benchmark.py
    │   ├─ 理论验证: bash scripts/theory_experiments.sh
    │   └─ OPT实验: bash scripts/opt/opt_full_pipeline.sh
    └─ 做完了 → 更新 paper.tex / PAPER.md → git push
```

## 核心数据

| 条件 | ΔW | code_bar | med_bar | 含义 |
|------|-----|----------|---------|------|
| Standard | 1.4% | 0.053±0.011 | 0.051±0.013 | LMC 成立 |
| High div | 8.0% | 0.118±0.031 | 0.228±0.102 | barrier 升高但适中 |
| 域内(code) | — | 0.048±0.000 | — | Code: 域差异≈0 额外 barrier |
| 域内(med) | — | 0.147±0.027 | — | Medical: 训练不稳定性 > 域差异 |
| Identical | — | ~0.000 | — | 噪声地板 |
| Random | — | 0.033 | 0.150 | 上界（3-seed 校准） |
| **OPT-2ep** | — | **0.251** | **0.896** | 跨架构: 绝对值 4-8×, 比值一致 (3.6× vs 3.7×) |

> 论文 §4.9 + DATA_INVENTORY.md 有完整三模型对比表。

## 模型目录

```
experiments/trained_models/
  code_lr1e-4_s0/   code_lr1e-4_s1/   code_lr1e-4_s2/
  code_lr5e-4_s0/   code_lr5e-4_s1/   code_lr5e-4_s2/
  medical_lr1e-4_s0/ ... medical_lr5e-4_s2/
```

## 训练后必做验证

```bash
python -c "
import torch
c = torch.load('experiments/trained_models/code/W_code_final.pt', map_location='cpu', weights_only=True)
# Compare to a known different model (e.g., medical)
m = torch.load('experiments/trained_models/medical_lr1e-4_s0/W_medical_final.pt', map_location='cpu', weights_only=True)
d = sum((c[k]-m[k]).float().norm().item()**2 for k in c if k in m)**0.5
n = sum(m[k].float().norm().item()**2 for k in m if k in c)**0.5
print(f'ΔW={(d/(n+1e-8))*100:.2f}%' if d/n > 0.001 else 'BUG: model is base!')
"
```

## 已知 Bug

详见 `docs/internal/ENGINEERING.md`（Bug 1-23 + 共享内存警告）。

| # | 最关键的 |
|---|---------|
| 22 | code_e1 曾是 base 模型，所有早期 LMC 无效 |
| 23 | 多 seed 训练静默覆盖（输出路径无 seed/lr 区分） |
| 21 | 训练后保存的模型可能是 base（做验证！） |
| 18 | code epoch 2+ 退化 |
| — | ⚠️ 绝对不能清除 CPU-GPU 共享内存（Tailscale 永久断连） |
```
