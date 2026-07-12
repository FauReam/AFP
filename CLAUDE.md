# AFP — Claude Code 项目上下文

## 研究问题

**Domain fine-tuning 产生的权重差异有多大？不同差异度下 LMC 是否成立？**

Pythia-1.4B 分别在 code 和 medical 上 full-FT。测量权重偏移量级 + 线性插值 loss barrier。

## 核心发现

| 条件 | ΔW | Code barrier | Med barrier |
|------|-----|-------------|------------|
| Standard | 1.4% | 0.053 ± 0.011 | 0.051 ± 0.013 |
| High divergence | 8.0% | 0.118 ± 0.031 | 0.228 ± 0.102 |
| 域内基线 | — | 0.048 (code) | 0.147 (medical) |
| 噪声地板 | — | ~0.000 (identical) | 0.222 (random init) |

1. **标准 FT 下 LMC 成立**（barrier ≈ 0.05）。
2. **高差异下 barrier 升高 2-5× 但仍适中**。域 FT 不足以轻易打破连通性。
3. **Code 跨域 ≈ 域内** → 域差异不产生额外 barrier；**Medical 域内 > 跨域** → 训练不稳定性 > 域差异。
4. **Per-block 模式高度相关** (r=0.995) — 两个域改变相同的 block，只差幅度。
5. **High-divergence medical (0.23) ≈ random init 上界 (0.22)** → 8% ΔW 已接近连通极限。

## ⚠️ 不要做的事

- **不要把 LR 写进结论或标题** — LR 只是制造差异的旋钮，不是贡献。
- **不要提 AFP/IVN/gate** — 该方向已废弃。`docs/internal/VISION.md` 为历史存档。
- **不要追 "LMC 破裂" 叙事** — 实验证明它没破。
- **不要用 symlink 管理模型目录** — 每个模型独立保存到 `{domain}_lr{lr}_s{seed}/`。
- **训练后必须验证 ΔW > 0.1%** — Bug 22 曾导致 base 模型被当作训练结果。

## 🛡️ 批量实验强制规则

**任何 >1h 的批量实验启动前必须逐条确认。违反即 Bug。**

| # | 规则 | 来源 |
|---|------|------|
| R1 | 必须 `nohup ... &` 启动，只返回 PID | ENGINEERING §5.3 |
| R2 | stdout/stderr 重定向到 `experiments/` 下时间戳日志 | ENGINEERING §5.3 |
| R3 | 脚本零 stdin 读取（无 `input()`, `sys.stdin.read()`） | ENGINEERING §5.1 |
| R4 | `HF_DATASETS_OFFLINE=1` — 全程零网络 | ENGINEERING §5.1 |
| R5 | **禁止 `set -e`** — 单个实验崩溃不得杀死整个 batch | Bug 23 教训 |
| R6 | 单步失败处理：`|| { log "ERROR"; return 1; }` — 不得 `exit 1` | Bug 23 教训 |
| R7 | 每步输出文件存在即 skip（`[ -f "$out" ] && continue`） | 幂等重跑 |
| R8 | 训练完成后验证 ΔW > 0.1%，失败则返回 1（不 exit） | Bug 21/22/23 |
| R9 | 启动后 `sleep 3 && ps -p $PID` 验证进程存活 | ENGINEERING §5.3 |
| R10 | 写结构化的失败日志到 `crashes/` 目录 | ENGINEERING §5.6 |

> **详见 `docs/internal/ENGINEERING.md` §5（训练代码规范）**

## 环境

| 项 | 值 |
|----|-----|
| 模型 | EleutherAI/pythia-1.4b (1.31B params) |
| 训练 | full-FT, bf16, L=256, batch=128, 1ep |
| 硬件 | DGX Spark GB10, 121GB, ARM64 CUDA 13.0 |
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| 速度 | ~42min/训练, ~15min/LMC 扫描 |

## 入口

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

# 训练单个模型
python -u scripts/train_agent.py --domain code --lr 1e-4 --model-id EleutherAI/pythia-1.4b

# LMC 扫描（需 code_e1 和 medical_e1 symlink）
python -u scripts/lmc_barrier_scan.py

# 查看状态
bash scripts/monitor.sh
```

## 关键文档

| 文件 | 内容 |
|------|------|
| `docs/internal/PAPER.md` | 论文草稿 v4 |
| `docs/internal/EXPERIMENT_PLAN.md` | 实验数据 + 待做 |
| `docs/internal/EXPERT_PANEL_FINDINGS.md` | **6-Expert 评审报告** — 11 条弱点 + 7 大优势 + venue strategy |
| `docs/internal/ICLR_SPRINT_PLAN.md` | **ICLR 2028 冲刺计划** — 3-phase 实验清单 + timeline + GPU budget |
| `docs/internal/ENGINEERING.md` | 22 条已知 Bug |
| `experiments/RESTART_PROMPT.md` | 新会话重启指南 |

## 项目结构

```
AFP/
├── scripts/
│   ├── train_agent.py           # 训练 full-FT
│   ├── lmc_barrier_scan.py      # LMC 扫描
│   ├── lmc_6scans.sh            # 批量跨域 LMC
│   ├── lmc_within_domain.sh     # 域内 LMC 基线
│   └── noise_floor.sh           # 噪声地板校准
├── experiments/
│   ├── trained_models/          # 独立目录: {domain}_lr{lr}_s{seed}/
│   └── phase0_ivn/results/      # LMC 结果 JSON
├── docs/internal/               # 设计文档
└── docs/reports/                # 图表
```
