# AFP 项目清理清单 — 2026-07-19

> 论文已迭代至 v19，以下为已过时/废弃/冗余文件，共 46 项已删除。
> 同日修复论文理论部分 5 处过时问题 + 12 处完整度问题。
> 2026-07-20 添加 Fig 4 OPT 跨架构图 + DGX 主机待办清单。

---

# ⚠️ DGX 实验主机 — 待办清单

> **读者**：DGX 主机操作者。
> **仓库**：`https://github.com/FauReam/AFP`，分支 `main`。
> **论文版本**：v19（文件：`docs/internal/paper.tex`，已编译 PDF：`paper.pdf`）。
> **背景**：本地 Mac 已做大量清理和论文修改，但无法直接访问 DGX 上的实验输出。以下任务需在 DGX 上完成。

---

## 步骤 0：先同步仓库（必须先做）

```bash
cd /home/jiayu/AFP
git pull origin main
# 如果本地有未提交的修改，先 stash：
#   git stash && git pull origin main && git stash pop
```

---

## 任务 A：恢复 OPT 原始实验数据（最高优先级）

> **为什么需要**：论文 §4.9 报告了 OPT-1.3B 跨架构复制实验。三模型对比表（Pythia-2ep vs OPT-2ep vs GPT-Neo-2ep）的汇总统计已在论文中，但以下三类原始逐点数据未提交到 GitHub。当前论文的 Fig 4 Panel C 只能用文字摘要代替折线图/散点图。拿到原始数据后可以补全定量图表。

### A1. 先定位 OPT 实验的原始输出

OPT 实验由 `scripts/opt/opt_full_pipeline.sh` 统一调度。先看这个脚本把结果写到了哪里：

```bash
cd /home/jiayu/AFP
grep -nE 'OUT|output|results|save|RESULT|\.json|\.pt|\.npy|\.pkl' scripts/opt/opt_full_pipeline.sh | head -30
```

同时也检查各个 phase 脚本的输出位置：

```bash
for f in scripts/opt/phase*.sh scripts/opt/theory_experiments.sh; do
    echo "=== $f ==="
    grep -nE 'OUT|OUTPUT|output_dir|results|save' "$f" | head -10
done
```

用这些信息确定 OPT 实验的输出根目录（可能是 `experiments/opt_results/`、`experiments/opt/`、`experiments/phase4_opt/` 等）。

### A2. 恢复训练轨迹数据

**论文中的描述**：21 checkpoints × 840 training steps。Code domain barrier: 0.14→0.20（单调上升）。Medical domain barrier: 0.09→0.14（相对平坦）。

⚠️ 注意：OPT 轨迹形状和 Pythia 不同。Pythia code 是倒 U 形，OPT code 是单调上升。不要混用 Pythia 的数值。

**查找**（从 A1 确定的输出目录开始）：
```bash
# 如果 A1 找到了输出目录，直接进去看
ls -la <输出目录>/

# 否则广撒网
find experiments/ -maxdepth 3 -type f \( -name '*trajectory*' -o -name '*traj*' -o -name '*checkpoint*' \) | grep -v '.git'
find experiments/ -maxdepth 3 -type d | grep -iE 'traj|phase4'
```

**可能的数据形式**：
- 一个 `.json` 文件包含所有 checkpoint 的 (step, loss, barrier) 数组 → 最好
- 多个 `.pt` / `.npy` 文件，每文件一个 checkpoint → 需要合并
- 脚本输出日志中有打印的数值 → 需要手动提取（最差情况）

**无论如何**，把找到的原始文件（不修改）复制到统一位置：
```bash
mkdir -p experiments/opt_recovery/trajectory
cp <原始文件路径> experiments/opt_recovery/trajectory/
```

**验证**：打开文件，确认 barrier 的端点与论文一致：
- code trajectory 第一个点 barrier ≈ 0.14，最后一个点 ≈ 0.20
- medical trajectory 第一个点 barrier ≈ 0.09，最后一个点 ≈ 0.14
- 应有 ~21 个数据点（840 steps ÷ 40 steps/checkpoint）

### A3. 恢复逐层散度数据

**论文中的描述**：24 层 transformer × 2 个域（code + medical）。Pythia 的 per-block 模式跨域高度相关（r=0.995），OPT 的相关性稍低（r=0.91）。

**查找**：
```bash
find experiments/ -maxdepth 3 -type f \( -name '*per_block*' -o -name '*layer*' -o -name '*block*div*' \) | grep -v '.git'
find experiments/ -maxdepth 3 -type d | grep -iE 'block|layer|div'
```

**可能的数据形式**：
- 一个 (24,) 或 (2, 24) 的 numpy 数组
- 一个 JSON，key 为 `code_divergence` / `medical_divergence`

**导出**：
```bash
mkdir -p experiments/opt_recovery/per_block
cp <原始文件路径> experiments/opt_recovery/per_block/
```

**验证**：确认 24 个值大致递减（浅层散度 > 深层散度），且 code 和 medical 的两条曲线形状相似（r ≈ 0.91）。

### A4. 恢复 Gaussian 扰动数据

**论文中的描述**：在 ΔW ∈ {0.5%, 1%, 2%, 4%, 8%} 五个级别上做 Gaussian 噪声扰动。障碍极小：从 0.0003（ΔW=0.5%）到 0.055（ΔW=8%）。

**查找**：
```bash
find experiments/ -maxdepth 3 -type f \( -name '*gaussian*' -o -name '*noise*' -o -name '*calib*' -o -name '*perturb*' \) | grep -v '.git'
```

**导出**：
```bash
mkdir -p experiments/opt_recovery/gaussian
cp <原始文件路径> experiments/opt_recovery/gaussian/
```

**验证**：确认 barrier 值在 0.0003 到 0.055 之间，且基本上随 ΔW 增大而增大。

### A5. 如果找不到任何数据

检查 DGX 上是否有其他工作目录：

```bash
find /home/jiayu -maxdepth 3 -name '*opt*' -type d 2>/dev/null
find /tmp -name '*opt*' -o -name '*AFP*' 2>/dev/null
ls -la ~/AFP/experiments/
```

如果确实丢失了，记录结论：**"OPT 原始数据已丢失，论文中的汇总数字是唯一留存记录。如有需要可重新运行实验。"**

### A6. 提交到 GitHub

```bash
cd /home/jiayu/AFP
git add experiments/opt_recovery/
git commit -m "data: recover OPT raw experiment outputs from DGX

- Trajectory: 21-ckpt code (0.14→0.20) + medical (0.09→0.14)
- Per-block divergence: 24-layer × 2-domain (r=0.91)
- Gaussian calibration: 5-level ΔW (0.0003–0.055)
- Raw files preserved as-is; format may vary"
git push origin main
```

---

## 任务 B：确认 GPT-Neo 实验状态

**背景**：论文三模型对比表中 GPT-Neo 行曾为全 "—" 占位符，已被删除。Limitations 节写 "GPT-Neo replication is pending"。

```bash
cd /home/jiayu/AFP

# 检查是否已有训练好的 GPT-Neo 模型
ls experiments/trained_models/ | grep -i neo
ls experiments/ | grep -i neo
find experiments/ -name '*neo*' -o -name '*gptneo*' | grep -v '.git'

# 检查批量脚本是否定义了 GPT-Neo 输出路径
grep -nE 'neo|GPT.Neo|EleutherAI' scripts/gptneo_pipeline.sh scripts/gptneo_v5.sh
```

**如果已有数据**：参照任务 A 的方式导出到 `experiments/gptneo_results/`，提交。

**如果没有**：OPT 已经足以支撑跨架构复制的结论（两个不同架构 + 一致的 3.6-3.7× med/code 比值）。可以标记 GPT-Neo 为 "未执行"，无需阻塞。

---

## 任务 C：同步本地清理

> 本地 Mac 已删除 46 个过时文件（详见下方类别 1-9）。拉取后 DGX 本地的对应文件也会被删除。

```bash
cd /home/jiayu/AFP
git pull origin main

# 确认清理后状态
git status

# 预览 DGX 本地未追踪的残留文件（不会自动删除）
git clean -nd
```

---

## 检查清单

| # | 任务 | 状态 |
|---|------|------|
| 0 | `git pull` 同步仓库 | ☐ |
| A1 | 定位 OPT 实验输出目录 | ☐ |
| A2 | 恢复训练轨迹原始数据 | ☐ |
| A3 | 恢复逐层散度原始数据 | ☐ |
| A4 | 恢复 Gaussian 扰动原始数据 | ☐ |
| A5 | 如找不到数据，记录结论 | ☐ |
| A6 | 提交 `experiments/opt_recovery/` | ☐ |
| B | 确认 GPT-Neo 状态 | ☐ |
| C | `git pull` 同步本地清理 | ☐ |

---

---

---

## 类别 1：AFP 协议废弃代码（10项）
> CLAUDE.md 明确声明 "AFP/IVN/gate 已废弃"。VISION.md 标注 "历史存档"。

| # | 文件 | 原因 |
|---|------|------|
| 1 | `src/afp/` (整个目录) | AFP 协议核心代码 — agent.py, integrator.py, trust.py |
| 2 | `scripts/run_ivn_phase0.py` | IVN (weight-space) 实验 |
| 3 | `scripts/run_fivn_phase0.py` | F-IVN (function-space) 实验 |
| 4 | `scripts/check_importance_quality.py` | IVN importance 质量检查 |
| 5 | `scripts/generate_phase0_report.py` | IVN Phase 0 报告生成 |
| 6 | `scripts/run_phase0_master.sh` | IVN/F-IVN 主控启动器 |
| 7 | `scripts/train_and_run_phase0.sh` | 训练+IVN 全自动管道 |
| 8 | `scripts/pipeline.sh` | IVN → F-IVN → Report 管道 |
| 9 | `scripts/queue_experiments.sh` | IVN 实验队列 |
| 10 | `experiments/phase0_ivn/ivn_results.json` | IVN 实验结果 (已不用) |

## 类别 2：根目录旧版理论 tex（2项）
> 内容已完全融入 `docs/internal/paper.tex` §5 理论框架。

| # | 文件 | 原因 |
|---|------|------|
| 11 | `lmc_theory_framework.tex` | 早期 LMC 理论框架草稿 |
| 12 | `theoretical_model_barrier_dynamics.tex` | 早期 barrier 动力学模型 |

## 类别 3：已过时内部文档（5项）

| # | 文件 | 原因 |
|---|------|------|
| 13 | `docs/internal/VISION.md` | 明确标注 "⚠️ AFP/IVN/gate 协议暂不推进...历史存档" |
| 14 | `docs/internal/EXPERIMENT_PLAN.md` | 日期 "2026-07-12 \| 论文 v2"，严重过时 |
| 15 | `docs/internal/DIRECTION.md` | 标注 "AFP 协议部分为历史存档"，文献地图已融入论文 |
| 16 | `docs/internal/OPT_SECOND_MODEL.md` | OPT-1.3B 复制计划，已在 v15-v19 完成 |
| 17 | `docs/internal/THEORY.md` | 完整理论内容在 paper.tex §5，纯重复 |

## 类别 4：LaTeX 编译产物（4项）

| # | 文件 | 原因 |
|---|------|------|
| 18 | `docs/internal/paper.aux` | pdflatex 辅助文件 |
| 19 | `docs/internal/paper.log` | pdflatex 日志 |
| 20 | `docs/internal/paper.out` | pdflatex 输出 |
| 21 | `docs/internal/paper_v14_backup.pdf` | 旧版备份，已有 v19 PDF |

## 类别 5：冗余PNG图片（3项）
> PDF 矢量版本已用于论文，PNG 不再需要。

| # | 文件 | 大小 |
|---|------|------|
| 22 | `docs/reports/fig1_lmc_overview.png` | 332K |
| 23 | `docs/reports/fig2_trajectory_analysis.png` | 204K |
| 24 | `docs/reports/fig3_seedpair_stability.png` | 128K |

## 类别 6：一次性 Fixup 脚本（3项）
> 针对已修复 bug 的一次性补救脚本。

| # | 文件 | Bug |
|---|------|-----|
| 25 | `scripts/fixup_A_rescan.sh` | 修复实验 A LMC 扫描路径 |
| 26 | `scripts/fixup_BC.sh` | 修复 B+C Hessian zero_grad bug |
| 27 | `scripts/fixup_final.sh` | 最终批量修复 |

## 类别 7：旧版 Batch/Shell 脚本（7项）

| # | 文件 | 原因 |
|---|------|------|
| 28 | `scripts/pythia160m_replicate.sh` | 160M 实验已放弃，改用 1.4B |
| 29 | `scripts/offline_12h.sh` | 旧版 12 小时离线训练队列 |
| 30 | `scripts/lmc_6scans.sh` | 硬编码 `/home/jiayu/AFP` 路径，不可移植 |
| 31 | `scripts/lmc_within_domain.sh` | 硬编码路径，功能已被更完整脚本覆盖 |
| 32 | `scripts/noise_floor.sh` | 硬编码路径，一次性校准已完成 |
| 33 | `scripts/lmc_scans_only.sh` | 功能已合并到 `final_batch.sh` |
| 34 | `scripts/download_models.sh` | 一次性模型下载，已完成 |

## 类别 8：Python 缓存 + 系统垃圾（13项）

| # | 文件 | 原因 |
|---|------|------|
| 35 | `scripts/__pycache__/` | Python 字节码缓存 |
| 36-43 | `.DS_Store` (8个) | macOS 系统垃圾文件 |
| 44 | `papers/missfont.log` | LaTeX 字体缺失日志 |

## 类别 9：空目录（2项）

| # | 文件 | 原因 |
|---|------|------|
| 45 | `data/versaprm/` | 空目录 |
| 46 | `experiments/phase0_diagnostic/` | 空目录 |

---

## 保留的核心文件

| 类别 | 文件 | 用途 |
|------|------|------|
| 论文 | `paper.tex`, `paper.pdf`, `PAPER.md`, `APPENDIX.md` | v19 论文 |
| 图表 | `fig1/2/3_lmc_*.pdf`, `figure_data.json`, `FIGURE_SPEC.md` | matplotlib 生成 |
| 内部文档 | `ENGINEERING.md`, `EXPERT_PANEL_FINDINGS.md`, `DATA_INVENTORY.md`, `ICLR_SPRINT_PLAN.md`, `reference_audit.md` | 工程/评审/数据/计划 |
| 活跃脚本 | `train_agent.py`, `lmc_barrier_scan.py`, `lmc_3pt_scan.py`, `generate_figures.py`, `merge_benchmark.py`, `fisher_curvature.py`, `hypothesis_tests.py`, `accuracy_eval.py`, `layer_stiffness.py`, `git_rebasin_check.py`, `lmc_barrier.py`, `generate_reports.py`, `prepare_versaprm.py`, `monitor.sh`, `watchdog.sh`, `pythia_2ep.sh`, `gptneo_*.sh`, `theory_experiments.sh`, `final_batch.sh`, `med_high_seeds.sh`, `phase*.sh` | 实验与分析 |
| 子目录 | `scripts/opt/`, `scripts/pythia/` | OPT/Pythia 专用脚本 |
| 实验数据 | `experiments/phase0_ivn/results/`, `experiments/lmc_trained/`, `experiments/expert_panel/` | LMC 结果/模型/评审 |
| 参考 | `papers/*.pdf` (9篇), `models/` (tokenizer), `CLAUDE.md`, `README.md`, `requirements.txt`, `.gitignore` | 基础配置 |

---

# 理论修复 — 2026-07-19

> 论文 v19 理论部分 (§5) 发现 5 处过时/错误，已全部修复。

## 🔴 #1: "目前正在运行" — 事实性错误
**位置**: Discussion §6  
**原文**: "Three theory-driven experiments are **currently running** to validate these predictions"  
**问题**: 实验 A/B/C 早已完成，结果已在同节报告  
**修复**: 改为过去时，并追加实验结论摘要

## 🟡 #2: "可测试预测" — 标题过时
**位置**: §5.4 (原标题)  
**原文**: `\subsection{Testable Predictions}`  
**修复**: 改为 `Theory Verification Experiments`

## 🟡 #3: Code inverted-U 理论缺 OPT 架构限定
**位置**: §5.2 SGD Dynamics  
**原文**: "In the code domain ... barrier declines (inverted-U)" 作为普遍规律  
**修复**: (a) "code domain" → "code domain on Pythia"; (b) 追加段落说明 trajectory shape 是架构依赖的

## 🟡 #4: Per-block r=0.995 当普遍值
**位置**: §5.1 Barrier Decomposition  
**原文**: "near-identical divergence patterns (r = 0.995)"  
**修复**: 改为 "on Pythia (r = 0.995)"，追加 OPT r=0.91

## 🟡 #5: `\ref{sec:theory}` 引用断裂
**位置**: Discussion §6 + 理论子节  
**原因**: 理论子节缺少 section header 和 label  
**修复**: 添加 `\section{Theoretical Framework}\label{sec:theory}`，使 Theory 成为独立 §5

---

# 完整度修复 — 2026-07-19

> 论文完整度检查发现 12 处问题，全部修复（除 #10 小问题跳过）。

## 🔴 严重问题

| # | 问题 | 修复 |
|---|------|------|
| **C1** | 无作者 `\author{}` 为空 | → `\author{FauReam}` |
| **C2** | GPT-Neo 行全是 "—" 占位符 | → 删除该行，表格精简为 2 模型 |
| **C3** | 三模型表 Cross (med) 列全空 | → 删除该列，表格从 5 列减为 4 列 |
| **C4** | Abstract code inverted-U 作普遍规律 | → 加 "On Pythia" + OPT 轨迹差异引用 |

## 🟡 中等问题

| # | 问题 | 修复 |
|---|------|------|
| **C5** | 无 Conclusion 章节 | → 新增 §6 Conclusion (Discussion 整合为结论) |
| **C6** | Discussion inverted-U 无 OPT 限定 | → L563、L579 两处加 OPT architecture-dependent 声明 |
| **C7** | Introduction 贡献未提 OPT | → 新增第 4 条贡献：跨架构复制 |
| **C8** | Training config 只有 1-epoch | → 追加 2-epoch cross-architecture 说明 |
| **C9** | §5 Theory 子节挂在 §4 Results 下 | → 添加 `\section{Theoretical Framework}` 独立成节 |

## 🟢 小问题

| # | 问题 | 修复 |
|---|------|------|
| **C10** | OPT 大量分析无图表 | ⏭️ 跳过（需生成新 matplotlib 图表，非文本修复范围） |
| **C11** | 无 Data Availability / Acknowledgments | → 新增两个 `\section*{}` 于 References 前 |
| **C12** | "GPT-Neo replication is pending" | 保留于 Limitations 中，表格 caption 提及 "in progress" |

## 编译结果

- **页数**: 18 → 19 页（+Conclusion +Data +Acknowledgments -GPT-Neo行）
- **错误**: 0
- **警告**: 0 (仅 float specifier/font 等非实质性警告)
- **引用**: 全部解析

---

# OPT 图表修复 — 2026-07-20

> 完整度检查 C10: "OPT 大量分析无图表" — 已修复。

## 数据来源

OPT 原始数据（轨迹 21 checkpoint、逐层散度、Gaussian 各 ΔW 值）未存储在 repo 中，只有论文正文中的汇总统计数据。基于现有数据生成了 Fig 4。

## Fig 4: Cross-Architecture Replication

| Panel | 内容 | 数据类型 |
|-------|------|----------|
| A | Pythia-2ep vs OPT-2ep 三指标对比柱状图 (code within, med within, cross) + error bars | 论文 Table §4.9 精确值 |
| B | Med/code 比值对比 (Pythia 3.7× vs OPT 3.6×) | 论文 Table §4.9 计算值 |
| C | OPT 扩展实验定性摘要 (Gaussian/trajectory/per-block) | 论文正文文本摘要 |

## 生成

- 脚本: `scripts/generate_fig4_opt.py`
- 输出: `docs/reports/fig4_opt_replication.pdf` (39KB vector) + `.png`
- 论文: 插入 §4.9，`\ref{fig:fig4}` 引用

## 编译

- 页数: 19 → 20 (+Fig 4 图表)

## ⚠️ 待补充：OPT 原始实验数据

> Fig 4 Panel C 目前仅为文本摘要，缺少定量图表（折线图/散点图）。以下数据未存储在 repo 中，需要从 OPT 实验输出中恢复或重新采集：

| 数据项 | 当前状态 | 需要的内容 | 用途 |
|--------|----------|------------|------|
| OPT 训练轨迹 | 仅有端点范围 (code 0.14→0.20, med 0.09→0.14) | 21 checkpoints × 840 steps 的 (step, ΔW, barrier) 三元组 | Panel C 折线图：Pythia inverted-U vs OPT monotonic 对比 |
| OPT 逐层散度 | 仅有 r=0.91 | 24 layers × 2 domains 的各层 ΔW 值 | Panel C 散点图：Pythia r=0.995 vs OPT r=0.91 逐层对比 |
| OPT Gaussian 扰动 | 仅有端点 (0.0003@0.5% → 0.055@8%) | ΔW ∈ {0.5%, 1%, 2%, 4%, 8%} 各级 barrier 值 | Panel C 折线图：Pythia vs OPT Gaussian barrier 对比 |

**恢复路径**：
1. 检查 `scripts/opt/` 下脚本的输出目录（`opt_full_pipeline.sh`, `theory_experiments.sh` 等）
2. 如原始 JSON 仍在 DGX 机器上，导出至 `experiments/opt_results/`
3. 补充后更新 `generate_fig4_opt.py` 的 Panel C，将文本替换为 3 个子图

**当前 Fig 4 的价值**：Panel A+B 的定量对比已完整（数据来自论文 Table §4.9），可独立支撑跨架构复制结论。

---

# ⚠️ Fig 4 数据可信度声明

> **这条给 DGX 操作者看。**

## Fig 4 的数据链路

```
DGX 原始 JSON 数据
    │
    │  （谁抄的？抄对了吗？）
    ▼
论文 .tex / PAPER.md 正文中的汇总数字
    │
    │  （Claude 从论文文本中提取，非一手数据）
    ▼
generate_fig4_opt.py 硬编码的数据 → Fig 4 PDF
```

## 我（Claude）能做的和不能做的

| 层面 | 能力 | 限制 |
|------|------|------|
| 从论文正文提取数字写入脚本 | ✅ 做到了 | 信源是二手文本，不是原始数据 |
| 数字内部的算术一致性检验 | ✅ 做了 | 0.231/0.063=3.67→3.7×，0.896/0.251=3.57→3.6×，自洽 |
| 验证论文数字 = DGX 原始数据 | ❌ 做不到 | 我没有 DGX 文件系统的访问权限 |
| 检测转录错误 | ❌ 做不到 | 如果论文把 0.251 写成了 0.251 但原始是 0.215，我无法发现 |

## 隐患场景

| 场景 | 后果 |
|------|------|
| OPT 实验跑了多个 run，论文用了 run A 但 JSON 里 run B 最好 | Fig 4 数字准确但代表的是次优 run |
| 论文抄数字时漏了一个标准差或抄错行 | Fig 4 柱状图高度和 error bar 都错 |
| 均值/std 是从 3 个 seed 算的，但其中一个 seed 模型损坏 | 论文和 Fig 4 同时继承了这个错误 |
| 实际实验中 code within 和 med within 用了不同的 ΔW 基准 | 比值 3.6× 可能不可比 |

## DGX 操作者验证步骤

拿到原始 JSON 后，**逐个数比对论文**：

```
论文 §4.9 Table:
  Pythia-2ep  code within = 0.063 ± 0.005
  Pythia-2ep  med  within = 0.231 ± 0.096
  Pythia-2ep  cross       = 0.079 ± 0.025
  OPT-2ep     code within = 0.251 ± 0.108
  OPT-2ep     med  within = 0.896 ± 0.042
  OPT-2ep     cross       = 0.485 ± 0.010

从原始 JSON 重新计算上述 6 个 (mean, std)
→ 如果全部匹配 → Fig 4 可信，不需要改
→ 如果有任何一条不匹配 → 更新 generate_fig4_opt.py 并重新生成 Fig 4
```

**如果发现不一致**：以原始 JSON 为准更新论文和图表。不要反过来"修正"原始数据去凑论文数字。
