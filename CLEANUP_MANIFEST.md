# AFP 项目清理清单 — 2026-07-19

> 论文已迭代至 v19，以下为已过时/废弃/冗余文件，共 46 项已删除。
> 同日修复论文理论部分 5 处过时问题，见底部 §理论修复。

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

# 理论修复 — 2026-07-19

> 论文 v19 理论部分 (§4.10–4.13) 发现 5 处过时/错误，已全部修复。

## 🔴 #1: "目前正在运行" — 事实性错误
**位置**: Discussion §6  
**原文**: "Three theory-driven experiments are **currently running** to validate these predictions"  
**问题**: 实验 A/B/C 早已完成，结果已在同节报告  
**修复**: 改为过去时，并追加实验结论摘要

## 🟡 #2: "可测试预测" — 标题过时
**位置**: §4.13 (原标题)  
**原文**: `\subsection{Testable Predictions}`  
**问题**: 内容实际已是实验结果，不是预测  
**修复**: 改为 `Theory Verification Experiments`

## 🟡 #3: Code inverted-U 理论缺 OPT 架构限定
**位置**: §4.12 SGD Dynamics  
**原文**: "In the code domain ... barrier declines (inverted-U)" 作为普遍规律呈现  
**问题**: OPT-1.3B code 轨迹是 monotonic (0.14→0.20)，不是 inverted-U  
**修复**: (a) 将 "code domain" 改为 "code domain on Pythia"; (b) 追加段落说明 inverted-U 是架构依赖的，OPT 表现为 monotonic，domain stability ordering 一致但 trajectory shape 不通用

## 🟡 #4: Per-block r=0.995 当普遍值
**位置**: §4.10 Barrier Decomposition  
**原文**: "near-identical divergence patterns (r = 0.995) follow from..."  
**问题**: OPT 只有 r=0.91，0.995 是 Pythia 特定值  
**修复**: 改为 "on Pythia (r = 0.995 across domains)"，追加 OPT r=0.91 提及并解释架构差异

## 🟡 #5: `\ref{sec:theory}` 引用断裂
**位置**: Discussion §6 + §4.10  
**原文**: `Section~\ref{sec:theory}` 引用的 label 不存在  
**问题**: 理论子节缺少 `\label{sec:theory}`，引用指向空  
**修复**: 在 `\subsection{Barrier Decomposition}` 添加 `\label{sec:theory}`，编译后引用正常解析
