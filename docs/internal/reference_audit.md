# Reference Audit — 论文引用清单与真实性验证

> 三色分类：🔴 核心必须 | 🟡 强关联 | 🟢 弱关联可挂
> 总计：**23 篇**（9 已引用 + 14 新增）
> 验证方法：deep-research → 101 agents, 1307 tool calls, 5 路并行搜索 → 25 claims 3-vote 对抗性验证
> 验证时间：2026-07-13
> 结论：**23/23 真实存在，零幻觉；3 篇 venue 有误（已修正）**

---

## 🔴 Category 1 — 核心奠基论文（9 篇）

| # | 论文 | 在本文中的作用 | 验证 |
|---|------|---------------|------|
| 1 | Frankle, J., et al. (ICML 2020). "Linear Mode Connectivity and the Lottery Ticket Hypothesis." | LMC 的定义来源，barrier 公式的原出处 | ✅ |
| 2 | Garipov, T., et al. (NeurIPS 2018). "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs." | Bezier 曲线；mode connectivity 开创工作 | ✅ |
| 3 | Draxler, F., et al. (ICML 2018). "Essentially No Barriers in Neural Network Energy Landscapes." | 首次实证 NN 极小值间几乎无障碍 | ✅ |
| 4 | Entezari, R., et al. (ICLR 2022). "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks." | 排列对称性是 LMC 失败主因 | ✅ |
| 5 | Ainsworth, S., et al. (ICLR 2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries." | 排列匹配，对齐独立训练模型到同一 basin | ✅ |
| 6 | Wortsman, M., et al. (ICML 2022). "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." | Model Soups — 权重平均提升鲁棒性 | ✅ |
| 7 | Yadav, P., et al. (NeurIPS 2023). "TIES-Merging: Resolving Interference When Merging Models." | TIES-Merging: Trim → Elect Sign → Merge 三步法 | ✅ |
| 8 | Biderman, S., et al. (NeurIPS 2023). "Pythia: A Suite for Analyzing Large Language Models." | 本文基座模型 Pythia | ✅ |
| 9 | Li, H., et al. (NeurIPS 2018). "Visualizing the Loss Landscape of Neural Nets." | Loss landscape 可视化标准方法 | ✅ |

---

## 🟡 Category 2 — 强关联（8 篇）

| # | 论文 | 关联性 | 验证 |
|---|------|--------|------|
| 10 | Wortsman, M., et al. (CVPR 2022). "Robust fine-tuning of zero-shot models." (WiSE-FT) | 权重插值提升 FT 鲁棒性 | ✅ |
| 11 | Ilharco, G., et al. (NeurIPS 2023). "Editing Models with Task Arithmetic." | 权重空间向量运算 | ✅ |
| 12 | Matena, M. & Raffel, C. (**NeurIPS 2022**). "Merging Models with Fisher-Weighted Averaging." | Fisher 加权合并；Laplace 近似后验 | ⚠️ venue 修正: ACL→NeurIPS |
| 13 | Jin, X., et al. (ICLR 2023). "Dataless Knowledge Fusion by Merging Weights of Language Models." | 无数据 LM 权重合并 | ✅ |
| 14 | Stoica, G., et al. (ICLR 2024). "ZipIt! Merging Models from Different Tasks without Training." | 多任务模型合并 | ✅ |
| 15 | Izmailov, P., et al. (UAI 2018). "Averaging Weights Leads to Wider Optima and Better Generalization." (SWA) | 权重平均→更宽极小值，解释低 barrier | ✅ |
| 16 | Neyshabur, B., et al. (NeurIPS 2020). "What is being transferred in transfer learning?" | FT 中哪些参数变化 | ✅ |
| 17 | Gururangan, S., et al. (ACL 2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." (DAPT) | 领域自适应 FT 标准化方法 | ✅ |

---

## 🟢 Category 3 — 弱关联但可挂（6 篇）

| # | 论文 | 关联性 | 验证 |
|---|------|--------|------|
| 18 | Zhang, C., et al. (ICLR 2017). "Understanding Deep Learning Requires Rethinking Generalization." | 泛化与权重空间的关系 | ✅ |
| 19 | Kirkpatrick, J., et al. (PNAS 2017). "Overcoming catastrophic forgetting in neural networks." (EWC) | 权重差异作为"遗忘"度量 | ✅ |
| 20 | Fort, S., et al. (NeurIPS 2019). "Stiffness: A New Perspective on Generalization in Neural Networks." | 参数对扰动的敏感度 | ✅ |
| 21 | Mirzadeh, S. I., et al. (**ICLR 2021**). "Linear Mode Connectivity in Multitask and Continual Learning." | 持续学习中的 LMC 基线 | ⚠️ venue 修正: ICML→ICLR |
| 22 | Lubana, E. S., et al. (**ICML 2023**). "Mechanistic Mode Connectivity." | mode connectivity 机制解释 | ⚠️ venue 修正: ICLR→ICML |
| 23 | Singh, S. P. & Jaggi, M. (NeurIPS 2020). "Model Fusion via Optimal Transport." | OT 非线性融合 vs 线性 LMC | ✅ |

---

## 验证总结

| 指标 | 数值 |
|------|------|
| 总数 | 23 |
| ✅ 完全正确 | **20** |
| ⚠️ venue 有误 | **3**（#12, #21, #22） |
| ❌ 不存在/幻觉 | **0** |
| 幻觉率 | **0%** |

### 三处 venue 修正

| # | 原文错误 | 正确 venue | 证据 |
|---|---------|-----------|------|
| 12 | ACL 2022 | **NeurIPS 2022** | NeurIPS 2022 proceedings (Vol 35, pp. 17703-17716); ACL Anthology 无记录 |
| 21 | ICML 2021 | **ICLR 2021** | DBLP path `conf/iclr/`; OpenReview ICLR 2021 poster 3142; Google Research 标注 ICLR 2021 |
| 22 | ICLR 2023 | **ICML 2023** | PMLR Vol 202 = ICML 2023 proceedings; icml.cc poster #24778; arXiv comments: "ICML, 2023" |

### 对抗性验证结果

25 条 claim 经 3-vote 对抗性验证：
- **18 confirmed** (3-0 或 2-1 存活)
- **7 refuted** (被 ≥2/3 驳倒) — 均为具体性能数据声明（如 "TIES-Merging 提升 +2.3%"），非论文存在问题
- **0 unverified**

### 背景

NeurIPS 2025 发现 100 篇 AI 生成的幻觉引用逃过 3-5 位审稿人，出现在 53 篇被接收论文中（~1%）。66% 为完全捏造，27% 为部分属性篡改。本清单 23 篇论文全部通过这项验证标准。
