# AMP — Claude Code 项目上下文

> **新会话必读**：[VISION.md](VISION.md) — AMP 的核心愿景与协议形式化

## 项目简介

**AMP (Agentic Mutual Protocol)** — 去中心化的 P2P 互学习协议。

核心洞察：现行联邦学习的 FedAvg 聚合是"初等运算"（凸组合），假定了服务器全知可信、所有客户端同等可靠、知识可以直接加和。AMP 推翻这些假定：

- **Agent 自主决策**：每个模型自己决定学不学、学什么、学多少、信谁
- **选择性子空间更新**：不是线性插值 W'=(1-α)W_A+α·W_B，而是 W'=W_A+M⊙(W_B-W_A)
- **信任是学出来的**：通过交互历史建立 trust，不是预设超参数
- **P2P 拓扑**：去中心化，无中心服务器，模型直接"对话"
- **类比学术界**：学者通过论文（权重）交流，选择性引用（整合），信任基于同行评议（交互验证）

## 项目结构

```
AMP/
├── VISION.md              # 核心愿景（必读）
├── DIRECTION.md           # 研究方向与文献地图
├── CLAUDE.md              # 本文件
├── EXPERIMENT_PLAN.md     # 实验计划
├── README.md              # 项目概述
├── src/amp/
│   ├── protocol/          # AMP 交互协议核心
│   │   ├── agent.py       # 自主 agent（决策、学习策略）
│   │   ├── trust.py       # 信任/声誉模块
│   │   ├── extractor.py   # 权重知识提取器
│   │   └── integrator.py  # 选择性子空间整合
│   ├── models/            # 模型定义
│   ├── data/              # 数据处理
│   └── experiments/       # 实验脚本
├── experiments/           # 实验输出
├── configs/               # 实验配置
└── docs/                  # 文档
```

## 设备

NVIDIA DGX Spark GB10（121GB 统一内存，ARM64）

## ⚠️ 工程手册（必读）

**[docs/ENGINEERING.md](docs/ENGINEERING.md)** — 从旧项目完整迁移的工程知识：
- 7 条真实踩过的 Bug 及修复（torch.cuda.synchronize、loss.item 多次调用、frozen backbone autograd、bf16→fp32 crash、YAML 类型陷阱、stale checkpoint、CD-SPI 静默空返回）
- 性能权衡记录（max_length、batch_size 决策）
- 全参数 FT 内存参考
- 11 条已知陷阱（Opacus、AttnRes、torch.compile、设备不匹配等）
- 训练代码强制规范（nohup 启动、tqdm、checkpoint、crash recovery、日志双通道、修改日志）
- VersaPRM 数据速查

**新 agent 在写任何训练代码之前必须先读 ENGINEERING.md。不要复现已知 Bug。**

## 旧项目参考

- 路径：`/Users/Apple/project/FCL-PRM-cdspi`
- 可复用：训练脚本结构、联邦模拟器框架、模型定义（base_wrapper, LoRA）、数据分区逻辑
- 已吸取的教训：CD-SPI 诊断框架被实验证伪（全配置 CD-SPI sym ≈ 0.001）
- 旧项目是"诊断已有系统中的问题"，AMP 是"重新定义系统本身"

## Phase 0 关键路径

Phase 0 目标：2-agent 最简原型，证明 AMP 的可行性

1. 定义 `protocol/agent.py` — Agent 类，封装自主决策逻辑
2. 定义 `protocol/integrator.py` — 选择性子空间整合（对比线性平均）
3. 定义 `protocol/trust.py` — 初始固定 trust 模式
4. 实验：两个 agent 各自在私有数据上训练，用 AMP 协议交换知识
5. Baseline：FedAvg 线性平均作为对比

## 工作约定

- 提交信息用英文，遵循 `feat(scope): description` 格式
- 旧项目 `FCL-PRM-cdspi` 的工程惯例可参考，但不强制沿用
- 训练 >1h 用 nohup + &，日志写入 `experiments/`
