# Experiment Triage — AFP 实验评估

在跑任何实验之前和之后强制评估。防止浪费 GPU 时间。

## 触发条件

当用户说"跑实验"、"训练"、"IVN"、"grid search"、"测一下"、任何涉及启动训练/评估的命令时，**必须先调此 skill 做 pre-triage**。

当实验完成、用户问"结果怎么样"、"跑完了"时，**调此 skill 做 post-triage**。

## Pre-triage（跑之前）

必须输出以下四行，缺一不可：

```
假设:    [跑这个实验预期验证什么？一句话。]
论文价值: [HIGH / MED / LOW / NONE]
         HIGH = 能进 paper figure/table
         MED   = 支撑性证据
         LOW   = 探索/调试
         NONE  = 纯工程（修 bug、调参）——不需要 skill 审批
方向判定: [EXPECTED ON TRACK / DRIFTING RISK]
         DRIFTING RISK = 即使实验成功，也不回答论文核心问题
预计耗时: [分钟]
```

规则：
- **论文价值 NONE**（纯工程）→ 直接跑，不需要等用户确认
- **论文价值 LOW** → 提醒用户，但仍可跑
- **论文价值 MED/HIGH** 但 **方向判定 DRIFTING RISK** → **阻止**，要求用户明确确认
- **预计耗时 > 60 分钟且论文价值 LOW** → **阻止**，警告时间投入产出比

## Post-triage（跑完之后）

在 `EXPERIMENT_PLAN.md` 中追加一行评估：

```
| E{N} | {名字} | {假设} | {论文价值} | {实际结果} | {ON TRACK/DRIFTING} | {分钟} |
```

并更新以下计数器：
- 有效 GPU 时间累计
- 浪费 GPU 时间累计
- 如果标记 DRIFTING：必须写一行"为什么偏离 + 如何避免"

## 批量实验规则

当用户想"排队跑 N 个实验"时：
1. **先跑第 1 个**，看 post-triage 结果
2. 如果 DRIFTING → **终止队列**，不跑后续
3. 如果 ON TRACK → 跑下一个
4. 禁止"一口气跑完 N 个"除非论文价值全是 HIGH

## 已知 DRIFTING 模式（直接阻止）

以下实验设计已被证实是 DRIFTING，看到直接阻止：
- 不对称差异实验（一方 1ep + 一方 3/5ep）→ 弱方会被淹没，不测试选择性
- 模型未经验证就 IVN（B→code < 0.76）→ 训练质量不够，先修训练
- 改 importance 指标但不改 τ → 已证实所有指标 cosine > 0.94
