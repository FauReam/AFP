# AFP Phase 0 — 新会话重启 Prompt

> 粘贴给新 Claude Code 会话即可继续。

---

```markdown
# AFP Phase 0 实验 — 状态恢复

## 你是谁

`claude-code`，运行在 DGX Spark GB10 (ARM64, 121GB, CUDA 13.0)。

## 项目路径

`/home/jiayu/AFP`

## 环境（必须）

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
VENV=/home/jiayu/AFP/venv/bin/python3
```

## 检查状态

```bash
cd /home/jiayu/AFP && bash scripts/monitor.sh
```

## 有用的检查命令

```bash
# 训练/实验进程
ps aux | grep -E "train_agent|run_ivn|train_and_run|queue" | grep -v grep

# 已训练的模型
ls -d experiments/trained_models/*/

# IVN 结果
ls experiments/phase0_ivn/results/

# 最新训练日志
tail -20 experiments/phase0_training/pipeline_*.log
```

## 如果实验在跑

**不要动。** 等它完成。

## 如果实验已完成

```bash
cd /home/jiayu/AFP
$VENV scripts/generate_phase0_report.py
git add -f experiments/phase0_ivn/results/ docs/reports/
git commit -m "feat(phase0): experiment results"
git push origin main
```

## 重启全流程

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

nohup bash scripts/train_and_run_phase0.sh > /dev/null 2>&1 &
echo $! > experiments/pipeline_pid.txt
```

训练后自动排队多实验：
```bash
# pipeline 完成后自动触发，或手动：
nohup bash scripts/queue_experiments.sh > experiments/queue.log 2>&1 &
```

## 已知 Bug（新会话必读）

### Bug 17: IVN 脚本 importance 未赋值给 agent
- `run_ivn_phase0.py` 用 L2 算 importance 但只存到局部变量
- `integrate_afp()` 检测到 `_importance is None` → 回退到 L1 mean（错）
- 已修：`agent_a._importance = imp_a`

### Bug 18: code 训练 epoch 2+ 退化
- code_e2-e5 权重偏移反而减小，e3 直接保存 base 模型
- 根因未定位
- 当前只信任 code_e1

### Bug 19: pipeline `rm -f` 删不掉目录
- `rm -f code/` 对目录无效（需要 `-r`）
- 导致 IVN 步骤失败
- 已修：`rm -rf`

### Bug 20: VENV 路径变更
- 旧路径 `/home/jiayu/FCL-PRM-cdspi/venv/bin/python3` 已删除
- 新路径 `/home/jiayu/AFP/venv/bin/python3`（torch 2.12）

### Bug 21: code/ 目录被 base 模型覆盖
- 未训练的 base 模型被保存到 `trained_models/code/`
- 已修：保存前验证 Δ > 1e-3

### 旧 Bug 8-12 & 1-7
详见 `docs/internal/ENGINEERING.md`。

## 训练速度（GB10 实测）

| 配置 | 速度 | 总时间 |
|------|------|--------|
| Pythia-1.4B full-FT, L=256, batch=128, 1ep | ~5.5s/step × 433 steps | ~40 min |
| IVN 实验 (c1m1) | — | ~10 min |

## 实测结果速查

| 实验 | 模型对 | FedAvg(α=0.1) | AFP(τ=0.5) | IVN |
|------|--------|---------------|-------------|-----|
| c1m1 | code_e1 + med_e1 | **+0.014** | -0.149 | -0.158 |
| c1m5 | code_e1 + med_e5 | -0.016 | -0.807 | -0.156 |

只有 FedAvg α=0.1 (10% 混合) 有正收益。根因：gate 范围 [0.33, 0.83] 无法做到足够保护。
```
