# AFP Phase 0 — 新会话重启 Prompt

> 把这个文件内容粘贴给新 Claude Code 会话，即可无缝继续。

---

```markdown
# AFP Phase 0 实验 — 状态恢复

## 你是谁

你是 `claude-code`，运行在 DGX Spark GB10 (ARM64, 121GB 统一内存, CUDA 13.0) 上。

## 项目路径

`/home/jiayu/AFP`

## 目标

完成 AFP Phase 0 两个实验，生成报告，提交代码。

## 当前状态（2026-07-02）

### 实验进程

**检查命令:** `cd /home/jiayu/AFP && bash scripts/monitor.sh`

| 组件 | 查找方式 | 状态 |
|------|---------|------|
| IVN 实验 | `ps aux \| grep run_ivn` | 运行中/已完成 |
| F-IVN 实验 | `ps aux \| grep run_fivn` | 等待中/运行中/已完成 |
| F-IVN watcher | `ps aux \| grep fivn_trigger` | 后台轮询 |
| IVN PID 文件 | `cat experiments/ivn_pid.txt` | — |
| IVN 结果 | `cat experiments/phase0_ivn/ivn_results.json` | 存在=完成 |
| F-IVN 结果 | `cat experiments/phase0_fivn/fivn_results.json` | 存在=完成 |

### 日志

- IVN: `experiments/phase0_ivn/logs/run_*.log` （最新）
- F-IVN: `experiments/phase0_fivn/logs/run_*.log` （最新）
- 流水线: `experiments/fivn_trigger.log`
- 崩溃: `experiments/phase0_ivn/crashes/`, `experiments/phase0_fivn/crashes/`

### 结果文件

- `experiments/phase0_ivn/ivn_results.json`
- `experiments/phase0_fivn/fivn_results.json`
- `docs/reports/phase0-results-*.html`

## 如果实验还在跑

1. 先 `bash scripts/monitor.sh` 看状态
2. 如果 IVN 在跑、F-IVN watcher 存活：**不要动，等它完成**
3. 如果 IVN 已死：检查 `experiments/phase0_ivn/crashes/` 找原因，修 bug，重启
4. 如果 IVN 完成但 F-IVN 没自动启动：手动启动 F-IVN

## 如果实验已完成

```bash
cd /home/jiayu/AFP
# 生成报告（如果没自动生成）
/home/jiayu/FCL-PRM-cdspi/venv/bin/python3 scripts/generate_phase0_report.py
# 提交
git add -f experiments/phase0_ivn/ivn_results.json experiments/phase0_fivn/fivn_results.json docs/reports/
git commit -m "feat(phase0): experiment results"
git push origin main
```

## 如果要重启实验

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com  # 必须！无 VPN
export PYTHONUNBUFFERED=1

# 用 FCL-PRM 的 Python（有 torch 2.11 + transformers）
VENV=/home/jiayu/FCL-PRM-cdspi/venv/bin/python3

# 启动 IVN
nohup $VENV -u scripts/run_ivn_phase0.py \
    --importance mas --gate rational --tau 0.5 --max-rounds 30 \
    >> experiments/phase0_ivn/logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > experiments/ivn_pid.txt

# 启动 F-IVN 自动触发器
nohup bash -c '
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com
VENV=/home/jiayu/FCL-PRM-cdspi/venv/bin/python3
while [ ! -f experiments/phase0_ivn/ivn_results.json ]; do sleep 120; done
FIVN_LOG="experiments/phase0_fivn/logs/run_$(date +%Y%m%d_%H%M%S).log"
$VENV -u scripts/run_fivn_phase0.py --tau 0.5 --max-rounds 30 >> "$FIVN_LOG" 2>&1
RC=$?
if [ $RC -eq 0 ] && [ -f experiments/phase0_fivn/fivn_results.json ]; then
    $VENV scripts/generate_phase0_report.py >> experiments/fivn_trigger.log 2>&1
    git add -f experiments/phase0_ivn/ivn_results.json experiments/phase0_fivn/fivn_results.json docs/reports/ 2>/dev/null
    git commit -m "feat(phase0): IVN + F-IVN experiment results (auto)" >> experiments/fivn_trigger.log 2>&1 || true
fi
' &
```

## 环境要点

| 事项 | 值 |
|------|-----|
| Python | `/home/jiayu/FCL-PRM-cdspi/venv/bin/python3` |
| torch | 2.11.0+cu128 |
| transformers | 5.11.0 |
| HF mirror | `https://hf-mirror.com` |
| 设备 | DGX Spark GB10, CUDA 13.0, ARM64, 121GB |
| Remote | `https://github.com/FauReam/AFP.git` |

## 进程保护规则（必须遵守）

1. 所有训练/实验进程 **必须** `nohup ... &` 启动
2. stdout/stderr 重定向到 `experiments/` 下的时间戳日志
3. Python 启动加 `-u` 标志（无缓冲输出）
4. 启动后验证进程存活：`ps -p $PID`
5. 禁止 stdin 读取、GUI、浏览器依赖
6. 当前 Claude Code 会话也需要保护——建议用户在 tmux 中重连

## 已修复的 Bug 清单（11 条）

新会话必须知道这些坑，不要复现：

### Bug 8: `AFP.protocol.__init__` 缺少 gate 函数导出
- `gate_linear` 和 `gate_rational` 没有在 `__init__.py` 中导出 → `ImportError`
- 已修：`src/AFP/protocol/__init__.py` 补充导出

### Bug 9: `AutoTokenizer` 无 `local_files_only` → HF 超时
- 跨域评估触发缓存未命中，tokenizer 下载路径连 HF → `ConnectTimeout`
- 已修：所有 `AutoTokenizer.from_pretrained` 加 `local_files_only=True`
- 教训：无 VPN 环境，所有 HF API 调用必须 local-only

### Bug 10: `N_BLOCKS=24` 不支持 Qwen2.5 28 层 → IndexError
- Qwen2.5-1.5B 有 28 层，代码硬编码 24（Pythia 的值）
- Noise control 中 `gates_a_noise[24]` → `IndexError`
- 已修：`N_BLOCKS=32`（安全上界），本地 `_block_index()` 加越界检查

### Bug 11: MAS `per-param .item()` 导致 10,416 次 GPU sync
- `param.grad.float().abs().mean().item()` 在 336 个参数 × 31 batch 循环中
- 每次 `.item()` = GPU→CPU 同步，GB10 上 ~1.7s/次 → 5+ 小时
- 已修：GPU 端 `imp_sum[blk] += ...` 累积，最后一次性 `.cpu()` 同步

### Bug 12: MAS 在 CPU 上运行（`to_device()` 调用太晚）
- `agent.to_device()` 在 baseline eval 前才调用，MAS 在之前就跑完了
- 模型全程在 CPU，GPU 内存 0GB，backward 极慢
- 已修：模型加载后立即 `to_device()`

### Bug 13: `run_ivn_phase0.py` 默认使用 Qwen2.5 而非 Pythia-1.4B
- 脚本 `--teacher`/`--student` 默认值是 Qwen2.5-Coder/Math，但方案要求 Pythia-1.4B full-FT on code+medical
- 直接 `python scripts/run_ivn_phase0.py` 得到错误的实验（importance cosine=0.975，不互补）
- 正确入口是 `bash scripts/train_and_run_phase0.sh`（会覆盖参数）
- 已修：默认值待改为 Pythia（目前 pipeline 脚本覆盖）

### Bug 14: `train_agent.py` 超参偏离 CLAUDE.md
- EPOCHS=2（应为 1），LR=5e-4（应为 1e-4），MAX_LEN=384（优化为 256）
- 导致训练步数 1,242（应为 433），12h/domain
- 已修：EPOCHS=1, LR=1e-4→3e-6, MAX_LEN=256

### Bug 15: 训练缓存不绑定 MAX_LEN
- 缓存文件名 `train_{domain}_pythia.pt` 不含配置参数
- 修改 MAX_LEN 后静默复用旧缓存 → "改参数重跑"是假重跑
- 已修：缓存文件名改为 `train_{domain}_pythia_L{MAX_LEN}.pt`

### Bug 16: `train_agent.py` SyntaxError — `global` 声明顺序
- `p.add_argument("--max-len", default=MAX_LEN)` 在 `global MAX_LEN` 之前使用变量
- Python 报 `SyntaxError: name 'MAX_LEN' is used prior to global declaration`
- 已修：`global MAX_LEN` 移到 `main()` 开头

### 旧项目 Bug 1-7
详见 `docs/internal/ENGINEERING.md` 第二节。都是真实踩过的坑，新 agent 必读。

## 正确启动方式（Pythia full-FT）

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 唯一入口：train + IVN 全流程
nohup bash scripts/train_and_run_phase0.sh > /dev/null 2>&1 &
echo $! > experiments/pipeline_pid.txt
```

**不要直接跑 `run_ivn_phase0.py`**（默认值仍是 Qwen2.5）。

## 训练速度参考（DGX Spark GB10, Pythia-1.4B full-FT）

| 配置 | 步数/domain | 每步时间 | 总时间/domain |
|------|------------|---------|-------------|
| MAX_LEN=384, EPOCHS=2, batch=128 | 1,242 | ~35s | ~12h |
| MAX_LEN=256, EPOCHS=1, batch=128 | 433 | ~23s | ~2.8h |

## 重要观察

1. **Importance cosine = 0.96~0.97** — Coder 和 Math 的重要性模式几乎相同。根据实验方案，这不算"领域互补"。
2. **实验耗时**：MAS ~5min + baseline ~6min + grid search ~54min + IVN ~10min ≈ **75 分钟**（修复后）。之前因 Bug 11+12 需要 8+ 小时。
3. **F-IVN 预估**：head training ~10min + negotiation ~5min + distill ~5min ≈ **20 分钟**
```
