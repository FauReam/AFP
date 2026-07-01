# AFP 工程手册

> 从 FCL-PRM 旧项目完整迁移 + AFP Phase 0 v2 新增。
> 以下每一条都是真实踩过的坑。
> **新 agent 必读。不要复现已知 Bug。**

---

## 一、硬件

| 项目 | 值 |
|------|-----|
| 机器 | NVIDIA DGX Spark (GB10) |
| GPU | Blackwell, compute 12.1, CUDA 13.0 |
| 内存 | 121 GB 统一内存 (CPU+GPU 共享) |
| CPU | ARM64 (Cortex-X925 + A725) |
| torch.compile | **不可用** (Triton 不兼容 ARM64) |
| 实测速度 | Pythia-1.4B frozen backbone, head-only, batch=128, max_len=256 → **~8s/batch** |

---

## 二、关键 Bug 及修复（7 条，全部真实）

### Bug 1: `torch.cuda.synchronize()` 每 batch 调用
- **位置**: `client.py:168`（旧项目）
- **症状**: 训练速度被强制 GPU 同步拖慢
- **修复**: 删除 `torch.cuda.synchronize()`，提速 ~2x
- **教训**: 不要在训练循环的 inner loop 里调 synchronize

### Bug 2: `loss.item()` 调 3 次
- **症状**: 每次调用触发 CPU-GPU 传输
- **修复**: 合并为 1 次 `loss_val = loss.item()`，复用变量

### Bug 3: Frozen backbone 走了完整 autograd
- **位置**: `base_wrapper.py` forward
- **症状**: 冻结的 backbone 仍在构建计算图，浪费 ~30% 时间
- **修复**: forward 中加 `torch.no_grad()` 包裹 frozen backbone

### Bug 4: bf16 backbone output → fp32 head crash
- **症状**: `mat1 and mat2 must have the same dtype`
- **修复**: forward 中 `last_hidden.float()` 转换

### Bug 5: YAML `1e-4` 被解析为字符串
- **症状**: PyYAML 1.1 将 `1e-4` 解析为 `"1e-4"` 字符串
- **修复**: (a) YAML 中写 `1.0e-4` 或 `0.0001`；(b) 代码入口加 `float()` 防御

### Bug 6: Stale checkpoint 导致空跑
- **症状**: 失败运行残留 `r25` checkpoint，恢复逻辑读到 `start_round=25` 直接跳过
- **修复**: 恢复时检测 `>= num_rounds` 并自动删除 stale checkpoint

### Bug 7: CD-SPI 静默返回空
- **症状**: anchor steps 只在 `aggregation: anchor_prm` 时生成
- **修复**: 条件改为 `needs_anchor_steps = anchor_prm or compute_cd_spi or compute_sym_cd_spi`

### Bug 8: `AFP.protocol.__init__` 缺少 gate 函数导出 — 2026-07-01
- **位置**: `src/AFP/protocol/__init__.py`
- **症状**: `ImportError: cannot import name 'gate_linear' from 'AFP.protocol'`
- **修复**: 补充 `gate_linear, gate_rational` 到 `from .trust import` 行
- **教训**: 新增公开函数必须更新 `__init__.py` 导出

### Bug 9: `AutoTokenizer.from_pretrained` 无 `local_files_only` → HF 超时 — 2026-07-01
- **位置**: `scripts/run_ivn_phase0.py:64`, `scripts/run_fivn_phase0.py:64`
- **症状**: 跨域评估 `evaluate(Coder, math)` 触发缓存未命中 → tokenizer 下载路径尝试连 HF → `httpx.ConnectTimeout`
- **根因**: 数据缓存以 `(domain, model_id)` 为 key。跨域组合未预热，fallback 到下载路径中的 `AutoTokenizer.from_pretrained()` 无 `local_files_only=True`
- **修复**: 所有 `AutoTokenizer.from_pretrained` 加 `local_files_only=True`
- **教训**: 无 VPN 时任何 HF API 调用都可能超时。离线可用操作必须显式 local-only

### Bug 10: `N_BLOCKS=24` 不支持 Qwen2.5 的 28 层 → IndexError — 2026-07-01
- **位置**: `scripts/run_ivn_phase0.py:44`, `src/AFP/protocol/trust.py:20`, `src/AFP/protocol/integrator.py:11`
- **症状**: Noise control 中 `gates_a_noise[blk]` — `IndexError: list index out of range`（blk=24, len=24）
- **根因**: Qwen2.5-1.5B 有 28 层，但 `N_BLOCKS=24` 是 Pythia-1.4B 的值。本地 `_block_index()` 无越界检查
- **修复**: `N_BLOCKS` → 32；本地 `_block_index()` 加 bounds check；noise control 加 `blk < len(gates)` 保护
- **教训**: 架构常量必须从 model config 动态获取。两个 `_block_index` 实现不同行为是代码坏味道

### Bug 11: MAS `per-param .item()` 导致 10,416 次 GPU sync → 5+ 小时 — 2026-07-01
- **位置**: `src/AFP/protocol/trust.py:137` `mas_importance()`
- **症状**: MAS 计算在 DGX Spark GB10 上耗时 5+ 小时（500 samples, batch=16）。GPU 利用率 0%，CPU 1,400%+。
- **根因**: 内层循环对每个参数的梯度调用 `.float().abs().mean().item()`——每次 `.item()` 触发 GPU→CPU 同步。Qwen2.5-1.5B 有 336 个 layer 参数 × 31 batches = 10,416 次 GPU sync。GB10 统一内存下每次 sync ~1.7s。
- **修复**: GPU 端累积 `imp_sum[blk] += param.grad.float().abs().mean()`（无 `.item()`），循环结束后一次性 CPU sync。总 sync 次数从 10,416 → 1，MAS 耗时从 5h → ~1min。
- **教训**: 任何在 GPU tensor 上循环调用 `.item()` 的模式都是性能杀手。先用 GPU tensor 累积，最后一次性同步。对 1.5B+ 参数模型尤其致命。

### Bug 12: MAS 在 CPU 上运行（`to_device()` 调用太晚）— 2026-07-01
- **位置**: `scripts/run_ivn_phase0.py:314-318`
- **症状**: MAS 计算时 GPU 内存 0GB，利用率 0%，CPU 1,400%+。1.5B 模型 backward 在 CPU 上跑，极慢。
- **根因**: `agent_a.to_device()` / `agent_b.to_device()` 在 baseline 评估前才调用（line 345），但 MAS 在 line 322 就运行了。`AFPAgent.__init__` 不会自动 `.to(device)`，模型默认在 CPU。
- **修复**: 在模型加载后、MAS 计算前调用 `to_device()`。
- **教训**: 构造函数不应隐式移动模型到 GPU（会导致加载失败时状态不一致），但调用方必须在计算前显式 `.to(device)`。可以在 `compute_*` 方法内部做防御性 GPU 移动。

---

## 三、性能权衡记录

| 改法 | 效果 | 风险 | 决策 |
|------|------|------|------|
| max_length 256→128 | ~3x 加速 | **截断 78% 数据** | ❌ 已否决 |
| max_length 256→384 | ~1.3x 变慢 | 丢弃 8.7%，无截断 | ✅ 旧项目采用 |
| batch_size 4→128 | ~32x 加速 | 仅 head-only/LoRA 安全 | ✅ head-only/LoRA 采用 |
| batch_size 4 (保留) | 安全 | 慢 | partial-FT / full-FT 保留 |

### PRM 训练内存参考

| 配置 | batch_size | 内存 |
|------|-----------|------|
| Pythia-1.4B head-only | 128 | ~5 GB |
| Pythia-1.4B LoRA r=8-256 | 128 | ~6-8 GB |
| Pythia-1.4B partial-FT (last 2) | 4 | ~15 GB |
| Pythia-1.4B full-FT (BF16+Adam) | 4 | ~21 GB |
| Pythia-1.4B full-FT (BF16+Adam) | **1024** | **~71 GB**（v2 新增） |
| Pythia-2.8B full-FT (BF16+Adam) | 4 | ~40 GB |

### AFP Phase 0 v2 内存精算（full-FT @ batch=1024）

```
模型权重 (bf16):         2.8 GB
Master 权重 (fp32):      5.6 GB
梯度 (fp32):             5.6 GB
Adam m (fp32):           5.6 GB
Adam v (fp32):           5.6 GB
─────────────────────────────
基础开销:               ~25 GB

每 sample activation (seq_len=384, 1 checkpoint/layer):
  384 × 2048 × 2B × 24层 ≈ 38 MB
  1024 samples:          ~39 GB
  attention mask 等:      ~5 GB
─────────────────────────────
Activations:            ~44 GB
总计:                   ~69 GB

121 GB 统一内存 → 安全
```

### 大 batch 训练速度预估

```
batch=1024, len=384, full-FT:
  forward:            ~15s
  backward:           ~30s
  optimizer step:     ~3s
  overhead:           ~5s
  ───────────────────────
  每 batch:           ~50-55s

Agent A (medical): 85 batches × 55s ≈ 1.3h
Agent B (code):    203 batches × 55s ≈ 3.1h
串行总计:          4.4h
```

### 全参数 FT 内存建议
- BF16 加载 backbone
- batch_size=4 起步（旧）；batch=1024 用 gradient checkpointing
- 检查点体积: 1.4B ~2.8 GB, 2.8B ~5.6 GB
- 4 客户端串行全参数: 2.8B 约 1-2 天/25 rounds

---

## 四、已知陷阱（从旧项目 CLAUDE.md）

1. **全参数 FT 内存**：BF16 加载 + batch_size=4 起步
2. **检查点体积**：注意磁盘空间
3. **客户端串行训练**：4 客户端全参数串行耗时长
4. **设备不匹配**：`_eval_per_domain` 需在返回前调 `.cpu()`
5. **检查点恢复**：崩溃发生在聚合前则不生成该轮检查点
6. **Opacus DP-SGD**：包覆后 `model._module` 才是原始模型
7. **对称化测量**：CD-SPI 须从 backbone 倒数第二层 hidden state 统一提取
8. **AttnRes 仅支持 GPTNeoX**：不支持其他架构
9. **AttnRes 零初始化必须**：伪查询向量初始化为 0
10. **torch.compile 时机**：在 StepRewardModel 构造之后
11. **AttnRes checkpoint 兼容性**：state_dict key 与标准模型不兼容

---

## 五、训练代码规范（强制）

### 5.1 终端断开存活
- 代码中**零 stdin 读取**（无 `input()`, `sys.stdin.read()`, 交互式确认）
- stdout/stderr 全量落盘（Tee 机制）
- crash report 写结构化 JSON 到 `crashes/`
- 零 GUI/浏览器依赖
- 启动**必须用** `nohup ... &`

### 5.2 启动模板

```bash
nohup python scripts/run_protocol.py \
    --config configs/xxx.yaml \
    --rounds 10 \
    >> experiments/xxx/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID: $!"
```

### 5.3 >1h 训练规则

1. 必须 `nohup` + `&` 启动，只返回 PID
2. stdout/stderr 重定向到 `experiments/` 下的时间戳日志
3. 不可用 Bash 工具长时间阻塞等待输出
4. 启动后验证进程存活（`ps -p $PID`）
5. 终端断开不影响进程

### 5.4 进度条 (tqdm)

| 层级 | desc | leave | postfix |
|------|------|-------|---------|
| 顶层循环 | `"Rounds"` | True | loss, time |
| 中层 (clients/eval) | `"[Eval] xxx"` | True | — |
| 内层 (epochs) | `[Client X] epoch N/M` | False | — |
| 最内层 (batches) | 不设 desc | False | loss (每 log_interval 刷新) |

### 5.5 检查点保存

强制性顺序：
1. `model.state_dict()` → `.cpu()`
2. `optimizer.state_dict()` → `.cpu()`（注意嵌套结构）
3. `torch.save()` 到磁盘
4. `del cpu_state, cpu_optim`
5. `gc.collect()`
6. `torch.cuda.empty_cache()`

文件命名：
```
{checkpoint_dir}/
├── model_r{round}_c-1.pt           # global model
├── clients/
│   ├── model_r{round}_c0.pt        # client 0
│   └── ...
├── history.json                     # 原子写入（.tmp → rename）
└── crashes/
    └── crash_r{round}_{ts}.json
```

### 5.6 错误处理三层

```
Layer 1: 进程级 — try/except 包裹 main()
Layer 2: 调度级 — per-agent try/except，单 agent 崩溃不影响其他
Layer 3: 系统级 — SIGINT handler，优雅退出 + 保存 snapshot
```

Crash report 格式：
```json
{
  "timestamp": "2026-06-19T14:30:00",
  "stage": "round_3_agent_B",
  "round": 3,
  "error_type": "RuntimeError",
  "error_message": "...",
  "traceback": "..."
}
```

### 5.7 日志双通道

```
通道 A: Tee(stdout) → 文本日志（含 tqdm 输出）
通道 B: ExperimentLogger → JSONL 结构化指标
```

日志目录：
```
experiments/{exp_name}/
├── logs/
│   ├── run_20260619_143000.log
│   └── metrics.jsonl
├── checkpoints/
└── crashes/
```

### 5.8 修改日志

效率/算法类代码变更**必须**记录 `docs/CHANGELOG.md`：
- 日期、类型、影响范围
- 变更前/后行为
- 预期效果（量化）
- 风险评估
- commit message 中引用 CHANGELOG 条目

---

## 六、旧项目关键文件索引

以下文件在 AFP 开发中可能需要参考：

| 组件 | 旧项目路径 | 说明 |
|------|-----------|------|
| StepRewardModel | `src/fclprm/models/base_wrapper.py` | PRM 模型结构（LoRA/partial-FT/AttnRes） |
| 联邦模拟器 | `src/fclprm/federated/simulator.py` | 多客户端调度、tqdm、checkpoint、crash recovery |
| 集中式训练 | `scripts/train_centralized_prm.py` | 本地训练逻辑 |
| 联邦入口 | `scripts/run_federated.py` | Tee 日志、torch.compile、配置解析 |
| 数据分区 | `src/fclprm/data/heterogeneity.py` | Dirichlet 分区 |
| 检查点 | `src/fclprm/models/checkpoint.py` | save/load 逻辑 |
| 日志 | `src/fclprm/utils/logging.py` | ExperimentLogger |
| CD-SPI | `src/fclprm/metrics/cd_spi.py` | 旧诊断度量（已被证伪，仅作代码参考） |
| 配置 | `configs/` | YAML 配置模板 |

---

## 七、VersaPRM 数据速查

```
84,098 CoT 样本 → 669,218 个 step
平均 question+step: 217 tokens, 中位数 185
max_length 分布:
  192: 52.9%    256: 73.5%    384: 91.3%    512: 97.1%
```

- 净化数据: `data/versaprm/versa_prm.jsonl`
- 净化方式: tokenize 后丢弃 >384 token 的 step（~8.7% 丢弃），不截断
- max_length=384 覆盖 91.3%

### Domain 分布

| Domain | Steps | 训练 (70%) |
|--------|-------|-----------|
| code | 296K | 207K |
| math | 200K | 140K |
| general | 188K | 132K |
| medical | 124K | 87K |

### AFP Phase 0 v2 使用的 domain 对

```
code + medical: 重叠最小，测试 AFP 在最正交领域对上的表现
```
