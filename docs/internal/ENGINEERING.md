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
| 实测速度 | Pythia-1.4B full-FT, batch=128, max_len=256, torch 2.12 → **~5.5s/step** |

---

## 二、关键 Bug 及修复（23 条，全部真实）

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

### Bug 13: `run_ivn_phase0.py` 默认使用 Qwen2.5 而非 Pythia-1.4B — 2026-07-03
- **位置**: `scripts/run_ivn_phase0.py:294-297`
- **症状**: 新会话直接运行 `python scripts/run_ivn_phase0.py`，加载 Qwen2.5-Coder/Math（而非 Pythia），importance cosine=0.975（Qwen 同基座 fine-tune 变体高度相似，不互补）。
- **根因**: `--teacher default="Qwen/Qwen2.5-Coder-1.5B-Instruct"`, `--student default="Qwen/Qwen2.5-Math-1.5B-Instruct"`。但 CLAUDE.md 和 Phase 0 方案要求使用 **Pythia-1.4B full-FT on code + medical**。`train_and_run_phase0.sh` 正确覆盖了参数，但直接调用脚本不会。
- **修复**: 将默认值改为 `EleutherAI/pythia-1.4b`，默认 domain 改为 `code`/`medical`；或至少在脚本顶部加 comment 警告。
- **教训**: 脚本默认值必须与项目文档（CLAUDE.md）一致。不一致的默认值会导致新 agent 在不知情的情况下跑错实验，浪费数小时 GPU 时间。

### Bug 14: `train_agent.py` 超参偏离 CLAUDE.md v2 spec — 2026-07-03
- **位置**: `scripts/train_agent.py:40-46`
- **症状**: 训练速度慢于预期。`EPOCHS=2`（spec 要求 1），`LR_MAX=5e-4`（spec 要求 1e-4），`LR_MIN=1.5e-5`（spec 要求 3e-6）。2 epochs × 621 batches/epoch × 35s/step ≈ 12h/domain。
- **修复**: 对齐 CLAUDE.md：`EPOCHS=1`, `LR_MAX=1e-4`, `LR_MIN=3e-6`。同时 `MAX_LEN=384→256`（保留 86% 数据，减少 33% FLOPs）。总步数从 1,242 → 433，每步从 35s → 23s，总时间 12h → 2.8h。
- **教训**: 脚本常量必须在文件顶部引用来源文档。超参变更要同步更新文档。

### Bug 15: 训练缓存文件不绑定 MAX_LEN → 静默复用旧缓存 — 2026-07-03
- **位置**: `scripts/train_agent.py:58-59`
- **症状**: 修改 `MAX_LEN` 后重跑训练，数据量没变（仍使用旧 `MAX_LEN=384` tokenize 的缓存）。
- **根因**: 缓存文件名 `train_{domain}_pythia.pt` 不含配置参数。`prepare_data()` 检测到文件存在即返回，不重新 tokenize。
- **修复**: 缓存文件名加入 MAX_LEN：`train_{domain}_pythia_L{MAX_LEN}.pt`。不同配置生成不同缓存，避免静默冲突。
- **教训**: 任何影响数据内容的参数变化必须体现在缓存 key 中。否则"改参数重跑"是假重跑。

### Bug 16: `train_agent.py` SyntaxError — `global MAX_LEN` 声明顺序 — 2026-07-03
- **位置**: `scripts/train_agent.py:320-323`
- **症状**: `SyntaxError: name 'MAX_LEN' is used prior to global declaration`。`p.add_argument("--max-len", default=MAX_LEN)` 在 `global MAX_LEN` 之前使用了模块级变量。
- **根因**: Python 要求 `global` 声明在函数内首次使用该变量名之前。`default=MAX_LEN` 是对 `MAX_LEN` 的读取使用。
- **修复**: 将 `global MAX_LEN` 移到 `main()` 函数开头（`p = argparse.ArgumentParser(...)` 之前）。
- **教训**: 如果函数内需要 `global` 覆写模块级常量，声明必须在所有对该变量的引用之前——包括函数参数的默认值表达式。

### Bug 17: IVN 脚本 importance 未赋值给 agent → 回退到错误指标 — 2026-07-05
- **位置**: `scripts/run_ivn_phase0.py:375-376`
- **症状**: 指定 `--importance magnitude_l2`，但 AFP/IVN 实际使用的是 L1 mean importance。所有 τ 值下 AFP 输出相同结果（gate 无区分度）。
- **根因**: `imp_a = compute_importance_l2_from_models(...)` 只存到局部变量。`integrate_afp()` 检查 `self._importance is None` → 调用 `self.compute_importance(init)` 使用 L1 mean。命令行指定的 importance 指标从未生效。
- **修复**: 在 importance 计算后添加 `agent_a._importance = imp_a; agent_b._importance = imp_b`。
- **教训**: 独立函数计算的值如果下游要通过对象方法使用，必须显式赋值。对象方法内部的 fallback 逻辑是静默 bug 的温床。

### Bug 18: code 训练 epoch 2+ 退化，e3 保存 base 模型 — 2026-07-05
- **位置**: `scripts/train_agent.py` 训练循环
- **症状**: code_e2 权重偏移小于 e1，code_e3 与 base 完全一致（291/291 tensors max_diff < 1e-8），code_e4-e5 偏移也极小。只有 code_e1 是有效的。
- **根因**: 未完全定位。可能原因：(a) 训练循环中模型状态未正确恢复；(b) val_loss 从 epoch 1 后就持续上升，后续保存的都是 epoch 1 的最佳模型，但 checkpoint 逻辑有误保存了初始权重。
- **当前方案**: 只信任 code_e1。e2+ 的 checkpoint 忽略。训练时加 Δ 验证（Bug 21），Δ < 1e-3 拒绝保存。
- **教训**: 每个 epoch 保存 checkpoint 前应验证模型权重确实发生了变化。不能假设"跑了 N 个 epoch 就训练了 N 轮"。

### Bug 19: pipeline `rm -f` 删不掉目录 → IVN 步骤失败 — 2026-07-05
- **位置**: `scripts/train_and_run_phase0.sh` symlink 创建
- **症状**: `rm: cannot remove 'experiments/trained_models/code': Is a directory`。IVN 步骤在训练完成后中止。
- **根因**: training 保存到 `code/` 和 `medical/` 目录，后续想创建同名 symlink。`rm -f` 无 `-r` 不能删目录。
- **修复**: `rm -rf experiments/trained_models/code medical`。
- **教训**: 用 `rm -rf` 而非 `rm -f` 删除可能为目录的路径。或者先判断 `test -d && rm -rf || rm -f`。

### Bug 20: VENV 路径变更 — 2026-07-05
- **位置**: 所有脚本中的 `VENV` 变量
- **症状**: `/home/jiayu/FCL-PRM-cdspi/venv/bin/python3` 不存在，pipeline 静默失败。
- **根因**: FCL-PRM venv 被删除，项目有了自己的 venv。
- **修复**: 全部改为 `/home/jiayu/AFP/venv/bin/python3`（torch 2.12.1, transformers 5.12.1）。
- **教训**: 环境路径变更时务必全局搜索替换。RESTART_PROMPT.md 的 VENV 引用必须与实际一致。

### Bug 21: `trained_models/code/` 被未训练 base 模型覆盖 — 2026-07-05
- **位置**: `scripts/train_agent.py:290-292`
- **症状**: IVN 实验加载的 code agent 实际是未训练的 Pythia base。importance 全为 0 → gate 全开 → AFP 灾难性遗忘（B_cross=0.125）。
- **根因**: pipeline 重启时 `train_agent.py` 重新初始化 agent，如果训练在第一个 epoch 完成前崩溃，或 val_loss 未改善，保存到 `code/` 的可能不是最优模型。
- **修复**: 保存前验证模型权重变化 `changed = Σ||W - W_init||`，若 changed < 1e-3 则跳过保存并打印警告。同时改为每 epoch 保存 checkpoint（`--save-every-n-epochs 1`），IVN 从 checkpoint 加载而非 `code/`。
- **教训**: 永远不要信任训练脚本的"best model"保存逻辑——加守卫条件。文件夹命名约定（`code_e{N}`）比模糊的 `code/` 更可靠。

### Bug 23: `train_agent.py` 输出路径无 seed/lr 区分 → 多轮训练互相覆盖 — 2026-07-13
- **位置**: `scripts/train_agent.py:316`, `save_dir = OUT_DIR / args.domain`
- **症状**: 批量训练 `math_lr1e-4_s0, math_lr1e-4_s1, math_lr1e-4_s2` 全部保存到同一个目录 `experiments/trained_models/math/`。第二、第三个训练会静默覆盖第一个训练的模型。最终只有一个模型存活，丢失 n-1 个 seed 的全部训练结果。**比 Bug 22 更隐蔽——不会报错，不会崩溃，但所有 seed 数据都是假的。**
- **根因**: `train()` 函数只有 `save_dir = OUT_DIR / args.domain`，没有 `lr` 和 `seed` 参数。现有模型目录 `{domain}_lr{lr}_s{seed}/` 是训练后手动 `mv` 改名的——新训练的模型必须先保存再改名，容易遗漏。
- **修复**: (a) 新增 `--output-dir` CLI 参数，覆盖默认保存路径；(b) 批量脚本 `phase1_batch.sh` 传入 `--output-dir experiments/trained_models/{domain}_lr{lr}_s{seed}/`，不再依赖手动改名。
- **教训**: 训练脚本的输出路径必须在启动时确定且唯一。不要依赖"跑完再改名"的后处理步骤——批量训练时必然会忘记或交错。如果将来要做 grid search，应在 `train()` 内部自动生成 `{domain}_lr{lr}_s{seed}/` 路径。

### Bug 22: code_e1 是 base 模型 — 所有实验用了未训练的 code agent — 2026-07-05
- **位置**: `experiments/trained_models/code_e1/`
- **症状**: code_e1 与 base 完全一致（291/291 tensors, max_diff < 1e-8, ||Δ||/||base|| = 0.000000）。E2-E7 所有实验的"code agent"实际上是未训练的 Pythia-1.4B base。
- **影响范围**: 
  - LMC barrier 扫描（E8）实测的是 base Pythia ⇄ medical-trained Pythia，不是 code ⇄ medical
  - 所有 c1mX 实验结果不代表"两个 domain-specific 模型之间的连通性"
  - E1（Jul 3）有过真正的 code 模型（B→code=0.768），已被后续 pipeline 覆盖丢失
- **根因**: Jul 5 pipeline 的 train_agent.py 在保存时触发了 Bug 18（epoch 2+ 退化）和 Bug 21（保存前无 Δ 验证），导致训练完成的模型被 base 覆盖。code_e1 的 checkpoint 保存和 best-model 保存可能指向了同一个被污染的 state_dict。
- **修复**: 需要从头训练 code agent。训练完成后立即备份 checkpoint。验证 Δ > 0 再标记为有效。
- **教训**: 每次训练完成后必须验证保存的模型确实改变了（与 base 做 diff）。不能假设"日志显示训练成功 = 模型保存正确"。训练和实验之间插入验证步骤。

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
