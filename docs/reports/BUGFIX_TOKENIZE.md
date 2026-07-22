# 🩹 Bug Fix Report: GPT-Neo Tokenize 内存爆炸修复

> **日期**: 2026-07-23
> **影响文件**: `scripts/gptneo_pipeline.sh`, `scripts/train_agent.py`
> **目标**: 峰值内存 ≤ 80GB（121GB 统一内存 DGX Spark GB10）
> **严重度**: 🔴 致命 — 原始代码在 >1M 步骤时 OOM 超过 121GB

---

## 1. 总览

原始代码在两个地方同时把内存打爆：

| 位置 | 数据结构 | 每 1000 万步骤的内存 |
|------|--------|-------------------|
| `gptneo_pipeline.sh` inline Python | `samples: list[dict]` | ~5 GB（纯 dict 开销） |
| `gptneo_pipeline.sh` inline Python | `ids: list[tensor(256)]` | ~23 GB（N 个独立 tensor 对象） |
| `train_agent.py` `prepare_data()` | `input_ids_list: list[list[int]]` | ~72 GB（Python int × 256 × N） |
| `train_agent.py` `prepare_data()` | `inp: tensor(N, 256)` | ~20 GB |

**叠加峰值**: 5 + 23 + 72 + 20 = **~120 GB**，加上模型训练 (~66 GB) 远超 121 GB → OOM。

修复策略：零中间对象 + 硬内存上限 + clone 后释放全量 tensor。

---

## 2. 修复清单

### 🔴 Bug 1 — 逐条 tokenize + `return_tensors='pt'` + `torch.stack`

**位置**: `gptneo_pipeline.sh` inline Python（原始）

**问题**: 每个 tokenize 调用返回 `(1,256)` tensor → N 个 tensor 堆积 → `torch.stack` 再拷贝。
峰值 ~3-5x 最终缓存。

**修复**: `batch_encode_plus(return_tensors=None)` 返回 Python list，写入一次性预分配 tensor：
```python
inp = torch.full((total, MAXL), PAD, dtype=torch.long)   # ← 一次性预分配
enc = tok.batch_encode_plus(b_texts, ..., return_tensors=None)  # ← Python list
inp[kept, :L] = torch.tensor(enc['input_ids'][k][:L])    # ← 直接写入
```

---

### 🔴 Bug 2 — JSONL 全文加载为 Python dict

**位置**: `gptneo_pipeline.sh` inline Python（原始）

**问题**: `samples.append(d)` 把每个 JSONL 行存为完整 dict（~300 bytes 开销 + 所有字段）。

**修复**: 只存 `(text, label)` tuple：
```python
RAW[dom].append((f'{q}\n{step_txt}', 1.0 if int(lbl) == 1 else 0.0))
del d  # ← 立即释放 dict，不保留引用
```

---

### 🔴 Bug 3 — `train_agent.py` `prepare_data()` list-of-lists 内存膨胀

**位置**: `scripts/train_agent.py:73-113`（原始）

**问题**: `input_ids_list.append(ids)` 把每个 tokenize 结果存为 `list[int]`。
Python int = 28 bytes，tensor int64 = 8 bytes → **3.5x 膨胀**。

**修复**: 双遍扫描——第一遍只计数 `lens: list[int]`（每个步骤 1 个 Python int），
第二遍直接写入预分配 tensor：
```
Pass 1: lens.append(len(ids))  →  N × 28 bytes (轻量)
Pass 2: inp[idx, :L] = torch.tensor(ids[:L])  →  直接写入，无中间 list
```

---

### 🔴 Bug 4 — Split 后全量 tensor 不释放（训练阶段 OOM）

**位置**: `scripts/train_agent.py:155-158`（原始和初版修复）

**问题**: `train["input_ids"] = inp[train_idx]` 是 **view**，`inp` 的全量存储不会释放。
训练时 model (~66GB) + 全量 data tensor (~20GB) = ~86GB > 80GB。

**修复**: `clone()` + `del` 全量 tensor：
```python
train = {"input_ids": inp[train_idx].clone(), ...}  # ← clone 断开引用
val   = {"input_ids": inp[val_idx].clone(), ...}
del inp, mask, labs, perm, lens  # ← 释放全量 tensor
gc.collect()
```

---

### 🛡️ 保护 1 — 硬内存上限（pipeline）

**位置**: `gptneo_pipeline.sh` inline Python

```python
MAX_STEPS_PER_DOMAIN = 8_000_000  # ~18.4 GB tensor
if n > MAX_STEPS_PER_DOMAIN:
    # random subsample（seed=42 可复现）
    keep = set(sorted(random.sample(range(n), MAX_STEPS_PER_DOMAIN)))
    pairs = [p for i, p in enumerate(pairs) if i in keep]
```

---

### 🛡️ 保护 2 — 硬内存上限（train_agent）

**位置**: `scripts/train_agent.py:93-105`

```python
MAX_DATA_GB = 30  # tensor budget
est_gb = n * BYTES_PER_STEP / 1e9
if est_gb > MAX_DATA_GB:
    cap_n = int(MAX_DATA_GB * 1e9 / BYTES_PER_STEP)
    # random subsample
```

---

### 🛡️ 保护 3 — `/proc/meminfo` 实时监控

两个脚本均在以下节点读取 `MemAvailable` 并打印日志：

1. 启动时
2. JSONL scan 完成后
3. 每个 domain tensor 分配前
4. 每个 domain 处理完成后
5. 结束前

若 `MemAvailable < 10 GB`，打印 `!! LOW MEM !!` 警告。

---

### 🛡️ 保护 4 — 每阶段强制 GC

```python
del inp, msk, labs; gc.collect()    # ← 每个 domain 保存后立即释放 tensor
del RAW[dom]; gc.collect()           # ← 每个 domain 处理后释放 raw data
```

---

## 3. 修复效果对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| `input_ids_list` (1000 万步) | **72 GB** (Python ints) | **0**（不存在） |
| 中间 tensor 对象 | N 个独立 tensor | **0** |
| 峰值内存（tokenize） | >121 GB (OOM) | **<25 GB** |
| 峰值内存（训练） | >121 GB (OOM) | **<80 GB**（data clone + cap） |
| JSONL 读取次数 | 4 次 | **1 次** |
| 训练次数/模型 | 2 次 | **1 次** |
| 硬上限 | 无 | **800 万步/domain（pipeline）30 GB（train_agent）** |
| 内存监控 | 无 | **每阶段 `/proc/meminfo`** |

---

## 4. 内存预算精算

DGX Spark GB10 — 121 GB 统一内存。目标 ≤ 80 GB。

```
Tokenize 阶段（无模型加载）:
  800 万步 tensor:  18.4 GB
  RAW text tuples:  ~5 GB (est.)
  其他开销:         ~2 GB
  ─────────────────────
  峰値:             ~25 GB  ✓

训练阶段（tokenize 后，模型加载前）:
  训练数据 tensor:   ≤18.4 GB (clone 后)
  验证数据 tensor:   ≤3 GB
  模型 + optimizer:  ~31 GB (batch=128, bf16)
  其他开销:          ~2 GB
  ─────────────────────
  峰値:             ~54 GB  ✓
```

修改 `MAX_STEPS_PER_DOMAIN` 和 `MAX_DATA_GB` 可调整预算分配。

---

## 5. 关键规则（给 AI 参考）

1. **绝不用** `tokenizer(return_tensors='pt')` 逐条 tokenize——每个返回一个 tensor 对象
2. **绝不用** `list.append(enc['input_ids'][0])` 攒 tensor——N 个 tensor 炸内存
3. **绝不用** `list[dict]` 存全量 JSONL——dict 开销 ~300 bytes/条
4. **必须用** `batch_encode_plus(return_tensors=None)` 返回 Python list
5. **必须用** `tokenizer.encode(text, truncation=True, max_length=...)` 
6. **必须** 在 split 后 `.clone()` + `del` 全量 tensor
7. **必须** 在循环中 `del big_var; gc.collect()`
8. 训练数据 tensor 用 `dtype=torch.long`（int64），因为词表可能 > 32767

---

## 6. 变更文件

```
M scripts/gptneo_pipeline.sh   — 流式 tokenize + batch_encode + 硬上限 + mem 监控
M scripts/train_agent.py       — 双遍扫描 + 预分配 + clone 释放 + 硬上限 + mem 日志
M docs/reports/BUGFIX_TOKENIZE.md — 本文档
```

---

*Generated for the experiment-machine AI.*
