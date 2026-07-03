# AFP 项目重启指南

> 新会话粘贴此文件即可继续。

---

## 你是谁

`claude-code`，运行在 DGX Spark GB10 (ARM64, 121GB 统一内存, CUDA 13.0)。

## 项目路径

`/home/jiayu/AFP`

## 当前进度（2026-07-04）

### Phase 0：已完成

Pythia-1.4B full-FT on VersaPRM code + medical，三种 importance 方法全部跑完：

| 方法 | Cosine | IVN vs FedAvg | 结论 |
|------|--------|---------------|------|
| MAS E[|g|] | 0.996 | = | 梯度衰减是架构属性 |
| STA |g·ΔW|| 0.994 | = | ΔW∝g，退化到 Fisher |
| L2 rel ||ΔW||/||W_base|| | 0.991 | = | 区分度仅 0.009 |

**结论：同架构 full-FT 下，importance cosine 必然 ≈ 1.0，AFP gate = FedAvg。** 理论解释见 LMC 文献。

### Phase 0.5：LMC Barrier（进行中）

**实验：** 测量两个模型间线性插值的 loss barrier。
- Group A（分化小）：code 1ep vs medical 1ep — **正在跑中**
- 结果输出：`experiments/lmc_trained/barrier_1ep.json`
- 评估脚本：`scripts/lmc_barrier.py`

**如果 barrier > 0：** 说明 LMC 在 1ep 就已不成立，实验可行。
**如果 barrier ≈ 0：** 需要更多 epoch 训练扩大分化。

---

## 环境

| 事项 | 值 |
|------|-----|
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| torch | 2.12.1+cu130 |
| transformers | 5.12.1 |
| datasets | 5.0.0 |
| 设备 | DGX Spark GB10, CUDA 13.0, ARM64, 121GB |
| Remote | `https://github.com/FauReam/AFP.git` |

## 关键路径

| 路径 | 内容 |
|------|------|
| `experiments/trained_models/code/` | code domain 1ep 训练好的模型 |
| `experiments/trained_models/medical/` | medical domain 1ep 训练好的模型 |
| `experiments/phase0_ivn/ivn_results.json` | Phase 0 最终结果 |
| `experiments/lmc_trained/` | LMC 实验输出 |
| `data/versaprm/` | Tokenized VersaPRM 数据 |

## 启动命令

```bash
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
VENV=/home/jiayu/AFP/venv/bin/python3

# LMC Barrier 评估（现有模型）
nohup $VENV scripts/lmc_barrier.py \
    --model-a experiments/trained_models/code \
    --model-b experiments/trained_models/medical \
    --domain-a code --domain-b medical \
    > experiments/lmc_trained/barrier.log 2>&1 &

# 多 epoch 训练（如需更大分化）
nohup $VENV scripts/train_agent.py --domain code --epochs 5 \
    --save-every-n-epochs 1 > experiments/lmc_trained/train_code_e5.log 2>&1 &

# IVN 实验（如需重新验证）
nohup $VENV scripts/run_ivn_phase0.py \
    --teacher EleutherAI/pythia-1.4b --student EleutherAI/pythia-1.4b \
    --base-model EleutherAI/pythia-1.4b \
    --teacher-domain code --student-domain medical \
    --weights experiments/trained_models \
    --importance magnitude_l2 --gate rational --tau 0.5 --max-rounds 30 \
    > experiments/phase0_ivn/run.log 2>&1 &
```

## 网络说明

无 VPN。hf-mirror.com 有时 SSL 证书异常，重试即可。直连 HF 超时。

## Bug 清单（20 条）

| Bug | 描述 | 状态 |
|-----|------|------|
| 1-7 | FCL-PRM 旧项目工程问题 | ✅ 见 ENGINEERING.md |
| 8 | gate 函数未导出 | ✅ |
| 9 | AutoTokenizer 无 local_files_only → HF 超时 | ✅ |
| 10 | N_BLOCKS=24 不支持 Qwen2.5 28 层 | ✅ |
| 11 | MAS per-param .item() → 10,416 次 GPU sync | ✅ |
| 12 | MAS 在 CPU 上运行（to_device 太晚） | ✅ |
| 13 | run_ivn_phase0.py 默认 Qwen2.5 非 Pythia | ✅ |
| 14 | train_agent.py 超参偏离 CLAUDE.md | ✅ |
| 15 | 训练缓存不绑定 MAX_LEN | ✅ |
| 16 | train_agent.py SyntaxError — global 声明顺序 | ✅ |
| **17** | eval n_test=15k → GB10 25min/eval | ✅ n_test 上限 1000 |
| **18** | F-IVN head 缺 h[:,-1,:] → shape mismatch crash | ✅ |
| **19** | Labels {-1,+1} 未转 {0,1} → 所有评估错误 | ✅ |
| **20** | state_dict() 在 dict comprehension 调用 582 次 | ✅ |

## 下载源

模型和数据集优先用 hf-mirror。如 SSL 持续失败：
- PyPI: `-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`
- ModelScope: `pip install modelscope` → `snapshot_download()`
- COCO 原始 HTTP: `http://images.cocodataset.org`

## 核心文档

- `docs/internal/IMPORTANCE_ANALYSIS.md` — 为什么梯度重要性方法结构失效
- `docs/internal/ROOT_CAUSE_IMPORTANCE.md` — 根因分析（共同模式问题 + LMC 视角）
- `docs/internal/REFERENCES.md` — LMC 文献 (Frankle 2020, Entezari 2022, Git Re-Basin 2023)
- `docs/internal/ENGINEERING.md` — 工程手册
- `docs/internal/EXPERIMENT_PLAN.md` — Phase 0 实验方案

## 训练速度参考（DGX Spark GB10）

| 配置 | 时间 |
|------|------|
| Pythia-1.4B full-FT 1 epoch | ~3h |
| Pythia-1.4B full-FT 5 epoch | ~15h |
| 单次 eval (500 samples) | ~3 min |
| LMC barrier 21λ × 2 domain | ~12 min |
