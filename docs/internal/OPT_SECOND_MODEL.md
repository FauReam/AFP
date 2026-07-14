# Second Model Family Plan — OPT-1.3B

> **Status:** Pythia-1.4B experiments complete (v13 paper). Start OPT-1.3B after theory exps finish.
> **Principle:** Never modify Pythia scripts directly. OPT gets its own parallel pipeline.

---

## Rationale

Pythia-1.4B gives us strong evidence that training stability determines LMC barrier height. To prove this is not a Pythia-specific phenomenon, we replicate on OPT-1.3B (Meta, GPT-2 architecture, ~1.3B params).

## Architecture Differences

| Property | Pythia-1.4B | OPT-1.3B |
|----------|-------------|----------|
| Architecture | GPT-NeoX (parallel attention) | GPT-2 (sequential) |
| Hidden dim | 2048 | 2048 |
| Layers | 24 | 24 |
| Vocab | 50304 | 50272 |
| Tokenizer | NeoX (GPT-NeoX) | GPT-2 |
| PRM head compatibility | ✅ Same hidden dim | ✅ Same hidden dim |
| Custom code needed | No | No |
| Download | HF cache (~2.8GB) | HF cache (~5GB, downloaded ✅) |

## Code Separation

```
scripts/
  train_agent.py            ← Pythia original (DO NOT MODIFY)
  lmc_barrier_scan.py       ← Pythia original
  lmc_3pt_scan.py           ← Pythia original
  lmc_barrier.py            ← Pythia original
  
  pythia/                   ← Pythia backups (frozen copies)
    train_agent.py
    lmc_barrier_scan.py
    lmc_3pt_scan.py
    lmc_barrier.py
  
  opt/                      ← NEW: OPT-specific scripts
    train_agent.py          ← modified from Pythia version
    lmc_barrier_scan.py     ← modified from Pythia version
    lmc_3pt_scan.py         ← modified from Pythia version
```

## What Needs to Change for OPT

### Minimal Changes (to existing scripts)

| File | Change | Reason |
|------|--------|--------|
| `train_agent.py` | `--model-id` default → `facebook/opt-1.3b` | Different model |
| `train_agent.py` | Tokenizer call → GPT-2 tokenizer | Different tokenizer |
| `train_agent.py` | Cache naming → `_opt_` instead of `_pythia_` | Separate data caches |
| `lmc_barrier_scan.py` | `--model-id` default → `facebook/opt-1.3b` | Different model |
| `lmc_3pt_scan.py` | `--model-id` default → `facebook/opt-1.3b` | Different model |
| `AFPAgent` | `model_id` parameter already supports override ✅ | No change |

### Better: Parameterize, Don't Duplicate

Instead of copying all scripts, add `--model-id` to the existing scripts where missing and use it everywhere. The Pythia defaults stay; OPT runs pass `--model-id facebook/opt-1.3b`.

## Experiment Plan (Minimal Viable Replication)

Replicate the CORE finding only — not all 50+ experiments.

| # | Experiment | Seeds | GPU-hours |
|---|-----------|:---:|:---:|
| 1 | Train code + medical (standard div) | 3 each | ~3h |
| 2 | Cross-domain LMC (code↔med) | 3 scans | ~0.75h |
| 3 | Within-domain LMC (code + medical) | 3 pairs each | ~1.5h |
| 4 | Per-block divergence analysis | 1 analysis | 0 |
| **Total** | | | **~5.25 GPU-hours** |

### Success Criteria

| Metric | Pythia-1.4B | OPT-1.3B must show |
|--------|:---:|------|
| Code within-domain barrier | 0.048 | ~0.05 (low, stable) |
| Medical within-domain barrier | 0.147 | > code within (unstable) |
| Code within ≈ code↔med cross | Yes | Same pattern |
| Per-block r (code vs med) | 0.995 | >0.95 |

If OPT replicates these 4 patterns, the thesis generalizes across architectures. If not, we have an interesting architecture-dependent finding.

## When to Start

After theory experiments (A+B+C) complete and results are written into paper v14.
