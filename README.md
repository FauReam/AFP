# Training Stability, Not Domain Difference, Determines Linear Mode Connectivity

**Pythia-1.4B / OPT-1.3B — Domain Fine-Tuning → LMC Barrier Analysis**

---

When a pretrained LM is fine-tuned on different domains, how far do the resulting models move in weight space? Do they remain linearly connected? This project measures the quantitative relationship between weight divergence and LMC barrier height.

- Paper: `docs/internal/paper.tex` (LaTeX v19, 20 pages) or `paper.pdf` (compiled)
- Markdown draft: `docs/internal/PAPER.md`

## Core Finding

> **Training stability, not domain difference, is the dominant driver of LMC barrier height — and stability varies dramatically across domains.**

| Condition | ΔW | Code barrier | Medical barrier |
|-----------|-----|:---:|:---:|
| Standard FT | 1.4% | 0.053 ± 0.011 | 0.051 ± 0.013 |
| High divergence | 8.0% | 0.118 ± 0.031 | 0.228 ± 0.102 |
| Within-domain | — | 0.048 (code) | **0.147** (medical) |
| Noise floor | — | ~0.000 (identical) | 0.150 (random init) |

Key asymmetry: medical within-domain barrier (0.147) is 3× larger than cross-domain barrier (0.051). Code shows no such gap (0.048 ≈ 0.053). Domain difference alone does not create barriers — training instability does.

## Cross-Architecture Replication

OPT-1.3B confirms the pattern: medical/code within-domain barrier ratio is 3.6× (vs Pythia's 3.7×). Absolute barriers are 4-8× larger, but relative ordering holds. GPT-Neo replication pending.

## Entry Points

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

# Train a domain-specialized model
python -u scripts/train_agent.py --domain code --lr 1e-4 --model-id EleutherAI/pythia-1.4b

# Run LMC barrier scan (11-point linear interpolation)
python -u scripts/lmc_barrier_scan.py

# Batch experiment pipelines
bash scripts/phase1_batch.sh      # ICLR sprint: LR sweeps, within-domain, baselines
bash scripts/final_batch.sh        # Merge benchmark, OPT trajectory, layer-selective
bash scripts/theory_experiments.sh # Label noise, extended putt, Hessian eigenvalues
```

## Environment

| Item | Value |
|------|-------|
| Hardware | NVIDIA DGX Spark GB10, 121GB unified memory, ARM64 CUDA 13.0 |
| Models | EleutherAI/pythia-1.4b, facebook/opt-1.3b |
| Training | full-FT, bf16, L=256, batch=128, 1-2 epochs, drive-putt LR schedule |
| Python | `/home/jiayu/AFP/venv/bin/python3` |
| Training speed | ~42 min/model (Pythia-1.4B, 1 epoch) |

## Key Documents

| File | Content |
|------|---------|
| `docs/internal/paper.tex` | LaTeX source v19 (authoritative) |
| `paper.pdf` | Compiled PDF (20 pages) |
| `docs/internal/PAPER.md` | Markdown mirror of paper |
| `docs/internal/APPENDIX.md` | Supplementary data tables |
| `docs/internal/ENGINEERING.md` | 23 documented bugs + training code standards |
| `docs/internal/EXPERT_PANEL_FINDINGS.md` | 6-expert adversarial review |
| `docs/internal/ICLR_SPRINT_PLAN.md` | ICLR 2028 3-phase sprint plan |
| `docs/internal/DATA_INVENTORY.md` | Complete data inventory (42+ models, 110+ scans) |
| `experiments/RESTART_PROMPT.md` | New session restart guide |

## Project Structure

```
AFP/
├── docs/internal/         # Paper, engineering manual, sprint plan, expert review
├── docs/reports/          # Figures (PDF vector), tutorial, literature survey
├── scripts/               # Training, LMC scanning, analysis, batch pipelines
│   ├── train_agent.py     # Full-FT with drive-putt LR schedule
│   ├── lmc_barrier_scan.py # 11-point LMC barrier measurement
│   ├── merge_benchmark.py  # TIES / Task Arithmetic / LayerSelect
│   └── opt/               # OPT-1.3B specific scripts
├── src/AFP/               # Protocol library (agent, integrator, trust)
├── experiments/
│   ├── trained_models/     # Pythia-1.4B checkpoints (288 GB, 240 .pt files)
│   ├── trained_models_opt/ # OPT-1.3B checkpoints (184 GB)
│   ├── phase0_ivn/results/ # 122 LMC barrier JSONs (primary data)
│   └── expert_panel/       # 6 expert review reports
└── data/versaprm/          # Tokenized training data (2.1 GB)
```
