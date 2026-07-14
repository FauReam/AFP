# Appendix — Supplementary Material

> For: "Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning"
> v7, 2026-07-14

---

## A. Complete Barrier Data (All Experiments)

### A.1 Cross-Domain LMC Barriers (Standard Divergence, lr=1e-4)

| Seed | Code barrier | Medical barrier |
|:---:|:---:|:---:|
| 0 | 0.047 | 0.066 |
| 1 | 0.068 | 0.035 |
| 2 | 0.044 | 0.052 |
| **Mean ± σ** | **0.053 ± 0.011** | **0.051 ± 0.013** |

### A.2 Cross-Domain LMC Barriers (High Divergence, lr=5e-4)

| Seed | Code barrier | Medical barrier |
|:---:|:---:|:---:|
| 0 | 0.135 | 0.339 |
| 1 | 0.144 | 0.094 |
| 2 | 0.075 | 0.252 |
| **Mean ± σ** | **0.118 ± 0.031** | **0.228 ± 0.102** |

### A.3 Cross-Domain LMC Barriers (Intermediate Divergence)

| lr | ΔW range | Seed | Code barrier | Medical barrier |
|:---:|:---:|:---:|:---:|:---:|
| 2e-4 | 3.0-3.8% | 0 | 0.073 | 0.312 |
| 2e-4 | 3.0-3.8% | 1 | 0.111 | 0.049 |
| 2e-4 | 3.0-3.8% | 2 | 0.054 | 0.118 |
| 3e-4 | 4.6-5.2% | 0 | 0.096 | 0.050 |
| 3e-4 | 4.6-5.2% | 1 | 0.056 | 0.719 |
| 3e-4 | 4.6-5.2% | 2 | 1.043 | 0.027 |

### A.4 Within-Domain LMC Baselines (Standard Divergence)

| Domain | Pair | Barrier (own-domain eval) |
|--------|:---:|:---:|
| Code | s0-s1 | 0.048 |
| Code | s0-s2 | 0.048 |
| Code | s1-s2 | 0.049 |
| Code | **Mean ± σ** | **0.048 ± 0.000** |
| Medical | s0-s1 | 0.109 |
| Medical | s0-s2 | 0.157 |
| Medical | s1-s2 | 0.173 |
| Medical | **Mean ± σ** | **0.147 ± 0.027** |

### A.5 Within-Domain LMC Baselines (High Divergence)

| Domain | Pair | Barrier (own-domain eval) |
|--------|:---:|:---:|
| Code | s0-s1 | 0.081 |
| Code | s0-s2 | 0.173 |
| Code | s1-s2 | 0.151 |
| Code | **Mean ± σ** | **0.135 ± 0.038** |
| Medical | s0-s1 | 0.207 |
| Medical | s0-s2 | 0.072 |
| Medical | s1-s2 | 0.183 |
| Medical | **Mean ± σ** | **0.154 ± 0.059** |

### A.6 Noise Floor Calibration

| Condition | Code barrier | Medical barrier |
|-----------|:---:|:---:|
| Identical copy | 0.000 | 0.000 |
| Random init (seed 0) | 0.034 | 0.176 |
| Random init (seed 1) | 0.033 | 0.143 |
| Random init (seed 2) | 0.033 | 0.131 |
| **Random init Mean ± σ** | **0.033 ± 0.001** | **0.150 ± 0.019** |

### A.7 Gaussian Perturbation Calibration

| ΔW | Code barrier | Medical barrier |
|:---:|:---:|:---:|
| 0.5% | 0.003 | 0.003 |
| 1.0% | 0.004 | 0.002 |
| 1.5% | 0.006 | 0.001 |
| 2.0% | 0.005 | 0.002 |
| 4.0% | 0.011 | 0.001 |
| 8.0% | 0.014 | 0.003 |

### A.8 Layer-Selective Interpolation

| Merged layers | Code barrier | Medical barrier |
|:---|:---:|:---:|
| Early (0-7) | 0.040 | 0.035 |
| Mid (8-15) | 0.003 | 0.014 |
| Late (16-23) | 0.000 | 0.004 |
| All (0-23) | 0.053 | 0.051 |

---

## B. Training Trajectory Complete Data

### B.1 Code Domain Trajectory

| Step | ΔW (%) | bar_code | L(0) | L(1) |
|:---:|:---:|:---:|:---:|:---:|
| 40 | 0.31 | 0.029 | 0.757 | 0.699 |
| 80 | 0.58 | 0.027 | 0.757 | 0.703 |
| 120 | 0.82 | 0.037 | 0.757 | 0.683 |
| 160 | 1.01 | 0.043 | 0.757 | 0.671 |
| 200 | 1.13 | 0.043 | 0.757 | 0.671 |
| 240 | 1.22 | 0.043 | 0.757 | 0.671 |
| 280 | 1.30 | 0.038 | 0.757 | 0.681 |
| 320 | 1.36 | 0.034 | 0.757 | 0.689 |
| 360 | 1.40 | 0.035 | 0.757 | 0.688 |
| 400 | 1.41 | 0.033 | 0.757 | 0.692 |

### B.2 Medical Domain Trajectory

| Step | ΔW (%) | bar_med | L(0) | L(1) |
|:---:|:---:|:---:|:---:|:---:|
| 40 | 0.35 | 0.173 | — | — |
| 80 | 0.63 | 0.173 | — | — |
| 120 | 0.87 | 0.151 | — | — |
| 160 | 1.07 | 0.187 | — | — |
| 200 | 1.20 | 0.199 | — | — |
| 240 | 1.31 | 0.202 | — | — |
| 280 | 1.40 | 0.213 | — | — |
| 320 | 1.48 | 0.218 | — | — |
| 360 | 1.54 | 0.208 | — | — |
| 400 | 1.59 | 0.211 | — | — |

---

## C. Weight Divergence Complete Data

### C.1 All Trained Model ΔW Values

| Domain | lr | Seed | ΔW (%) |
|--------|:---:|:---:|:---:|
| code | 1e-4 | s0 | 1.39 |
| code | 1e-4 | s1 | 1.32 |
| code | 1e-4 | s2 | 1.43 |
| code | 2e-4 | s0 | 3.84 |
| code | 2e-4 | s1 | 3.05 |
| code | 2e-4 | s2 | 3.09 |
| code | 3e-4 | s0 | 4.58 |
| code | 3e-4 | s1 | 4.59 |
| code | 3e-4 | s2 | 4.65 |
| code | 5e-4 | s0 | 8.28 |
| code | 5e-4 | s1 | 7.62 |
| code | 5e-4 | s2 | 8.11 |
| medical | 1e-4 | s0 | 1.59 |
| medical | 1e-4 | s1 | 1.49 |
| medical | 1e-4 | s2 | 1.47 |
| medical | 2e-4 | s0 | 3.22 |
| medical | 2e-4 | s1 | 3.57 |
| medical | 2e-4 | s2 | 3.16 |
| medical | 3e-4 | s0 | 5.22 |
| medical | 3e-4 | s1 | 4.72 |
| medical | 3e-4 | s2 | 4.68 |
| medical | 5e-4 | s0 | 8.79 |
| medical | 5e-4 | s1 | 8.47 |
| medical | 5e-4 | s2 | 8.36 |

### C.2 Newly Trained Models

| Domain | lr | Seed | ΔW (%) |
|--------|:---:|:---:|:---:|
| math | 1e-4 | s0 | 1.04 |
| math | 1e-4 | s1 | 0.88 |
| math | 1e-4 | s2 | 0.80 |
| general | 1e-4 | s0 | 0.95 |
| general | 1e-4 | s1 | 0.88 |
| general | 1e-4 | s2 | — |
| medical | 5e-4 | s3 | 8.80 |
| medical | 5e-4 | s4 | — |

---

## D. Bootstrap Confidence Intervals

100,000 resamples, 95% CI, n=3 seeds per condition.

| Condition | Mean | 95% CI | Range |
|-----------|:---:|:---:|:---:|
| Standard code | 0.053 | [0.044, 0.068] | 0.024 |
| Standard med | 0.051 | [0.035, 0.066] | 0.031 |
| High code | 0.118 | [0.075, 0.144] | 0.069 |
| High med | 0.228 | [0.094, 0.340] | 0.246 |
| Within code | 0.048 | [0.048, 0.049] | 0.001 |
| Within med | 0.147 | [0.109, 0.173] | 0.064 |

---

## E. Hardware and Reproducibility

| Item | Detail |
|------|--------|
| GPU | NVIDIA DGX Spark GB10, Blackwell SM 12.1 |
| CUDA | 13.0 |
| Memory | 121 GB unified (CPU+GPU) |
| CPU | ARM64 (Cortex-X925 + A725) |
| Python | 3.12 |
| PyTorch | 2.12.1 |
| Transformers | 5.12.1 |
| Training time | ~42 min / model (Pythia-1.4B full-FT) |
| LMC scan time | ~15 min / scan (11 α × 2 domains × 2000 samples) |
| Total GPU-hours | ~80 (all training + all LMC scans) |

All experiments use `HF_DATASETS_OFFLINE=1` with `local_files_only=True` — zero network access during training and evaluation. Raw JSON loss data for all 50+ LMC scans is preserved in `experiments/phase0_ivn/results/`, enabling independent barrier recomputation via the Frankle formula. All 24+ trained model checkpoints are available with verified ΔW > 0.1%.

---

## F. Bug Register (Engineering Context)

The project maintains a 23-entry engineering log at `docs/internal/ENGINEERING.md`. Key bugs affecting data integrity were:

- **Bug 22**: code_e1 checkpoint was identical to base model (detected by ΔW verification)
- **Bug 23**: train_agent.py saved all runs to same directory, silently overwriting models
- **Bug 21**: training saved base model instead of trained model (no ΔW guard)

All three were discovered and fixed before any reported measurements were taken. Every model file in `experiments/trained_models/` has been independently verified as genuinely fine-tuned (ΔW > 0.1%) by at least two independent reviewers during the expert panel process.
