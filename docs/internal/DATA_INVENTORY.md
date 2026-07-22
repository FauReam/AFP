# Complete Data Inventory — AFP Phase 0 LMC Paper

> v19 — 2026-07-19 (data current as of paper v19)
> All data verified via Frankle formula recomputation from raw JSON

---

## 1. Core Measurements (Pythia-1.4B, 1-epoch)

### 1.1 Weight Divergence
| Condition | Code ΔW | Medical ΔW | Code↔Med Cross |
|-----------|:---:|:---:|:---:|
| Standard | 1.4 ± 0.0% | 1.5 ± 0.1% | 2.0 ± 0.1% |
| High | 8.0 ± 0.3% | 8.5 ± 0.2% | 11.6 ± 0.3% |
| Intermediate (lr=2e-4) | ~3.0-3.8% | ~3.2-3.6% | — |
| Intermediate (lr=3e-4) | ~4.6-5.2% | ~4.7-5.2% | — |
| **Source:** `experiments/trained_models/{domain}_lr{lr}_s{seed}/W_{domain}_final.pt` |
| **Models trained:** 24 (2 domains × 4 LRs × 3 seeds) |

### 1.2 LMC Barriers (Cross-Domain)
| Condition | Code barrier | Medical barrier | Seeds |
|-----------|:---:|:---:|:---:|
| Standard | 0.053 ± 0.011 | 0.051 ± 0.013 | 3 |
| High | 0.118 ± 0.031 | 0.228 ± 0.102 | 3 |
| **Source:** `experiments/phase0_ivn/results/lmc_lr{lr}_s{seed}.json` |

### 1.3 Within-Domain LMC (4 Domains)
| Domain | Barrier | Pairs |
|--------|:---:|:---:|
| Code | 0.048 ± 0.000 | 3 |
| General | 0.071 ± 0.007 | 3 |
| Math | 0.087 ± 0.032 | 3 |
| Medical | 0.147 ± 0.027 | 3 |
| **Source:** `experiments/phase0_ivn/results/lmc_{domain}_s{s1}_s{s2}.json` |

### 1.4 Seed-Pair Analysis (Medical High-Div, n=10 pairs)
| Pairs | Barrier range | Mean |
|-------|:---:|:---:|
| s0-s1 to s3-s4 | 0.072 — 1.213 | 0.304 |
| **Key:** s4↔s1 = 1.213 (catastrophic), s4↔s2 = 0.080 (normal) |
| **Source:** `experiments/phase0_ivn/results/lmc_medical_high_s{s1}_s{s2}.json` |

---

## 2. Mechanistic Experiments (Pythia-1.4B)

### 2.1 Training Trajectory (1-epoch, 11-point scans)
| Domain | Steps | Barrier range | Shape |
|--------|:---:|:---:|------|
| Code | 10 (40-400) | 0.027 — 0.043 | Inverted-U (peak step 200) |
| Medical | 10 (40-400) | 0.151 — 0.218 | Monotonic growth |
| **Source:** `experiments/phase0_ivn/results/lmc_traj_{domain}_step{step}.json` |

### 2.2 Gaussian Perturbation Calibration
| ΔW | Barrier |
|:---:|:---:|
| 0.5% | 0.003 |
| 1.0% | 0.004 |
| 2.0% | 0.005 |
| 4.0% | 0.011 |
| 8.0% | 0.014 |
| **Source:** `experiments/phase0_ivn/results/noise_gaussian_dw{dw}.json` |

### 2.3 Layer-Selective Interpolation (Cross-Domain)
| Merged layers | Code barrier | % of full |
|:---|---:|---:|
| Early (0-7) | 0.040 | 75% |
| Mid (8-15) | 0.003 | 6% |
| Late (16-23) | 0.000 | 0% |
| All | 0.053 | 100% |
| **Source:** `experiments/phase0_ivn/results/lmc_layers_{layer}.json` |

### 2.4 Layer-Selective Interpolation (Within-Domain)
| Domain | Early | Mid | Late | All |
|--------|:---:|:---:|:---:|:---:|
| Code | 0.026 (87%) | 0.002 | 0.001 | 0.030 |
| Medical | 0.020 (77%) | 0.001 | 0.002 | 0.026 |
| **Source:** `experiments/phase0_ivn/results/lmc_layers_within_{domain}_{layer}.json` |

### 2.5 Noise Floor
| Condition | Code barrier | Medical barrier |
|-----------|:---:|:---:|
| Identical copy | ~0.000 | ~0.000 |
| Random init (3 seeds) | 0.033 ± 0.001 | 0.150 ± 0.019 |
| **Source:** `experiments/phase0_ivn/results/noise_identical.json`, `noise_random_s{seed}.json` |

### 2.6 Cross-Domain Asymmetry
| Model | loss_code | loss_med |
|-------|:---:|:---:|
| Base Pythia | 0.757 | 0.423 |
| Code-trained (mean 3 seeds) | 0.599 | 0.584 |
| Medical-trained (mean 3 seeds) | 0.686 | 0.663 |
| **Source:** `experiments/phase0_ivn/results/lmc_lr1e-4_s{seed}.json` (endpoints) |

### 2.7 Hypothesis Tests
| Test | t | p | d | Verdict |
|------|:---:|:---:|:---:|:---:|
| Code within = cross | -0.60 | 0.55 | 0.49 | ns |
| Med within > cross | 4.49 | <0.001 | 3.66 | *** |
| Code within ≠ med within | 5.11 | <0.001 | 4.17 | *** |
| Gaussian ≠ zero | 3.44 | <0.001 | — | *** |
| Code high > std | 2.83 | 0.005 | 2.31 | ** |
| Med high > std | 2.45 | 0.014 | 2.06 | * |

### 2.8 Accuracy Barriers
| Pair | acc barrier | Model Soup? |
|------|:---:|:---:|
| Code↔Med cross (code eval) | +0.061 | No |
| Code↔Med cross (med eval) | −0.058 | **Yes** |
| Code within | 0.005 | — |
| Medical within | 0.015 | — |
| **Source:** `experiments/phase0_ivn/results/accuracy_barriers.json` |

### 2.9 TIES Merge Benchmark
| Method | Best code acc | Best med acc |
|--------|:---:|:---:|
| Individual (best) | 0.660 | 0.895 |
| Pure averaging | 0.629 | 0.885 |
| TIES-Merging | ~0.62 | ~0.88 |
| Task Arithmetic | ~0.62 | ~0.87 |
| **Evaluations:** 44 total | **Source:** `experiments/phase0_ivn/results/merge_benchmark.json` |

---

## 3. Theory Verification Experiments

### 3.1 Experiment A: 2-Epoch Medical
| Pairs | Barrier |
|-------|:---:|
| s10-s11 | 0.214 |
| s10-s12 | 0.133 |
| s11-s12 | 0.198 |
| **Mean** | **0.182** (vs 1-epoch 0.147) |
| **Finding:** Instability persists with additional training |

### 3.2 Experiment B: Noisy Code (15% label flip)
| Pairs | Barrier |
|-------|:---:|
| s10-s11 | 0.045 |
| s10-s12 | 0.041 |
| s11-s12 | 0.053 |
| **Mean** | **0.046** (vs clean 0.048) |
| **Finding:** Random noise does NOT induce medical-like instability |

### 3.3 Experiment C: Fisher Curvature
| Model | Mean Fisher | Early:late ratio |
|------|:---:|:---:|
| Code | 1.44e-8 | 4.5:1 |
| Medical | 5.76e-9 | 21:1 |
| **Ratio:** code 2.5× > medical |
| **Finding:** Medical minima are flatter but more distant from pretrained |

---

## 4. Cross-Architecture Replication

### 4.1 Pythia-1.4B (2-epoch)
| Metric | Value |
|--------|:---:|
| Code within | 0.063 ± 0.005 |
| Medical within | 0.231 ± 0.096 |
| Cross code↔med | 0.079 ± 0.025 |
| med/code ratio | **3.7×** |

### 4.2 OPT-1.3B (2-epoch)
| Metric | Value |
|--------|:---:|
| Code within | 0.251 ± 0.108 |
| Medical within | 0.896 ± 0.042 |
| Cross code↔med | 0.485 ± 0.010 |
| med/code ratio | **3.6×** |
| **Gaussian calibration** | 0.0003 — 0.055 (5 levels) |
| **Trajectory** | Code monotonic (0.14→0.20), Medical flat (0.09→0.14) |
| **Per-block r** | 0.91 (vs Pythia 0.995) |

### 4.3 GPT-Neo-1.3B (2-epoch)
| Status | 🔄 Running |
|--------|-----------|
| Expected | Cross + within LMC, ~10h |

---

## 5. Summary: All Numbers in One Table

### Tri-Model Comparison (Core Finding)
| Model | Code within | Med within | Cross | med/code |
|-------|:---:|:---:|:---:|:---:|
| Pythia-1.4B (1ep) | 0.048 | 0.147 | 0.053 | 3.1× |
| Pythia-1.4B (2ep) | 0.063 | 0.231 | 0.079 | 3.7× |
| OPT-1.3B (2ep) | 0.251 | 0.896 | 0.485 | 3.6× |
| GPT-Neo-1.3B (2ep) | 🔄 | 🔄 | 🔄 | 🔄 |

### Experiment Scale
| Model | Trained models | LMC scans | GPU-hours |
|-------|:---:|:---:|:---:|
| Pythia-1.4B | 30+ | 70+ | ~50 |
| OPT-1.3B | 6 | 30+ | ~25 |
| GPT-Neo-1.3B | 6 (running) | 9 (running) | ~10 |
| **Total** | **42+** | **110+** | **~85** |
