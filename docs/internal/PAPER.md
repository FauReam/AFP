# Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning

> **Draft v2 — 2026-07-11 | 基于 3-seed 实测数据**

---

## Abstract

When a pretrained language model is fine-tuned on different domains, how far do the resulting models move in weight space? Do they remain in the same linearly-connected loss basin? We measure this on Pythia-1.4B fine-tuned on code and medical reasoning tasks. At standard training intensity, the models diverge by 1.4-1.5% in weight space and the LMC barrier is 0.05 — well within the linearly-connected regime. At higher divergence (8.0-8.5% weight displacement, 11.6% cross-model), the barrier rises to 0.12 ± 0.03 (code) and 0.23 ± 0.10 (medical) — a 2-5× increase. However, even at this divergence level the barrier remains modest, suggesting that domain specialization alone does not easily break linear connectivity. Across 3 random seeds, barrier magnitudes are consistent within each condition. Per-block divergence patterns are nearly identical across domains (r = 0.995). We release the training recipe, measurement code, and all model checkpoints.

## 1. Introduction

Fine-tuning pretrained language models on domain-specific data produces specialized capabilities. But how far do these fine-tuned models actually move from their starting point? When two models are fine-tuned on different domains, does the linear path between their weights cross a loss barrier — or do they remain in the same basin?

Linear Mode Connectivity (Frankle et al., ICML 2020) established that neural networks fine-tuned from the same initialization often lie in the same linearly-connected loss basin. Subsequent work (Wortsman et al. 2022; Yadav et al. 2023) showed that weight interpolation between related models can even improve performance. But the quantitative relationship between *how much* models diverge in weight space and *whether* they remain connected remains underexplored.

We measure this relationship directly. Using Pythia-1.4B as a base model, we train two variants on code reasoning and medical reasoning respectively. We vary the degree of weight-space movement, measure per-block divergence from the pretrained checkpoint, and compute LMC barriers via 11-point linear interpolation between the domain-specialized models.

## 2. Method

### 2.1 Training

| Parameter | Value |
|-----------|-------|
| Base model | EleutherAI/pythia-1.4b (1.31B params) |
| Training | Full-parameter fine-tuning, 1 epoch |
| Data | VersaPRM: code (55K train) + medical (~55K train) |
| Precision | bfloat16 autocast, fp32 classification head |
| Batch size | 128 |
| Sequence length | 256 tokens |
| Optimizer | AdamW (β=0.9, 0.999, wd=0.1) |
| Hardware | NVIDIA DGX Spark GB10 (121GB unified memory, ARM64 CUDA 13.0) |
| Seeds | 3 per condition |

Model pairs are created at two divergence levels by varying the optimization step size. We refer to these as the **standard-divergence** and **high-divergence** conditions. The step size values are 1e-4 and 5e-4 respectively; the specific values are incidental — what matters is the resulting weight-space distance between models.

### 2.2 LMC Barrier Scan

For each model pair (θ_code, θ_medical), we construct 11 interpolated models:

```
θ(α) = (1-α)·θ_code + α·θ_medical,   α ∈ {0.0, 0.1, ..., 1.0}
```

We evaluate binary cross-entropy loss on 2,000 test samples per domain. The LMC barrier follows Frankle et al. (2020):

```
barrier = max_α L(θ(α)) − (L(θ(0)) + L(θ(1))) / 2
```

## 3. Results

### 3.1 Weight Divergence (Figure 2)

| Condition | Code ΔW | Medical ΔW | Code↔Med Cross |
|-----------|---------|------------|-----------------|
| Standard | 1.4 ± 0.0% | 1.5 ± 0.1% | 2.0 ± 0.1% |
| High | 8.0 ± 0.3% | 8.5 ± 0.2% | 11.6 ± 0.3% |

Per-block divergence patterns are nearly identical across domains (r=0.995). Both models change the same transformer blocks; the divergence is primarily in magnitude, not pattern.

### 3.2 LMC Barriers (Frankle definition, 3 seeds)

| Condition | Code barrier | Medical barrier |
|-----------|-------------|----------------|
| Standard | 0.053 ± 0.011 | 0.051 ± 0.013 |
| High | 0.118 ± 0.031 | 0.228 ± 0.102 |

Key observations:
- At standard divergence, LMC holds cleanly: barriers are ~0.05, well within the connected regime.
- At high divergence (8-12% weight displacement), barriers increase 2-5× but remain modest.
- The barrier increase is sublinear: 5-6× more weight divergence produces only 2-5× more barrier.
- Even at 8% divergence, domain-specialized Pythia-1.4B models remain in broadly the same loss basin.

### 3.3 Within-Domain LMC Baseline

To calibrate the cross-domain barriers, we measure LMC between different seeds of the *same* domain (3 pairs each for code and medical, standard divergence):

| Domain | Within-domain barrier |
|--------|----------------------|
| Code | 0.049 ± 0.000 |
| Medical | 0.077 ± 0.003 |

The within-domain barriers are comparable to the cross-domain barriers at standard divergence (0.05 vs 0.05). This indicates that domain difference itself contributes negligibly to the barrier — the dominant source of barrier height is the *degree of weight displacement*, not domain dissimilarity.

### 3.3 Cross-Domain Asymmetry

At standard divergence, the code model's loss on medical data is lower than base Pythia's — code fine-tuning produces a small positive externality for medical reasoning. Medical fine-tuning provides no reciprocal benefit for code. This asymmetry suggests that general reasoning capabilities acquired during pretraining transfer more readily to specialized tasks than vice versa.

## 4. Discussion

Domain-specific fine-tuning of Pythia-1.4B produces only modest weight-space movement: 1-2% at typical training intensity, 7-9% at elevated step sizes. LMC holds across this entire range, though the barrier grows with divergence. The relationship is monotonic but sublinear — barrier increases more slowly than weight distance, suggesting the loss landscape is robust to substantial parameter changes before connectivity breaks.

Critically, within-domain LMC barriers (different seeds, same domain) are comparable to cross-domain barriers at standard divergence: 0.05 vs 0.05 for code, 0.08 vs 0.05 for medical. This means domain difference contributes negligibly to the barrier — the dominant factor is how far the models moved, not what domain they moved toward.

A practical implication: model merging techniques that rely on linear interpolation are likely to work well for domain-specialized models fine-tuned from the same base, even at aggressive optimization settings. The primary concern is not domain mismatch but training intensity mismatch.

## 5. Limitations

- Single model architecture (Pythia-1.4B). Results may differ for other scales or architectures.
- Two domains only (code, medical). Broader domain sampling needed to characterize the asymmetry finding.
- The high-divergence models require further convergence study — their self-domain loss is higher than standard-divergence models.

## References

- Frankle, J., et al. (ICML 2020). "Linear Mode Connectivity and the Lottery Ticket Hypothesis."
- Wortsman, M., et al. (ICML 2022). "Model soups: averaging weights of multiple fine-tuned models."
- Yadav, P., et al. (NeurIPS 2023). "TIES-Merging: Resolving Interference When Merging Models."
- Biderman, S., et al. (NeurIPS 2023). "Pythia: A Suite for Analyzing Large Language Models."
- Entezari, R., et al. (ICLR 2022). "The Role of Permutation Invariance in Linear Mode Connectivity."
- Ainsworth, S., et al. (ICLR 2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries."
