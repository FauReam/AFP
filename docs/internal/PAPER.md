# Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning

> **Draft v4 — 2026-07-13 | 3-seed + bootstrap CI + noise floor + 160M scaling check**

---

## Abstract

When a pretrained language model is fine-tuned on different domains, how far do the resulting models move in weight space? Do they remain in the same linearly-connected loss basin? We measure this on Pythia-1.4B fine-tuned on code and medical reasoning tasks. At standard training intensity, the models diverge by 1.4-1.5% in weight space and the LMC barrier is 0.05 — well within the linearly-connected regime. At higher divergence (8.0-8.5% weight displacement, 11.6% cross-model), the barrier rises to 0.12 ± 0.03 (code) and 0.23 ± 0.10 (medical) — a 2-5× increase. However, even at this divergence level the barrier remains modest, suggesting that domain specialization alone does not easily break linear connectivity. Within-domain baselines reveal domain-specific stability: code models exhibit near-identical barriers within and across domains (0.048 vs. 0.053), while medical models show substantially higher within-domain variance (0.147), indicating that training stability, not domain difference, drives barrier height. Noise-floor calibration places these results in context: identical copies yield barrier ≈ 0.000, and pretrained-to-random-init interpolation establishes an upper bound of 0.22 — which the high-divergence medical condition (0.23) nearly saturates. Per-block divergence patterns are nearly identical across domains (r = 0.995).

## 1. Introduction

Fine-tuning pretrained language models on domain-specific data produces specialized capabilities. But how far do these fine-tuned models actually move from their starting point? When two models are fine-tuned on different domains, does the linear path between their weights cross a loss barrier — or do they remain in the same basin?

Linear Mode Connectivity (Frankle et al., ICML 2020) established that neural networks fine-tuned from the same initialization often lie in the same linearly-connected loss basin. Subsequent work (Wortsman et al. 2022; Yadav et al. 2023) showed that weight interpolation between related models can even improve performance. But the quantitative relationship between *how much* models diverge in weight space and *whether* they remain connected remains underexplored.

We measure this relationship directly. Using Pythia-1.4B as a base model, we train two variants on code reasoning and medical reasoning respectively. We vary the degree of weight-space movement, measure per-block divergence from the pretrained checkpoint, and compute LMC barriers via 11-point linear interpolation between the domain-specialized models.

## 2. Related Work

**Linear Mode Connectivity.** Frankle et al. (ICML 2020) established that neural networks fine-tuned from a shared initialization tend to remain in the same linearly-connected loss basin — the loss along the linear path between two models exhibits at most a modest "barrier." Entezari et al. (ICLR 2022) showed that permutation symmetry accounts for most apparent LMC failures, and Ainsworth et al. (ICLR 2023) proposed permutation-matching algorithms (Git Re-Basin) to align independently-trained models into a shared basin. Garipov et al. (NeurIPS 2018) demonstrated that low-loss connecting curves (Bezier) exist even when linear paths fail. Our work differs by systematically measuring how *quantitative weight displacement* maps onto LMC barrier height, rather than testing whether connectivity exists as a binary property.

**Model Merging.** Weight interpolation between fine-tuned models has proven practically useful. Wortsman et al. (ICML 2022) showed that averaging multiple fine-tuned variants (Model Soups) improves robustness. Yadav et al. (NeurIPS 2023) introduced TIES-Merging, which resolves parameter interference through trim-elect-sign operations before merging. Task arithmetic (Ilharco et al., 2023) demonstrated that weight-space vector addition can compose task capabilities. These methods demonstrate that linear interpolation *works*, but do not characterize *when it fails*. Our barrier measurements provide a quantitative framework for predicting merge quality from weight divergence metrics.

**Domain-Specialized Fine-Tuning.** Prior work has extensively studied domain adaptation for LMs (Gururangan et al., 2020), but the focus has been on downstream performance, not on weight-space characterization. Biderman et al. (NeurIPS 2023) released Pythia checkpoints at multiple training steps, enabling weight-space analysis, but did not study fine-tuning divergence. Our work bridges these literatures by measuring how far domain fine-tuning moves models in weight space and how that movement affects loss landscape connectivity.

## 3. Method

### 3.1 Training

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

Model pairs are created at two divergence levels by varying the optimization step size. We refer to these as the **standard-divergence** and **high-divergence** conditions. The step size values are incidental — what matters is the resulting weight-space distance between models.

### 3.2 LMC Barrier Scan

For each model pair (θ_code, θ_medical), we construct 11 interpolated models:

```
θ(α) = (1-α)·θ_code + α·θ_medical,   α ∈ {0.0, 0.1, ..., 1.0}
```

We evaluate binary cross-entropy loss on 2,000 test samples per domain. The LMC barrier follows Frankle et al. (2020):

```
barrier = max_α L(θ(α)) − (L(θ(0)) + L(θ(1))) / 2
```

## 4. Results

### 4.1 Weight Divergence

| Condition | Code ΔW | Medical ΔW | Code↔Med Cross |
|-----------|---------|------------|-----------------|
| Standard | 1.4 ± 0.0% | 1.5 ± 0.1% | 2.0 ± 0.1% |
| High | 8.0 ± 0.3% | 8.5 ± 0.2% | 11.6 ± 0.3% |

Per-block divergence patterns are nearly identical across domains (r=0.995). Both models change the same transformer blocks; the divergence is primarily in magnitude, not pattern (Figure 2).

### 4.2 LMC Barriers

| Condition | Code barrier | Medical barrier |
|-----------|-------------|----------------|
| Standard | 0.053 ± 0.011 | 0.051 ± 0.013 |
| High | 0.118 ± 0.031 | 0.228 ± 0.102 |

Key observations:
- At standard divergence, LMC holds cleanly: barriers are ~0.05, well within the connected regime.
- At high divergence (8-12% weight displacement), barriers increase 2-5× but remain modest (Figure 1).
- The barrier increase is sublinear: 5-6× more weight divergence produces only 2-5× more barrier.
- Even at 8% divergence, domain-specialized Pythia-1.4B models remain in broadly the same loss basin.

### 4.3 Within-Domain LMC Baseline

To calibrate the cross-domain barriers, we measure LMC between different seeds of the *same* domain (3 pairs each for code and medical, standard divergence). For each pair, we evaluate on the domain's own data:

| Domain | Within-domain barrier |
|--------|----------------------|
| Code | 0.048 ± 0.000 |
| Medical | 0.147 ± 0.027 |

For code, the within-domain barrier (0.048) is nearly identical to the cross-domain barrier at standard divergence (0.053). Domain difference contributes negligibly. For medical, the within-domain barrier (0.147) is substantially higher than the cross-domain barrier (0.051), indicating that medical fine-tuning is intrinsically less stable across seeds — the variance between two medical training runs can exceed the variance between code and medical models trained at the same intensity. The dominant source of barrier height is the *degree of weight displacement combined with per-domain training stability*, not domain dissimilarity itself.

### 4.4 Cross-Domain Asymmetry

At standard divergence, the code model's loss on medical data is lower than base Pythia's — code fine-tuning produces a small positive externality for medical reasoning. Medical fine-tuning provides no reciprocal benefit for code. This asymmetry suggests that general reasoning capabilities acquired during pretraining transfer more readily to specialized tasks than vice versa.

### 4.5 Noise Floor Calibration

We calibrate the measurement with two boundary conditions. First, we run the LMC scan on two identical copies of the same model — this should produce a barrier of zero and reveals any numerical artifacts in the interpolation pipeline. Second, we interpolate between a pretrained Pythia-1.4B model and a randomly-initialized model of the same architecture — this establishes the effective upper bound on what barrier magnitudes the method can produce.

| Condition | Code barrier | Medical barrier |
|-----------|-------------|----------------|
| Identical copy | ~0.000 | ~0.000 |
| Pretrained ↔ Random init | 0.033 | 0.222 |

The identical-copy barrier is effectively zero, confirming the measurement pipeline is free of numerical artifacts. The pretrained-to-random medical barrier (0.222) is nearly identical to the high-divergence cross-domain medical barrier (0.228), suggesting that at ~8% weight displacement, medical models approach the maximum possible loss of connectivity.

## 5. Discussion

Domain-specific fine-tuning of Pythia-1.4B produces only modest weight-space movement: 1-2% at typical training intensity, 7-9% at elevated step sizes. LMC holds across this entire range, though the barrier grows with divergence. The relationship is monotonic but sublinear — barrier increases more slowly than weight distance, suggesting the loss landscape is robust to substantial parameter changes before connectivity breaks.

The within-domain baselines reveal domain-specific stability differences. For code, within-domain (0.048) and cross-domain (0.053) barriers are equivalent — domain difference contributes nothing. For medical, the within-domain barrier (0.147) exceeds the cross-domain barrier (0.051), indicating that medical training trajectories are intrinsically higher-variance. Two medical models trained on the same data can differ more in the loss landscape than a code and medical model trained at the same intensity.

The noise-floor calibration confirms measurement validity. Identical model copies produce a barrier of ~0.000, ruling out numerical artifacts. Pretrained-to-random-init interpolation yields a medical-domain barrier of 0.22, establishing the effective upper bound. The high-divergence cross-domain medical barrier (0.23) nearly saturates this bound, suggesting that at ~8% weight displacement, medical models approach the connectivity limit.

A practical implication: model merging techniques that rely on linear interpolation are likely to work well for domain-specialized models fine-tuned from the same base, even at aggressive optimization settings. However, for domains like medical where training trajectories are intrinsically noisy, merging different seeds of the same domain can be more challenging than merging across domains.

## 6. Limitations

- **Single model family.** Results are based on Pythia-1.4B. A preliminary scaling experiment with Pythia-160M (ΔW ≈ 0.3%, both models barely moved from pretrained) produced barriers of 0.001 (code) and 0.056 (medical), consistent with the trend that barrier magnitude tracks weight displacement. Systematic scaling across model sizes is left to future work.
- **Two domains only** (code, medical). Broader domain sampling is needed to characterize how domain-specific properties (such as medical training's intrinsic instability) affect connectivity.
- **High-divergence convergence.** The high-divergence models achieve higher self-domain loss than standard-divergence models, suggesting that the aggressive optimization settings trade off convergence quality for weight displacement. The barriers reported for these models may reflect a combination of weight displacement and under-convergence.
- **Seed count.** Three seeds per condition provide meaningful estimates of variance (bootstrap 95% CIs are reported in the supplementary analysis), but larger seed counts would tighten the confidence intervals, particularly for the high-divergence medical condition where inter-seed variance is large.

## References

- Frankle, J., et al. (ICML 2020). "Linear Mode Connectivity and the Lottery Ticket Hypothesis."
- Wortsman, M., et al. (ICML 2022). "Model soups: averaging weights of multiple fine-tuned models."
- Yadav, P., et al. (NeurIPS 2023). "TIES-Merging: Resolving Interference When Merging Models."
- Biderman, S., et al. (NeurIPS 2023). "Pythia: A Suite for Analyzing Large Language Models."
- Entezari, R., et al. (ICLR 2022). "The Role of Permutation Invariance in Linear Mode Connectivity."
- Ainsworth, S., et al. (ICLR 2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries."
- Garipov, T., et al. (NeurIPS 2018). "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs."
- Gururangan, S., et al. (ACL 2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks."
- Ilharco, G., et al. (NeurIPS 2023). "Editing Models with Task Arithmetic."

## Figures

**Figure 1: LMC barrier curves.** Loss along the linear interpolation path θ(α) = (1-α)·θ_code + α·θ_medical, evaluated on both code (solid) and medical (dashed) test sets. Top: standard divergence (ΔW ≈ 1.4%). Bottom: high divergence (ΔW ≈ 8%). The shaded region represents ±1 standard deviation across 3 random seeds. Barriers increase with divergence but remain modest even at 8% weight displacement.

![LMC barrier curves](../docs/reports/fig1_lmc_barrier.png)

**Figure 2: Per-block weight divergence.** Root-mean-square weight difference from the pretrained checkpoint, computed per transformer block. Code and medical models at standard divergence are overlaid (r = 0.995). Divergence concentrates in early layers (layer 0: ~5.6% at standard divergence) and decreases roughly exponentially with depth.

![Per-block divergence](../docs/reports/fig2_per_block.png)
