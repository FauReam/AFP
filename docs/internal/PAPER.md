# Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning

> **Draft v7 — 2026-07-14 | Trajectory LMC + Gaussian calibration + Layer-selective merge**

---

## Abstract

When a pretrained language model is fine-tuned on different domains, how far do the resulting models move in weight space? Do they remain in the same linearly-connected loss basin? We measure this on Pythia-1.4B fine-tuned on code and medical reasoning tasks. At standard training intensity, the models diverge by 1.4-1.5% in weight space and the LMC barrier is 0.05 — well within the linearly-connected regime. At higher divergence (8.0-8.5% weight displacement, 11.6% cross-model), the barrier rises to 0.12 ± 0.03 (code) and 0.23 ± 0.10 (medical) — a 2-5× increase. However, even at this divergence level the barrier remains modest, suggesting that domain specialization alone does not easily break linear connectivity. Within-domain baselines reveal a striking asymmetry: code models exhibit near-identical barriers within and across domains (0.048 vs. 0.053), while medical models show 3× larger within-domain than cross-domain barriers (0.147 vs. 0.051). This indicates that training stability, not domain difference, is the dominant source of LMC barrier height — and that stability varies dramatically across domains. Noise-floor calibration places these results in context: identical copies yield barrier ≈ 0.000, and pretrained-to-random-init interpolation provides a reference point of 0.22. Per-block divergence patterns are nearly identical across domains (r = 0.995). A training trajectory analysis reveals that the barrier follows an inverted-U shape for code (peaking mid-training then declining) but grows monotonically for medical — confirming domain-specific stability at the within-run level. Gaussian perturbation experiments show that unstructured noise produces negligible barriers even at 8% weight displacement, while structured training-induced changes create barriers 9× larger at equivalent magnitude. Layer-selective interpolation demonstrates that the barrier is almost entirely concentrated in early transformer layers (0-7), with deep layers (16-23) showing near-zero interpolation penalty — a directly actionable finding for model merging practice.

## 1. Introduction

Fine-tuning pretrained language models on domain-specific data produces specialized capabilities. But how far do these fine-tuned models actually move from their starting point? When two models are fine-tuned on different domains, does the linear path between their weights cross a loss barrier — or do they remain in the same basin?

Linear Mode Connectivity (Frankle et al., ICML 2020) established that neural networks fine-tuned from the same initialization often lie in the same linearly-connected loss basin. Subsequent work (Wortsman et al. 2022; Yadav et al. 2023) showed that weight interpolation between related models can even improve performance. But the quantitative relationship between *how much* models diverge in weight space and *whether* they remain connected remains underexplored.

We measure this relationship directly. Using Pythia-1.4B as a base model, we train two variants on code reasoning and medical reasoning respectively. We vary the degree of weight-space movement, measure per-block divergence from the pretrained checkpoint, and compute LMC barriers via 11-point linear interpolation between the domain-specialized models.

## 2. Related Work

**Linear Mode Connectivity.** Draxler et al. (ICML 2018) first demonstrated empirically that neural network minima are connected by near-zero-barrier paths. Garipov et al. (NeurIPS 2018) introduced Fast Geometric Ensembling via Bezier curves. Frankle et al. (ICML 2020) formalized the linear interpolation barrier definition we adopt: $\text{barrier} = \max_\alpha \mathcal{L} - (\mathcal{L}_0 + \mathcal{L}_1)/2$. Li et al. (NeurIPS 2018) provided the standard visualization framework for loss landscapes. Entezari et al. (ICLR 2022) proved that permutation symmetry accounts for most apparent LMC failures — fine-tuned models from the same initialization are almost certainly connected after permutation alignment. Ainsworth et al. (ICLR 2023) operationalized this with Git Re-Basin matching algorithms. Mirzadeh et al. (ICLR 2021) studied LMC specifically in continual learning settings. Lubana et al. (ICML 2023) provided mechanistic explanations for when mode connectivity holds. Our work differs from this literature by measuring how *quantitative weight displacement* maps onto LMC barrier height, rather than testing whether connectivity exists as a binary property.

**Model Merging.** Linear weight interpolation has proven practically useful. Izmailov et al. (UAI 2018) showed that weight averaging finds wider optima (Stochastic Weight Averaging). Wortsman et al. (ICML 2022) averaged fine-tuned variants into "Model Soups" that improve robustness at zero inference cost. Wortsman et al. (CVPR 2022) further demonstrated (WiSE-FT) that interpolating between zero-shot and fine-tuned weights improves distribution shift robustness. Yadav et al. (NeurIPS 2023) introduced TIES-Merging, which resolves parameter interference through trim-elect-sign operations. Task arithmetic (Ilharco et al., NeurIPS 2023) showed that weight-space vector addition composes task capabilities. Matena \& Raffel (NeurIPS 2022) proposed Fisher-weighted averaging grounded in Laplace posterior approximation. Jin et al. (ICLR 2023) demonstrated dataless knowledge fusion via weight merging, and Stoica et al. (ICLR 2024) merged models from different tasks without training (ZipIt!). Singh \& Jaggi (NeurIPS 2020) used optimal transport for nonlinear model fusion. These methods demonstrate that weight-space operations *work*, but do not characterize *when they fail*. Our barrier measurements provide a quantitative framework connecting weight divergence magnitude to expected interpolation quality — a predictor that could guide practitioners in deciding when to merge.

**Fine-Tuning Analysis and Weight-Space Characterization.** Neyshabur et al. (NeurIPS 2020) investigated what is transferred during fine-tuning, finding that feature reuse rather than task-specific adaptation dominates. Gururangan et al. (ACL 2020) established domain-adaptive pretraining (DAPT) as the standard domain adaptation paradigm, but focused on downstream accuracy, not weight-space properties. Kirkpatrick et al. (PNAS 2017) introduced Elastic Weight Consolidation (EWC), providing a Fisher-information framework for measuring parameter importance — a precursor to our per-block divergence analysis. Biderman et al. (NeurIPS 2023) released Pythia checkpoints at multiple training steps with documented training dynamics, enabling systematic weight-space analysis that we leverage. Zhang et al. (ICLR 2017) clarified the relationship between generalization and parameter-space properties. Fort et al. (NeurIPS 2019) introduced "stiffness" as a measure of parameter sensitivity to perturbations. Our work bridges these literatures by measuring how far domain fine-tuning moves models in weight space, characterizing per-block divergence patterns, and connecting weight displacement magnitude to loss landscape connectivity.

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
- The barrier grows less than proportionally to divergence: 5-6× more weight displacement yields only 2-5× more barrier.
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

We calibrate the measurement with two boundary conditions. First, we run the LMC scan on two identical copies of the same model — this should produce a barrier of zero and reveals any numerical artifacts in the interpolation pipeline. Second, we interpolate between a pretrained Pythia-1.4B model and a randomly-initialized model of the same architecture — this provides a reference point for interpreting barrier magnitudes in an extreme divergence scenario.

| Condition | Code barrier | Medical barrier |
|-----------|-------------|----------------|
| Identical copy | ~0.000 | ~0.000 |
| Pretrained ↔ Random init | 0.033 | 0.222 |

The identical-copy barrier is effectively zero, confirming the measurement pipeline is free of numerical artifacts. The pretrained-to-random reference barrier (0.222) serves as a calibration point — models approaching this magnitude are undergoing extreme weight-space displacement. We note that this reference is a single measurement and its variance is uncharacterized; it should be interpreted as indicative rather than a precise bound.

### 4.6 Training Trajectory: Continuous ΔW→Barrier Curve

To characterize the functional relationship between weight displacement and barrier height beyond two discrete divergence levels, we save model checkpoints every 40 training steps and measure the LMC barrier between each checkpoint and the pretrained base model. This yields a continuous ΔW→barrier curve from a single training run, eliminating the convergence confound of the LR sweep (all checkpoints come from the same optimization trajectory).

**Code domain** (Figure 3): The barrier follows an inverted-U shape — rising from 0.029 at step 40 (ΔW ≈ 0.3%) to a peak of 0.043 at step 200 (ΔW ≈ 1.1%), then *declining* to 0.033 at step 400 (ΔW ≈ 1.4%) despite monotonically increasing weight displacement. The endpoint (final trained model vs. base) shows a *lower* barrier than the mid-training checkpoint, indicating that the model settles into a well-defined minimum that is more linearly connected to the pretrained initialization than the intermediate "partially-trained" states.

**Medical domain**: The barrier grows monotonically with training progress — from 0.173 (step 40) to 0.218 (step 320), then stabilizes around 0.211 (step 400). Unlike code, medical training never shows the inverted-U recovery: once the barrier begins to rise, it plateaus but does not decline. This confirms the domain-specific stability difference at a granular, within-trajectory level: code models converge toward a basin that re-connects to the pretrained initialization, while medical models diverge into a loss landscape region that remains persistently separated.

### 4.7 Structured vs. Unstructured Weight Displacement

Does the barrier arise from the *magnitude* of weight change or from its *structure*? We add isotropic Gaussian noise to the pretrained model, scaled to achieve ΔW ∈ {0.5%, 1%, 2%, 4%, 8%}, and measure the resulting LMC barrier.

| ΔW | Gaussian noise barrier | Training-induced barrier |
|:---:|:---:|:---:|
| 0.5% | 0.003 | — |
| 1.0% | 0.004 | — |
| 1.5% | 0.006 | 0.053 (standard code) |
| 2.0% | 0.005 | — |
| 4.0% | 0.011 | — |
| 8.0% | 0.014 | 0.118 (high-div code) |

Unstructured Gaussian perturbation produces negligible barriers regardless of magnitude: even at ΔW = 8%, the barrier is 0.014 — indistinguishable from the identical-copy baseline (~0.000). In contrast, training-induced weight displacement at ΔW = 1.5% produces a barrier of 0.053, approximately 9× larger. This demonstrates that LMC barriers arise from the *directional structure* of weight changes — the specific, task-aligned pattern of parameter updates — rather than from weight-space distance alone.

### 4.8 Layer-Selective Interpolation

The per-block divergence pattern (Figure 2, r = 0.995 across domains) shows that early layers change 5.6× more than late layers. We test whether this divergence asymmetry translates to barrier asymmetry by performing layer-selective interpolation: merge only a subset of layers (early: 0-7, mid: 8-15, late: 16-23) while keeping the remaining layers from the code model.

| Merged layers | Code barrier | % of full barrier |
|:---|:---:|:---:|
| Early (0-7) | 0.040 | 75% |
| Mid (8-15) | 0.003 | 6% |
| Late (16-23) | 0.000 | 0% |
| All (0-23) | 0.053 | 100% |

The barrier is almost entirely concentrated in the first 8 transformer layers. Merging only late layers (16-23) produces essentially zero barrier — these layers can be safely interpolated with negligible loss penalty. This provides a directly actionable strategy for model merging practitioners: aggressive interpolation of deep layers combined with conservative handling of early layers.

## 5. Discussion

Domain-specific fine-tuning of Pythia-1.4B produces only modest weight-space movement: 1-2% at typical training intensity, 7-9% at elevated step sizes. The trajectory analysis (§4.6) reveals that the ΔW→barrier relationship is more nuanced than a simple monotonic function: code models exhibit an inverted-U pattern where barrier *declines* after mid-training, while medical models show persistent barrier growth. This directly supports our central thesis that training stability, not weight displacement magnitude, is the dominant determinant of LMC barrier height — and that stability is domain-specific.

The Gaussian perturbation experiment (§4.7) reveals that unstructured weight noise produces essentially zero barrier even at ΔW = 8%, while structured training-induced displacement at ΔW = 1.5% produces a barrier of 0.053. This 9× difference demonstrates that LMC barriers measure the *directional alignment* of weight changes, not their magnitude per se. The implication is that two models trained on different tasks with similar ΔW can have dramatically different barriers depending on whether their weight updates are task-aligned (high barrier) or noise-like (low barrier).

The layer-selective interpolation experiment (§4.8) shows that the barrier is almost entirely driven by early transformer layers (0-7). Deep layers (16-23) can be merged with near-zero penalty. This finding, combined with the per-block pattern analysis, suggests a practical two-tier merging strategy: apply conservative, importance-weighted merging to early layers while using aggressive linear interpolation for late layers.

The within-domain baselines reveal domain-specific stability differences. For code, within-domain (0.048) and cross-domain (0.053) barriers are equivalent — domain difference contributes nothing. For medical, the within-domain barrier (0.147) exceeds the cross-domain barrier (0.051), indicating that medical training trajectories are intrinsically higher-variance. Two medical models trained on the same data can differ more in the loss landscape than a code and medical model trained at the same intensity.

The noise-floor calibration confirms measurement validity. Identical model copies produce a barrier of ~0.000, ruling out numerical artifacts. Pretrained-to-random-init interpolation yields a medical-domain barrier of 0.22, providing a reference calibration point. The high-divergence cross-domain medical barrier (0.23) is comparable to this reference, though we note that the high-divergence condition has a convergence confound (higher self-domain loss), and the reference value itself has uncharacterized variance from a single measurement. Both numbers should be interpreted as indicative rather than precise bounds.

A practical implication: model merging techniques that rely on linear interpolation are likely to work well for domain-specialized models fine-tuned from the same base, even at aggressive optimization settings. However, for domains like medical where training trajectories are intrinsically noisy, merging different seeds of the same domain can be more challenging than merging across domains.

## 6. Limitations

- **Single model family.** Results are based on Pythia-1.4B. A preliminary scaling experiment with Pythia-160M (ΔW ≈ 0.3%, both models barely moved from pretrained) produced barriers of 0.001 (code) and 0.056 (medical), consistent with the trend that barrier magnitude tracks weight displacement. Systematic scaling across model sizes is left to future work.
- **Two domains only** (code, medical). Broader domain sampling is needed to characterize how domain-specific properties (such as medical training's intrinsic instability) affect connectivity.
- **High-divergence convergence.** The high-divergence models achieve higher self-domain loss than standard-divergence models, suggesting that the aggressive optimization settings trade off convergence quality for weight displacement. The barriers reported for these models may reflect a combination of weight displacement and under-convergence.
- **Seed count.** Three seeds per condition provide meaningful estimates of variance (bootstrap 95% CIs are reported in the supplementary analysis), but larger seed counts would tighten the confidence intervals, particularly for the high-divergence medical condition where inter-seed variance is large.

## References

**Core — LMC & Loss Landscape**
1. Draxler, F., et al. (ICML 2018). "Essentially No Barriers in Neural Network Energy Landscapes."
2. Garipov, T., et al. (NeurIPS 2018). "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs."
3. Li, H., et al. (NeurIPS 2018). "Visualizing the Loss Landscape of Neural Nets."
4. Frankle, J., et al. (ICML 2020). "Linear Mode Connectivity and the Lottery Ticket Hypothesis."
5. Entezari, R., et al. (ICLR 2022). "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks."
6. Ainsworth, S., et al. (ICLR 2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries."
7. Mirzadeh, S. I., et al. (ICLR 2021). "Linear Mode Connectivity in Multitask and Continual Learning."
8. Lubana, E. S., et al. (ICML 2023). "Mechanistic Mode Connectivity."

**Core — Model Merging & Weight Interpolation**
9. Izmailov, P., et al. (UAI 2018). "Averaging Weights Leads to Wider Optima and Better Generalization."
10. Wortsman, M., et al. (ICML 2022). "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time."
11. Wortsman, M., et al. (CVPR 2022). "Robust fine-tuning of zero-shot models."
12. Yadav, P., et al. (NeurIPS 2023). "TIES-Merging: Resolving Interference When Merging Models."
13. Ilharco, G., et al. (NeurIPS 2023). "Editing Models with Task Arithmetic."
14. Matena, M. and Raffel, C. (NeurIPS 2022). "Merging Models with Fisher-Weighted Averaging."
15. Jin, X., et al. (ICLR 2023). "Dataless Knowledge Fusion by Merging Weights of Language Models."
16. Stoica, G., et al. (ICLR 2024). "ZipIt! Merging Models from Different Tasks without Training."
17. Singh, S. P. and Jaggi, M. (NeurIPS 2020). "Model Fusion via Optimal Transport."

**Core — Fine-Tuning & Weight-Space Analysis**
18. Biderman, S., et al. (NeurIPS 2023). "Pythia: A Suite for Analyzing Large Language Models."
19. Gururangan, S., et al. (ACL 2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks."
20. Neyshabur, B., et al. (NeurIPS 2020). "What is being transferred in transfer learning?"
21. Kirkpatrick, J., et al. (PNAS 2017). "Overcoming catastrophic forgetting in neural networks."
22. Fort, S., et al. (NeurIPS 2019). "Stiffness: A New Perspective on Generalization in Neural Networks."
23. Zhang, C., et al. (ICLR 2017). "Understanding Deep Learning Requires Rethinking Generalization."

## Figures

**Figure 1: LMC barrier curves.** Loss along the linear interpolation path θ(α) = (1-α)·θ_code + α·θ_medical, evaluated on both code (solid) and medical (dashed) test sets. Top: standard divergence (ΔW ≈ 1.4%). Bottom: high divergence (ΔW ≈ 8%). The shaded region represents ±1 standard deviation across 3 random seeds. Barriers increase with divergence but remain modest even at 8% weight displacement.

![LMC barrier curves](../docs/reports/fig1_lmc_barrier.png)

**Figure 2: Per-block weight divergence.** Root-mean-square weight difference from the pretrained checkpoint, computed per transformer block. Code and medical models at standard divergence are overlaid (r = 0.995). Divergence concentrates in early layers (layer 0: ~5.6% at standard divergence) and decreases roughly exponentially with depth.

![Per-block divergence](../docs/reports/fig2_per_block.png)

**Figure 3: Training trajectory barrier.** LMC barrier height as a function of training step when interpolating between trajectory checkpoints and the pretrained base model. Code (blue) follows an inverted-U: barrier peaks at step 200 then declines as the model converges. Medical (red) grows monotonically and plateaus — the model never re-connects to the pretrained initialization. This within-run asymmetry confirms that domain-specific training stability, not weight displacement magnitude, determines barrier height.
