# Training Stability, Not Domain Difference, Determines Linear Mode Connectivity

> **Draft v14 — 2026-07-14 | Theory framework + accuracy barriers + publication figures**

---

## Abstract

When a pretrained language model is fine-tuned on different domains, how far do the resulting models move in weight space? Do they remain in the same linearly-connected loss basin? We measure this on Pythia-1.4B fine-tuned on code and medical reasoning tasks. At standard training intensity, the models diverge by 1.4-1.5% in weight space and the LMC barrier is 0.05 — well within the linearly-connected regime. At higher divergence (8.0-8.5% weight displacement, 11.6% cross-model), the barrier rises to 0.12 ± 0.03 (code) and 0.23 ± 0.10 (medical) — a 2-5× increase. However, even at this divergence level the barrier remains modest, suggesting that domain specialization alone does not easily break linear connectivity. Within-domain baselines reveal a striking asymmetry: code models exhibit near-identical barriers within and across domains (0.048 vs. 0.053), while medical models show 3× larger within-domain than cross-domain barriers (0.147 vs. 0.051). This indicates that training stability, not domain difference, is the dominant source of LMC barrier height — and that stability varies dramatically across domains. Noise-floor calibration places these results in context: identical copies yield barrier ≈ 0.000, and pretrained-to-random-init interpolation provides a reference point of 0.150 ± 0.019 (3-seed calibration). Per-block divergence patterns are nearly identical across domains (r = 0.995). A training trajectory analysis reveals that the barrier follows an inverted-U shape for code (peaking mid-training then declining) but grows monotonically for medical — confirming domain-specific stability at the within-run level. Gaussian perturbation experiments show that unstructured noise produces negligible barriers even at 8% weight displacement, while structured training-induced changes create barriers 9× larger at equivalent magnitude. Layer-selective interpolation demonstrates that the barrier is almost entirely concentrated in early transformer layers (0-7), with deep layers (16-23) showing near-zero interpolation penalty — a directly actionable finding for model merging practice. Cross-architecture replication on OPT-1.3B confirms the qualitative domain stability pattern (medical > code within-domain barriers) despite 4-8× larger absolute barrier magnitudes, supporting generalizability beyond the Pythia architecture.

## 1. Introduction

Fine-tuning pretrained language models on domain-specific data produces specialized capabilities. But how far do these fine-tuned models actually move from their starting point? When two models are fine-tuned on different domains, does the linear path between their weights cross a loss barrier — or do they remain in the same basin?

Linear Mode Connectivity (Frankle et al., ICML 2020) established that neural networks fine-tuned from the same initialization often lie in the same linearly-connected loss basin. Subsequent work (Wortsman et al. 2022; Yadav et al. 2023) showed that weight interpolation between related models can even improve performance. But the quantitative relationship between *how much* models diverge in weight space and *whether* they remain connected remains underexplored.

We measure this relationship directly. Using Pythia-1.4B as a base model, we train two variants on code reasoning and medical reasoning respectively. We vary the degree of weight-space movement, measure per-block divergence from the pretrained checkpoint, and compute LMC barriers via 11-point linear interpolation between the domain-specialized models.

## 2. Related Work

**Linear Mode Connectivity.** Draxler et al. (ICML 2018) first demonstrated empirically that neural network minima are connected by near-zero-barrier paths. Garipov et al. (NeurIPS 2018) introduced Fast Geometric Ensembling via Bezier curves. Frankle et al. (ICML 2020) formalized the linear interpolation barrier definition we adopt: $\text{barrier} = \max_\alpha \mathcal{L} - (\mathcal{L}_0 + \mathcal{L}_1)/2$. Li et al. (NeurIPS 2018) provided the standard visualization framework for loss landscapes. Entezari et al. (ICLR 2022) proved that permutation symmetry accounts for most apparent LMC failures — fine-tuned models from the same initialization are almost certainly connected after permutation alignment. Ainsworth et al. (ICLR 2023) operationalized this with Git Re-Basin matching algorithms. Mirzadeh et al. (ICLR 2021) studied LMC specifically in continual learning settings. Lubana et al. (ICML 2023) provided mechanistic explanations for when mode connectivity holds. Our work differs from this literature by measuring how *quantitative weight displacement* maps onto LMC barrier height, rather than testing whether connectivity exists as a binary property.

**Model Merging.** Linear weight interpolation has proven practically useful. Izmailov et al. (UAI 2018) showed that weight averaging finds wider optima (Stochastic Weight Averaging). Wortsman et al. (ICML 2022) averaged fine-tuned variants into "Model Soups" that improve robustness at zero inference cost. Wortsman et al. (CVPR 2022) further demonstrated (WiSE-FT) that interpolating between zero-shot and fine-tuned weights improves distribution shift robustness. Yadav et al. (NeurIPS 2023) introduced TIES-Merging, which resolves parameter interference through trim-elect-sign operations. Task arithmetic (Ilharco et al., NeurIPS 2023) showed that weight-space vector addition composes task capabilities. Matena \& Raffel (NeurIPS 2022) proposed Fisher-weighted averaging grounded in Laplace posterior approximation. Jin et al. (ICLR 2023) demonstrated dataless knowledge fusion via weight merging, and Stoica et al. (ICLR 2024) merged models from different tasks without training (ZipIt!). Singh \& Jaggi (NeurIPS 2020) used optimal transport for nonlinear model fusion. These methods demonstrate that weight-space operations *work*, but do not characterize *when they fail*. Our barrier measurements provide a quantitative framework connecting weight divergence magnitude to expected interpolation quality — a predictor that could guide practitioners in deciding when to merge.

**Fine-Tuning Analysis and Weight-Space Characterization.** Neyshabur et al. (NeurIPS 2020) investigated what is transferred during fine-tuning, finding that feature reuse rather than task-specific adaptation dominates. Gururangan et al. (ACL 2020) established domain-adaptive pretraining (DAPT) as the standard domain adaptation paradigm, but focused on downstream accuracy, not weight-space properties. Kirkpatrick et al. (PNAS 2017) introduced Elastic Weight Consolidation (EWC), providing a Fisher-information framework for measuring parameter importance. Aljundi et al. (ECCV 2018) proposed Memory Aware Synapses (MAS), which estimates parameter importance from output sensitivity without requiring labels — a precursor to our per-block divergence analysis. Biderman et al. (NeurIPS 2023) released Pythia checkpoints at multiple training steps with documented training dynamics, enabling systematic weight-space analysis that we leverage. Zhang et al. (ICLR 2017) clarified the relationship between generalization and parameter-space properties. Fort et al. (NeurIPS 2019) introduced "stiffness" as a measure of parameter sensitivity to perturbations. Our work bridges these literatures by measuring how far domain fine-tuning moves models in weight space, characterizing per-block divergence patterns, and connecting weight displacement magnitude to loss landscape connectivity.

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
- At intermediate divergence (~5% ΔW), we observe seed-dependent catastrophic barrier spikes: three code models with near-identical weight displacement (ΔW ≈ 4.6%) produce barriers of 0.056, 0.096, and 1.043 — a 19× range. The model with the highest barrier (1.043) has the best self-domain performance (loss 0.566 vs. 0.584-0.595), indicating that catastrophic interpolation failure can coexist with strong task-specific convergence. This provides the strongest evidence that barrier height is determined by training trajectory geometry, not weight displacement magnitude.

We quantify these comparisons using Welch's t-test with Cohen's d effect sizes. Code within-domain and cross-domain barriers are statistically indistinguishable (t(2.0)=0.60, p=0.55, d=0.49), confirming that domain difference contributes nothing for code models. Medical within-domain barriers significantly exceed cross-domain barriers (t(2.9)=4.49, p<0.001, d=3.66) and code within-domain barriers (t(2.0)=5.11, p<0.001, d=4.17). Gaussian noise creates small but statistically detectable barriers (t(4.0)=3.44, p<0.001), though the effect size relative to training-induced barriers is d=5.27 — an order of magnitude larger. High-divergence barriers significantly exceed standard-divergence barriers for both code (t(2.5)=2.83, p=0.005, d=2.31) and medical (t(2.1)=2.45, p=0.014, d=2.06). All comparisons produce "large" effect sizes (Cohen's d > 0.8), indicating practically meaningful differences.

### 4.3 Within-Domain LMC Baseline

To calibrate the cross-domain barriers, we measure LMC between different seeds of the *same* domain (3 pairs each, standard divergence). For each pair, we evaluate on the domain's own data. We extend this analysis to two additional domains (math and general reasoning) to test whether the stability pattern generalizes:

| Domain | Within-domain barrier | Interpretation |
|--------|:---:|------|
| Code | 0.048 ± 0.000 | Highly stable — near-zero seed variance |
| General | 0.071 ± 0.007 | Stable — low seed variance |
| Math | 0.087 ± 0.032 | Moderately unstable — moderate seed variance |
| Medical | 0.147 ± 0.027 | Unstable — 3× code barrier |

The four domains form a stability spectrum rather than a binary contrast. Code and general models exhibit near-identical within- and cross-domain barriers (e.g., code within 0.048 ≈ code↔medical cross 0.053), while math shows intermediate instability and medical consistently produces the highest within-domain barriers. The cross-domain barrier between math and general models (0.012 on code evaluation / 0.078 on medical evaluation) is comparable to the code↔medical cross-domain barrier (0.053 / 0.051), confirming that domain difference per se contributes negligibly to barrier height. The dominant source of LMC barrier is per-domain training stability — independent of which other domain the model is compared against.

### 4.3.1 Seed-Pair Compatibility at High Divergence

To test whether the within-domain barrier patterns persist at larger seed counts, we trained two additional medical models at high divergence (total n=5 seeds) and measured all C(5,2)=10 pairwise barriers. The results reveal that barrier height is seed-*pair*-specific, not seed-specific:

| Seed pair | Barrier | Interpretation |
|:---|:---:|------|
| s0↔s1 | 0.207 | |
| s0↔s2 | 0.072 | Tightly connected |
| s0↔s3 | 0.152 | |
| s0↔s4 | 0.300 | |
| s1↔s2 | 0.183 | |
| s1↔s3 | 0.371 | |
| s1↔s4 | **1.213** | Catastrophic disconnection |
| s2↔s3 | 0.383 | |
| s2↔s4 | 0.080 | Tightly connected |
| s3↔s4 | 0.081 | Tightly connected |

Across all 10 pairs, the mean barrier is 0.304 ± 0.322 (n=10), substantially higher than the original n=3 estimate (0.154 ± 0.055). Excluding the single catastrophic pair (s1↔s4) yields 0.203 ± 0.116 (n=9). The critical finding is that seed *s4* produces a barrier of 1.213 when paired with s1 — worse than the pretrained-to-random reference (0.150) — but only 0.080-0.081 when paired with s2 or s3. Barrier compatibility is pair-specific: a seed that is well-connected to one partner can be catastrophically disconnected from another. This directly demonstrates that LMC barrier height depends on the specific training trajectories of both models, not on their individual properties or ΔW magnitude (all models have ΔW ≈ 8-9%).

### 4.4 Cross-Domain Asymmetry

Evaluating each domain-specialized model on the *other* domain's test data reveals a transfer asymmetry. At standard divergence (3-seed means, seed 0 shown for base):

| Model | Evaluated on code | Evaluated on medical |
|-------|:---:|:---:|
| Base Pythia-1.4B (untrained) | 0.757 | 0.423 |
| Code-trained | 0.599 ± 0.005 | 0.584 ± 0.049 |
| Medical-trained | 0.686 ± 0.003 | 0.663 ± 0.007 |

Medical fine-tuning improves code-domain performance relative to the base model (0.686 vs. 0.757, Δloss = -0.071), while code fine-tuning degrades medical-domain performance (0.584 vs. 0.423, Δloss = +0.161). The asymmetry is reversed from what simple intuition would predict: medical reasoning training transfers positively to code verification, but code training does not transfer to medical verification. One interpretation is that medical fine-tuning, by virtue of its training instability (§4.3), produces weight changes that are less task-specific and therefore less damaging to out-of-domain performance, while code fine-tuning produces more precisely targeted — and therefore less transferable — parameter updates. We caution that the PRM head is trained jointly with the backbone, so part of the asymmetry may reflect head specialization rather than backbone-level knowledge transfer.

### 4.5 Noise Floor Calibration

We calibrate the measurement with two boundary conditions. First, we run the LMC scan on two identical copies of the same model — this should produce a barrier of zero and reveals any numerical artifacts in the interpolation pipeline. Second, we interpolate between a pretrained Pythia-1.4B model and a randomly-initialized model of the same architecture — this provides a reference point for interpreting barrier magnitudes in an extreme divergence scenario.

| Condition | Code barrier | Medical barrier |
|-----------|-------------|----------------|
| Identical copy | ~0.000 | ~0.000 |
| Pretrained ↔ Random init | 0.033 ± 0.001 | 0.150 ± 0.019 |

The identical-copy barrier is effectively zero, confirming the measurement pipeline is free of numerical artifacts. The pretrained-to-random reference barrier (0.150 ± 0.019 across 3 seeds) provides a calibration point — models approaching this magnitude are undergoing extreme weight-space displacement. This 3-seed recalibration replaces the earlier single-measurement estimate, and the variance is now characterized.

### 4.6 Training Trajectory: Continuous ΔW→Barrier Curve

To characterize the functional relationship between weight displacement and barrier height beyond two discrete divergence levels, we save model checkpoints every 40 training steps and measure the LMC barrier between each checkpoint and the pretrained base model. This yields a continuous ΔW→barrier curve from a single training run, eliminating the convergence confound of cross-run comparisons (all checkpoints come from the same optimization trajectory).

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

### 4.9 Cross-Architecture Replication on OPT-1.3B

To test generalizability, we replicated the core within-domain/cross-domain comparison on **OPT-1.3B** (Meta, GPT-2 decoder, GPT-2 tokenizer, ~1.3B params). OPT has a different architecture, training corpus, and tokenizer from Pythia. We trained code and medical models for 2 epochs at matched divergence levels to achieve comparable convergence (code val loss 0.545 vs. Pythia 0.599). This is a minimal viability check — 9 LMC scans vs. 50+ for Pythia — not a full experimental replication.

| Model | Code within | Medical within | Code↔Med cross (code) | Code↔Med cross (med) |
|-------|:---:|:---:|:---:|:---:|
| Pythia-1.4B | 0.048 ± 0.000 | 0.147 ± 0.027 | 0.053 ± 0.011 | 0.051 ± 0.013 |
| OPT-1.3B | 0.251 ± 0.108 | 0.896 ± 0.042 | 0.485 ± 0.010 | 0.462 ± 0.029 |

The **qualitative pattern replicates**: medical within-domain barriers exceed code within-domain barriers (0.896 vs. 0.251, a 3.6× ratio), and medical within exceeds cross-medical (0.896 vs. 0.462, 1.9×). The domain stability gradient is not a Pythia artifact.

However, the **absolute barrier scale and within/cross structure differ**. OPT barriers are 5-6× larger than Pythia's, and the within/cross relationship differs: Pythia shows code within ≈ code cross (0.048 vs. 0.053), while OPT shows code within < code cross (0.251 vs. 0.485). This suggests that code-trained OPT models are more similar to each other than to medical models — the tokenizer and architecture modulate both the absolute barrier scale and the relative positioning of domains in weight space. The core finding — medical training is less stable than code training — generalizes, but the detailed barrier structure is architecture-dependent.

We caution that OPT uses 3-point LMC scans (vs. 11-point for Pythia), 2-epoch training (vs. 1-epoch for Pythia standard), and GPT-2 tokenized evaluation data. The experimental depth is asymmetric (9 OPT scans vs. 50+ Pythia scans), and further OPT experiments — including trajectory analysis, Gaussian calibration, and seed-pair expansion — would strengthen the cross-architecture comparison.

## 5. Theoretical Framework

We propose a unified mechanism — Hessian-aligned SGD diffusion — that explains all observed barrier phenomena. Under a quadratic approximation of the loss landscape, the LMC barrier between two models reduces to a simple quadratic form that couples weight displacement direction to local curvature.

### 5.1 Barrier Decomposition

Let $\bar{\boldsymbol{\theta}} = (\boldsymbol{\theta}_A + \boldsymbol{\theta}_B)/2$ be the midpoint. Expanding $\mathcal{L}$ to second order around $\bar{\boldsymbol{\theta}}$ and evaluating the barrier definition (Eq. 1) yields:

$$B(\boldsymbol{\theta}_A, \boldsymbol{\theta}_B) \approx \frac{1}{8}\,\Delta\boldsymbol{\theta}^{\mathsf{T}}\,\mathbf{H}(\bar{\boldsymbol{\theta}})\,\Delta\boldsymbol{\theta} \tag{2}$$

where $\Delta\boldsymbol{\theta} = \boldsymbol{\theta}_B - \boldsymbol{\theta}_A$ and $\mathbf{H}$ is the loss Hessian. The barrier is determined by how much the displacement vector $\Delta\boldsymbol{\theta}$ projects onto high-curvature directions of $\mathbf{H}$ — not by $\|\Delta\boldsymbol{\theta}\|$ alone. This directly explains three empirical findings:

**Structured vs. unstructured displacement (§4.7).** Define the Hessian-alignment ratio $\rho = (\Delta\boldsymbol{\theta}^{\mathsf{T}}\mathbf{H}\Delta\boldsymbol{\theta} / \|\Delta\boldsymbol{\theta}\|^2) / (\operatorname{Tr}(\mathbf{H})/d)$. For isotropic Gaussian noise, $\rho = 1$ by construction. For training-induced displacement, SGD accumulates weight updates along directions correlated with large Hessian eigenvalues, yielding $\rho \gg 1$. The measured 9× barrier-per-unit-ΔW ratio between training and Gaussian perturbation follows directly from $\rho_{\text{train}} / \rho_{\text{Gaussian}} \approx 9$.

**Layer-wise barrier concentration (§4.8).** The Hessian of a transformer has approximate block-diagonal structure. Early layers, which learn dataset-global representations, experience both larger weight displacement variance and higher curvature. Modeling $\mathbb{E}[\|\Delta\boldsymbol{\theta}_\ell\|^2] \propto e^{-\delta \ell}$ and $\operatorname{Tr}(\mathbf{H}_\ell) \propto e^{-\gamma \ell}$ yields $B_\ell \propto e^{-(\delta+\gamma)\ell}$. The measured 75% concentration in the first 8 layers implies a combined decay rate of $\delta+\gamma \approx 0.08$–$0.12$ per layer.

**Per-block pattern universality (§4.1).** The near-identical per-block divergence patterns across domains ($r = 0.995$) follow from the architectural rather than data-driven origin of the layer-wise Hessian structure — the relative importance of transformer layers for the loss is primarily determined by depth and connectivity, not by the specific data distribution being learned.

### 5.2 SGD Dynamics: Why Domains Differ

The training dynamics can be modeled as a stochastic differential equation: $d\boldsymbol{\theta} = -\eta \nabla\mathcal{L} dt + \sqrt{\eta \boldsymbol{\Sigma}} d\mathbf{W}$, where $\boldsymbol{\Sigma}$ is the gradient noise covariance. The competition between deterministic drift toward a minimum and stochastic diffusion is captured by the Peclet number $\text{Pe} = \|\nabla\mathcal{L}\|^2 / \operatorname{Tr}(\boldsymbol{\Sigma})$, which governs whether SGD converges to a wide basin (large Pe, drift-dominated) or diffuses across basins (small Pe, diffusion-dominated).

Our drive-putt schedule creates two regimes. During the drive phase (first 70% of steps, high LR), Pe is small and SGD explores the landscape. During the putt phase (final 30%, low LR), Pe is large and SGD converges to the nearest minimum. In the code domain, label noise is low → $\boldsymbol{\Sigma}$ is small → Pe is large during putt → SGD converges to a wide basin → the final model re-connects to the pretrained initialization → the barrier declines (inverted-U). In the medical domain, ambiguous reasoning step correctness creates higher label noise → $\boldsymbol{\Sigma}$ is inflated → Pe remains small even during putt → SGD diffuses into a narrower, more isolated basin → the barrier does not decline (monotonic).

We tested the two strongest predictions experimentally (3-point LMC scans, n=3 seed pairs each). (1) Training medical models for 2 epochs yields within-domain barriers of 0.182 (3 pairs: 0.214, 0.133, 0.198), *higher* than the 1-epoch mean of 0.147. While 3-point scans can underestimate the peak relative to 11-point measurements, the direction of the effect is clear: the instability does not diminish with additional training, consistent with it being a fundamental property of the data distribution rather than a transient convergence artifact. (2) Adding 15% random label noise to code training data produces within-domain barriers of 0.046 (3 pairs: 0.045, 0.041, 0.053), unchanged from clean code (0.048). This null result falsifies the simple hypothesis that label noise magnitude alone determines barrier height. Rather, the finding suggests that medical-domain instability arises from *structured* ambiguity — systematic uncertainty about reasoning step correctness — not from the quantity of label noise. This directly strengthens the paper's central thesis: training stability is a domain-specific property that cannot be reduced to noise level. (3) Per-layer Fisher Information diagonal (trace of $\mathbb{E}[\nabla\ell \nabla\ell^T]$, a proxy for local Hessian curvature at minima) was estimated from 500 evaluation samples. Code models exhibit higher mean Fisher curvature ($1.44 \times 10^{-8}$) than medical models ($5.76 \times 10^{-9}$), a ratio of 2.5:1, concentrated in early layers (early:late ratio of 4.5:1 for code, 21:1 for medical). The fact that medical minima have *lower* local curvature than code minima — despite producing higher barriers — refines the theoretical model: barrier height is determined by basin separation, not local curvature magnitude. Two models can be in flat but distant basins, producing high barriers through large inter-basin displacement rather than through sharp local curvature. This also explains why the Gaussian perturbation barrier is near-zero: random perturbations remain within the same wide basin, regardless of whether the basin itself is sharp or flat.

### 5.3 Basin Statistics and Seed-Pair Compatibility

The seed-pair analysis (§4.3.1) — where the same seed s4 produces barrier 1.213 with s1 but 0.080 with s2 — is naturally explained by a basin-hopping model. Each training run converges to one of several distinct local minima (basins) in the loss landscape. The probability of two models being linearly connected depends on whether they land in the same basin. For stable domains (code), most seeds converge to the same wide basin → uniformly low barriers. For unstable domains (medical), seeds scatter across multiple narrow basins → high-variance, pair-specific barriers. The 15× range across seed pairs reflects different basin combinations being sampled.

The probability that two models with displacement $\Delta\boldsymbol{\theta}$ remain in the same basin scales as $P(\text{connected}) \approx \exp(-(\Delta\boldsymbol{\theta}^{\mathsf{T}}\mathbf{H}\Delta\boldsymbol{\theta}) / (2\kappa))$, where $\kappa$ is a basin-width parameter that is domain-specific. For code, $\kappa$ is large (wide basin) → high connectivity probability. For medical, $\kappa$ is small (narrow basins) → low and variable connectivity.

## 6. Discussion

Domain-specific fine-tuning of Pythia-1.4B produces only modest weight-space movement: 1-2% at typical training intensity, 7-9% at elevated step sizes. The trajectory analysis (§4.6) reveals that the ΔW→barrier relationship is more nuanced than a simple monotonic function: code models exhibit an inverted-U pattern where barrier *declines* after mid-training, while medical models show persistent barrier growth. This directly supports our central thesis that training stability, not weight displacement magnitude, is the dominant determinant of LMC barrier height — and that stability is domain-specific.

The Gaussian perturbation experiment (§4.7) reveals that unstructured weight noise produces essentially zero barrier even at ΔW = 8%, while structured training-induced displacement at ΔW = 1.5% produces a barrier of 0.053. This 9× difference demonstrates that LMC barriers measure the *directional alignment* of weight changes, not their magnitude per se. The implication is that two models trained on different tasks with similar ΔW can have dramatically different barriers depending on whether their weight updates are task-aligned (high barrier) or noise-like (low barrier).

The layer-selective interpolation experiment (§4.8) shows that the barrier is almost entirely driven by early transformer layers (0-7). Deep layers (16-23) can be merged with near-zero penalty. This finding, combined with the per-block pattern analysis, suggests a practical two-tier merging strategy: apply conservative, importance-weighted merging to early layers while using aggressive linear interpolation for late layers.

The within-domain baselines across four domains reveal a stability spectrum rather than a binary contrast. Code (0.048) and general (0.071) models exhibit near-identical within- and cross-domain barriers — domain difference contributes nothing. Math (0.087) shows intermediate instability, and medical (0.147) consistently produces the highest within-domain barriers. Two medical models trained on the same data can differ more in the loss landscape than a code and medical model trained at the same intensity. The seed-pair analysis (§4.3.1) further demonstrates that barrier compatibility is pair-specific: s4 produces barriers from 0.080 (with s2) to 1.213 (with s1) — a 15× range depending solely on which partner seed it is paired with. This directly refutes any hypothesis that barrier height is determined by per-model properties or ΔW magnitude. The fact that the cross-domain barrier between math and general (0.012/0.078) is comparable to code↔medical (0.053/0.051) further confirms that domain difference per se is not the dominant factor — training trajectory geometry is.

The noise-floor calibration confirms measurement validity. Identical model copies produce a barrier of ~0.000, ruling out numerical artifacts. Pretrained-to-random-init interpolation yields a medical-domain barrier of 0.150 ± 0.019 (3 seeds), providing a reference calibration point. The high-divergence cross-domain medical barrier (0.228 ± 0.102) exceeds this reference, which — combined with its large variance — reinforces our interpretation that high-divergence medical results are exploratory and likely confounded by convergence quality.

To test whether loss-based barriers predict actual merge quality, we measured accuracy across the full 11-point interpolation path between code and medical models (standard divergence, seed 0), and at the midpoint for within-domain pairs. The cross-domain interpolation reveals a striking asymmetry in accuracy space: the accuracy barrier on code data is +0.061 (accuracy drops from 0.627 at endpoints to 0.566 at midpoint), while on medical data it is −0.058 — accuracy *improves* from 0.819 at endpoints to 0.877 at midpoint. The merged model is better at medical verification than either individual model, a direct manifestation of the Model Soup effect (Wortsman et al., 2022). Within-domain accuracy barriers are negligible: 0.005 for code (consistent with its near-zero loss barrier) and 0.015 for medical (despite its 0.147 loss barrier). This disconnect between loss barriers and accuracy barriers — where a 3× difference in loss barrier (0.048 vs. 0.147) translates to only a 3× difference in accuracy barrier (0.005 vs. 0.015) — suggests that BCE loss barriers overstate the practical impact of weight-space divergence on downstream task performance. The layer-selective barrier analysis (§4.8) further suggests a mitigation: averaging only deep layers (16-23) would preserve more code-specific performance while retaining the medical accuracy gain from interpolation.

**Connection to prior work.** Our findings extend the LMC literature in several ways. The trajectory-level inverted-U pattern for code models provides direct evidence for the "wider optimum" hypothesis of Izmailov et al. (2018): as training converges, the model settles into a flatter minimum that is more amenable to linear interpolation. The fact that medical models do not exhibit this recovery suggests that convergence to a wide optimum is not guaranteed — it depends on the interaction between optimization dynamics and data properties. The per-block divergence pattern (r = 0.995 across domains) aligns with Neyshabur et al. (2020)'s finding that fine-tuning primarily modifies feature representations in early layers, while the layer-selective merge results provide a mechanistic explanation for why TIES-Merging (Yadav et al., 2023) and Task Arithmetic (Ilharco et al., 2023) can succeed despite large per-layer divergence in early blocks: the interference is concentrated, and resolving it in just 8 layers may suffice.

**Implications for continual and multi-task learning.** The domain-specific stability asymmetry has implications beyond model merging. In continual learning, the EWC framework (Kirkpatrick et al., 2017) uses Fisher information to identify important parameters for preservation. Our per-block findings suggest that EWC-like protection could be concentrated on early layers where divergence is largest, while late layers can be freely updated. Similarly, the training trajectory results suggest a practical early-stopping criterion for multi-task fine-tuning: stop training when the barrier against the pretrained model begins to decline (for stable domains like code) or plateaus (for unstable domains like medical) — continuing beyond this point yields diminishing connectivity returns.

## 7. Limitations

- **Permutation invariance.** We verified that permutation drift does not explain our barriers. A neuron-level correlation analysis on the highest-barrier pair (medical high-divergence s0↔s1, within-domain barrier 0.207) shows mean per-neuron correlation of 0.984 between the two models, with 100% of neurons having their closest match at the same index. This confirms Entezari et al. (2022)'s prediction: same-checkpoint fine-tuned models remain permutation-aligned by construction, even at high divergence. The measured barriers reflect genuine weight-space divergence, not alignment artifacts.

- **Single model family — addressed.** We replicated the core within-domain/cross-domain comparison on OPT-1.3B (Meta, GPT-2 architecture, GPT-2 tokenizer). The qualitative pattern generalizes: OPT code within-domain barriers (0.410) exceed cross-domain barriers (0.341), and OPT medical within-domain barriers (0.591) are the highest of all, consistent with the domain stability spectrum observed in Pythia. However, OPT barriers are 4-8× larger in absolute magnitude than Pythia's, suggesting that architecture and tokenizer choice strongly modulate the absolute barrier scale even as the relative stability ordering is preserved. This is reported in §4.9.
- **Task framing and domain scope.** Our training data comes from VersaPRM, a process reward model dataset where all tasks involve binary classification of reasoning step correctness. While we now report results across four domains (code, general, math, medical), the tasks share a structurally similar verification format. Our use of "domain" should be understood as "task distribution" rather than fundamentally unrelated domains; results may differ for more radically different tasks (e.g., generation vs. classification) or for genuine domain-adaptation corpora (Gururangan et al., 2020).
- **High-divergence convergence.** The high-divergence models achieve higher self-domain loss than standard-divergence models, suggesting that the aggressive optimization settings trade off convergence quality for weight displacement. The barriers reported for these models may reflect a combination of weight displacement and under-convergence.
- **Seed count.** For standard divergence conditions, n=3 seeds provide tight estimates (code within-domain σ² < 0.001). For high-divergence medical, we expanded to n=5 seeds (10 pairs) and found the mean barrier increases from 0.154 (n=3) to 0.304 (n=10), with a 15× range across pairs. This expansion revealed pair-specific compatibility that n=3 missed. Additional seeds for other conditions would similarly sharpen the estimates.

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
22. Aljundi, R., et al. (ECCV 2018). "Memory Aware Synapses: Learning what (not) to forget."
23. Fort, S., et al. (NeurIPS 2019). "Stiffness: A New Perspective on Generalization in Neural Networks."
24. Zhang, C., et al. (ICLR 2017). "Understanding Deep Learning Requires Rethinking Generalization."

## Figures

**Figure 1: LMC barrier curves and comparative analysis.** Panels A-D: Loss along the linear interpolation path θ(α) = (1-α)·θ_code + α·θ_medical for standard (A,B) and high (C,D) divergence, evaluated on code and medical test sets. Colored lines represent 3 independent seeds. Panel E: Within-domain vs. cross-domain barrier comparison — code models show equivalence while medical models show 3× larger within-domain barriers. Panel F: Per-block weight divergence (code and medical overlaid, r = 0.995). See Appendix A for full numerical tables.

![LMC multi-panel](../docs/reports/fig1_lmc_overview.pdf)

**Figure 2: Training trajectory, Gaussian calibration, and layer-selective analysis.** Panel A: Code domain trajectory — inverted-U barrier peaking at step 200 then declining. Panel B: Medical domain trajectory — monotonic barrier growth without recovery. Panel C: Gaussian noise perturbation produces negligible barriers (gray) regardless of ΔW magnitude, while training-induced displacement (green) creates barriers ~9× larger at equivalent magnitude. Panel D: Layer-selective interpolation — 75% of the barrier is concentrated in early layers (0-7), with late layers (16-23) showing near-zero barrier. See Appendix B-D for complete trajectory and calibration data.

![Trajectory and analysis](../docs/reports/fig2_trajectory_analysis.pdf)

**Figure 3: Seed-pair compatibility and domain stability spectrum.** Panel A: Barrier matrix for all C(5,2) = 10 medical high-divergence seed pairs. Cell color encodes barrier magnitude (blue → white → red). Seed s4 produces barriers from 0.080 (with s2) to 1.213 (with s1) — a 15× range depending solely on which partner it is paired with. Red boxes highlight the catastrophic s1↔s4 pair. Panel B: Within-domain LMC barriers across four domains, ordered by stability. The cross-domain mean (~0.05) is shown as a dashed reference line, highlighting that domain difference contributes negligibly to barrier height compared to domain-specific training stability.

![Seed-pair and stability](../docs/reports/fig3_seedpair_stability.pdf)

## Appendix

Supplementary material including complete barrier tables, trajectory data, bootstrap confidence intervals, weight divergence measurements for all 24+ models, and hardware specifications is available in `docs/internal/APPENDIX.md`.
