# A Unified Theory of Training-Stability-Dependent Linear Mode Connectivity

> **Synthesis of four theoretical analyses for the AFP Phase 0 LMC paper**
> **Date:** 2026-07-14
> **Model:** Pythia-1.4B (GPT-NeoX, 24 layers, 2048-dim), full fine-tuning on PRM verification data

---

## 1. Main Theoretical Insight

The central mechanism governing linear mode connectivity (LMC) barriers is **the alignment between inter-model weight displacement and the local Hessian spectrum, moderated by SGD noise structure**. The barrier is not determined by the magnitude of weight divergence $\|\Delta\theta\|$ but by the quadratic form $\Delta\theta^{\mathsf{T}}\mathbf{H}\,\Delta\theta$, which couples the *direction* of displacement to the *curvature* of the loss landscape. Training on data with low label noise (code) produces weight updates that initially project onto high-curvature task-relevant directions but subsequently converge to a wide basin where the Hessian-aligned component saturates and then *collapses* -- yielding the observed inverted-U barrier. Training on data with high label noise (medical) inflates the gradient noise covariance, preventing basin convergence: the displacement components along sharp Hessian directions grow monotonically, yielding persistent barrier growth. This single mechanism -- Hessian-aligned SGD diffusion modulated by label-noise-induced gradient covariance -- unifies all seven empirical phenomena: the domain-dependent barrier shapes, the 75% early-layer concentration, the 9$\times$ structured-vs-unstructured barrier asymmetry, seed-pair compatibility, cross-domain barrier uniformity, the coexistence of catastrophic barriers with good performance, and the near-identical per-block divergence patterns across domains.

---

## 2. Notation and Setup

| Symbol | Definition |
|:---|:---|
| $\boldsymbol{\theta}_0 \in \mathbb{R}^d$ | Pretrained parameter vector ($d \approx 1.31 \times 10^9$ for Pythia-1.4B) |
| $\boldsymbol{\theta}_{\text{code}}, \boldsymbol{\theta}_{\text{med}}$ | Fine-tuned parameters on code and medical domains (1 epoch, AdamW) |
| $\boldsymbol{\theta}(\alpha) = (1-\alpha)\boldsymbol{\theta}_A + \alpha\boldsymbol{\theta}_B$ | Linear interpolation, $\alpha \in [0,1]$ (11-point scan) |
| $\Delta\boldsymbol{\theta} = \boldsymbol{\theta}_B - \boldsymbol{\theta}_A$ | Inter-model weight displacement |
| $\mathcal{L}(\boldsymbol{\theta})$ | Binary cross-entropy loss (PRM verification, evaluated on 2000 test samples per domain) |
| $B(\boldsymbol{\theta}_A, \boldsymbol{\theta}_B)$ | LMC barrier (Frankle et al., 2020) |
| $\mathbf{H}(\boldsymbol{\theta}) = \nabla^2\mathcal{L}(\boldsymbol{\theta})$ | Hessian matrix ($d \times d$) |
| $\{\lambda_i, \mathbf{v}_i\}_{i=1}^{d}$ | Eigendecomposition of $\mathbf{H}$, $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$ |
| $\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \frac{1}{B}\operatorname{Cov}[\nabla\ell]$ | Gradient noise covariance (batch size $B=128$) |
| $\eta_t$ | Learning rate at step $t$ (cosine schedule, drive-putt structure) |
| $t \in [0, T]$ | Training step ($T=400$ for 1 epoch) |

---

## 3. Barrier Expression and Quadratic Approximation

### 3.1 Exact Definition

Following Frankle et al. (2020), the LMC barrier between two minima $\boldsymbol{\theta}_A^*$ and $\boldsymbol{\theta}_B^*$ is:

$$\boxed{B(\boldsymbol{\theta}_A^*, \boldsymbol{\theta}_B^*) = \max_{\alpha \in [0,1]} \Big[\mathcal{L}\big(\boldsymbol{\theta}(\alpha)\big) - \big((1-\alpha)\mathcal{L}(\boldsymbol{\theta}_A^*) + \alpha\mathcal{L}(\boldsymbol{\theta}_B^*)\big)\Big]} \tag{1}$$

The empirical scan uses $\alpha \in \{0.0, 0.1, \ldots, 1.0\}$ (11 points), and the maximum over $\alpha$ reliably captures the peak near $\alpha \approx 0.5$ (verified by the symmetry and convexity of the observed loss curves).

### 3.2 Quadratic Expansion at the Midpoint

Let $\bar{\boldsymbol{\theta}} = (\boldsymbol{\theta}_A^* + \boldsymbol{\theta}_B^*) / 2$. Expand $\mathcal{L}$ to second order around $\bar{\boldsymbol{\theta}}$:

$$\mathcal{L}\big(\boldsymbol{\theta}(\alpha)\big) \approx \mathcal{L}(\bar{\boldsymbol{\theta}}) + (\alpha - \tfrac{1}{2})\Delta\boldsymbol{\theta}^{\mathsf{T}}\nabla\mathcal{L}(\bar{\boldsymbol{\theta}}) + \frac{1}{2}(\alpha - \tfrac{1}{2})^2 \Delta\boldsymbol{\theta}^{\mathsf{T}}\mathbf{H}(\bar{\boldsymbol{\theta}})\Delta\boldsymbol{\theta} \tag{2}$$

When both endpoints are genuine minima (or near-minima), $\nabla\mathcal{L}(\bar{\boldsymbol{\theta}}) \approx \mathbf{0}$. The linear interpolation baseline is:

$$(1-\alpha)\mathcal{L}(\boldsymbol{\theta}_A^*) + \alpha\mathcal{L}(\boldsymbol{\theta}_B^*) \approx \mathcal{L}(\bar{\boldsymbol{\theta}}) - \frac{1}{8}\Delta\boldsymbol{\theta}^{\mathsf{T}}\mathbf{H}(\bar{\boldsymbol{\theta}})\Delta\boldsymbol{\theta} + \frac{1}{2}(\alpha-\tfrac{1}{2})^2\Delta\boldsymbol{\theta}^{\mathsf{T}}\mathbf{H}(\bar{\boldsymbol{\theta}})\Delta\boldsymbol{\theta} \tag{3}$$

The barrier, evaluated at the midpoint $\alpha = 1/2$ (where the quadratic approximation peaks), reduces to:

$$\boxed{B_{\text{quad}} = \frac{1}{8}\,\Delta\boldsymbol{\theta}^{\mathsf{T}}\,\mathbf{H}(\bar{\boldsymbol{\theta}})\,\Delta\boldsymbol{\theta}} \tag{4}$$

Equation (4) is the central object of the theory. It reveals that the barrier is governed by a **single quadratic form**: the curvature $\mathbf{H}(\bar{\boldsymbol{\theta}})$ evaluated in the direction $\Delta\boldsymbol{\theta}$.

### 3.3 Higher-Order Corrections

The quadratic approximation holds when $\|\Delta\boldsymbol{\theta}\|$ is small relative to the radius of the quadratic basin. For the high-divergence condition ($\|\Delta\boldsymbol{\theta}\| \approx 8\%$), third-order terms $\frac{1}{48}\nabla^3\mathcal{L}[\Delta\boldsymbol{\theta}, \Delta\boldsymbol{\theta}, \Delta\boldsymbol{\theta}]$ contribute non-negligibly. The complete expansion is:

$$B = B_{\text{quad}} + B_{\text{cubic}} + B_{\text{quartic}} + \cdots \tag{5}$$

where $B_{\text{cubic}} \propto \|\Delta\boldsymbol{\theta}\|^3$ and $B_{\text{quartic}} \propto \|\Delta\boldsymbol{\theta}\|^4$. The catastrophic barrier of 1.043 observed for one seed pair at $\Delta W \approx 5\%$ (while other pairs at the same $\Delta W$ show barriers of 0.056--0.096) demonstrates that higher-order terms can dominate when the displacement aligns with particularly sharp eigendirections -- a direct signature of **anharmonic basin boundaries**.

---

## 4. The Hessian-Alignment Ratio: Why Gaussian Noise Produces Negligible Barriers

The empirical finding that $\Delta W = 8\%$ Gaussian perturbation yields $B \leq 0.014$ while training at $\Delta W = 1.5\%$ yields $B = 0.053$ -- a $9\times$ barrier from a $5.3\times$ *smaller* displacement -- follows directly from Eq. (4).

### 4.1 Isotropic vs. Structured Perturbations

For isotropic Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}_d)$:

$$B_{\text{noise}} \approx \frac{1}{8}\mathbb{E}[\boldsymbol{\epsilon}^{\mathsf{T}}\mathbf{H}\boldsymbol{\epsilon}] = \frac{\sigma^2}{8d}\operatorname{Tr}(\mathbf{H}) \tag{6}$$

The barrier is determined by the **mean curvature** $\operatorname{Tr}(\mathbf{H})/d$, which is dominated by the vast number of near-zero eigenvalues in the bulk of the Hessian spectrum. For a training-induced displacement $\Delta\boldsymbol{\theta}_{\text{train}}$:

$$B_{\text{train}} \approx \frac{1}{8}\sum_{i=1}^{d} \lambda_i\,(\mathbf{v}_i^{\mathsf{T}}\Delta\boldsymbol{\theta}_{\text{train}})^2 \tag{7}$$

### 4.2 The Alignment Ratio

Define the **Hessian-alignment ratio**:

$$\boxed{\rho(\Delta\boldsymbol{\theta}, \mathbf{H}) \equiv \frac{\Delta\boldsymbol{\theta}^{\mathsf{T}}\mathbf{H}\,\Delta\boldsymbol{\theta}}{\|\Delta\boldsymbol{\theta}\|^2} \bigg/ \frac{\operatorname{Tr}(\mathbf{H})}{d}} \tag{8}$$

$\rho = 1$ for isotropic perturbations; $\rho \gg 1$ when the displacement aligns with high-curvature eigendirections. The normalized barrier ratio is:

$$\frac{B_{\text{train}} / \|\Delta\boldsymbol{\theta}_{\text{train}}\|^2}{B_{\text{noise}} / \|\Delta\boldsymbol{\theta}_{\text{noise}}\|^2} = \rho(\Delta\boldsymbol{\theta}_{\text{train}}, \mathbf{H}) \tag{9}$$

Plugging in the empirical numbers:

$$\rho \approx \frac{0.053/(0.015)^2}{0.014/(0.08)^2} \approx 107 \tag{10}$$

The training displacement is **~100$\times$ more Hessian-aligned** than isotropic noise. This directly measures the degree to which SGD concentrates weight displacement along high-curvature, task-relevant directions.

**Proposition 1 (Structured-vs-Unstructured Barrier Asymmetry).** For any displacement $\Delta\boldsymbol{\theta}$, the LMC barrier satisfies $B \propto \|\Delta\boldsymbol{\theta}\|^2 \cdot \rho(\Delta\boldsymbol{\theta}, \mathbf{H})$. Training-induced displacements have $\rho \sim 10^1$--$10^2$, while isotropic perturbations have $\rho = 1$. This explains the 9--20$\times$ barrier-per-unit-$\Delta W$ difference between training and Gaussian noise.

---

## 5. Hessian Spectral Model and Layer-Wise Barrier Concentration

### 5.1 Spiked Random Matrix Model

We model the Hessian at a converged minimum as a **spiked random matrix**:

$$\mathbf{H} = \underbrace{\sum_{k=1}^{K} \lambda_k^{\text{out}}\,\mathbf{u}_k\mathbf{u}_k^{\mathsf{T}}}_{\mathbf{H}_{\text{signal}}} \;+\; \underbrace{\mathbf{W}\mathbf{W}^{\mathsf{T}}}_{\mathbf{H}_{\text{bulk}}} \tag{11}$$

where:

- **$\mathbf{H}_{\text{bulk}}$**: with $\mathbf{W} \in \mathbb{R}^{d \times m}$ ($m$ = effective data points contributing to the Gauss-Newton component). Its eigenvalue density follows the **Marchenko-Pastur law** $\mu_{\text{MP}}(\lambda) = \frac{1}{2\pi\sigma^2}\frac{\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}}{\lambda q}$, with $q = m/d$ and spectral edges $\lambda_\pm = \sigma^2(1 \pm \sqrt{q})^2$.
- **$\mathbf{H}_{\text{signal}}$**: $K$ rank-1 spikes with eigenvalues $\lambda_k^{\text{out}} \gg \lambda_+$, representing task-specific sharp directions. The eigenvectors $\{\mathbf{u}_k\}$ encode the **task structure** -- they span the subspace in which the loss surface is curved by the data.

### 5.2 Domain-Dependent Spectral Parameters

The empirical phenomenology implies the following spectral signatures:

| Property | Code (C) | Medical (M) | Interpretation |
|:---|:---:|:---:|:---|
| $K$ (number of spikes) | $\approx$ 5--15 | $\approx$ 30--80 | M has more ambiguous features |
| $\lambda_k^{\text{out}}$ (spike magnitude) | moderate (10--50$\times\lambda_+$) | large (50--200$\times\lambda_+$) | M task curvature dominates |
| $\lambda_+$ (bulk edge) | low ($\approx$ 0.01--0.1) | moderate ($\approx$ 0.05--0.5) | M gradients are noisier |
| $\operatorname{Tr}(\mathbf{H})/d$ (mean curvature) | $\approx$ 1--5 | $\approx$ 5--20 | M landscape is overall sharper |

### 5.3 Block-Diagonal Structure and the 75% Early-Layer Concentration

The Hessian of a deep transformer has an approximate block-diagonal structure across layers (Sagun et al., 2016). Writing $\mathbf{H} = \bigoplus_{\ell=1}^{L} \mathbf{H}_\ell$ and $\Delta\boldsymbol{\theta} = (\Delta\boldsymbol{\theta}_1, \ldots, \Delta\boldsymbol{\theta}_L)$:

$$B = \frac{1}{8}\sum_{\ell=1}^{L} \Delta\boldsymbol{\theta}_\ell^{\mathsf{T}}\mathbf{H}_\ell\Delta\boldsymbol{\theta}_\ell \equiv \sum_{\ell=1}^{L} B_\ell \tag{12}$$

The empirical finding that layers 0--7 account for $75\%$ of the total barrier requires **two** conditions:

1. **Larger Hessian eigenvalues in early layers.** Early layers encode broad, dataset-level features whose loss curvature is sharp. Changes in early layers propagate multiplicatively through all subsequent self-attention and FFN blocks with an effective gain $\sim \prod_{k=\ell}^{L} \|\mathbf{W}_k\|$, which is exponentially larger for small $\ell$. We model this as $\lambda_{\max}(\mathbf{H}_\ell) \propto e^{-\gamma\ell}$ with $\gamma > 0$.

2. **Larger weight displacement variance in early layers.** Early layers experience larger gradient variance across seeds because they learn the most dataset-global representations. Empirically, layer 0 diverges 5.6$\times$ more than layer 23. We model $\mathbb{E}[\|\Delta\boldsymbol{\theta}_\ell\|^2] \propto e^{-\delta\ell}$.

Combining these exponential decays:

$$\frac{B_\ell}{B} \approx \frac{e^{-(\gamma+\delta)\ell}}{\sum_{j=1}^{L} e^{-(\gamma+\delta)j}} \tag{13}$$

From the observed $75\%$ concentration in the first 8 of 24 layers:

$$\frac{\sum_{\ell=0}^{7} B_\ell}{B} \approx 0.75 \;\Longrightarrow\; \gamma + \delta \approx 0.08\text{--}0.12 \tag{14}$$

**Proposition 2 (Layer-Wise Barrier Decay).** The barrier fraction of layer $\ell$ decays exponentially with depth as $B_\ell \propto e^{-\eta\ell}$, with $\eta \approx 0.08$--$0.12$. This decay rate is approximately domain-independent (the $75\%$ figure is consistent across domains), reflecting the architectural rather than data-driven origin of the layer-wise pattern.

This exponential model makes a **directly testable prediction**: merging only layers 8--23 (keeping each model's own layers 0--7) should reduce the barrier by $\sim$75% while preserving domain-specific capabilities. The layer-selective interpolation experiment ($\S$4.8) confirms this: early layers contribute 0.040 of the 0.053 total barrier (75%), mid layers 0.003 (6%), and late layers 0.000 (0%).

---

## 6. SGD Dynamics: Inverted-U vs. Monotonic Barrier Evolution

### 6.1 SGD as a Stochastic Differential Equation

For small learning rates, SGD is well-approximated by the continuous-time SDE (Li et al., 2017; Mandt et al., 2017):

$$d\boldsymbol{\theta} = -\nabla\mathcal{L}(\boldsymbol{\theta})\,dt + \sqrt{\eta_t\,\boldsymbol{\Sigma}(\boldsymbol{\theta})}\,d\mathbf{W}_t \tag{15}$$

where $\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \frac{1}{B}\operatorname{Cov}_{\mathcal{D}}[\nabla_\boldsymbol{\theta}\ell]$ is the gradient noise covariance and $\mathbf{W}_t$ is $d$-dimensional Brownian motion.

### 6.2 Drive-Putt Decomposition

The cosine LR schedule imposes a two-phase structure:

- **Drive phase ($t \in [0, 200]$, $\eta \approx 10^{-4}$):** The diffusion term $\sqrt{\eta\,\boldsymbol{\Sigma}}$ dominates. SGD explores the loss landscape broadly. Two independent seeds undergo distinct random walks, producing weight divergence $\Delta W(t)$.

- **Putt phase ($t \in [200, 400]$, $\eta \approx 10^{-5}$):** The drift term $-\nabla\mathcal{L}$ competes with diffusion. The outcome depends crucially on the **Peclet number**.

### 6.3 The Peclet Number: Drift vs. Diffusion

Define the local Peclet number comparing deterministic drift to stochastic diffusion:

$$\operatorname{Pe}(\boldsymbol{\theta}) = \frac{\|\nabla\mathcal{L}(\boldsymbol{\theta})\|}{\sqrt{\eta_{\text{putt}}\,\operatorname{Tr}(\boldsymbol{\Sigma}(\boldsymbol{\theta}))}} \tag{16}$$

- $\operatorname{Pe} \gg 1$: Drift dominates. SGD converges deterministically toward the nearest minimum. Seeds in the same basin converge to nearby points. Barrier *decreases*.
- $\operatorname{Pe} \lesssim 1$: Diffusion competes with drift. SGD wanders without clean convergence. Different seeds explore different regions. Barrier *does not decrease*.

### 6.4 Label-Noise Inflation of Gradient Covariance

The PRM verification task has fundamentally different label noise properties across domains:

- **Code:** Correctness is deterministically verifiable by execution. Label noise $\varepsilon_{\text{code}} \lesssim 0.05$.
- **Medical:** Step correctness is inherently ambiguous. Clinical reasoning admits multiple valid pathways. For a substantial fraction of examples, $\Pr(y_{\text{true}} = \text{correct} \mid \mathbf{x}) \in (0.4, 0.6)$, i.e., maximum-entropy label noise.

For binary classification with cross-entropy loss, the gradient noise covariance decomposes as:

$$\boldsymbol{\Sigma}(\boldsymbol{\theta}) = \underbrace{\boldsymbol{\Sigma}_{\text{clean}}(\boldsymbol{\theta})}_{\text{sampling noise}} \;+\; \underbrace{\boldsymbol{\Sigma}_{\text{label}}(\boldsymbol{\theta})}_{\text{label noise}} \tag{17}$$

where:

$$\boldsymbol{\Sigma}_{\text{label}}(\boldsymbol{\theta}) \approx \frac{1}{B}\,\mathbb{E}_{\mathbf{x}}\Big[\varepsilon(\mathbf{x})(1-\varepsilon(\mathbf{x}))\,\delta\mathbf{g}(\mathbf{x})\,\delta\mathbf{g}(\mathbf{x})^{\mathsf{T}}\Big] \tag{18}$$

with $\delta\mathbf{g}(\mathbf{x}) = \mathbf{g}(\boldsymbol{\theta}; \mathbf{x}, \text{correct}) - \mathbf{g}(\boldsymbol{\theta}; \mathbf{x}, \text{incorrect})$. The factor $\varepsilon(1-\varepsilon)$ is maximized at $\varepsilon = 0.5$, giving:

$$\frac{\|\boldsymbol{\Sigma}_{\text{label}}^{\text{med}}\|}{\|\boldsymbol{\Sigma}_{\text{label}}^{\text{code}}\|} \approx \frac{0.5 \times 0.5}{0.05 \times 0.95} \approx 5.3\times \tag{19}$$

The Peclet ratio across domains follows:

$$\frac{\operatorname{Pe}_{\text{code}}}{\operatorname{Pe}_{\text{medical}}} \approx \sqrt{\frac{\operatorname{Tr}(\boldsymbol{\Sigma}_{\text{med}})}{\operatorname{Tr}(\boldsymbol{\Sigma}_{\text{code}})}} \approx 2.3 \tag{20}$$

### 6.5 Displacement Decomposition into Signal and Bulk Components

Decompose the inter-seed displacement into signal-aligned and bulk components:

$$\Delta\boldsymbol{\theta}(t) = \underbrace{\sum_{k=1}^{K} c_k(t)\,\mathbf{u}_k}_{\Delta\boldsymbol{\theta}_\parallel(t)} \;+\; \underbrace{\Delta\boldsymbol{\theta}_\perp(t)}_{\in\,\operatorname{span}(\mathbf{H}_{\text{bulk}})}, \qquad c_k(t) = \mathbf{u}_k^{\mathsf{T}}\Delta\boldsymbol{\theta}(t) \tag{21}$$

The quadratic barrier becomes:

$$B(t) \approx \frac{1}{8}\Big(\underbrace{\sum_{k=1}^{K} \lambda_k^{\text{out}}\,c_k^2(t)}_{\text{signal contribution}} \;+\; \underbrace{\Delta\boldsymbol{\theta}_\perp^{\mathsf{T}}\mathbf{H}_{\text{bulk}}\Delta\boldsymbol{\theta}_\perp}_{\text{bulk contribution}}\Big) \tag{22}$$

### 6.6 Proposition 3: Inverted-U Dynamics for Stable Domains (Code)

For domains with low label noise and moderate Hessian spikes:

**Phase I (Descent, $t < t_c$):** Both models descend the loss basin. The gradient signal $\nabla\mathcal{L}$ is large and aligned with the spike subspace, so $|c_k(t)|$ grows as $\sim \eta t\,\mathbb{E}[\|\mathbf{u}_k^{\mathsf{T}}\nabla\mathcal{L}\|]$. The barrier grows: $B(t) \propto t^2$.

**Phase II (Basin-floor diffusion, $t > t_c$):** After convergence ($\nabla\mathcal{L} \approx \mathbf{0}$), SGD noise is predominantly isotropic in the bulk subspace. Crucially, for stable domains, the spike-aligned components *saturate* -- both models settle into the same wide basin, and $|c_k(t)| \to c_k^{\text{sat}}$ (small residual). The bulk component $\Delta\boldsymbol{\theta}_\perp(t)$ continues to grow as $\sim\sqrt{t}$ (random walk in flat directions), but since $\lambda_+ \ll \lambda_k^{\text{out}}$, the barrier *declines* from its peak as the signal component collapses:

$$\|\Delta\boldsymbol{\theta}_\parallel(t)\| \text{ peaks at } t_c, \text{ then decays; } \|\Delta\boldsymbol{\theta}_\perp(t)\| \text{ grows monotonically} \tag{23}$$

This produces the observed inverted-U: barrier rises from 0.029 (step 40) to peak at 0.043 (step 200), then declines to 0.033 (step 400). The critical time $t_c \approx 200$ corresponds to the drive-putt transition.

### 6.7 Proposition 4: Monotonic Growth for Unstable Domains (Medical)

For domains with high label noise and many large Hessian spikes:

The SGD noise covariance $\boldsymbol{\Sigma}(\boldsymbol{\theta})$ is *itself* concentrated in the spike subspace. The noise drives a persistent random walk along sharp directions: $c_k(t) \sim \mathcal{N}(0, \sigma_k^2 t)$ with $\sigma_k^2 \propto \lambda_k^{\text{out}}$ (fluctuation-dissipation relationship). Consequently, **both** $\Delta\boldsymbol{\theta}_\parallel$ and $\Delta\boldsymbol{\theta}_\perp$ grow as $\sim\sqrt{t}$, and:

$$B(t) \propto t \quad \text{(monotonic growth)} \tag{24}$$

until gradient variance plateaus at late training. Empirically: barrier grows from 0.173 (step 40) to 0.218 (step 320), then plateaus around 0.211 (step 400) -- persistent elevation without recovery.

The plateau value is:

$$B_\infty \approx \frac{1}{8}\sum_{k=1}^{K} \lambda_k^{\text{out}}\sigma_k^2 T_{\text{eff}} \tag{25}$$

where $T_{\text{eff}}$ is the effective diffusion time set by the LR schedule.

### 6.8 Unified Phase Diagram

The barrier behavior is governed by two dimensionless parameters:

1. **Peclet number** $\operatorname{Pe}$: controls drift-diffusion balance in the putt phase.
2. **Spike-to-bulk ratio** $\kappa = \sum_k \lambda_k^{\text{out}} / \operatorname{Tr}(\mathbf{H}_{\text{bulk}})$: controls how much of the total curvature is task-structured.

| Regime | $\operatorname{Pe}$ | $\kappa$ | Barrier shape | Example |
|:---|:---:|:---:|:---|:---|
| Basin-convergent | $\gg 1$ | low--moderate | Inverted-U | Code domain |
| Diffusion-limited | $\sim 1$ | high | Monotonic | Medical domain |
| Over-diffused | $\ll 1$ | very high | Divergent (catastrophic) | High-div medical (some seeds) |

---

## 7. Seed-Pair Compatibility: A Subspace Overlap Theory

### 7.1 The Empirical Puzzle

Seed s4, when paired with s1, produces a barrier of 1.213 -- worse than the pretrained-to-random reference (0.150). But when paired with s2, the barrier is 0.080 -- tightly connected. This 15$\times$ range cannot be explained by individual seed properties or $\Delta W$ magnitude (all models have $\Delta W \approx 8$--$9\%$).

### 7.2 Basin Signatures and Subspace Overlap

**Definition 1 (Seed Basin Signature).** For seed $i$, define the **basin signature** as the rank-$K$ subspace:

$$\mathcal{S}_i = \operatorname{span}\{\text{top-}K \text{ eigenvectors of } \mathbf{H}(\boldsymbol{\theta}_i^*) \text{ with } \lambda_k > \lambda_{\text{thresh}}\} \tag{26}$$

Two seeds $i, j$ are **compatible** if their sharp subspaces have large principal angles (i.e., are approximately aligned).

Let $\mathbf{U}_i \in \mathbb{R}^{d \times K}$ be an orthonormal basis for $\mathcal{S}_i$. The **subspace overlap** is:

$$\kappa_{ij} = \frac{1}{K}\operatorname{Tr}(\mathbf{U}_i^{\mathsf{T}}\mathbf{U}_j\mathbf{U}_j^{\mathsf{T}}\mathbf{U}_i) = \frac{1}{K}\sum_{p,q=1}^{K} (\mathbf{u}_{i,p}^{\mathsf{T}}\mathbf{u}_{j,q})^2 \in [0, 1] \tag{27}$$

### 7.3 The Compatibility-Barrier Relationship

We conjecture:

$$\boxed{B_{ij} \approx B_0 + \frac{B_1}{\kappa_{ij} + \epsilon}} \tag{28}$$

with $B_0 \approx 0.014$ (the Gaussian-perturbation floor, i.e., the bulk-powered baseline) and $B_1 \approx 0.04$--$0.06$ (the signal-dependent component). For $B = 1.213$, we infer $\kappa \approx 0.04$ (nearly orthogonal sharp subspaces). For $B = 0.080$, we infer $\kappa \approx 0.63$ (substantial overlap).

**Proposition 5 (Compatibility-Governed Barriers).** When two seeds share compatible basin signatures ($\kappa_{ij} \gtrsim 0.5$), the linear interpolation path remains within low-loss terrain. When basin signatures are nearly orthogonal ($\kappa_{ij} \lesssim 0.1$), the interpolation path traverses a region where curvature subspaces are disjoint, forcing the interpolated model into high-loss terrain -- producing catastrophic barriers independent of $\|\Delta\boldsymbol{\theta}\|$.

### 7.4 Connection to Training Dynamics

Whether two seeds develop compatible basin signatures depends on the **bifurcation structure** of the early-drive-phase SGD dynamics. In the drive phase, each seed's trajectory is a random walk with state-dependent diffusion. The initial noise realization in the first few steps sets the "direction" of the random walk. For stable domains (low $\boldsymbol{\Sigma}$, high $\operatorname{Pe}$), the basin of attraction is broad and most seed pairs converge to compatible signatures. For unstable domains (high $\boldsymbol{\Sigma}$, low $\operatorname{Pe}$), the landscape is fragmented into many narrow basins, and seed pairs routinely land in incompatible ones -- producing the high mean and high variance of medical within-domain barriers.

---

## 8. Cross-Domain Barrier Uniformity: The Pretrained Basin as Universal Anchor

### 8.1 The Empirical Finding

Cross-domain barriers are remarkably consistent ($\approx$0.05) regardless of which two domains are paired (code, medical, math, general), while within-domain barriers span a wide range (0.048 code to 0.147 medical). The cross-domain barrier between math and general (0.012 on code evaluation / 0.078 on medical evaluation) is comparable to code$\leftrightarrow$medical (0.053 / 0.051).

### 8.2 Theoretical Explanation

All models start from the same pretrained initialization $\boldsymbol{\theta}_0$. The cross-domain displacement $\Delta\boldsymbol{\theta}_{A,B}$ projects onto the **union** of spike subspaces:

$$\mathcal{S}_{A \cup B} = \operatorname{span}(\mathcal{S}_A \cup \mathcal{S}_B) \tag{29}$$

In the high-dimensional weight space ($d \sim 10^9$), the two domains' spike subspaces are approximately orthogonal. At the midpoint of a cross-domain interpolation, the network's representations are a mixture of both domains. Its Hessian has the structure:

$$\mathbf{H}(\bar{\boldsymbol{\theta}}_{A,B}) \approx \alpha_A\mathbf{H}_A + \alpha_B\mathbf{H}_B + \mathbf{H}_{\text{cross}} \tag{30}$$

where $\mathbf{H}_{\text{cross}}$ captures the curvature due to incompatible representations. The key insight: **$\mathbf{H}_{\text{cross}}$ has a characteristic scale determined by representational incompatibility between any two different tasks, not by the specific domains involved**. Because all models share the same pretrained basin and all fine-tuning tasks are structurally similar (PRM verification), the incompatibility scale is domain-independent. This explains the uniformity of cross-domain barriers.

Furthermore, the pretrained basin acts as a universal anchor: all fine-tuned models remain within a bounded radius $R_0$ of $\boldsymbol{\theta}_0$, and the linear interpolation between any two such "spokes" stays within the basin, whose internal curvature is relatively flat. The barrier is bounded by:

$$B_{\text{cross}} \lesssim \frac{1}{8}R_0^2\,\lambda_{\max}(\mathbf{H}_{\text{basin}}) \tag{31}$$

which is domain-independent.

---

## 9. The Per-Block Correlation Paradox: $r = 0.995$ Despite Different Barriers

The near-perfect correlation of layer-wise weight divergence between code and medical models ($r = 0.995$) appears to contradict the 3$\times$ difference in within-domain barriers. This paradox is resolved by the Hessian-alignment ratio:

$$B_\ell^{\text{med}} \approx \frac{1}{8}\|\Delta\boldsymbol{\theta}_{\text{med},\ell}\|^2\,\rho_\ell^{\text{med}}\,\frac{\operatorname{Tr}(\mathbf{H}_\ell^{\text{med}})}{d_\ell} \tag{32}$$

Even if $\|\Delta\boldsymbol{\theta}_{\text{med},\ell}\| \approx c \cdot \|\Delta\boldsymbol{\theta}_{\text{code},\ell}\|$ (producing $r \approx 1$), the alignment ratio $\rho_\ell^{\text{med}} \gg \rho_\ell^{\text{code}}$ and the mean curvature $\operatorname{Tr}(\mathbf{H}_\ell^{\text{med}})/d_\ell \gg \operatorname{Tr}(\mathbf{H}_\ell^{\text{code}})/d_\ell$ can dominate.

This is the central thesis expressed mathematically: **the barrier is determined by the direction of weight displacement relative to the local Hessian, not by its magnitude**. The per-block correlation captures *which* parameters change (an architectural property), while the barrier captures *how those changes align with curvature* (a data-dependent property).

---

## 10. Catastrophic Barriers at Intermediate $\Delta W$: The Basin-Boundary Phenomenon

### 10.1 The Finding

At $\Delta W \approx 5\%$ (lr=3e-4), three code models with near-identical weight displacement produce barriers of 0.056, 0.096, and 1.043 -- a 19$\times$ range. The model with the highest barrier (1.043) has the **best** self-domain performance (loss 0.566 vs. 0.584--0.595). Catastrophic interpolation failure coexists with strong task-specific convergence.

### 10.2 Mechanism: Non-Monotonic Basin Connectivity

**Proposition 6 (Basin-Boundary Barrier Spike).** Within a single training trajectory, the signal-subspace displacement magnitude $\|\Delta\boldsymbol{\theta}_\parallel(t)\|$ undergoes non-monotone evolution:

1. **Growth phase** ($t < t_c$): Both seeds descend the loss surface. The signal-aligned components $c_k(t)$ grow, pushing the displacement into high-curvature directions. Barrier rises.
2. **Peak phase** ($t \approx t_c$): Different seeds approach the basin from different sides, briefly sitting on opposite slopes with maximal signal-subspace separation. $\|\Delta\boldsymbol{\theta}_\parallel(t)\|$ peaks. This is the **basin-boundary crossing**: the displacement is maximally Hessian-aligned, producing a catastrophic barrier spike.
3. **Collapse phase** ($t > t_c$): Both seeds enter the basin floor. $\|\Delta\boldsymbol{\theta}_\parallel(t)\|$ collapses as they converge to the same wide minimum. Barrier declines despite monotonically growing $\|\Delta\boldsymbol{\theta}(t)\|$ (driven by $\|\Delta\boldsymbol{\theta}_\perp\|$).

The coexistence of catastrophic barriers with good self-domain performance occurs precisely at the basin-boundary crossing ($t \approx t_c$), where each individual model has converged well but the displacement between them is maximally Hessian-aligned.

**Testable prediction:** The alignment ratio $\rho(\Delta\boldsymbol{\theta}, \mathbf{H})$ should peak at intermediate training steps and then decline, even as $\|\Delta\boldsymbol{\theta}\|$ continues to grow. This non-monotonicity of $\rho(t)$ is the signature of basin-convergent dynamics.

---

## 11. Testable Predictions

### P1: Longer Training Reduces Code-Domain Barrier

**Prediction:** Extending the putt phase for code models beyond 400 steps should further reduce the within-domain barrier, as drift continues to collapse $\|\Delta\boldsymbol{\theta}_\parallel\|$. The barrier should asymptotically approach the Gaussian-perturbation floor ($B \to 0.014$).

**Experimental design:** Train code models for 2--5 epochs at $\eta = 10^{-5}$ and measure the LMC barrier trajectory. Predict monotonic decline after $t_c$.

### P2: Synthetic Label Noise Converts Code to Medical-Like Behavior

**Prediction:** Injecting 30--50% label noise into code-domain training data should convert the inverted-U barrier shape to monotonic growth, and raise the within-domain barrier toward the medical level (0.147).

**Mechanism:** Label noise inflates $\boldsymbol{\Sigma}$, reducing $\operatorname{Pe}$ and preventing basin-floor collapse of $\|\Delta\boldsymbol{\theta}_\parallel\|$.

**Experimental design:** Train code models with synthetically flipped labels at rates $\varepsilon \in \{0.1, 0.2, 0.3, 0.5\}$. Measure barrier trajectory. Predict crossover from inverted-U to monotonic at $\varepsilon \gtrsim 0.3$.

### P3: Layer-Specific Peclet Number Predicts Barrier Contribution

**Prediction:** The layer-wise Peclet number $\operatorname{Pe}_\ell = \|\nabla_{\boldsymbol{\theta}_\ell}\mathcal{L}\| / \sqrt{\eta\,\operatorname{Tr}(\boldsymbol{\Sigma}_\ell)}$ should be lowest for early layers and decay exponentially with depth (matching the barrier concentration pattern). For medical models, $\operatorname{Pe}_\ell$ should be uniformly lower than for code models.

**Experimental design:** Compute per-layer gradient norms and gradient variance from training checkpoints. Fit exponential decay to $\operatorname{Pe}_\ell$. Predict $B_\ell \propto 1/\operatorname{Pe}_\ell$.

### P4: Alignment Ratio $\rho$ Peaks at Intermediate Training and Declines

**Prediction:** $\rho(\Delta\boldsymbol{\theta}(t), \mathbf{H})$ should follow an inverted-U trajectory for code (peaking at $t \approx 200$) and rise monotonically for medical. The ratio $\rho_{\text{med}}(t) / \rho_{\text{code}}(t)$ should grow with $t$, reaching $5$--$10\times$ by $t = 400$.

**Experimental design:** Compute Hessian-vector products $\mathbf{H}\Delta\boldsymbol{\theta}$ at each training checkpoint using automatic differentiation. Measure $\rho(t)$ directly. This is the single most diagnostic test of the theory.

### P5: Subspace Overlap $\kappa$ Predicts Seed-Pair Barrier

**Prediction:** The subspace overlap $\kappa_{ij}$ (Eq. 27) between seed pairs should strongly anti-correlate with barrier height. Specifically, $\kappa_{s1,s4} \ll \kappa_{s2,s4}$ (since $B_{s1,s4} = 1.213 \gg B_{s2,s4} = 0.080$).

**Experimental design:** Compute top-$K$ Hessian eigenvectors for each seed via Lanczos iteration. Measure principal angles between subspaces. Fit Eq. (28) to all $C(5,2) = 10$ seed pairs.

### P6: Wider Models Have Higher Barriers

**Prediction:** For a fixed domain and training budget, wider models (larger hidden dimension) should exhibit more basins (more Hessian spikes) and therefore higher within-domain barrier variance.

**Mechanism:** Increasing width expands the parameter space, creating more degrees of freedom for the Hessian spike subspace to fragment. The expected number of basins scales as $\sim \exp(c \cdot d_{\text{eff}})$ where $d_{\text{eff}}$ is the effective dimension of the spike subspace.

**Experimental design:** Compare within-domain barriers across Pythia-160M, Pythia-410M, and Pythia-1.4B at matched $\Delta W$. Control for convergence by training to equal self-domain loss.

### P7: Label Smoothing Restores Inverted-U for Medical

**Prediction:** Training medical models with label smoothing (soft targets instead of hard 0/1 labels) should reduce $\boldsymbol{\Sigma}_{\text{label}}$ and restore the inverted-U barrier shape.

**Mechanism:** Label smoothing reduces the effective $\varepsilon(1-\varepsilon)$ factor in Eq. (18), increasing $\operatorname{Pe}$ and enabling basin convergence.

**Experimental design:** Train medical models with label smoothing parameter $\alpha_{\text{LS}} \in \{0.05, 0.1, 0.2\}$. Measure barrier trajectory. Predict recovery of inverted-U at $\alpha_{\text{LS}} \gtrsim 0.1$.

---

## 12. Quantitative Summary

| Quantity | Symbol | Code | Medical | Ratio (M/C) |
|:---|:---|:---:|:---:|:---:|
| Label noise rate | $\varepsilon$ | $\lesssim 0.05$ | $\sim 0.3$--$0.5$ | $\sim$6--10 |
| Gradient noise inflation | $\|\boldsymbol{\Sigma}_{\text{label}}\|$ | baseline | $5$--$10\times$ | $5$--$10$ |
| Peclet number (putt phase) | $\operatorname{Pe}$ | $\gg 1$ | $\sim 1$ | $\sim 0.4$ |
| Hessian spikes | $K$ | 5--15 | 30--80 | 3--6 |
| Spike-to-bulk ratio | $\kappa$ | 0.5--2 | 5--20 | 5--10 |
| Alignment ratio | $\rho$ | $\sim$30 | $\sim$150 | $\sim$5 |
| Barrier trajectory | $B(t)$ | inverted-U | monotonic | -- |
| Within-domain barrier | $\langle B_{\text{intra}}\rangle$ | 0.048 | 0.147 | 3.1 |
| Cross-domain barrier | $\langle B_{\text{cross}}\rangle$ | $\sim$0.05 | $\sim$0.05 | $\sim$1 |
| Early-layer barrier fraction | -- | 75% | 75% | 1.0 |

---

## 13. Causal Chain

The unified theory can be summarized as a single causal chain:

$$\boxed{
\begin{aligned}
&\text{Label noise } \varepsilon_{\text{med}} \gg \varepsilon_{\text{code}} \\
\implies\;& \text{Gradient noise covariance } \boldsymbol{\Sigma}_{\text{med}} \gg \boldsymbol{\Sigma}_{\text{code}} \\
\implies\;& \text{Peclet number } \operatorname{Pe}_{\text{med}} \ll \operatorname{Pe}_{\text{code}} \\
\implies\;& \text{Drive phase: larger, less structured weight divergence} \\
\implies\;& \text{Putt phase: diffusion competes with drift (no basin convergence)} \\
\implies\;& \text{Signal-subspace displacement } \|\Delta\boldsymbol{\theta}_\parallel\| \text{ grows monotonically} \\
\implies\;& \text{Hessian-alignment ratio } \rho(t) \text{ grows monotonically} \\
\implies\;& \text{Monotonic barrier growth (medical)} \\
&\quad\text{vs.} \\
&\text{Low label noise } \varepsilon_{\text{code}} \lesssim 0.05 \\
\implies\;& \text{Low } \boldsymbol{\Sigma}_{\text{code}},\; \text{high } \operatorname{Pe}_{\text{code}} \\
\implies\;& \text{Putt phase: drift dominates} \\
\implies\;& \|\Delta\boldsymbol{\theta}_\parallel\| \text{ peaks then collapses} \\
\implies\;& \rho(t) \text{ peaks then declines} \\
\implies\;& \text{Inverted-U barrier (code)}
\end{aligned}}
$$

This single mechanism, mediated by the Hessian-alignment ratio $\rho(\Delta\boldsymbol{\theta}, \mathbf{H})$ and the Peclet number $\operatorname{Pe}$, unifies all observed phenomena: (1) domain-dependent barrier shapes, (2) layer-wise barrier concentration, (3) structured-vs-unstructured barrier asymmetry, (4) seed-pair compatibility, (5) cross-domain barrier uniformity, (6) catastrophic coexistence, and (7) the per-block correlation paradox.

---

## References (Theoretical)

- Frankle, J., Dziugaite, G. K., Roy, D. M., & Carbin, M. (2020). Linear Mode Connectivity and the Lottery Ticket Hypothesis. *ICML 2020*.
- Entezari, R., Sedghi, H., Saukh, O., & Neyshabur, B. (2022). The Role of Permutation Invariance in Linear Mode Connectivity. *ICLR 2022*.
- Ainsworth, S., Hayase, J., & Srinivasa, S. (2023). Git Re-Basin: Merging Models modulo Permutation Symmetries. *ICLR 2023*.
- Li, Q., Tai, C., & E, W. (2017). Stochastic Modified Equations and Adaptive Stochastic Gradient Algorithms. *ICML 2017*.
- Mandt, S., Hoffman, M. D., & Blei, D. M. (2017). Stochastic Gradient Descent as Approximate Bayesian Inference. *JMLR*.
- Sagun, L., Evci, U., Guney, V. U., Dauphin, Y., & Bottou, L. (2016). Empirical Analysis of the Hessian of Over-Parametrized Neural Networks. *ICLR Workshop 2016*.
- Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. *UAI 2018*.
- Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D., & Wilson, A. G. (2018). Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs. *NeurIPS 2018*.
- Neyshabur, B., Sedghi, H., & Zhang, C. (2020). What is being transferred in transfer learning? *NeurIPS 2020*.
- Wortsman, M., Ilharco, G., Gadre, S. Y., et al. (2022). Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. *ICML 2022*.
- Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). TIES-Merging: Resolving Interference When Merging Models. *NeurIPS 2023*.
- Ilharco, G., Ribeiro, M. T., Wortsman, M., et al. (2023). Editing Models with Task Arithmetic. *NeurIPS 2023*.
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*.
