# Reviewer 2 — Adversarial Review

## Paper Under Review

**Title:** Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning
**Draft:** v4 (2026-07-13)

---

## Overall Assessment: REJECT

**Score: 3/10** (borderline reject, major revisions insufficient without new experiments)

This paper addresses an interesting question: how does weight-space divergence during domain-specific fine-tuning relate to loss landscape connectivity? However, the evidence presented does not support the paper's central claims. The experimental design has fundamental flaws, the core quantitative results contain internal contradictions, and the analysis overstates conclusions that the data cannot sustain. I detail the issues below in order of severity.

---

## CRITICAL FLAW 1: The "Upper Bound" is Empirically Exceeded — Invalidating a Core Calibration

**Severity: 9/10 (fatal)**

The paper calibrates its measurements with a noise floor: interpolating between a pretrained Pythia-1.4B and a randomly initialized model of the same architecture. This yields barrier values of 0.033 (code) and 0.222 (medical), which the paper characterizes as the "effective upper bound on what barrier magnitudes the method can produce" (Section 4.5).

The high-divergence cross-domain medical barrier is reported as 0.228 +/- 0.102. The paper states this "nearly saturates" the upper bound of 0.222 (Abstract, Section 5).

**The problem: 0.228 > 0.222.** A real model exceeds the claimed "upper bound." This is logically impossible by the definition of "upper bound." The variance makes this worse: individual seeds produce barriers of 0.340, 0.094, and 0.252 (verified from raw data in `lmc_lr5e-4_s*.json`). The seed producing 0.340 exceeds the bound by 53%.

There are only three possible interpretations, none of which are flattering to the paper:
1. The "upper bound" characterization is wrong — pretrained-to-random interpolation does NOT establish an upper bound. In that case, the entire noise-floor calibration section is misleading.
2. The 0.228 measurement is itself noisy/unreliable — but then the central high-divergence result is untrustworthy.
3. The measurements are correct, meaning that fine-tuning a model on medical data can push it further from linear connectivity than random initialization — an extraordinary claim that would require extraordinary evidence (which 3 seeds with std=0.10 cannot provide).

The paper never acknowledges this contradiction. The phrase "nearly saturates" is used to paper over what is, mathematically, a bound violation. This is a serious analytical error.

---

## CRITICAL FLAW 2: Only Two Valid Divergence Levels — "Sublinearity" Cannot Be Established

**Severity: 9/10 (fatal to the sublinearity claim)**

The paper reports results at two divergence levels: "standard" (LR=1e-4, ΔW ≈ 1.4%) and "high" (LR=5e-4, ΔW ≈ 8%). The Discussion claims:

> "The relationship is monotonic but sublinear — barrier increases more slowly than weight distance" (Section 5)

> "5-6x more weight divergence produces only 2-5x more barrier" (Section 4.2)

**Verification results from raw data:** The intermediate-divergence results (`lmc_lr2e-4_s*.json` and `lmc_lr3e-4_s*.json`) are **bit-for-bit identical** across all 3 seeds. Every loss value at every alpha and every seed matches to 6+ significant figures between the lr=2e-4 and lr=3e-4 conditions. This is impossible under genuine measurement and indicates a pipeline duplication bug.

This means the paper has exactly **2 data points** for its central quantitative relationship. Two points cannot:
- Establish sublinearity (requires at least 3)
- Establish monotonicity (2 points are trivially monotonic)
- Fit any functional form
- Distinguish between linear, sublinear, or superlinear scaling

The actual model weights ARE different between lr=2e-4 and lr=3e-4 (verified: 3.56% vs 4.19% divergence from base). So intermediate divergence levels exist in the trained models — but the LMC scans for them were corrupted. The paper cannot claim these results don't exist when the trained checkpoints are present and simply weren't measured correctly.

The intermediate divergence results are not mentioned in the paper at all, which is itself a form of selective reporting. If they were measured and discarded, the reader deserves to know why. If they were never measured despite trained models being available, the experimental design is incomplete.

---

## CRITICAL FLAW 3: The Core Claim Is Contradicted by Half the Evidence

**Severity: 8/10 (fatal to the main thesis)**

The paper's central claim (stated in the Abstract) is:

> "barrier is driven by weight displacement, not domain difference"

The within-domain baselines are the key evidence for this claim. Let us examine them:

| Domain | Within-domain barrier | Cross-domain barrier (standard) | Conclusion |
|--------|----------------------|-------------------------------|------------|
| Code | 0.048 +/- 0.000 | 0.053 +/- 0.011 | Domain difference negligible (supports claim) |
| Medical | 0.147 +/- 0.027 | 0.051 +/- 0.013 | Domain difference is NEGATIVE? (contradicts claim) |

For code, the evidence supports the claim: within-domain barrier (0.048) nearly equals cross-domain barrier (0.053).

For medical, the evidence **directly contradicts** it: within-domain barrier (0.147) is **3x higher** than cross-domain (0.051). If barrier were driven solely by weight displacement, two medical models trained on the same data should have a barrier similar to a code-medical pair trained at the same intensity. Instead, two medical models differ MORE from each other than a medical model differs from a code model.

The paper attempts to rescue the claim by reinterpreting this as "training stability" rather than domain difference:

> "medical fine-tuning is intrinsically less stable across seeds — the variance between two medical training runs can exceed the variance between code and medical models"

But this reinterpretation undermines the paper's own framework. If "training stability" (a property of the domain) drives barrier height MORE than weight displacement, then the claim "barrier is driven by weight displacement" is false. The correct claim would be: "Barrier is driven by a combination of weight displacement, domain-dependent training stability, and possibly domain difference — and we cannot disentangle these factors with N=2 domains."

Furthermore: the cross-domain medical barrier (0.051) represents the barrier between a code-trained and medical-trained model. The within-domain medical barrier (0.147) represents the barrier between two medical-trained models. If the code model were truly "base" (as the asymmetry analysis at `asymmetry_analysis.json` line 8 confirms it was in earlier experiments), then the comparison is between base-medical (0.051) vs medical-medical (0.147). This has a different interpretation entirely: base-to-medical interpolation is smoother than medical-to-medical interpolation because the base model sits in a "neutral" region of the loss landscape while medical models scatter into higher-loss regions. This is interesting but contradicts the paper's framing.

The paper has exactly 2 domains. For one domain, the evidence supports the claim. For the other, it contradicts it. This is a 50-50 split. You cannot claim a general principle from evidence that is split exactly down the middle.

---

## MAJOR ISSUE 4: The Script Computes Barriers Incorrectly

**Severity: 6/10 (methodological, partially mitigated)**

The main LMC scan script (`scripts/lmc_barrier_scan.py`, lines 101-102) computes the barrier as:

```python
barrier_code = max(losses_code) - losses_code[0]
```

This is **not** the Frankle et al. (2020) definition that the paper claims to use (Section 3.2):

```
barrier = max_α L(θ(α)) − (L(θ(0)) + L(θ(1))) / 2
```

The script's formula uses `L(0)` as the baseline instead of the midpoint `(L(0) + L(1))/2`. This inflates barriers by approximately 2x whenever `L(0) < L(1)` (which is the typical case when one model is better than the other on the evaluation domain). For the high-divergence medical condition seed 0, the script would report ~0.68 instead of the correct Frankle barrier of ~0.34.

**Mitigation:** The paper's reported numbers (0.053, 0.051, 0.118, 0.228) use the correct Frankle definition (verified against raw data in `bootstrap_ci.json`). So the paper itself is not wrong — but the code that generated the data used for the paper computes a different, incorrect value. This means either:

1. The barriers were computed externally (e.g., in the bootstrap analysis), and the script's printed barriers were ignored — but the script's printed output is what experimenters see, creating opportunity for confusion.
2. Or there exists another computation path that was used for the paper.

Either way, a reviewer checking reproducibility would find that the main experimental script computes a different quantity than what the paper reports. This is a reproducibility hazard.

The secondary script (`scripts/lmc_barrier.py`, line 159) has the same issue:

```python
barrier_a = max_loss_a - loss_a_on_a
```

---

## MAJOR ISSUE 5: Enormous Variance in the Key Result

**Severity: 7/10 (statistical)**

The high-divergence medical barrier is reported as 0.228 +/- 0.102. The coefficient of variation is 45%. With N=3 seeds, the 95% bootstrap CI is [0.094, 0.340] — spanning nearly the entire range from "modest barrier" to "exceeding the random-init upper bound."

The individual per-seed barriers (computed from raw data, Frankle definition) are:

| Seed | Code barrier | Medical barrier |
|------|-------------|----------------|
| 0 | 0.135 | 0.340 |
| 1 | 0.144 | 0.094 |
| 2 | 0.075 | 0.252 |
| **Mean** | **0.118** | **0.228** |

The medical barrier spans a factor of 3.6x across seeds (0.094 to 0.340). With only 3 seeds, the mean is dominated by the outlier. Seed 1 (barrier=0.094) is BELOW the pretrained-to-random bound (0.222), while seed 0 (barrier=0.340) far exceeds it.

This level of variance means that:
1. No reliable conclusion can be drawn about the high-divergence medical condition
2. The comparison between high-divergence medical (0.228) and the random-init bound (0.222) is within one standard deviation — statistically indistinguishable
3. The paper's claim that this "nearly saturates" the bound could be an artifact of one high-variance seed

Three seeds is simply insufficient for a condition with this much inter-run variation. The paper acknowledges this in Limitations but understates the severity: this is not about "tightening confidence intervals," it is about whether the point estimate is meaningful at all.

---

## MAJOR ISSUE 6: Missing Permutation Invariance Check

**Severity: 7/10 (critical gap)**

The paper explicitly cites Entezari et al. (ICLR 2022) and Ainsworth et al. (ICLR 2023) on permutation symmetry and Git Re-Basin, but never applies these methods. This is not merely a missing ablation — it undercuts the very validity of the LMC measurement.

The LMC barrier depends on how neurons are aligned between two models. Two models trained on different domains may converge to functionally equivalent but permuted representations. Without permutation matching, the measured barrier may be an artifact of misalignment rather than genuine loss landscape structure.

This is especially relevant given the high within-domain medical barrier (0.147): if two medical models trained on the SAME data have a 3x higher barrier than code-medical cross-domain models, this could indicate that medical training trajectories explore different permutation basins, while code-medical paths happen to preserve alignment. Without permutation matching, we cannot distinguish between:
- Genuine loss landscape barriers
- Artifacts of weight-space permutation misalignment

The paper should at minimum report LMC barriers after Git Re-Basin matching, or explain why permutation invariance is not expected to matter in this setting.

---

## MAJOR ISSUE 7: Single Architecture, Single Model Size

**Severity: 6/10 (limited generalizability)**

All primary experiments use Pythia-1.4B. The paper's single "scaling check" with Pythia-160M is described as:

> "A preliminary scaling experiment with Pythia-160M (ΔW ≈ 0.3%, both models barely moved from pretrained) produced barriers of 0.001 (code) and 0.056 (medical)"

Several problems:
1. The models "barely moved" — meaning this experiment measures barriers at near-zero weight displacement, which is a different regime from the main results.
2. The medical barrier (0.056) at 0.3-0.6% ΔW is already comparable to the 1.4B standard-divergence medical barrier (0.051) at 1.5% ΔW. This contradicts the weight-displacement narrative: if barrier tracks weight displacement, why does 0.3% displacement at 160M produce a barrier similar to 1.5% at 1.4B?
3. This is a single run with no seeds, so variance is unknown.
4. The 160M result is mentioned once in Limitations and never analyzed — it is essentially non-evidence.

Without a second architecture or model size with meaningful weight displacement, the claim that the findings generalize beyond Pythia-1.4B is unsupported.

---

## MODERATE ISSUE 8: The Asymmetry Analysis Uses a Known-Buggy Code Model

**Severity: 5/10 (historical, affects earlier experiments)**

The asymmetry analysis (`asymmetry_analysis.json`, line 3) explicitly states:

> "critical_caveat: code_e1 checkpoint IS identical to base model (Pythia-1.4B). LMC scans are actually base<->medical interpolations, not code<->medical."

This was subsequently fixed (verified: `code_lr1e-4_s0` has 1.24% divergence from base), but the paper's asymmetry claims (Section 4.4) may have been developed from the buggy data. The paper claims:

> "code fine-tuning produces a small positive externality for medical reasoning"

If the "code" model was actually the base model (per Bug 22), then this finding reduces to: "interpolating between the base model and an undertrained medical model sometimes improves medical performance." This is trivially expected if the medical model is undertrained — it has lost general reasoning capabilities that the base model retains. The finding has nothing to do with "code fine-tuning" at all.

While the current paper draft uses correctly-trained models for the main LMC results, the asymmetry analysis section may still reflect the buggy interpretation. This should be explicitly verified and stated.

---

## MODERATE ISSUE 9: Missing Baselines

**Severity: 5/10**

Several baselines are conspicuously absent:

1. **No interpolation of untrained (same-init, different-seed) models.** Two copies of Pythia-1.4B with different random seeds but NO training would provide a true zero-displacement baseline for LMC. The paper only has "identical copy" (which is trivial) and "pretrained-random" (which is one endpoint).

2. **No random perturbation baseline.** If you add Gaussian noise to a model's weights to achieve ΔW ≈ 1.5%, what barrier do you get? This would separate the effect of weight displacement magnitude from the effect of structured (training-induced) weight changes.

3. **No intermediate training checkpoints.** The paper measures LMC only at trained endpoints. Measuring LMC at intermediate checkpoints (e.g., every 20% of an epoch) would provide the within-training-dynamics view of how barrier evolves — and would give multiple divergence levels for free.

4. **No cross-architecture LMC.** Even within Pythia, comparing 160M vs 1.4B would provide insight into whether the findings are architecture-size dependent.

---

## ADDITIONAL CONCERNS

### A. The Per-Block Correlation Is Nearly Tautological

**Severity: 3/10**

The paper reports per-block divergence correlation r=0.995 between code and medical models. I verified this from properly trained models: r=0.9946. However, this correlation is nearly tautological: both models start from the same initialization, train on similar-sized datasets for similar durations, and apply the same optimizer. The per-block divergence pattern is overwhelmingly determined by the architecture (early layers diverge more due to larger gradient magnitudes), not by domain. Finding that two models trained on different data show similar per-block weight-change patterns is about as surprising as finding that two different cars driving on two different roads both experience more tire wear on the front wheels.

The paper presents this as evidence that domains "change the same blocks" — but this would be true for almost any fine-tuning of the same architecture from the same initialization.

### B. The "2-Point Sublinearity" Is a Category Error

**Severity: 4/10**

The computation "5-6x more weight divergence produces only 2-5x more barrier" is comparing ratios at two points. For code: 8.0/1.4 = 5.7x divergence increase produces 0.118/0.053 = 2.2x barrier increase. The paper calls this "sublinear." But with two data points, ANY ratio where the barrier multiplier is less than the divergence multiplier is "sublinear" — including multiplicative relationships like barrier ∝ sqrt(ΔW), barrier ∝ log(ΔW), or barrier ∝ ΔW^0.5. The paper uses the language of functional form without the data to support it.

### C. The Evaluation Protocol Uses Only 1 Epoch of Training

**Severity: 4/10**

All models are trained for exactly 1 epoch. The asymmetric analysis confirms that medical models at epoch 1 are undertrained (loss improves from 0.623 to 0.485 across epochs 1-5). The LMC barriers measured at epoch 1 may reflect transient training dynamics rather than converged model properties. A model that has barely begun to specialize may naturally show lower barriers than a fully converged specialist.

---

## Summary: Three Reasons to Reject

1. **Internal contradiction invalidates the calibration framework.** The paper establishes a pretrained-to-random-init barrier of 0.222 as an "upper bound," then reports a high-divergence medical barrier of 0.228 +/- 0.102 that exceeds it. A bound cannot be exceeded by an in-bounds measurement. Either the bound characterization is wrong, the measurement is unreliable, or the paper is making an extraordinary claim without extraordinary evidence.

2. **The core claim is not supported by the evidence.** The claim that "barrier is driven by weight displacement, not domain difference" is contradicted by one of the two available domains (medical: within-domain 0.147 vs cross-domain 0.051). A claim supported by 50% of the evidence cannot be the paper's central thesis. The paper's attempt to reframe this as "training stability" rather than domain difference does not rescue the original claim — it replaces it with a different, unsubstantiated one.

3. **The experimental design is too thin to support the paper's quantitative analysis.** Two divergence levels (after discarding the corrupted intermediate results), two domains, one architecture, one model size, and N=3 seeds with 45% coefficient of variation on the key measurement. This is insufficient to establish any quantitative relationship between weight displacement and LMC barriers.

---

## Minimum Additional Work for Acceptance at a Top Venue

The following is the minimum I would consider necessary for a revise-and-resubmit. Each item is listed with estimated effort.

| # | Required Work | Effort |
|---|--------------|--------|
| 1 | Fix the "upper bound" contradiction. Either (a) remeasure the pretrained-to-random barrier with more seeds to establish a true confidence interval that contains 0.228, (b) acknowledge that 0.222 is not an upper bound and reframe the calibration, or (c) demonstrate that 0.228 is within the sampling distribution of the bound measurement. | Low |
| 2 | Add at least 2 more intermediate divergence levels (e.g., LR 2e-4 and 3e-4) with verified, non-duplicate LMC scans. Without this, all quantitative claims about the barrier-vs-divergence relationship are unfounded. | Medium (models already trained; just need scans) |
| 3 | Apply permutation alignment (Git Re-Basin or weight matching) before LMC measurement, and report barriers with and without alignment. This addresses the most glaring methodological gap. | Medium |
| 4 | Add a second model architecture (e.g., GPT-2-Medium or OPT-1.3B) with at least standard-divergence runs on both domains. | High |
| 5 | Increase seeds to at least 5 for the high-divergence medical condition, given the enormous observed variance. Or, if the variance is real, report the full per-seed results transparently and adjust claims accordingly. | Medium |
| 6 | Add a random perturbation baseline: add Gaussian noise to base model to achieve ΔW ≈ 1.5% and ≈ 8%, measure LMC barrier. This disentangles structured (training) from unstructured (noise) weight displacement effects. | Low |
| 7 | Fix the barrier calculation in `lmc_barrier_scan.py` and `lmc_barrier.py` to use the Frankle definition, ensuring code-paper consistency. | Trivial |
| 8 | Retract or heavily qualify the "sublinear" claim unless at least 4 divergence levels are measured. Report just the numbers without functional-form claims. | Trivial (text change) |
| 9 | Report the asymmetry analysis with verified trained code models (not base), or explicitly state which experiments use base-model proxies. | Medium |

---

## Detailed Scores

| Weakness | Severity (1-10) | Category |
|----------|----------------|----------|
| Upper bound exceeded by measurement | 9 | Analytical error |
| Only 2 valid divergence levels; intermediate results corrupted | 9 | Experimental design |
| Core claim contradicted by 50% of evidence | 8 | Claim-evidence mismatch |
| Enormous variance in key result (CV=45%) | 7 | Statistical |
| Missing permutation invariance check | 7 | Methodological gap |
| Script computes wrong barrier formula | 6 | Reproducibility |
| Single architecture / model size | 6 | Generalizability |
| Missing baselines (random noise, untrained seeds) | 5 | Experimental design |
| Asymmetry analysis may use buggy data | 5 | Data integrity |
| Per-block correlation near-tautological | 3 | Overclaimed finding |
| 2-point "sublinearity" claim | 4 | Logical error |
| Single-epoch training may not reflect convergence | 4 | Experimental design |

**Overall score: 3/10** — The paper addresses a genuinely interesting question, but the evidence is insufficient to support its claims. The most charitable reading is that this is preliminary work that needs substantially more experiments before it can contribute to the literature.
