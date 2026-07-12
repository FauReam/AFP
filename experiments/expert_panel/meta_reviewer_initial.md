# Senior Area Chair — Initial Independent Assessment

**Paper:** Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning
**Draft:** v4, 2026-07-13
**Reviewer:** Meta-Reviewer (Senior AC, 50+ papers handled at NeurIPS/ICML/ICLR)
**Date:** 2026-07-13

---

## 1. Overall Contribution

**Verdict: "Careful measurement paper with one genuinely non-obvious finding, bounded by scale."**

On the spectrum from "incremental measurement paper" to "surprising fundamental finding," this paper lands at approximately the 35th percentile. It does not overturn existing understanding; it refines it. The core findings align with expectations:
- More weight divergence produces higher LMC barriers (expected).
- The relationship is sublinear (reassuring but not surprising — loss landscapes are known to be locally convex around minima).
- Early layers diverge more than late layers (consistent with known fine-tuning dynamics).

The one genuinely non-obvious result is the **domain-specific training stability asymmetry**: within-domain medical barriers (0.147) substantially exceed cross-domain code-medical barriers (0.051), while code shows the opposite pattern (within = 0.048, cross = 0.053). This reveals that training trajectory variance, not domain dissimilarity, is the dominant source of LMC barrier height. This is a legitimate finding.

However, `finding != paper`. A NeurIPS/ICML paper needs either a new method, a new theory, or a finding that changes how the community thinks about a problem. "Domain-specific training stability affects LMC more than domain difference" is interesting, but it is descriptive, not mechanistic. The paper tells us *that* this happens but not *why* medical fine-tuning is intrinsically less stable.

## 2. Conference Fit and Estimated Acceptance Probability

| Venue | Est. Accept Rate | Fit Assessment | Acceptance Probability |
|-------|-----------------|----------------|----------------------|
| **NeurIPS 2027** | 15-20% | Moderate fit. NeurIPS values "wow" and novelty. This paper is careful but unsurprising. Would need an outstanding framing to clear the bar. | **15-20%** (high-risk) |
| **ICML 2027** | 20-25% | Slightly better than NeurIPS. ICML is more receptive to careful empirical work, especially with practical implications (model merging). Still a reach. | **20-25%** (high-risk) |
| **ICLR 2028** | 25-30% | **Best top-tier venue.** ICLR has a strong loss-landscape / mode-connectivity community (Frankle, Entezari, Ainsworth, Garipov all published there). The paper speaks their language. A sympathetic AC assignment is plausible. | **30-35%** (moderate-risk) |
| **ACL 2027** | 20% | **Poor fit.** The paper uses NLP data but makes zero NLP contribution — no linguistic analysis, no downstream task improvement, no domain adaptation insights. Would be desk-rejected or savaged by reviewers asking "what's the NLP contribution?" | **10-15%** (do not submit) |
| **EMNLP 2027** | 25% | Marginally better than ACL (EMNLP is slightly more methods-tolerant) but same fundamental mismatch. | **15%** (do not submit) |
| **TMLR** | ~60% desk accept | **Excellent fit.** TMLR explicitly values rigorous measurement studies without requiring conceptual novelty. The noise-floor calibration, bootstrap CIs, within-domain baselines, and clear experimental design would be well-received. Rolling submission means no deadline pressure. | **65-75%** (strong candidate) |

**Recommendation:** Target ICLR 2028 as primary, with TMLR as the fallback. If the paper is strengthened to 2+ model families and 4 domains with systematic divergence sweeps, ICLR acceptance becomes 40-50%. Do not submit to ACL/EMNLP — the venue mismatch guarantees negative reviews regardless of paper quality.

## 3. Strongest Part

**The within-domain LMC baseline (Section 4.3).** This is the paper's most defensible contribution. The discovery that two medical models trained on the same data under identical conditions exhibit greater loss-landscape dissimilarity (barrier = 0.147) than a code model and a medical model trained at the same intensity (barrier = 0.051) is genuinely counterintuitive. Most readers, myself included, would have predicted the opposite: that domain difference is the primary driver of barrier height. Refuting that intuition is a real contribution.

The **noise-floor calibration** (Section 4.5) is also methodologically strong and addresses a common critique of LMC work — that measured barriers might be measurement artifacts. The identical-copy barrier of ~0.000 and pretrained-to-random upper bound of 0.22 provide credible calibration. These are exactly the kind of rigor checks that distinguish careful work from sloppy work.

The **per-block divergence pattern** (r = 0.995 across domains) is a crisp, replicable result that suggests architecture, not training data, determines which parameters change during fine-tuning. This has implications for parameter-efficient fine-tuning design.

## 4. Weakest Part

**The high-divergence condition is contaminated.** The paper acknowledges (Limitations, Section 6) that "the high-divergence models achieve higher self-domain loss than standard-divergence models, suggesting that the aggressive optimization settings trade off convergence quality for weight displacement." This is a serious confound. The high-divergence barriers (0.12 code, 0.23 medical) reflect not just weight-space distance but also suboptimal convergence. A reader cannot tell whether the barrier increase from 0.05 to 0.12 is due to (a) genuine loss of linear connectivity at larger weight distances, or (b) both endpoints being worse models, which makes any convex combination worse. These are different mechanisms with different implications.

Furthermore, the medical high-divergence barrier (0.23) nearly saturating the random-init upper bound (0.22) is presented as a finding, but it could also be interpreted as "the high-divergence medical model is so poorly converged that it's barely better than random." The paper doesn't resolve this ambiguity.

**The 160M scaling check is trivial.** Both models "barely moved from pretrained" (ΔW ≈ 0.3%). The barriers (0.001 code, 0.056 medical) are consistent with the 1.4B findings but add almost no information — they merely confirm that when models barely change, LMC holds. A proper scaling study would ensure comparable ΔW across sizes, which would require adjusting training hyperparameters per model size.

## 5. GPU-Days Needed for NeurIPS Acceptance

**Approximately 0.8-1.2 additional GPU-days** (on DGX Spark GB10 equivalent):

| Experiment | Runs | Time/run | Total |
|-----------|------|----------|-------|
| 2nd model family (e.g., Llama-3.2-1B) | 2 domains × 2 div levels × 3 seeds = 12 trainings | ~30min | 6h |
| 3rd model family (e.g., Qwen-2.5-1.5B) | 12 trainings | ~30min | 6h |
| 2 additional divergence levels (systematic sweep) | 2 domains × 2 levels × 3 seeds = 12 trainings | ~42min | 8.4h |
| Additional domains (math, general) | 2 new domains × 2 div levels × 3 seeds = 12 trainings | ~42min | 8.4h |
| Git Re-Basin alignment comparison | 6 LMC scans with permutation alignment | ~15min | 1.5h |
| All LMC scans (for all new models) | ~24 scans | ~15min | 6h |
| **Total** | | | **~36 GPU-hours ≈ 1.5 GPU-days** |

However, the compute is not the bottleneck. **The conceptual gap is larger than the experimental gap.** Even with perfect experiments, a pure measurement paper without a mechanistic contribution faces an uphill battle at NeurIPS. The additional GPU-days would increase confidence in generalizability, but they wouldn't transform the paper from measurement to insight.

To close the conceptual gap, the authors would need either:
- (a) A theoretical model that predicts barrier height from weight divergence (e.g., a quadratic approximation to the loss landscape that explains the sublinear relationship), or
- (b) A practical demonstration showing that the measured divergence-barrier relationship predicts downstream merge quality for TIES-Merging / Task Arithmetic across multiple model families.

Option (b) is more achievable and would take approximately 2-3 additional GPU-days for merging experiments.

## 6. Best Secondary Venue and Required Work

**Primary recommendation: TMLR** (rolling submission).

TMLR is the strongest venue where this paper, in its current form, has a high probability of acceptance. The venue explicitly values methodological rigor and careful measurement over conceptual novelty. The 22-bug engineering log demonstrates the kind of attention to experimental validity that TMLR reviewers appreciate.

**To "lock in" TMLR acceptance**, add:
1. **One additional model architecture** (Llama-3.2-1B or OLMo-1B). This addresses the most common single-model-family criticism. Approximately 6 hours of GPU time.
2. **Two additional domains** (math and general from VersaPRM, which are already in the dataset). This costs almost nothing — the data is already cached. Approximately 6 hours of GPU time.
3. **Remove or clearly frame the high-divergence confound.** Either (a) re-run high-divergence with a schedule that achieves comparable convergence (e.g., train longer with the same high LR), or (b) restructure the paper so high-divergence is framed as an exploratory analysis rather than a primary result.
4. **Add Git Re-Basin comparison.** Run one LMC scan after permutation-aligning the two models. This addresses the obvious "have you tried alignment?" question. Approximately 1.5 hours of GPU time plus implementation effort.

**With these additions, TMLR acceptance probability rises to ~85%.**

For **ICLR 2028** as the ambitious target:
- All of the above, plus
- A systematic sweep of 4-5 divergence levels to characterize the functional form ΔW → barrier
- A theoretical analysis or at minimum a functional fit (e.g., barrier ∝ ΔW^α with α < 1)
- Total: approximately 2-3 additional GPU-days

## 7. What the Paper Currently Lacks

1. **Generalizability evidence.** Single model family (Pythia-1.4B). The 160M check doesn't count — both models barely moved. Any reviewer will ask: "Does this hold for Llama? For encoder-decoder models? For MoE architectures?"

2. **Domain breadth.** Two domains (code, medical) is the bare minimum to claim "domain specialization." Adding math and general from the same dataset would quadruple the domain coverage at near-zero data preparation cost (VersaPRM already has these domains tokenized and cached).

3. **A mechanistic story.** The paper reports that barrier ∝ ΔW, sublinearly. But it never asks *why*. Is it because the Hessian is locally flat around the pretrained minimum? Is it because early-layer changes dominate and early layers are less sensitive to perturbation? Is it because the loss landscape has a single broad basin that both models stay within? No hypothesis is tested.

4. **Resolution of the high-divergence confound.** The paper's most dramatic quantitative result (0.23 barrier ≈ random-init upper bound) is from a condition where the model is under-converged. A reviewer will call this out as a confound that invalidates the comparison.

5. **Connection to model merging practice.** The Discussion (Section 5) claims practical implications for model merging, but the paper never actually runs a merging experiment. A TIES-Merging or Task Arithmetic evaluation on the code+medical pair would close the loop between measurement and application. This is the most impactful missing experiment.

6. **Comparison to permutation alignment.** Entezari et al. (2022) and Ainsworth et al. (2023) showed that permutation symmetry accounts for most apparent LMC failures. The paper never checks whether the measured barriers would be lower after Git Re-Basin alignment. This is a conspicuous omission given the Related Work.

7. **Statistical formalism.** Three seeds with bootstrap CIs is a reasonable minimum, but the paper doesn't report effect sizes, doesn't do hypothesis tests (e.g., is the within-domain medical barrier significantly different from the cross-domain barrier?), and doesn't report power analysis. The bootstrap CIs for high-divergence medical are [0.09, 0.34] — a range so wide that it includes both "essentially connected" and "essentially disconnected." More seeds are needed for the high-variance condition.

8. **Ablation on training hyperparameters.** The paper varies LR to produce different divergence levels, but the confound is that LR also affects convergence quality. A cleaner design would hold LR constant and vary training duration (e.g., 0.25, 0.5, 1.0, 2.0 epochs) to control weight displacement independently of the optimization trajectory's stability.

9. **A clear, falsifiable hypothesis stated upfront.** The introduction describes the paper as measurement ("we measure this relationship"), which is honest but weak. A stronger paper would stake out a clear prediction (e.g., "we hypothesize that domain difference, not weight displacement magnitude, determines LMC barrier height — and we show this is false"). The paper implicitly tests this but never states it.

10. **Code and data release plan.** No mention of whether trained models, evaluation code, or the VersaPRM subset will be released. For a measurement paper at a top venue, reproducibility artifacts are table stakes.

## 8. Summary and Strategic Recommendation

This is a solid, honest, carefully executed measurement study. The within-domain stability finding is genuinely interesting. The noise-floor calibration is exemplary. The per-block divergence pattern is crisp. But the paper is fundamentally bounded by its scale (one model family, two domains) and its descriptive nature (no mechanism, no method, no theory).

**My recommendation as Senior AC:**

- **Do not submit to NeurIPS 2027 in current form.** The acceptance probability is too low to justify the effort. One hostile reviewer who says "this is expected" or "single model family" will tank the paper, and there will always be at least one such reviewer.

- **Preferred path: Strengthen for ICLR 2028.** Add 1-2 model families, 2 more domains, resolve the high-divergence confound, and add a Git Re-Basin comparison. This is 1.5-2 GPU-days of work and transforms the paper from "interesting measurement on one model" to "general phenomenon across architectures and domains." With a strong framing around the domain-stability finding as the primary contribution, ICLR acceptance becomes plausible.

- **Safe path: TMLR with minimal additions.** Add one model family and two domains (~0.5 GPU-days). Submit to TMLR rolling review. High acceptance probability, no deadline pressure. This is the risk-minimizing strategy.

- **High-risk/high-reward alternative:** Recast the paper as a **benchmark and diagnostic toolkit** for model merging research. Package the noise-floor calibration, within-domain baselines, and per-block divergence analysis as a standardized protocol that other researchers can apply to their own model pairs. This transforms the paper from "we measured one thing" to "here's how to measure this thing properly." A benchmarks paper at NeurIPS Datasets & Benchmarks track would be a natural fit and has a different (often more forgiving) reviewer pool.

**Bottom line:** The paper is publishable in its current form at TMLR. With 1.5-2 GPU-days of additional experiments and a sharper narrative, it is competitive at ICLR. The authors should not waste a submission cycle on NeurIPS without substantially strengthening the contribution.
