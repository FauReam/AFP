# Model Merging Practitioner's Review: AFP Phase 0 LMC Paper

**Reviewer:** Merging Practitioner (Model Soups, TIES-Merging, Task Arithmetic)
**Paper:** "Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning" (Draft v4, 2026-07-13)
**Date:** 2026-07-13

---

## 1. Executive Summary

This paper measures the relationship between weight-space divergence and LMC barrier height when Pythia-1.4B is fine-tuned on code and medical reasoning. The core finding — that 1-2% weight displacement produces barriers of ~0.05 and 8% displacement produces barriers of 0.12-0.23 — is methodologically sound. However, **the paper stops short of what a practitioner actually needs**: it never connects these barrier numbers to real merge quality. As a result, I can read this paper, understand the measurements, and still not know whether I should merge my two fine-tuned models.

---

## 2. Practical Utility: Does Knowing the ΔW→Barrier Relationship Help a Practitioner?

### What the paper tells us

The paper establishes a monotonic, sublinear relationship: 5-6x more weight divergence produces only 2-5x more barrier. This is a descriptive finding about the geometry of the loss landscape between two fine-tuned models. It tells us that domain specialization of this kind (code and medical) does not easily break linear connectivity.

### What a practitioner actually needs to know

A practitioner deciding whether to merge two models wants to know:

1. **Will the merged model be better than either of the individual models?** (The Model Soups question — Wortsman et al. showed merging can *improve* performance, not just preserve it.)
2. **If I merge, what is the expected downstream accuracy?** (Barrier height on BCE loss tells me nothing about downstream task accuracy.)
3. **Should I use pure averaging, or do I need TIES/slerp/task arithmetic?** (The paper evaluates zero merging methods — just raw linear interpolation.)
4. **At what barrier threshold should I NOT merge?** (The paper gives no actionable threshold.)

The paper answers none of these. The barrier is measured on the same BCE loss used for training — this is an *interpolation loss*, not a *merge utility metric*. A merged model with barrier 0.2 might still work fine on downstream tasks, or it might be catastrophically degraded. We simply do not know from the data presented.

### Bottom line

Knowing the ΔW→barrier relationship is **scientifically interesting but not actionable**. A practitioner would not change their merging workflow based on this paper. They would still need to try the merge and evaluate it — which is what they already do.

---

## 3. The Paper's Practical Claim: "Model Merging Likely Works Well"

> *"A practical implication: model merging techniques that rely on linear interpolation are likely to work well for domain-specialized models fine-tuned from the same base, even at aggressive optimization settings."* (Section 5, Discussion)

### Is this claim supported?

**Partially, with important caveats.**

The data supports the claim that the *interpolation loss barrier* is modest — the models are demonstrably in the same linearly-connected basin. This is a necessary condition for linear interpolation to work at all, and the paper confirms that condition holds.

However, the claim jumps from "the barrier is modest" to "merging likely works well." These are not the same thing:

- A low barrier means the interpolated model's loss is close to the endpoint losses. It does NOT mean the interpolated model is *better* than either endpoint (the Model Soups finding).
- The paper never evaluates the merged model on any downstream benchmark — no accuracy, no F1, no reasoning score. The only metric is BCE training loss, which is a terrible proxy for merge quality.
- There is no comparison to individual model performance on any held-out task. The asymmetry analysis (Section 4.4) hints at this — the code model sees a slight medical improvement from code fine-tuning — but doesn't quantify it as a merge benefit.

### Does the data tell us about merge quality, or only about interpolation loss?

**Only about interpolation loss.** This is the central gap. The LMC barrier (`max_α L(θ(α)) − avg(L(θ(0)), L(θ(1)))`) measures how much worse the midpoint is than the endpoints on the *training loss*. It does not measure:

- Whether the merged model generalizes
- Whether the merged model composes capabilities from both domains
- Whether the merged model is better than either parent on downstream tasks
- Whether a 50-50 average is even the right merge ratio

A practitioner reading this paper would reasonably ask: *"If the barrier is 0.23 on medical data, what does that mean for my merged model's diagnostic accuracy? Does it still work? Is it useless?"* The paper cannot answer this.

---

## 4. Missing Comparisons: TIES-Merging and Task Arithmetic

### The gap

The paper cites TIES-Merging (Yadav et al., NeurIPS 2023) and Task Arithmetic (Ilharco et al., 2023) in both the Related Work and References. The Related Work section correctly notes: *"These methods demonstrate that linear interpolation works, but do not characterize when it fails."* This is accurate — those papers showed *that* merging works, and the current paper aims to characterize *when* it fails.

**But the paper never tests whether TIES or Task Arithmetic actually change the barrier.**

### Why this matters

Pure linear interpolation (`θ(α) = (1-α)θ_A + αθ_B`) is a strawman for modern merging. Practitioners do not use raw linear interpolation for two models trained on different domains. They use:

1. **TIES-Merging:** Trim small-magnitude changes, resolve sign conflicts (elect), then merge only the agreed-upon directions. This explicitly removes the exact kind of interference that would cause a barrier.
2. **Task Arithmetic:** Construct task vectors (`τ = θ_finetuned − θ_pretrained`), then add scaled vectors: `θ_new = θ_pretrained + λ_code·τ_code + λ_medical·τ_medical`. This is structurally different from linear interpolation of the fine-tuned weights.

The paper's barrier measurements, while valid for pure linear interpolation, tell us nothing about what happens under TIES's trim-elect-sign pipeline or under task vector addition. The sign-resolution step in TIES could eliminate a large fraction of the barrier, since parameter interference is precisely what TIES is designed to resolve. The per-block divergence data (r=0.995 correlation between domains) actually suggests that TIES might struggle here — the two domains change the same blocks in the same pattern, which means sign conflicts could be widespread.

### What the paper should have done

At minimum, run one TIES merging experiment and one task arithmetic experiment on the standard-divergence models, and compare the resulting model's performance on downstream tasks against the pure linear interpolation. This would:

- Show whether the barrier measured under linear interpolation is predictive of merge quality under modern methods
- Test whether TIES's sign resolution actually helps when both domains change the same parameters (the r=0.995 correlation suggests it might not)
- Give practitioners an actual comparison point they can use

### Is a pure linear interpolation barrier the right metric?

**No, not for modern merging.** The field has moved past pure linear interpolation. The barrier under linear interpolation is a valid geometric measurement — it tells us something about the loss landscape — but it is not the right metric for predicting merge quality under TIES, Task Arithmetic, or even weighted Model Soups. The right metric would be the difference between the merged model's downstream performance and the best individual model's downstream performance.

---

## 5. The Medical Within-Domain Finding

> *"For domains like medical where training trajectories are intrinsically noisy, merging different seeds of the same domain can be more challenging than merging across domains."*

### Is this useful?

**Yes — it is the most interesting finding in the paper** — but it needs more support.

The data shows:
- Code within-domain barrier: 0.048 (tight, ±0.000)
- Code cross-domain barrier (standard): 0.053 (very close to within-domain)
- Medical within-domain barrier: 0.147 (±0.027, much larger variance)
- Medical cross-domain barrier (standard): 0.051 (far lower than within-domain)

This is genuinely surprising: two medical models trained on the same data with different seeds are *less* connected in the loss landscape than a code model and a medical model trained at the same intensity. The paper interprets this as "medical training is intrinsically less stable across seeds."

### Has this been observed before?

Not to my knowledge in the context of domain fine-tuning LMC measurements. The model merging literature has been largely domain-agnostic — Wortsman et al. merged models fine-tuned on the same task; Yadav et al. merged models fine-tuned on different tasks within the same general domain (vision). The observation that within-domain seed variance can exceed cross-domain difference is novel and worth highlighting.

However, there are important caveats that weaken this finding:

1. **The high-divergence models are under-converged.** The paper's own Limitation section (Section 6) acknowledges: "The high-divergence models achieve higher self-domain loss than standard-divergence models, suggesting that the aggressive optimization settings trade off convergence quality for weight displacement." This means the within-domain comparison for medical is confounded — we don't know if the high variance is due to domain properties or due to under-convergence.

2. **Only 3 seeds.** The bootstrap CI for within-medical is [0.109, 0.173] — wide. Three seeds is the minimum defensible number for computing a standard deviation.

3. **Only 2 domains.** We cannot say "medical specifically has this property" versus "some domains have it, some don't." It could be that the medical *dataset* (VersaPRM's ~55K medical samples) has higher variance than the code dataset, not that medical *reasoning* is inherently noisier to learn.

4. **The standard-medical cross-domain barrier (0.051) is measured against code models at *standard* divergence.** The within-medical barrier (0.147) uses standard-divergence medical seeds. These use the same optimization settings. So the comparison is fair at standard divergence. But the claim about "medical training being intrinsically noisy" should be verified with more than 3 seeds and ideally across different medical datasets.

### Practical implication

If this finding holds up, it means that for medical models, within-domain ensembling (averaging multiple seeds of the same domain) might actually be *worse* than cross-domain merging. This is counterintuitive and practically important. But it needs stronger evidence before a practitioner would act on it.

---

## 6. The Experiment That Would Make This Paper Immediately Useful

**Correlate LMC barrier heights with downstream merge quality across multiple merging methods.**

Specifically:

1. Train models on 4+ domains (code, medical, math, legal, etc.) at standard divergence.
2. For every pair of domains, compute:
   - The LMC barrier (as currently done)
   - Weight-space divergence (as currently done)
   - The downstream task accuracy of the merged model using: (a) pure averaging, (b) TIES-Merging, (c) Task Arithmetic, (d) Model Soups (greedy or uniform)
3. Plot merge quality (merged accuracy minus best single-model accuracy) against barrier height.
4. Identify the barrier threshold above which merging is counterproductive.

This would **operationalize the barrier measurement** — turning a geometric curiosity into a decision tool. A practitioner could then compute the barrier between their two models and know, before attempting any merge, whether it is likely to help or hurt.

A secondary experiment that would be immediately useful: **layer-selective interpolation**. The per-block data (Figure 2) shows that early layers diverge 5.6x more than late layers. If the barrier is primarily caused by early-layer divergence, then merging only layers 8+ (while keeping each model's own early layers) might dramatically reduce the barrier while preserving domain-specific capabilities. This is directly actionable: a practitioner could implement "merge except layers 0-7" as a simple rule.

---

## 7. Scores

| Dimension | Score (out of 10) | Rationale |
|-----------|-------------------|
| **Practical Utility** | 3/10 | The paper tells us that LMC holds for domain-specialized Pythia-1.4B models. This confirms what most practitioners already assume. It does not help them decide whether to merge, which method to use, or what to expect from the merged model. The ΔW→barrier relationship is a descriptive finding with no downstream utility given the current data. |
| **Actionability** | 2/10 | There is exactly one actionable takeaway: "linear interpolation probably works for domain-specialized models from the same base." But practitioners already assume this (it is why Model Soups exists). The paper provides no threshold, no decision rule, and no comparison of merging methods. The within-domain finding is intriguing but not robust enough to act on. |
| **Connection to Merging Literature** | 5/10 | The paper correctly cites the major merging papers (Model Soups, TIES, Task Arithmetic, Git Re-Basin) and positions itself as complementary ("these methods show merging works; we characterize when it fails"). But it never engages with the merging methods experimentally. The Related Work is well-written but the experiments do not follow through on the claimed connection. The paper cites TIES as part of its motivation but never runs TIES. |
| **Missing Experiments** | 7/10 | The list of missing experiments is substantial: (1) no downstream accuracy evaluation of merged models, (2) no comparison against TIES or Task Arithmetic, (3) no layer-selective interpolation (despite having the per-block data to motivate it), (4) only 2 domains, (5) only one model family (Pythia), (6) high-divergence models are under-converged (confounds the barrier measurements). The 160M scaling check is a start but too preliminary to count. The noise floor calibration is well done but establishes measurement validity, not practical utility. |

---

## 8. Overall Assessment

This paper does competent, careful measurement work. The experimental design is methodical: 3 seeds with bootstrap CIs, per-block divergence analysis, noise floor calibration with identical-copy and random-init baselines. The writing is clear and the claims are mostly appropriately hedged (the Discussion is more cautious than the Abstract).

However, the paper reads as if it was written by someone who has read the model merging literature but has not actually *done* model merging in practice. A practitioner who regularly merges models would immediately ask the questions this paper does not answer: *What does a barrier of 0.12 mean for my merge? Should I use TIES instead of averaging? At what barrier should I give up?*

The paper's biggest rhetorical move — citing TIES-Merging and Task Arithmetic as motivation but then never comparing against them — is a missed opportunity. The paper claims to characterize "when merging fails," but it measures the wrong thing (interpolation loss, not merge quality) using the wrong method (pure linear interpolation, not modern merging techniques).

**The within-domain medical finding is the most valuable contribution.** If replicated with more seeds, more domains, and proper convergence controls, it could become a standard caution in the model merging literature. But in its current form (3 seeds, under-converged high-divergence models), it is suggestive but not conclusive.

**Recommendation for the authors:** The fastest path to practical relevance is a single additional experiment: merge the standard-divergence code and medical models using (a) pure averaging and (b) TIES-Merging, evaluate both on downstream code and medical benchmarks, and report whether the barrier height predicts merge quality. This would take roughly one day of GPU time and would transform the paper from "interesting measurement" to "practitioner's decision tool."
