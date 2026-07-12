# LMC Theorist Expert Report

**Paper under review:** "Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning" (Draft v4, 2026-07-13)

**Reviewer role:** Senior researcher in loss landscape geometry and Linear Mode Connectivity (Frankle et al. 2020, Entezari 2022, Ainsworth 2023). This is an adversarial review — every claim is examined against the raw experimental data and source code.

**Date:** 2026-07-13

---

## 1. Methodology Verification: The Barrier Formula

### 1.1 A bug in `lmc_barrier_scan.py` (lines 101-102)

The core scanning script `/home/jiayu/AFP/scripts/lmc_barrier_scan.py` contains an error in its barrier summary computation:

```python
# lmc_barrier_scan.py, line 101-102
barrier_code = max(losses_code) - losses_code[0]    # BUG
barrier_med = max(losses_med) - losses_med[0]        # BUG
```

The Frankle et al. (ICML 2020) definition — correctly stated in the paper's Section 3.2 — is:

```
barrier = max_α L(θ(α)) − (L(θ(0)) + L(θ(1))) / 2
```

The script computes `max(L) - L(0)`, which uses only one endpoint as the baseline rather than averaging both. When L(0) and L(1) differ substantially, this produces inflated barriers (up to 2x for the standard-divergence condition). The same bug also appears in `/home/jiayu/AFP/scripts/lmc_barrier.py` (lines 159-160).

### 1.2 The bug does NOT affect the paper's numbers (but reveals a workflow gap)

I recomputed all barriers from the raw JSON data using the correct Frankle definition:

| Condition | Script bug (max - L0) | Correct Frankle | Paper reports |
|-----------|----------------------|-----------------|---------------|
| Std code s0 | 0.093 | 0.047 | (mean 0.053) |
| Std med s0 | 0.133 | 0.066 | (mean 0.051) |
| High code s0 | 0.270 | 0.135 | (mean 0.118) |
| High med s0 | 0.679 | 0.339 | (mean 0.228) |

The paper's reported numbers in all cases match the **correct Frankle definition**, not the buggy script output. I verified this by recomputing every barrier from the raw `.json` files and confirming exact numerical agreement with `bootstrap_ci.json`.

**Why the paper is correct despite the bug**: The orchestration shell scripts (`lmc_6scans.sh` line 34, `lmc_scans_only.sh` line 25, `lmc_within_domain.sh` line 36) all include inline Python snippets that compute and print **both** barrier definitions side by side:

```python
print(f'bar={cm-c0:.4f}/{cm-(c0+c1)/2:.4f}')
#          ^buggy   ^correct
```

The paper author(s) were clearly aware of both definitions and chose the correct one. The raw JSON files produced by `lmc_barrier_scan.py` contain only per-alpha loss values (not pre-computed barriers), so there is no contamination in the stored data. The bug is confined to the on-screen summary printed by the scan script.

**Recommendation**: The scan script should be fixed so that future experiments do not accidentally report wrong barriers. But this is a code hygiene issue, not a paper integrity issue.

---

## 2. The Sublinearity Claim: Insufficient Data

### 2.1 Only 2 divergence levels with real data

The paper claims (Section 5):

> "The relationship is monotonic but sublinear — barrier increases more slowly than weight distance."

This claim rests on exactly **two** divergence levels: standard (~1.4% ΔW) and high (~8% ΔW). With only two data points, one can assert that the relationship is *monotonic* (trivially true for 2 points) but cannot distinguish between *linear* and *sublinear* behavior. A linear function passing through the origin and the two measured points would show the same property — barrier rising slower than ΔW — unless the barrier were proportional to ΔW.

### 2.2 The lr2e-4 and lr3e-4 files are byte-identical duplicates

Six result files exist on disk (`lmc_lr2e-4_s0.json` through `lmc_lr3e-4_s2.json`), but all six share the **identical md5 hash**: `2c1ded409f5a4a245fb5611d49619a67`. They are literal byte-for-byte copies of a single scan. Even worse, the endpoint losses in these files (L0_code=0.677, L0_med=0.631) are substantially higher than even the high-divergence models (L0_code=0.607-0.622), suggesting this scan was run on a different (likely misconfigured or under-trained) model pair. These files provide **no useful intermediate divergence points**.

### 2.3 What additional data is needed

To support the sublinearity claim, the experiment needs at least **3-4 distinct divergence levels** (e.g., ΔW at 1%, 3%, 5%, 8%) with consistent training convergence quality. The current data cannot distinguish sublinear from linear behavior. The proper way to test sublinearity is to fit both linear and power-law (barrier = c × ΔW^k, k < 1) models and compare goodness-of-fit — which requires at minimum 3 points per domain.

**Severity**: This is the weakest empirical claim in the paper. A reviewer WILL notice that the "sublinear" assertion is unsubstantiated. The claim should be downgraded to something like "the barrier increase is less than proportional to the weight displacement" without claiming a specific functional form.

---

## 3. Permutation Invariance: A Missing Discussion

### 3.1 Why permutation invariance matters

Entezari et al. (ICLR 2022) demonstrated that most apparent LMC failures between independently trained networks disappear after applying appropriate permutation matching. Ainsworth et al. (ICLR 2023) provided algorithms (Git Re-Basin) to find these permutations. The paper cites both works (Section 2) but never discusses whether permutation alignment is relevant to the experimental setup.

### 3.2 Is it actually relevant here?

**Probably not**, but the reasoning should be explicit. The models in this study are fine-tuned from the **same pretrained checkpoint** using full-parameter fine-tuning with the **same optimizer and schedule** (only learning rate varies). Weight-space permutations arise primarily when networks are trained *from different random initializations*. Since fine-tuning preserves the neuron alignments established during pretraining, the models should be permutation-aligned by construction. This is consistent with the standard-divergence results showing near-zero within-domain barriers for code (0.048).

However:
- The high-divergence medical models show substantial within-domain variance (0.147 ± 0.027), which could partially reflect partial permutation drift during aggressive fine-tuning.
- The paper should include a 1-2 sentence justification for why permutation matching was not performed, ideally with a citation to the fine-tuning context (e.g., Wortsman et al. 2022 also performed LMC without permutation matching for fine-tuned models).

**Recommendation**: Add a brief paragraph in the Limitations or Method section explaining why permutation matching is unnecessary when models share both initialization and training data domain (for within-domain) or just initialization (for cross-domain), and note that prior LMC work on fine-tuned models (Model Soups, TIES-Merging) also operates without permutation alignment.

---

## 4. Novelty Assessment

### 4.1 Has the quantitative ΔW → barrier mapping been measured before?

**Partially.** The general idea that larger weight displacement correlates with larger LMC barriers is intuitive and has been observed qualitatively in prior work. However, a systematic, controlled measurement of this relationship for LLM fine-tuning — varying only the optimization step size to produce different divergence levels, then measuring barriers with proper statistical treatment (3 seeds, bootstrap CIs) — is not, to my knowledge, directly present in the literature.

**What is novel:**
- The noise-floor calibration (identical copies → barrier ~0, random init → upper bound) is clean and methodologically sound. Using random-init interpolation as an empirical upper bound for the barrier is a nice idea.
- The finding that **within-domain medical barriers exceed cross-domain barriers** (medical within-domain 0.147 vs cross-domain 0.051 at standard divergence) is genuinely non-obvious and interesting. It reveals that domain-specific training instability can be a larger source of loss landscape disconnection than domain difference itself.
- The per-block correlation r=0.995 showing that code and medical fine-tuning change the **same** blocks (just different magnitudes) is a clean empirical result.

**What is NOT novel:**
- The qualitative claim that LMC holds for fine-tuned models from the same base. This is well-established (Frankle 2020, Wortsman 2022).
- The general idea that barriers grow with divergence. This is unsurprising and would be expected a priori.

### 4.2 Is this filling a real gap or reinventing known results?

The paper genuinely bridges two literatures (LMC theory and LLM domain adaptation) that have not been systematically connected. The **within-domain baseline insight** (medical instability > domain difference) is the strongest novel contribution and should be elevated in the narrative.

---

## 5. Comparison to Entezari (ICLR 2022)

Entezari et al.'s main finding is that **barrier heights between independently trained networks are primarily artifacts of permutation mismatch**, not genuine loss landscape disconnection. After applying optimal permutations, barriers collapse to near-zero.

This paper's context is fundamentally different:
- **Entezari concerns networks trained from different random initializations** (e.g., two CIFAR-10 ResNets from scratch). Permutation symmetries arise because neurons in layer k can be reordered arbitrarily without changing the function.
- **This paper concerns models fine-tuned from the same pretrained checkpoint.** Permutation alignment is preserved by the shared initialization and the continuity of gradient descent.

**However**, there is a subtle point the paper misses: even fine-tuning can induce permutation drift if the training is aggressive enough. The high-divergence medical condition (lr=5e-4, single-seed medical barriers up to 0.339 in the raw Frankle computation) generates enough weight displacement that some neuron reordering could occur. Entezari would argue that a quick Git Re-Basin check on the high-divergence medical models would either (a) confirm that permutation is irrelevant (barriers unchanged after matching) or (b) reveal that some of the high barrier is a permutation artifact. Either outcome strengthens the paper.

---

## 6. Additional Technical Issues

### 6.1 Asymmetry in experimental design

The paper evaluates LMC by interpolating between code and medical models and evaluating **both** domain test sets (loss_code, loss_med). However, within-domain baselines only evaluate on the domain's own data. This is appropriate for within-domain comparisons but creates an asymmetry: the cross-domain results show barriers on both code and medical evaluation sets, while within-domain results show barriers on only one. The paper should clarify this explicitly.

### 6.2 The lr2e-4/lr3e-4 data integrity problem

As documented in Section 2.2, all six intermediate-divergence scan files are byte-identical duplicates. If these files were generated automatically by `lmc_scans_only.sh` (which iterates over `LRS="1e-4 2e-4 3e-4 5e-4"`), the script may have failed silently for lr=2e-4 and 3e-4 (perhaps the trained models didn't exist), and the `cp` on line 24 would have copied stale `lmc_barrier_c1m1.json` results from a prior run. This is a data provenance issue that should be cleaned up before submission. **These files should either be regenerated with real data or removed.**

### 6.3 The 11-point interpolation is coarse but standard

The paper uses α ∈ {0.0, 0.1, ..., 1.0} for interpolation (11 points). Frankle et al. used 60 points. However, for the smooth LMC curves typically observed with fine-tuned models, 11 points is sufficient to detect the barrier peak — especially since the barrier is defined as max over α, and the curves appear convex (peaking near α=0.5). This is acceptable methodologically.

### 6.4 Self-domain loss for high-divergence models

The paper acknowledges (Section 6) that high-divergence models have higher self-domain loss than standard-divergence models (L0_code rises from 0.592 → 0.607-0.622). This means the high-divergence condition conflates two variables: weight displacement and convergence quality. A cleaner design would match self-domain loss across conditions by training longer at lower learning rates. The current design cannot cleanly attribute barrier increases to weight displacement vs. under-convergence. This is a significant confound.

---

## 7. Would a NeurIPS/ICML Reviewer Find This Sufficient?

### 7.1 As a full paper: Likely reject at NeurIPS/ICML

A top-tier venue reviewer would flag the following:

1. **Insufficient empirical scale**: One model family (Pythia-1.4B), two domains, two divergence levels (effectively — the intermediate LRs are ghost data). Compare to model merging papers at NeurIPS that evaluate across 5-10 model families and 8+ tasks.

2. **The sublinearity overclaim**: Claiming a functional form from two points is a red flag for reviewer scrutiny.

3. **The confound between ΔW and convergence quality**: High-divergence models have worse self-domain loss — was the barrier increase caused by weight displacement or under-training? The experiment cannot answer this.

4. **Novelty is modest**: The core insight ("more divergence → higher barrier") is unsurprising. The within-domain medical instability finding is the only genuinely non-obvious result.

5. **No predictive model**: The paper measures but does not model. There is no equation, scaling law, or theoretical framework for predicting barrier height from divergence metrics. This limits the conceptual contribution.

### 7.2 As a workshop paper or short paper: Potentially acceptable

The work is technically sound (numbers are correct, methodology is traceable), the noise-floor calibration is elegant, and the within-domain finding is surprising enough to generate discussion. A workshop venue (e.g., ICML WiML, NeurIPS FITML) would likely accept this with the sublinearity claim softened.

---

## 8. Scores (out of 10)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Novelty** | 5/10 | ΔW→barrier mapping not previously systematic for LLMs, but the qualitative finding is intuitive. Within-domain instability result is the novel part. |
| **Technical Depth** | 4/10 | Correct use of Frankle barrier definition, but no theoretical model, no permutation check, no scaling analysis beyond one model size. The confound between ΔW and convergence quality is not resolved. |
| **Experimental Rigor** | 6/10 | 3 seeds + bootstrap CIs + noise-floor calibration is good. But: duplicate lr2e-4/lr3e-4 data, 2-point sublinearity claim, and only 2 domains + 1 model family limit generalizability. |
| **Clarity** | 7/10 | Paper is well-structured and readable. Methods section is clear. Limitations section is honest. Figures referenced but not embedded in the draft. |
| **Importance** | 5/10 | Practical implication for model merging is real but incremental. Does not open a new research direction. The finding that training stability matters more than domain difference could inform future work. |
| **Overall** | **5.4/10** | A competent but incremental empirical study. Not competitive for NeurIPS/ICML as-is. |

---

## 9. Key Recommendations

1. **Fix the sublinearity claim immediately.** Replace with "the barrier increases less than proportionally to weight displacement" or, better, run intermediate-divergence experiments (lr=2e-4, 3e-4) properly and fit a functional form.

2. **Delete or regenerate the lr2e-4/lr3e-4 files.** Six byte-identical JSON files are a data integrity liability. If the scans were never run, remove the files. If they were run with wrong models, regenerate them.

3. **Add permutation invariance discussion.** 2-3 sentences in Section 3 or Section 6 explaining why permutation matching is unnecessary for fine-tuned-from-same-checkpoint models. Optionally run a single Git Re-Basin alignment on the highest-barrier pair (high-divergence medical, any seed pair) and report whether the barrier changes.

4. **Elevate the within-domain finding.** The medical instability result (0.147 within-domain > 0.051 cross-domain) is the most interesting empirical finding. Restructure the paper to make this the central narrative: "training instability, not domain difference, drives LMC barrier height."

5. **Address the convergence confound.** Either: (a) add a control experiment where models are trained to equal self-domain loss at different learning rates, or (b) discuss this confound more prominently as a limitation and argue why it does not invalidate the main conclusions.

6. **Fix the scan script barrier formula.** Change `lmc_barrier_scan.py` lines 101-102 to use `(losses_code[0] + losses_code[-1]) / 2` as baseline. This prevents future errors.

7. **Add more domains or model families.** Even one additional domain (e.g., math reasoning, legal text) or one additional model family (e.g., Llama-3.2-1B) would substantially strengthen generalizability claims.

---

## 10. Data Provenance Summary

All claims in this report were verified against:

- Raw LMC scan data: `/home/jiayu/AFP/experiments/phase0_ivn/results/lmc_lr1e-4_s{0,1,2}.json`, `lmc_lr5e-4_s{0,1,2}.json`, `lmc_code_s*_s*.json`, `lmc_medical_s*_s*.json`, `noise_*.json`
- Bootstrap CI summary: `/home/jiayu/AFP/experiments/phase0_ivn/results/bootstrap_ci.json` — values verified against recomputation from raw data
- Scan scripts: `/home/jiayu/AFP/scripts/lmc_barrier_scan.py` (lines 101-102 contain the barrier formula bug), `/home/jiayu/AFP/scripts/lmc_barrier.py` (lines 159-160, same bug), `/home/jiayu/AFP/scripts/lmc_6scans.sh` (lines 29-35, prints both barrier definitions)
- Paper draft: `/home/jiayu/AFP/docs/internal/PAPER.md`
- Project context: `/home/jiayu/AFP/CLAUDE.md`, `/home/jiayu/AFP/docs/internal/EXPERIMENT_PLAN.md`
- Per-block divergence: `/home/jiayu/AFP/experiments/phase0_ivn/results/per_block_divergence.json` (r=0.995 verified)
- Duplicate lr2e-4/lr3e-4 files: all six share md5 `2c1ded409f5a4a245fb5611d49619a67`
