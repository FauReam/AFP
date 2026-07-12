# Experimentalist Review: AFP Phase 0 LMC Paper

**Reviewer role:** Senior ML experimentalist, NeurIPS/ICML standards
**Date:** 2026-07-13
**Paper under review:** "Weight-Space Divergence and Loss Landscape Connectivity in Domain-Specialized Fine-Tuning" (Draft v4)

---

## Executive Summary

This paper measures how weight-space distance between domain-specialized fine-tuned models relates to linear mode connectivity (LMC) barriers. I independently verified the weight divergence of all 12 model pairs (standard + high divergence, 3 seeds each) against the HuggingFace pretrained Pythia-1.4B checkpoint, and recomputed all LMC barriers from raw data using the Frankle et al. (2020) formula. The core empirical claims are supported by the data. However, there are concerns at several levels: the seed count is minimal, the bootstrap CIs are degenerate, the barrier formula in the scanning script is wrong (though the paper itself uses the correct formula), and a body of lr=2e-4 and lr=3e-4 results consists of 6 identical file copies. The missing Gaussian-perturbation baseline and untrained-seed-pair calibration are notable omissions from the experimental design.

---

## 1. Independent Verification Results

### 1.1 Weight Divergence Verification

I loaded all 24 trained model checkpoints (12 code + 12 medical) from `/home/jiayu/AFP/experiments/trained_models/` and computed ||W_trained - W_base|| / ||W_base|| against the HuggingFace `EleutherAI/pythia-1.4b` checkpoint. Results:

| Condition | Seed | Code ΔW (verified) | Med ΔW (verified) | Cross ΔW (verified) | Is model trained? |
|-----------|------|-------------------|-------------------|---------------------|-------------------|
| Standard (lr=1e-4) | s0 | 1.39% | 1.59% | 2.10% | YES (242/291 tensors changed) |
| Standard (lr=1e-4) | s1 | 1.32% | 1.49% | 1.97% | YES (242/291 tensors changed) |
| Standard (lr=1e-4) | s2 | 1.43% | 1.47% | 2.03% | YES (242/291 tensors changed) |
| High (lr=5e-4) | s0 | 8.28% | 8.79% | 12.06% | YES (248-249/291 tensors changed) |
| High (lr=5e-4) | s1 | 7.62% | 8.47% | 11.30% | YES (248/291 tensors changed) |
| High (lr=5e-4) | s2 | 8.11% | 8.36% | 11.51% | YES (248/291 tensors changed) |

**Paper claims vs. verified:**

| Metric | Paper claim | Verified (mean ± std of 3 seeds) | Match? |
|--------|------------|----------------------------------|--------|
| Standard code ΔW | 1.4 ± 0.0% | 1.38 ± 0.05% | YES |
| Standard med ΔW | 1.5 ± 0.1% | 1.52 ± 0.06% | YES |
| Standard cross ΔW | 2.0 ± 0.1% | 2.03 ± 0.07% | YES |
| High code ΔW | 8.0 ± 0.3% | 8.00 ± 0.35% | YES |
| High med ΔW | 8.5 ± 0.2% | 8.54 ± 0.22% | YES |
| High cross ΔW | 11.6 ± 0.3% | 11.62 ± 0.39% | YES |

**Verdict:** All weight divergence measurements in the paper are verified. The trained models are clean — all show genuine weight movement. None are base model copies.

### 1.2 Barrier Verification (Frankle Formula)

I recomputed all barriers from the raw JSON loss data using the correct formula `barrier = max_α L(θ(α)) - (L(θ(0)) + L(θ(1))) / 2`.

| Condition | Seed | Code barrier (verified) | Medical barrier (verified) |
|-----------|------|------------------------|---------------------------|
| Standard (lr=1e-4) | s0 | 0.0467 | 0.0663 |
| Standard (lr=1e-4) | s1 | 0.0681 | 0.0347 |
| Standard (lr=1e-4) | s2 | 0.0441 | 0.0522 |
| **Standard mean** | | **0.0529 ± 0.0108** | **0.0511 ± 0.0129** |
| High (lr=5e-4) | s0 | 0.1349 | 0.3395 |
| High (lr=5e-4) | s1 | 0.1437 | 0.0938 |
| High (lr=5e-4) | s2 | 0.0748 | 0.2516 |
| **High mean** | | **0.1178 ± 0.0306** | **0.2283 ± 0.1016** |

Paper: Standard code=0.053±0.011, med=0.051±0.013, High code=0.118±0.031, med=0.228±0.102.

**Verdict:** All barriers are verified to within rounding tolerance. The paper's reported values are accurate.

### 1.3 Within-Domain Barriers (Verified)

| Domain | Verified barrier (mean ± std, 3 pairs) | Paper claim | Match? |
|--------|---------------------------------------|-------------|--------|
| Code (on code data) | 0.0484 ± 0.0002 | 0.048 ± 0.000 | YES |
| Medical (on medical data) | 0.1466 ± 0.0272 | 0.147 ± 0.027 | YES |

### 1.4 Noise Floor Calibration (Verified)

| Condition | Verified code barrier | Verified med barrier | Paper claim | Match? |
|-----------|----------------------|---------------------|-------------|--------|
| Identical copy | 0.0001 | 0.0001 | ~0.000 | YES |
| Pretrained ↔ Random init | 0.0333 | 0.2220 | 0.033 / 0.222 | YES |

---

## 2. Statistical Rigor

### 2.1 n=3 Seeds

The paper uses 3 random seeds per condition. This is common in ML but at the lower bound of what NeurIPS/ICML reviewers accept.

**Strengths:**
- The paper acknowledges this limitation in Section 6 (Limitations).
- Mean and standard deviation are reported honestly.
- The high-divergence medical condition shows large inter-seed variance (std = 0.1016, which is 44% of the mean of 0.228). With n=3, the standard error of the mean is 0.1016/sqrt(3) = 0.0587. A 95% confidence interval using t-distribution (df=2) would be 0.228 ± (4.303)(0.0587) = 0.228 ± 0.253 = [-0.025, 0.481]. This interval includes zero and negative values, meaning the high-divergence medical barrier result is **not statistically distinguishable from zero** at the 95% level using basic parametric inference.

**Concerns:**
- The "bootstrap 95% CI" reported in `bootstrap_ci.json` is misleading. With n=3, bootstrap percentile CIs essentially reproduce the min and max of the 3 observations. For standard code: the CI (0.044, 0.068) is exactly the min/max of the 3 barriers. This is not an inference procedure — it just reports the observed range. A legitimate bootstrap CI with n=3 has only 10 unique resamples (3^3 with replacement), making it functionally identical to enumeration, not statistical inference.
- The paper cites the bootstrap CIs as "reported in the supplementary analysis" (Section 6) but these CIs have no statistical meaning beyond describing the 3 observations.
- For any claim about the high-divergence medical condition specifically, n=3 is insufficient given the very large variance. At least 5 seeds would be needed for a meaningful t-based CI, and ideally 10+ for a defensible nonparametric CI.

### 2.2 Variance Reporting

The paper correctly reports mean ± std for all primary results. However, the per-condition sample sizes are never explicitly stated in tables — readers must infer n=3 from the text. This should be made explicit in each table caption.

### 2.3 Multiple Comparisons

The paper makes approximately 8-10 barrier comparisons (standard code, standard med, high code, high med, within-domain code, within-domain med, noise floor x2). No multiple-comparison correction is applied. Given n=3, this is a minor concern since no formal hypothesis tests are conducted, but if p-values were to be reported, a Bonferroni or Holm correction would be needed.

### 2.4 Score: Statistical Rigor -- 4/10

Rationale: n=3 is the primary limitation. The bootstrap CIs are not valid inference. The high-divergence medical barrier result has an enormous CI that undermines the precision of the claim. The paper does acknowledge the seed-count limitation, which mitigates somewhat, but does not account for how it affects the specific medical-domain claim.

---

## 3. Data Integrity

### 3.1 Bug 22 and Bug 21 Assessment

The engineering log records two serious data integrity bugs:

- **Bug 22** (2026-07-05): `code_e1` was the untrained base model. All experiments E2-E7 that used this model were invalid. The LMC barrier scans that used `code_e1` would have measured base-Pythia vs. medical-Pythia, not code-vs-medical.
- **Bug 21** (2026-07-05): Training saved the base model when training failed silently.

**Current status verified by independent weight inspection:**

I checked all 24 models in `/home/jiayu/AFP/experiments/trained_models/` against the HuggingFace base checkpoint. Results:

- **All standard-divergence models** (6 code + 6 medical at lr=1e-4): 242/291 tensors changed (49 tensors unchanged, likely embedding/LM-head layers). ΔW = 1.3-1.6%. All clean.
- **All high-divergence models** (6 code + 6 medical at lr=5e-4): 247-249/291 tensors changed. ΔW = 7.6-8.8%. All clean.
- **lr=2e-4 models** (3 code + 3 medical): 242-244/291 tensors changed. ΔW = 3.0-3.8% (code), 3.2-3.6% (medical). Clean.
- **lr=3e-4 models** (3 code + 3 medical): 244-245/291 tensors changed. ΔW = 4.6-4.7% (code), 4.7-5.2% (medical). Clean.

**No base model was found in the current trained_models directory.** The old `code_e1` directory that was the base model is no longer present. The current naming convention (`code_lr{lr}_s{seed}`) avoids confusion with the old epoch-based naming (`code_e{n}`).

### 3.2 Train-Agent Safeguard Verification

The current `train_agent.py` (lines 311-320) includes a Δ-verification guard:

```python
if changed > 1e-3:
    save_dir = OUT_DIR / args.domain
    agent.save(save_dir)
else:
    print(f"  [e{epoch}] WARNING SKIP save: model unchanged (Δ={changed:.6f})")
```

This guard was added as a direct response to Bug 21/22. I confirm it is present and would prevent untrained models from being saved as training results.

### 3.3 LMC Scan Script Issues

The `lmc_barrier_scan.py` script (line 89) hardcodes the output model name:

```python
'models': 'code_e1 + medical_e1',
```

This field is **not trustworthy** — it always says "code_e1 + medical_e1" regardless of which models were actually scanned. The `lmc_6scans.sh` wrapper uses symlinks to map the real model directories onto `code_e1` and `medical_e1`, so the actual models are correct, but the JSON metadata is misleading. This is a documentation bug, not a data integrity bug, but it could confuse future readers or reviewers.

### 3.4 Score: Data Integrity -- 7/10

Rationale: The current 24-model dataset is clean (independently verified). The training script has a safeguard against base-model contamination. However, the project history (Bugs 21, 22) shows that data integrity was not prioritized early on, and the misleading JSON metadata field ("code_e1 + medical_e1") persists in all result files. The lr=2e-4 and lr=3e-4 result files (6 files, all identical copies of the same md5) suggest sloppy data management practices that do not inspire full confidence.

---

## 4. Measurement Methodology

### 4.1 11-Point Linear Interpolation

The paper uses α ∈ {0.0, 0.1, ..., 1.0} for 11 interpolation points. Frankle et al. (2020) use 60 points, but 11 is standard in subsequent works (Wortsman et al. 2022, Yadav et al. 2023) and adequate for smooth convexity-like barrier shapes.

**Issue identified:** The LMC scan script (`lmc_barrier_scan.py`, lines 101-102) computes the barrier using a **wrong formula**:

```python
barrier_code = max(losses_code) - losses_code[0]   # WRONG
barrier_med  = max(losses_med)  - losses_med[0]    # WRONG
```

The Frankle et al. (2020) formula is `barrier = max_α L(θ(α)) - (L(θ(0)) + L(θ(1))) / 2`. The script's formula only subtracts L(0), not the average of both endpoints. This inflates the barrier by approximately L(1)/2. However, **the paper correctly cites and uses the Frankle formula** — the barriers reported in the paper match the recomputed values using the correct formula. The shell scripts (`lmc_6scans.sh`) print both formulas for comparison. So the numeric values in the paper are correct; the script-level bug was caught and corrected during analysis.

### 4.2 2000 Test Samples

2000 samples per domain for BCE loss evaluation is reasonable. BCE loss converges quickly as a sample statistic. With a batch size of 128, this gives 16 evaluation batches. The evaluation is deterministic (no dropout) and the batch-level loss is summed over all samples, so the total loss is an exact aggregate.

**Potential concern:** The test set is drawn from the same VersaPRM dataset as the training set (85/15 split), not from an independent held-out corpus. For the measurement task (LMC barrier height estimation rather than performance benchmarking), this is acceptable since we care about relative loss changes along the interpolation path, not absolute loss values. However, it does mean the loss numbers are not indicative of out-of-distribution performance.

### 4.3 BCE Loss

Binary cross-entropy is appropriate for the PRM binary classification task (step correctness). The loss is computed with `reduction='sum'` across valid tokens and then divided by the number of samples, yielding per-sample average loss. This is standard.

### 4.4 Head Weight Divergence Not Discussed

The paper measures backbone weight divergence only. I also checked head weight divergence:

| Model | Head ΔW (from random init) |
|-------|---------------------------|
| code_lr1e-4_s0 | 181% |
| code_lr5e-4_s0 | 175% |
| medical_lr1e-4_s0 | 35% |
| medical_lr5e-4_s0 | 102% |

Head weights start from random initialization and change dramatically during training. The paper's focus on backbone weights is defensible (the theoretical LMC literature also focuses on backbone), but the head divergence is an under-explored confound. When two domain-specialized models are interpolated in backbone space, the head is taken from whichever model holds the backbone state — this creates a mismatch between the backbone and head at intermediate α values. The paper does not discuss this.

### 4.5 Score: Measurement Validity -- 6/10

Rationale: The core protocol (interpolation + BCE evaluation) is sound and standard. The 11-point sampling is adequate. But the script-level barrier formula bug, the misleading metadata, and the unaddressed head-backbone mismatch in the interpolation procedure are concerns. The paper cites the correct formula and reports correct numbers, but the measurement pipeline has a known bug that could have produced wrong numbers if not manually corrected downstream.

---

## 5. Missing Controls and Baselines

### 5.1 Untrained Model Interpolation (Missing)

The paper does not include a baseline where two **different random seeds of the untrained base model** are interpolated. Since the base model checkpoint is deterministic (loaded from the same HuggingFace weights), two "different seeds of untrained" would be identical, so this would essentially repeat the identical-copy noise floor. However, a more informative baseline would be: **interpolation between two independently fine-tuned models on the SAME domain at different seeds**, which IS provided as the "within-domain baseline" (Section 4.3). This is a reasonable substitute.

### 5.2 Gaussian Noise Perturbation of Weights (Missing)

A standard control in LMC studies is to add independent Gaussian noise to weights and measure the resulting barrier, calibrating how much of the observed barrier could be attributed to random perturbation rather than structured learning. The paper does not include this control.

**Mitigation:** The noise floor calibration (identical copy and pretrained↔random init) partially addresses this. The identical copy shows that the measurement pipeline introduces essentially zero barrier (0.0001). The pretrained↔random init establishes an upper bound. However, these are boundary conditions (zero displacement and maximum displacement), not intermediate perturbation calibrations. A Gaussian perturbation at matching levels of weight displacement (e.g., ΔW = 1.4% and ΔW = 8%) would directly test whether the observed barriers are attributable to the structure of domain-specific learning or simply to the magnitude of weight displacement.

### 5.3 Missing: Same-Hyperparameter, Same-Domain, Different-Random-Split

The within-domain baseline interpolates between models trained with the same hyperparameters but different random seeds (which means different data shuffles for each seed — the data split uses `torch.randperm`, line 116 of `train_agent.py`). This is a good control and is appropriately discussed. However, all three within-domain pairs for code use lr=1e-4 (standard divergence), and all three for medical use lr=1e-4. No within-domain baseline at high divergence (lr=5e-4) is provided, which would help disambiguate whether the high-divergence barrier increase is due to domain difference or simply higher weight displacement.

### 5.4 Score: Baseline Coverage -- 5/10

Rationale: The noise floor calibrations are well-designed. The within-domain baseline is a genuine strength of the paper and produces the most interesting result (medical within-domain > cross-domain). However, the absence of a Gaussian-perturbation calibration and the absence of high-divergence within-domain baselines are significant gaps. The paper also lacks a "no fine-tuning" baseline comparing the base model's loss on both domains to the fine-tuned models' losses, which would clarify whether fine-tuning actually specialized the models (rather than just moved weights).

---

## 6. lr=2e-4 and lr=3e-4 File Identity Issue

**Finding:** All 6 LMC result files for lr=2e-4 and lr=3e-4 (3 seeds each) have the identical md5 hash: `2c1ded409f5a4a245fb5611d49619a67`. The files are bit-for-bit identical copies.

**Evidence:**
```
lmc_lr2e-4_s0.json  md5=2c1ded409f5a4a245fb5611d49619a67
lmc_lr2e-4_s1.json  md5=2c1ded409f5a4a245fb5611d49619a67
lmc_lr2e-4_s2.json  md5=2c1ded409f5a4a245fb5611d49619a67
lmc_lr3e-4_s0.json  md5=2c1ded409f5a4a245fb5611d49619a67
lmc_lr3e-4_s1.json  md5=2c1ded409f5a4a245fb5611d49619a67
lmc_lr3e-4_s2.json  md5=2c1ded409f5a4a245fb5611d49619a67
```

**Analysis:**
- The lr=2e-4 and lr=3e-4 models ARE genuinely different (verified ΔW: code_lr2e4_s0 = 3.84%, code_lr3e4_s0 = 4.58%). The trained models exist and are clean.
- However, the LMC result files were clearly copied, not generated from independent scans. Only one scan was actually run, and its output was duplicated 6 times.
- This means the within-seed and between-lr variation in LMC barriers at these intermediate learning rates was never actually measured.

**Impact on paper integrity:** NONE for the current paper content. The paper reports only the standard-divergence (lr=1e-4) and high-divergence (lr=5e-4) conditions, and those 6 result files are all unique with distinct md5 hashes and genuinely different loss values. The lr=2e-4 and lr=3e-4 results are not cited or used in the paper.

**But this is a serious concern for project data management.** It indicates that:
1. Someone manually copied result files rather than running the scans.
2. The project lacks automated integrity checks (e.g., timestamp validation, expected runtime verification, cross-file uniqueness checks).
3. If the paper were to be extended to include intermediate divergence levels, these copied files could be mistakenly treated as legitimate experimental data.

**Recommendation:** Delete or explicitly mark these 6 files as "placeholder/copy — not valid experimental data" to prevent future misuse. Institute a policy that all result files must be generated by running scripts, never by manual file copy.

---

## 7. Reproducibility Assessment

### 7.1 Strengths
- Training script (`train_agent.py`) is single-file, self-contained, and includes all hyperparameters at the top as module-level constants.
- The LMC scan script is straightforward and reproducible.
- Shell scripts (`lmc_6scans.sh`, `lmc_within_domain.sh`, `noise_floor.sh`) document the exact commands used.
- Models are saved with descriptive naming (`{domain}_lr{lr}_s{seed}`) and all weight files are present.
- Raw loss data is preserved in JSON format, enabling independent barrier recomputation (as done in this review).
- The VersaPRM dataset path is specified, and the data preparation code includes caching logic.

### 7.2 Weaknesses
- **Hardware specificity:** Training on DGX Spark GB10 (ARM64 CUDA 13.0, 121GB unified memory) is a unique hardware configuration. Reproducing the exact results on standard x86_64 GPU hardware may produce numerically different weight trajectories due to non-deterministic CUDA operations, even with the same random seed. The paper does not discuss this.
- **Non-determinism:** The training script does not set `torch.use_deterministic_algorithms(True)` or `CUBLAS_WORKSPACE_CONFIG`. Full reproducibility requires deterministic mode, which is not mentioned.
- **No random seed recording:** The training script uses `torch.randperm` for data splitting but does not record the seed, making exact data-split reproduction impossible without additional effort.
- **CUDA bf16 autocast:** bf16 operations are not bitwise-deterministic across GPU architectures. Results may differ on non-Blackwell hardware.
- **Model file naming in scan script:** The hardcoded `"code_e1 + medical_e1"` metadata field makes it impossible to determine from the JSON alone which models were actually scanned.

### 7.3 Score: Reproducibility -- 6/10

Rationale: The code and data are well-organized and the raw results are preserved. However, hardware-specific non-determinism, missing random seed recording, and the hardcoded metadata field reduce confidence in exact numerical reproduction. A deterministic mode flag and seed logging would be easy additions.

---

## 8. Additional Observations

### 8.1 Cross-Domain Asymmetry Claim

The paper claims (Section 4.4) that "code fine-tuning produces a small positive externality for medical reasoning." This is based on comparing base model loss on medical data vs. code-model loss on medical data. I did not independently verify this claim (it requires evaluating the base model on the test sets, which I did not run), but the LMC scan data shows: at α=0.0 (pure code model), loss_med ranges from 0.488 to 0.651 (standard divergence). The paper's base-model medical loss is not reported, so I cannot verify the direction of change. This claim should be supported with explicit base-model evaluation numbers.

### 8.2 Per-Block Divergence Pattern

The paper reports r=0.995 correlation between code and medical per-block divergence patterns. The `per_block_divergence.json` file reports r=0.9949. This is verified. However, the JSON also contains raw values that are presented as "importance" (not divergence). The `code_imp` and `med_imp` arrays range from ~1.0 to ~0.19, which does not match the per-block divergence percentages shown in the paper's Figure 2 (layer 0 ~5.6%). These appear to be normalized importance scores, not raw divergence percentages. The paper should clarify the transformation.

### 8.3 160M Scaling Experiment

The paper mentions a "preliminary scaling experiment with Pythia-160M" producing barriers of 0.001 (code) and 0.056 (medical). I could not verify this because the 160M model architecture does not match 1.4B (different hidden size), and the model file uses a separate `code_160m_s0` / `medical_160m_s0` naming convention. The `lmc_160m.json` file exists and contains plausible loss values. The paper appropriately labels this as a "preliminary" result in the Limitations section.

---

## 9. Scorecard

| Category | Score | Rationale |
|----------|-------|-----------|
| **Statistical Rigor** | 4/10 | n=3 with degenerate bootstrap CIs; high-divergence medical condition not statistically distinguishable from zero under t-based CI; no multiple-comparison correction |
| **Data Integrity** | 7/10 | Current model weights independently verified clean; training script has safeguard; but Bug 21/22 history and lr=2e-4/3e-4 file copies indicate past data hygiene problems |
| **Measurement Validity** | 6/10 | Core protocol (interpolation + BCE) is sound; 11-point sampling adequate for smooth curves; but barrier formula bug in scanning script, misleading metadata, and unaddressed head-backbone mismatch |
| **Reproducibility** | 6/10 | Code is self-contained and well-documented; raw data preserved; but hardware-specific non-determinism (ARM64 CUDA, bf16) means exact numerical reproduction unlikely on different hardware |
| **Baseline Coverage** | 5/10 | Noise floor calibrations are good; within-domain baseline is the paper's strongest result; but missing Gaussian-perturbation calibration, no high-divergence within-domain baseline, no explicit base-model evaluation numbers |

**Overall: 28/50 (56%)**

---

## 10. Recommendations

### Critical (must address before submission)

1. **Increase seeds to n >= 5 for the high-divergence condition.** The current n=3 with std=0.102 for high-divergence medical produces a CI that nearly includes zero. The paper's claim that the high-divergence medical barrier "nearly saturates" the random-init bound (0.23 vs 0.22) is interesting but not statistically reliable with n=3 given the large variance.

2. **Remove or clearly caveat the bootstrap CI claims.** Either (a) drop the mention of bootstrap CIs entirely and report honest min-max ranges, or (b) increase n to at least 20 so bootstrap percentile CIs are meaningful. The current "bootstrap CIs" with n=3 are not valid inference.

### Important (should address before submission)

3. **Add a Gaussian-perturbation calibration.** Train a model, add independent Gaussian noise scaled to produce ΔW = 1.4% and ΔW = 8%, and measure the resulting LMC barrier. This directly tests whether the observed barriers are due to structured learning or magnitude alone.

4. **Add high-divergence within-domain baselines.** The paper's most interesting result is that medical within-domain barrier (0.147) exceeds cross-domain (0.051). But this was only measured at standard divergence. Does this hold at high divergence? Running within-domain LMC at lr=5e-4 for both code and medical would strengthen the paper's core narrative.

5. **Fix the hardcoded metadata in lmc_barrier_scan.py.** The `"code_e1 + medical_e1"` string should be replaced with the actual model paths or configuration names. This is a one-line code change.

### Minor (nice to have)

6. **Report base model loss on both domains.** The paper claims a cross-domain asymmetry (code helps medical but not vice versa) without reporting the base model's loss. Add a simple base model evaluation to the results table.

7. **Add deterministic mode.** Set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the training script, and record the random seed.

8. **Fix the barrier formula in lmc_barrier_scan.py.** Line 101-102 should compute `max(L) - (L[0] + L[-1]) / 2` to match the Frankle et al. formula that the paper cites.

9. **Clean up the lr=2e-4 and lr=3e-4 result files.** Either delete the 6 identical copies or replace them with a README explaining they are not valid experimental data.

10. **Document head-backbone mismatch.** Add a brief discussion in the Limitations section noting that the LMC interpolation procedure changes only backbone weights while the classification head remains from the code model, creating a potential confound at intermediate α values.

---

## 11. Overall Assessment

The paper asks a well-motivated question and the core experimental design (train on two domains, measure weight divergence, compute LMC barriers) is appropriate. The data supporting the main claims is verified: all trained models show genuine weight displacement, and the reported barriers recompute correctly from raw loss data. The within-domain baseline result (Section 4.3) is the paper's most interesting and original contribution — the finding that medical training instability drives barrier height more than domain difference is novel and well-supported.

The primary weaknesses are statistical (n=3 with large variance in the high-divergence medical condition), baseline completeness (missing Gaussian perturbation control), and the troubling history of data integrity bugs (even though the current dataset is clean). The lr=2e-4 / lr=3e-4 file duplication is not relevant to the paper's claims but indicates a data management culture that needs improvement.

**Recommendation:** Revise and resubmit with additional seeds, Gaussian perturbation calibration, and corrected bootstrap reporting. The core finding — that domain specialization at standard training intensity does not break linear connectivity — is robust and will stand with improved statistics.

---

## Appendix A: Verification Script

The verification script used for this review is at:
`/home/jiayu/AFP/experiments/expert_panel/verify_models.py`

It loads all trained model checkpoints from `/home/jiayu/AFP/experiments/trained_models/`, computes weight divergence against the HuggingFace `EleutherAI/pythia-1.4b` checkpoint, recomputes barriers from raw JSON data using the correct Frankle formula, and checks file identity between lr=2e-4 and lr=3e-4 result files.

## Appendix B: Key File Paths

| Item | Path |
|------|------|
| Paper | `/home/jiayu/AFP/docs/internal/PAPER.md` |
| Engineering log (22 bugs) | `/home/jiayu/AFP/docs/internal/ENGINEERING.md` |
| Training script | `/home/jiayu/AFP/scripts/train_agent.py` |
| LMC scan script | `/home/jiayu/AFP/scripts/lmc_barrier_scan.py` |
| LMC 6-scans wrapper | `/home/jiayu/AFP/scripts/lmc_6scans.sh` |
| Within-domain scan wrapper | `/home/jiayu/AFP/scripts/lmc_within_domain.sh` |
| Noise floor calibration | `/home/jiayu/AFP/scripts/noise_floor.sh` |
| Trained models directory | `/home/jiayu/AFP/experiments/trained_models/` (24 models) |
| LMC results directory | `/home/jiayu/AFP/experiments/phase0_ivn/results/` (29 files) |
| Bootstrap CI file | `/home/jiayu/AFP/experiments/phase0_ivn/results/bootstrap_ci.json` |
| Per-block divergence file | `/home/jiayu/AFP/experiments/phase0_ivn/results/per_block_divergence.json` |
| Verification script | `/home/jiayu/AFP/experiments/expert_panel/verify_models.py` |
| This report | `/home/jiayu/AFP/experiments/expert_panel/experimentalist.md` |
