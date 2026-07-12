# Expert Panel Findings — AFP Phase 0 LMC Paper

> **Date:** 2026-07-13 | **Panel:** 6 experts, adversarial review
> **Verdict:** WEAK REJECT for NeurIPS/ICML | **Panel Mean Score:** 4.8/10
> **Source reports:** `experiments/expert_panel/*.md` (full individual reports)

---

## Expert Roster

| # | Role | Focus | Overall Score |
|---|------|-------|:---:|
| 1 | LMC/Loss Landscape Theorist | Novelty, theoretical grounding, sublinearity | 5.4 |
| 2 | Experimental Design & Statistics | Statistical rigor, data integrity, reproducibility | 5.6 |
| 3 | Domain/NLP Specialist | Domain framing, asymmetry, NLP contribution | 4.5 |
| 4 | Adversarial Reviewer 2 | Fundamental flaws, overclaims, missing baselines | 3.0 |
| 5 | Senior AC / Meta-Reviewer | Conference fit, venue strategy, contribution level | ~5 (35th %ile) |
| 6 | Model Merging Practitioner | Practical utility, actionability, merge quality | ~4.2 |

---

## Convergence Analysis

### What ALL Six Experts Agree On

1. **The numbers are correct.** Three experts independently re-derived barriers from raw JSON using the Frankle formula. All published values match to within rounding tolerance. Weight divergence verified against HuggingFace base checkpoint. All 24 trained models are clean (no base-model contamination).

2. **The within-domain medical finding is the paper's strongest contribution.** The discovery that two medical models trained on the *same* data (barrier = 0.147) exhibit larger loss-landscape dissimilarity than a code and medical model trained at the same intensity (barrier = 0.051) is counterintuitive, novel, and supported by the data. [[EXPERT_PANEL_FINDINGS#w5]]

3. **n=3 seeds is insufficient for the high-divergence medical condition.** Coefficient of variation = 45%. Bootstrap 95% CI with n=3: [0.094, 0.340]. t-based CI: [-0.025, 0.481] — includes zero. [[EXPERT_PANEL_FINDINGS#w3]]

4. **The "sublinear" claim is unsubstantiated.** Only two valid divergence levels exist (standard ≈1.4%, high ≈8%). The lr=2e-4 and lr=3e-4 LMC results are six byte-identical file copies (md5: `2c1ded40`). A functional-form claim requires ≥4 distinct levels. [[EXPERT_PANEL_FINDINGS#w1]]

5. **The high-divergence convergence confound is serious.** High-divergence models have higher self-domain loss than standard-divergence models (L0_code: 0.592 → 0.607-0.622). The barrier increase conflates: (a) genuine LMC degradation at larger ΔW, and (b) both endpoints being worse models whose convex combination is also worse. [[EXPERT_PANEL_FINDINGS#w4]]

6. **The noise-floor calibration is methodologically strong.** Identical-copy barrier ≈ 0.000 confirms artifact-free measurement. Pretrained-to-random-init baseline provides a reference for interpreting magnitudes. All six experts praise this as exemplary rigor.

7. **The barrier formula bug exists in scan scripts.** `lmc_barrier_scan.py:101-102` and `lmc_barrier.py:159-160` compute `max(L) - L[0]` instead of the Frankle definition `max(L) - (L[0]+L[-1])/2`. Paper numbers use the correct formula (manually corrected in analysis), giving different results from running the scripts directly. [[EXPERT_PANEL_FINDINGS#w10]]

8. **The paper does not belong at ACL/EMNLP.** Zero NLP contribution — no linguistic analysis, no NLP benchmarks, no domain adaptation insights. Domain Specialist gives NLP Relevance 3/10. Venue mismatch guarantees negative reviews regardless of paper quality.

9. **lr=2e-4/lr=3e-4 data is corrupted.** Six JSON files share identical md5 hashes. These are copy-pasted, not measured. The models DO exist and are genuinely different (verified ΔW = 3.0-5.2%), but LMC scans were never run — output files were duplicated. [[EXPERT_PANEL_FINDINGS#w1]]

### What Experts Disagree On

| Issue | Positions |
|-------|-----------|
| **Severity of the upper-bound contradiction** (0.228 > 0.222) | Adversarial: fatal logical error (9/10 severity). Theorist: notes in passing. Meta: entire high-div condition is confounded anyway. |
| **Acceptability at NeurIPS/ICML** | Adversarial: flat reject (3/10). Meta: 15-20%. Theorist: "not competitive as-is." Range reflects different priors about reviewer tolerance. |
| **Novelty of ΔW→barrier mapping** | Theorist: partially novel. Adversarial: entirely expected. Meta: 35th percentile. Merging: interesting but not actionable. |
| **Importance of permutation alignment** | Adversarial + Theorist: significant gap. Experimentalist: not flagged. Domain: irrelevant. All agree a justification paragraph is needed. |
| **Interpretation of medical within-domain finding** | Theorist + Domain: domain property. Adversarial: refutation of central claim. Merging: most valuable contribution but needs replication. |

### Evidence vs. Interpretation Boundary

The following claims are interpretive, not directly measured:

1. **"Barrier is driven by weight displacement, not domain difference."** Supported by code (within=0.048 ≈ cross=0.053), contradicted by medical (within=0.147 >> cross=0.051). The "training stability" reframe changes the claim rather than rescuing it.

2. **"The relationship is sublinear."** Two data points cannot distinguish linear from sublinear. This is a category error — using functional-form language without functional-form data.

3. **"Model merging is likely to work well."** The paper measures interpolation loss, not merge quality. No downstream accuracy, no TIES/Task Arithmetic comparison. The Merging Practitioner gives Actionability 2/10.

4. **"Code FT produces positive externality for medical."** This claim may originate from the Bug 22 era when "code" model was the untrained base. Whether the asymmetry survives with correctly-trained models is not independently verified across all seeds.

5. **"Pretrained-to-random barrier (0.222) is an upper bound."** The high-divergence medical barrier (0.228) exceeds it. Whether 0.222 is a true bound, an uncalibrated estimate, or a different measurement entirely is not resolved.

---

## Critical Weaknesses (Ranked by Severity)

### W1: Only 2 Valid Divergence Levels — Sublinearity Unsubstantiated <span style="color:#e74c3c">CRITICAL</span>

**Flagged by:** Theorist, Adversarial, Meta

**Evidence:** Two divergence levels (standard ≈1.4%, high ≈8%). Intermediate lr=2e-4 and lr=3e-4 LMC results are six byte-identical file copies (md5: `2c1ded40`). The models exist and are genuinely different (verified ΔW = 3.0-5.2%), but LMC scans were never run.

**Fix:** Run LMC scans at lr=2e-4 and lr=3e-4 (models already trained). Add at least one more divergence level. Soften claim from "sublinear" to "barrier grows less than proportionally to divergence." Delete the fake result files.

**Est. GPU cost:** ~1.5 GPU-hours (scans only, models exist)

---

### W2: Upper Bound (0.222) Exceeded by Measurement (0.228) <span style="color:#e74c3c">CRITICAL</span>

**Flagged by:** Adversarial, Theorist, Meta

**Evidence:** Pretrained-to-random-init medical barrier = 0.222 is characterized as "upper bound." High-divergence cross-domain medical barrier = 0.228 ± 0.102. Individual seed s0 yields 0.340, exceeding the "bound" by 53%. The paper papers over this with "nearly saturates."

**Fix:** Either (a) remeasure pretrained-to-random barriers with ≥5 seeds to establish a CI that contains 0.228, (b) drop "upper bound" and reframe as "reference calibration," or (c) acknowledge bound violation and attribute it to the convergence confound.

**Est. GPU cost:** ~0.5 GPU-hours (recalibration scans) + text revision

---

### W3: High-Divergence Medical Barrier Statistically Unreliable <span style="color:#e74c3c">CRITICAL</span>

**Flagged by:** Experimentalist, Adversarial, Theorist, Meta

**Evidence:** Per-seed barriers: 0.340, 0.094, 0.252 (span: 3.6×). Bootstrap CI with n=3 is degenerate — merely enumerates the 3 observations. t-based 95% CI: [-0.025, 0.481] includes zero. Mean (0.228) is dominated by one outlier seed.

**Fix:** Increase to n ≥ 5 seeds for the high-divergence medical condition. If the large variance is real, report full per-seed results transparently and adjust claims.

**Est. GPU cost:** ~7 GPU-hours (2 additional high-div medical trainings + LMC scans)

---

### W4: High-Divergence Convergence Confound <span style="color:#e74c3c">CRITICAL</span>

**Flagged by:** Theorist, Meta, Adversarial, Merging

**Evidence:** The paper acknowledges (Section 6) that high-divergence models have higher self-domain loss. The barrier increase from 0.05 to 0.12-0.23 conflates two mechanisms: (a) genuine LMC degradation at larger ΔW, and (b) both endpoints being worse models (under-converged) whose convex combination is worse. Cannot attribute barrier increase to weight displacement alone.

**Fix:** Either (a) re-run high-divergence with a schedule that achieves comparable convergence (e.g., train longer at same high LR, or use warmup+decay), or (b) restructure paper so high-divergence is exploratory analysis with the confound prominently discussed, while primary claims rest on standard-divergence results.

**Est. GPU cost:** Option (a): ~14 GPU-hours. Option (b): text revision only.

---

### W5: Core Claim Contradicted by 50% of Evidence <span style="color:#e74c3c">CRITICAL</span>

**Flagged by:** Adversarial, Domain Specialist, Theorist

**Evidence:** "Barrier is driven by weight displacement, not domain difference" holds for code (within=0.048 ≈ cross=0.053) but is directly contradicted for medical (within=0.147 >> cross=0.051). A claim supported by 50% of the evidence cannot be the central thesis.

**Fix:** Restructure narrative around the *domain-specific stability asymmetry*. The correct thesis: "Training stability, not domain difference, is the dominant source of LMC barrier height — and stability varies dramatically across domains, with medical training producing 3x higher within-domain barriers than cross-domain barriers."

**Est. cost:** Text revision only (major restructuring)

---

### W6: Single Model Family — Generalizability Unproven <span style="color:#f39c12">MAJOR</span>

**Flagged by:** Meta, Adversarial, Theorist

**Evidence:** All primary results on Pythia-1.4B only. The 160M "scaling check" is trivial — both models barely moved (ΔW ≈ 0.3%), making it a different experimental regime. No other architecture.

**Fix:** Add at least one additional model family (Llama-3.2-1B or OLMo-1B) at standard divergence on both domains. Train 2 domains × 1 div level × 3 seeds = 6 models, plus LMC scans.

**Est. GPU cost:** ~8 GPU-hours

---

### W7: Missing Permutation Invariance Check (Git Re-Basin) <span style="color:#f39c12">MAJOR</span>

**Flagged by:** Adversarial, Theorist, Meta

**Evidence:** Paper cites Entezari (ICLR 2022) [1] and Ainsworth (ICLR 2023) [2] but never applies permutation matching. The high within-domain medical barrier (0.147) could partially reflect permutation drift during aggressive fine-tuning rather than genuine loss landscape barriers. Without a Git Re-Basin check, we cannot distinguish.

**Fix:** Minimum: add 2-3 sentence justification in Methods/Limitations explaining why permutation matching is unnecessary for same-checkpoint fine-tuning. Recommended: run one Git Re-Basin alignment on the highest-barrier pair and report barrier with and without alignment.

**Est. GPU cost:** Text: 30 min. Git Re-Basin: ~2 GPU-hours + implementation

---

### W8: "Domain Specialization" Framing Is Weak <span style="color:#f39c12">MAJOR</span>

**Flagged by:** Domain Specialist, Meta

**Evidence:** VersaPRM is a process reward model dataset [3]. The task is binary classification of reasoning step correctness — meta-reasoning, not domain-specific generation. The domain label is a surface tag on the question. Both models learn fundamentally similar verification skills on different text distributions. This partially explains why cross-domain barriers are so low.

**Fix:** Reframe as "task-specialized fine-tuning" or "fine-tuning divergence analysis" rather than "domain-specialized." Alternatively, keep domain framing but add a paragraph in Limitations acknowledging that PRM verification data differs from standard domain adaptation corpora.

**Est. cost:** Text revision only

---

### W9: No Connection to Downstream Merge Quality <span style="color:#f39c12">MAJOR</span>

**Flagged by:** Merging Practitioner, Meta

**Evidence:** Paper claims practical implications for model merging (Section 5) but never runs a merge experiment. Barrier is measured on BCE training loss — it tells us nothing about whether the merged model works on downstream tasks. No TIES-Merging [4], Task Arithmetic [5], or Model Soups [6] evaluation. Practitioners cannot use the paper's numbers to make merge decisions.

**Fix:** Run one merge experiment: standard-divergence code+medical merged via (a) pure averaging and (b) TIES-Merging, evaluate on downstream code and medical benchmarks. Plot merge quality vs. barrier height. This converts a geometric measurement into a decision tool.

**Est. GPU cost:** ~12 GPU-hours (merge + evaluate)

---

### W10: Barrier Formula Bug in Scan Scripts <span style="color:#3498db">MODERATE</span>

**Flagged by:** Theorist, Experimentalist, Adversarial

**Evidence:** `lmc_barrier_scan.py:101-102` computes `max(L) - L[0]` instead of `max(L) - (L[0]+L[-1])/2`. Same bug in `lmc_barrier.py:159`. Paper numbers use correct formula (manually corrected), but any reviewer who runs the scripts gets different values. The hardcoded metadata string `"code_e1 + medical_e1"` is misleading when actual paths differ.

**Fix:** Fix both scripts to use the Frankle definition [7]. Fix hardcoded metadata to use actual model paths. Add assertion that printed barrier matches JSON-saved barrier.

**Est. cost:** ~30 min (code fix)

---

### W11: Missing Baselines <span style="color:#3498db">MODERATE</span>

**Flagged by:** Experimentalist, Adversarial, Merging

**Evidence:** No Gaussian perturbation calibration to separate structured (training) from unstructured (noise) weight displacement effects. No base model evaluation on both domains (asymmetry claim unverifiable). No high-divergence within-domain baselines (cannot disambiguate domain vs. displacement effects at high ΔW).

**Fix:** Add Gaussian perturbation at ΔW ≈ 1.5% and 8% (~1 GPU-hour). Evaluate base model on both test sets (~15 min). Add high-div within-domain scans for code and medical (~1 GPU-hour).

**Est. GPU cost:** ~2.5 GPU-hours

---

## Strengths

1. **The within-domain medical finding is genuinely novel.** "Two medical models trained on the same data exhibit larger loss-landscape dissimilarity than a code and medical model trained at the same intensity" — counterintuitive, data-supported, and has real implications for model merging practice. All six experts independently identify this as the paper's strongest contribution.

2. **Noise-floor calibration is exemplary.** Identical-copy (≈0.000) + pretrained-to-random-init baselines. Both the Theorist and Meta Reviewer praise this as the kind of rigor check that distinguishes careful work from sloppy work.

3. **All numerical claims are independently verified.** Three experts independently recomputed weight divergence against the HuggingFace base checkpoint and recomputed barriers using the correct Frankle formula from raw JSON data. Every reported number matches. All 24 trained models are clean.

4. **Per-block divergence pattern (r = 0.995) is crisp and replicable.** Code and medical fine-tuning change the same transformer blocks, differing only in magnitude. Has implications for parameter-efficient fine-tuning: domain-agnostic PEFT methods may be well-justified.

5. **Methodological transparency.** The 22-bug engineering log [8], the preservation of raw JSON loss data enabling independent barrier recomputation, and shell scripts documenting exact experimental commands represent a level of transparency above the field's norm.

6. **Clean experimental design.** Vary only optimization intensity to produce different weight displacement levels → measure LMC barriers at each level → calibrate with noise floor. The ablation logic (within-domain baselines as controls) is sound.

7. **Writing is clear and well-structured.** The Theorist gives Clarity 7/10, the highest dimension score. Methods section is precise, Limitations is honest, Discussion appropriately hedges most claims.

---

## Venue Strategy

### Tier 1: NeurIPS 2027 / ICML 2027 — NOT RECOMMENDED

| Metric | Value |
|--------|-------|
| Current readiness | Not competitive (Adversarial: 3/10, Meta: 15-20%) |
| What's needed | Second model family, 2+ additional domains, 4-5 divergence levels, Git Re-Basin, downstream merge evaluation, convergence confound resolution, n≥5 seeds, Gaussian perturbation baseline, mechanistic hypothesis |
| Timeline | 4-6 weeks experiments + 2 weeks writing (~3-5 GPU-days on DGX Spark) |
| Probability after strengthening | 35-45% |

**Risk:** One hostile reviewer saying "single model family, 2 domains, expected results" tanks the paper. NeurIPS always has at least one such reviewer. The ceiling is ~35% even with perfect experiments if the contribution remains purely empirical.

### Tier 1.5: ICLR 2028 — RECOMMENDED PRIMARY TARGET

| Metric | Value |
|--------|-------|
| Current readiness | Not ready (Meta: 30-35%) |
| What's needed | One additional model family (Llama-3.2-1B or OLMo-1B), two additional domains (math, general from VersaPRM), resolve upper-bound contradiction, n≥5 seeds for high-div medical, Git Re-Basin alignment check, systematic divergence sweep (≥4 levels) |
| Timeline | 2-3 weeks experiments + 1 week writing (~2-3 GPU-days) |
| Probability after strengthening | **40-50%** |

**Rationale:** ICLR has a strong loss-landscape community (Frankle, Entezari, Ainsworth, Garipov) and is more receptive to careful empirical work than NeurIPS. A sympathetic AC assignment is plausible. The medical instability finding would resonate with the mode-connectivity audience.

### Tier 2: TMLR (Rolling Submission) — RELIABLE FALLBACK

| Metric | Value |
|--------|-------|
| Current readiness | 65-75% as-is |
| Minimal additions for ~85% | Second model family, two more domains, resolve upper-bound contradiction, fix barrier formula bug, Git Re-Basin on one pair |
| Timeline | 1-2 weeks (~0.5 GPU-days for minimal; ~1.5 GPU-days for safe) |
| Probability | **85%** after minimal additions |

**Rationale:** TMLR explicitly values methodological rigor and careful measurement over conceptual novelty — precisely the paper's strengths. Rolling submission means no deadline pressure. TMLR reviewers would appreciate the noise-floor calibration, within-domain baselines, and data transparency.

### Venues to Avoid

| Venue | Reason |
|-------|--------|
| ACL/EMNLP/NAACL | Zero NLP contribution. NLP reviewers will ask "what's the NLP contribution?" Estimated probability: 10-15%. |
| NeurIPS workshops only | Undersells the contribution. The medical instability finding deserves archival publication. |

---

## Scorecard Summary

| Dimension | LMC Theorist | Avg Panel |
|-----------|:---:|:---:|
| Novelty | 5 | 5.0 |
| Technical Depth | 4 | 4.0 |
| Experimental Rigor | 6 | 6.0 |
| Clarity | 7 | 7.0 |
| Importance | 5 | 5.0 |
| **Overall** | **5.4** | **4.8** |

| Sub-Dimension | Score |
|---------------|:---:|
| Statistical Rigor | 4 |
| Data Integrity | 7 |
| Measurement Validity | 6 |
| Reproducibility | 6 |
| Baseline Coverage | 5 |
| Domain Framing Validity | 4 |
| NLP Relevance | 3 |
| Asymmetry Novelty | 5 |
| Interpretation Soundness | 6 |
| Practical Utility | 3 |
| Actionability | 2 |
| Literature Connection | 5 |

---

## References

1. Entezari, R., et al. (ICLR 2022). "The Role of Permutation Invariance in Linear Mode Connectivity." arXiv:2204.10882.
2. Ainsworth, S., et al. (ICLR 2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries." arXiv:2209.04836.
3. VersaPRM dataset. Process reward model training data with domain-labeled reasoning steps. Used in AFP experiments, `/home/jiayu/AFP/data/versaprm/versa_prm.jsonl`.
4. Yadav, P., et al. (NeurIPS 2023). "TIES-Merging: Resolving Interference When Merging Models." arXiv:2306.01708.
5. Ilharco, G., et al. (NeurIPS 2023). "Editing Models with Task Arithmetic." arXiv:2212.04089.
6. Wortsman, M., et al. (ICML 2022). "Model soups: averaging weights of multiple fine-tuned models." arXiv:2203.05482.
7. Frankle, J., et al. (ICML 2020). "Linear Mode Connectivity and the Lottery Ticket Hypothesis." arXiv:1912.05671.
8. AFP Engineering Manual. `/home/jiayu/AFP/docs/internal/ENGINEERING.md`. 22 recorded bugs from project history.
