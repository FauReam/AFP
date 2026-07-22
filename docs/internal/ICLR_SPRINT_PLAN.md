# ICLR 2028 Sprint Plan — AFP Phase 0 LMC Paper

> **Target:** ICLR 2028 (estimated deadline: Oct 2027)
> **Fallback:** TMLR (rolling submission)
> **Hardware:** NVIDIA DGX Spark GB10, 121GB, ARM64 CUDA 13.0
> **Current date:** 2026-07-22 | **Paper:** v19 (20 pages, tri-model + theory verification done)
> **Based on:** 6-expert adversarial panel findings (see `EXPERT_PANEL_FINDINGS.md`)

---

## TL;DR

**2-3 GPU-days of experiments + 1 week of writing → ICLR 2028 at 40-50% acceptance probability.**

The paper has a genuinely novel finding (medical training instability drives larger LMC barriers than domain difference) buried inside a measurement study that overclaims. The fix: (1) run the experiments the expert panel flagged, (2) restructure the narrative around the medical stability finding, (3) downgrade/remove unsupported claims.

---

## Phase 0: Immediate Fixes (0 GPU-hours, Today)

These are code/text fixes that must be done before any new experiments.

| # | Task | Time |
|---|------|------|
| P0.1 | Fix barrier formula in `lmc_barrier_scan.py:101-102`: `max(L) - L[0]` → `max(L) - (L[0]+L[-1])/2` | 10 min |
| P0.2 | Fix barrier formula in `lmc_barrier.py:159-160` (same bug) | 5 min |
| P0.3 | Fix hardcoded metadata string `"code_e1 + medical_e1"` to use actual model paths | 10 min |
| P0.4 | Delete 6 fake result files: `lmc_lr2e-4_s*.json`, `lmc_lr3e-4_s*.json` | 1 min |
| P0.5 | Add assertion in scan script: printed barrier == JSON-saved barrier | 5 min |
| P0.6 | Remove "sublinear" language from paper, soften to "grows less than proportionally" | 15 min |
| P0.7 | Restructure central claim: elevate medical stability asymmetry, downgrade "ΔW drives barrier" | 30 min |
| P0.8 | Resolve upper-bound contradiction: reframe 0.222 as "reference calibration" not "upper bound" | 15 min |

---

## Phase 1: MUST Add (Any Venue, ~9 GPU-hours)

These address the four CRITICAL weaknesses. All models already trained — this is measurement-only work.

### E1: Run lr=2e-4 and lr=3e-4 LMC Scans (~1.5 GPU-h)

**Status:** Models trained ✅. LMC results are fake (byte-identical copies). Just need to run scans.

**Command:**
```bash
# For each lr in 2e-4, 3e-4; s in 0, 1, 2
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /home/jiayu/AFP
VENV=venv/bin/python3
MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results

lr=2e-4 seed=0
rm -f $MODELS/code_e1 $MODELS/medical_e1
(cd $MODELS && ln -sfn code_lr${lr}_s${seed} code_e1 && ln -sfn medical_lr${lr}_s${seed} medical_e1)
$VENV -u scripts/lmc_barrier_scan.py >> experiments/phase0_training/lmc_lr${lr}_s${seed}.log 2>&1
cp $RESULTS/lmc_barrier_c1m1.json $RESULTS/lmc_lr${lr}_s${seed}.json
# ... repeat for all 6 combinations
```

**Result:** 4 divergence levels (1.4%, 3.2%, 4.7%, 8.0%) → can test for monotonicity and fit functional form.

### E2: Resolve Upper-Bound Contradiction (~0.5 GPU-h)

**Option A (preferred):** Remeasure pretrained-to-random barrier with 3 seeds.
**Option B:** Text-only reframe as "reference calibration."

Run `noise_floor.sh` Test 2 with multiple random seeds:
```bash
# Modify noise_floor.sh to create 3 different random seeds
for s in 0 1 2; do
    # Create random model with seed s, run LMC scan
    # Save to noise_random_s${s}.json
done
# Report mean ± CI
```

**Result:** Either the CI contains 0.228 (no contradiction), or it doesn't (document the bound violation honestly).

### E3: Additional Seeds for High-Div Medical (~7 GPU-h)

Train 2 more medical models at high divergence:
```bash
$VENV -u scripts/train_agent.py --domain medical --lr 5e-4 --max-samples 0
# Seed 3 and seed 4 (increment from existing s0, s1, s2)
```

Then run LMC scans for the new seeds against their code counterparts.

**Result:** n=5 for high-div medical. Bootstrap CI becomes meaningful. If the large variance is real (domain property), report it transparently. If seed s0 was an outlier, the point estimate tightens.

### E4: Convergence Confound — Text Revision (0 GPU-h)

Restructure Section 4.2 to split findings:
- **Standard divergence:** primary, clean, convergence-matched
- **High divergence:** exploratory, with convergence confound prominently noted

Add a paragraph: "We note that the high-divergence models have higher self-domain loss than standard-divergence models (ΔL ≈ 0.02-0.03). Consequently, the higher barriers at high divergence may reflect a combination of genuine LMC degradation and endpoint quality degradation. We treat standard-divergence results as our primary findings and high-divergence results as exploratory upper-bound estimates."

---

## Phase 2: SHOULD Add (ICLR Target, ~37 GPU-hours)

These address the MAJOR weaknesses and elevate the paper to ICLR competitiveness.

### E5: Second Model Family (~8 GPU-h)

Train Llama-3.2-1B (or OLMo-1B) on code+medical at standard divergence:
```bash
# Train: 2 domains × 1 div level × 3 seeds = 6 models
$VENV -u scripts/train_agent.py --domain code --lr 1e-4 --model-id meta-llama/Llama-3.2-1B
$VENV -u scripts/train_agent.py --domain medical --lr 1e-4 --model-id meta-llama/Llama-3.2-1B
# × 3 seeds each
```
Then LMC scans + within-domain baselines.

**Result:** "These findings generalize beyond Pythia" — directly addresses W6.

### E6: Two Additional Domains (~8 GPU-h)

Use math and general domains from VersaPRM (data already tokenized and cached):
```bash
$VENV -u scripts/train_agent.py --domain math --lr 1e-4
$VENV -u scripts/train_agent.py --domain general --lr 1e-4
# × 3 seeds each
```
Then LMC scans.

**Result:** Test whether the medical instability finding is domain-specific. If math or general also shows high within-domain barriers, it's a broader phenomenon. If not, medical is a special case — also publishable.

### E7: Git Re-Basin Permutation Alignment (~2 GPU-h)

Run on the highest-barrier pair (high-div medical, worst seed pair):
```bash
# Implement or adapt git-re-basin weight matching
# Report barrier with and without permutation alignment
```

**Result:** Either permutation is irrelevant (strengthens paper — barrier is real) or reveals a correction (interesting finding — barrier partially explained by permutation drift). Either outcome publishable.

### E8: Gaussian Perturbation Baseline (~1 GPU-h)

Add structured vs. unstructured comparison:
```python
# Add Gaussian noise to base Pythia-1.4B to match ΔW ≈ 1.5% and ≈ 8%
# Run LMC scan between base and perturbed model
# Compare barrier to training-induced barrier at same ΔW
```

**Result:** If noise at ΔW=8% produces barrier 0.01 while training produces 0.12 → strong evidence that barrier reflects structured, domain-specific changes, not just weight magnitude.

### E9: Downstream Merge Quality Evaluation (~12 GPU-h)

The Merging Practitioner's core request:
```bash
# Merge standard-divergence code+medical using:
# (a) Pure weight averaging: W_merged = (W_code + W_medical) / 2
# (b) TIES-Merging
# Evaluate on downstream benchmarks (HumanEval for code, MedQA for medical)
# Plot: merge quality vs. barrier height
```

**Result:** Converts barrier from geometric curiosity to decision tool. Correlation between barrier and merge quality would be the paper's strongest practical contribution.

### E10: High-Div Within-Domain Baselines (~6 GPU-h)

Train additional high-div seeds and run within-domain scans:
```bash
# For code: 3 high-div seeds → LMC within-domain (3 pairs)
# For medical: need at least 5 seeds total → 3 pairs
```

**Result:** Shows whether medical instability persists or amplifies at larger ΔW. Directly tests whether instability is a domain property or a low-ΔW artifact.

---

## Phase 3: COULD Add (NeurIPS Target, ~56 GPU-hours)

These are optional enhancements for completeness. Only pursue if targeting NeurIPS.

| # | Experiment | GPU-hours | Why |
|---|-----------|:---:|------|
| E11 | Training-duration sweep (epochs instead of LR) | 18 | Disentangles ΔW from convergence quality |
| E12 | Layer-selective interpolation | 2 | Tests whether merging only late layers reduces barrier |
| E13 | Systematic 6-level divergence sweep | 28 | Fit `barrier = c × ΔW^k`, report k with CI |
| E14 | Third model family (Qwen-2.5-1.5B) | 8 | Three-family generalizability comparison |

---

## Narrative Restructuring Plan (0 GPU-hours, Critical Path)

The paper's biggest weakness is narrative, not data. The six experts converged on this:

### Current Narrative (v4) — WRONG
> "Weight displacement drives barrier height. The relationship is sublinear. Domain difference is negligible."

**Problems:** "Sublinear" unsupported by 2 points. "Domain difference negligible" contradicted by 50% of evidence.

### Target Narrative (v5) — CORRECT
> "Training stability, not domain difference, is the dominant source of LMC barrier height in fine-tuned models. This stability varies dramatically across domains: code fine-tuning is remarkably stable (within-domain ≈ cross-domain barriers), while medical fine-tuning produces 3× larger within-domain than cross-domain barriers. The degree of weight displacement modulates but does not determine barrier height — two models can differ by 1.5% in weight space yet exhibit barriers from 0.05 to 0.15 depending on the domain's training stability."

### Abstract Restructuring

**Drop:** "sublinear relationship," "barrier is driven by weight displacement," "upper bound"

**Keep:** Within-domain vs. cross-domain comparison, noise-floor calibration, per-block pattern

**Elevate:** Medical instability asymmetry as the central finding

### Section Restructuring

| Section | v4 | v5 Change |
|---------|-----|-----------|
| Abstract | Sublinearity + ΔW → barrier | Medical stability asymmetry |
| 4.2 LMC Barriers | Primary: cross-domain barriers | Split: standard (primary) / high (exploratory) |
| 4.3 Within-Domain | Calibration for cross-domain | **Central finding** — elevated to primary |
| 4.4 Asymmetry | Cross-domain loss externality | Move to appendix or fold into 4.3 |
| 5. Discussion | Merging implications first | Medical instability first, merging second |
| Title | "...in Domain-Specialized Fine-Tuning" | "...Training Stability vs. Domain Difference in Fine-Tuned LMs" (suggested) |

---

## Timeline

```
Week 1 (Jul 13-20):  Phase 0 (code fixes) + Phase 1 MUST (E1-E4)
                      └─ ~9 GPU-hours on DGX Spark

Week 2 (Jul 20-27):  Phase 2 SHOULD (E5-E8)
                      └─ ~19 GPU-hours

Week 3 (Jul 27-Aug 3): Phase 2 SHOULD continued (E9-E10)
                        └─ ~18 GPU-hours

Week 4 (Aug 3-10):   Writing sprint
                      └─ Narrative restructuring
                      └─ Revised abstract
                      └─ Updated figures
                      └─ Internal review pass

Week 5+ (buffer):     Polish, internal rounds, optional Phase 3
                      └─ Submit to TMLR as safety net (rolling)
                      └─ Prepare for ICLR 2028
```

### Fallback Trigger

If at the end of Week 3:
- E5 (second model family) fails to replicate the medical instability finding
- OR E9 (merge quality) shows zero correlation with barrier

→ **Pivot to TMLR immediately.** The paper still has sufficient contribution for TMLR: careful measurement + noise-floor calibration + within-domain baselines on one model family. Submit within 1 week.

---

## Experiment Pipeline — Execution Rules

Per `ENGINEERING.md`:

1. **All training must use `nohup ... &`** — zero stdin reads
2. **Verify ΔW > 0.1% after every training** — Bug 21/22 prevention
3. **`HF_DATASETS_OFFLINE=1`** — no network during training/eval
4. **No symlink management for real models** — use independent directories `{domain}_lr{lr}_s{seed}/`
5. **Clean temp dirs between LMC scans** — prevent stale symlink contamination
6. **Log to `experiments/phase0_training/`** with timestamps
7. **Verify barrier values match between script output and JSON** (after P0.5 fix)

### Monitoring

```bash
bash scripts/monitor.sh                    # one-off status
watch -n 60 bash scripts/monitor.sh        # poll every 60s
```

---

## Success Criteria

| Criterion | Threshold | How to Check |
|-----------|-----------|-------------|
| Barrier formula fix | Script output = Frankle definition | Run scan, verify `barrier = max(L) - (L[0]+L[-1])/2` |
| lr=2e-4/3e-4 re-scanned | 6 unique .json files with distinct md5 | `md5sum lmc_lr2e-4_s*.json` |
| Upper bound resolved | CI for pretrained→random contains or clearly bounds 0.228 | Report mean ± CI |
| High-div medical n≥5 | CV drops below 30% | Compute CV across seeds |
| Second model family | Llama barrier results qualitatively consistent with Pythia | Compare within-vs-cross domain pattern |
| Merge quality correlated | Spearman ρ between barrier and merge Δ(accuracy) > 0.5 | Run TIES merge, evaluate |
| Narrative restructured | Medical instability is central claim in abstract | Read abstract cold — is it the first thing mentioned? |

---

## GPU Budget Summary

| Phase | Experiments | GPU-hours | Cumulative |
|-------|------------|:---:|:---:|
| P0 (today) | Code fixes | 0 | 0 |
| P1 (MUST) | E1-E4 | 9 | 9 |
| P2 (ICLR) | E5-E10 | 37 | 46 |
| P3 (NeurIPS) | E11-E14 | 56 | 102 (~4.3 GPU-days) |

**Target: P0 + P1 + P2 = ~46 GPU-hours ≈ 2 GPU-days on DGX Spark.**

---

## References

- Frankle, J., et al. (ICML 2020). "Linear Mode Connectivity and the Lottery Ticket Hypothesis."
- Entezari, R., et al. (ICLR 2022). "The Role of Permutation Invariance in Linear Mode Connectivity."
- Ainsworth, S., et al. (ICLR 2023). "Git Re-Basin: Merging Models modulo Permutation Symmetries."
- Yadav, P., et al. (NeurIPS 2023). "TIES-Merging: Resolving Interference When Merging Models."
- Wortsman, M., et al. (ICML 2022). "Model soups: averaging weights of multiple fine-tuned models."
- Ilharco, G., et al. (NeurIPS 2023). "Editing Models with Task Arithmetic."
- AFP Expert Panel Findings (2026-07-13). `docs/internal/EXPERT_PANEL_FINDINGS.md`
- AFP Engineering Manual. `docs/internal/ENGINEERING.md`
