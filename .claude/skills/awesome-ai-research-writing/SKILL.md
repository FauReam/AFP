---
name: awesome-ai-research-writing
description: Academic paper writing, editing, and review. Use when drafting papers, writing sections (abstract/intro/method/results/discussion), checking completeness, or reviewing for submission. Enforces standard CS paper structure.
allowed-tools:
  - Read
  - Write
  - Bash
  - Edit
---

# AI Research Writing

Academic paper writing, review, and completeness checking.

## Paper Structure (CS/ML Conference Standard)

```
1. Abstract        — 150-250 words: problem, method, key results, significance
2. Introduction    — Why this problem? What's known? What's our contribution?
3. Related Work    — What others did, what gaps remain
4. Method          — What we did, reproducible detail
5. Results         — What we found, figures and tables
6. Discussion      — Why it matters, limitations, implications
7. Conclusion      — One-paragraph summary + future work
References         — Alphabetical, complete
```

## Completeness Checklist

Run this before considering any paper draft "done":

### Data & Results
- [ ] Every number in the paper is verifiable from experiment logs
- [ ] All tables have units and captions
- [ ] All figures referenced in text, with captions explaining what to see
- [ ] Statistical significance addressed (or limitation explicitly stated)

### Claims
- [ ] Every claim in abstract backed by results section
- [ ] Every claim in introduction backed by results or references
- [ ] No unsupported "we believe" statements without evidence or citation
- [ ] Limitations explicitly acknowledged

### Method
- [ ] Reproducible: model, hyperparams, data, hardware all specified
- [ ] Baselines justified (why these comparisons?)
- [ ] Metrics defined

### Narrative
- [ ] One clear thesis statement (what's the ONE thing reader should remember?)
- [ ] Intro tells reader what question we answer
- [ ] Discussion answers "so what?"
- [ ] No orphan sections (every section referenced or necessary)

## Writing Rules

1. **Numbers always with context**: "barrier = 0.071" not just "barrier is small". Say "barrier = 0.071 (13.8% of baseline loss)".
2. **Active voice**: "We measure..." not "Measurements were taken..."
3. **One idea per paragraph**: First sentence = topic. Rest = evidence.
4. **No hedging without evidence**: "may suggest" → provide the evidence or cut it.
5. **Figures before text**: Generate figures first, then write text describing them.

## Workflow for This Project

1. Read `docs/internal/PAPER.md` — current draft
2. Read `docs/internal/EXPERIMENT_PLAN.md` — experiments and results
3. Read `experiments/phase0_ivn/results/*.json` — raw data
4. Cross-check: every number in paper → traceable to raw data
5. Flag: missing figures, unsupported claims, stale data
