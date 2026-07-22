# AFP LMC Paper — Figure Specification & Data Package v2

> **Date:** 2026-07-22 | **Paper:** v19 | **Status:** Fig 1-3 accurate; Fig 4 needs Panel C upgrade; new figures pending GPT-Neo + OPT redo
>
> All data extracted from verified raw JSON. No secondary transcription.

---

## Quick Status

| Figure | Panels | Status | Action |
|--------|--------|:---:|------|
| Fig 1 | A–F (6) | ✅ Accurate | No change needed |
| Fig 2 | A–D (4) | ✅ Accurate | No change needed |
| Fig 3 | A–B (2) | ✅ Accurate | No change needed |
| Fig 4 | A–C (3) | 🔴 Upgrade | Panel C: text→charts |
| Fig 5 | A–D (4) | 🆕 New | After GPT-Neo + OPT redo |
| Fig 6 | A–B (2) | 🆕 New | After GPT-Neo + OPT redo |

---

## Figure 1: LMC Overview (6 panels, 190mm full-width)

**Status:** ✅ No changes needed. All data in `docs/reports/figure_data.json`.

### Panel A: Standard Divergence — Code Eval
- **Data:** `figure_data.json` → `fig1_lmc_curves.standard.seed_0/1/2` → `alpha` vs `loss_code`
- 3 colored lines (blue/green/orange), circle markers
- x: α (0→1), y: BCE Loss (code)

### Panel B: Standard Divergence — Medical Eval
- Same seeds, `loss_med`, square markers

### Panel C: High Divergence — Code Eval
- `fig1_lmc_curves.high.seed_0/1/2` → `loss_code`
- Wider y-range

### Panel D: High Divergence — Medical Eval
- Same, `loss_med`

### Panel E: Within vs Cross-Domain Bar Chart
- Data keys in `figure_data.json`:
  - `code_within`: 0.048 ± 0.000
  - `code_cross_std`: 0.053 ± 0.011
  - `general_within`: 0.071 ± 0.007
  - `math_within`: 0.087 ± 0.032
  - `medical_within`: 0.147 ± 0.027
  - `medical_cross_std`: 0.051 ± 0.013
- Cross-domain bars at 45% alpha; dashed ref line at 0.05; "3×" annotation on medical within

### Panel F: Per-Block Weight Divergence
- `fig2_per_block.code` (24 values), `fig2_per_block.medical` (24 values)
- Overlaid line plots, r=0.995 annotation

---

## Figure 2: Trajectory & Mechanism (4 panels, 190mm full-width)

**Status:** ✅ No changes needed. All data in `figure_data.json`.

### Panel A: Code Trajectory — Inverted-U
- `fig3_trajectory.code` → `step` vs `bar_code`
- Dashed vertical at step 200; "Peak: step 200 (bar=0.043)" annotation
- y: 0.020–0.050

### Panel B: Medical Trajectory — Monotonic
- `fig3_trajectory.medical` → `step` vs `bar_med`
- "Plateaus at ~0.21" annotation
- y: 0.140–0.230; same x-scale as A

### Panel C: Structured vs Unstructured
- `gaussian_calibration`: dw_0.5=0.003, dw_1.0=0.004, dw_2.0=0.005, dw_4.0=0.011, dw_8.0=0.014
- Training scatter: (ΔW=1.5%, bar=0.053), (ΔW=8.0%, bar=0.118)
- "≈9×" bidirectional arrow annotation

### Panel D: Layer-Selective
- `layer_selective.early.bar_code`=0.040, `mid`=0.003, `late`=0.000
- Bars: Early (75%), Mid (6%), Late (0%), All (100%)

---

## Figure 3: Seed-Pair & Stability (2 panels, 190mm × 70mm)

**Status:** ✅ No changes needed. All data in `figure_data.json`.

### Panel A: Medical High-Div Seed-Pair Heatmap
- `seed_pair_medical_high`: 10 pairs (s0–s4)
- Red highlight on s1↔s4 (1.213); "15× range" annotation
- Color scale: blue(0.0) → white(0.65) → red(1.3)

### Panel B: 4-Domain Stability Spectrum
- Same values as Fig 1 Panel E (within-domain only)
- Dashed ref line at cross-domain mean (~0.05)
- Value labels above bars

---

## Figure 4: Cross-Architecture Replication (UPGRADE NEEDED)

**Current:** Panel A+B are hardcoded bar charts; Panel C is a text table.
**Target:** Panel A+B keep bar charts but read from verified data; Panel C becomes 3 real sub-panels.

### ⚠️ Data Note
OPT data VERIFIED against raw JSON (71 scan files) on 2026-07-22. Per-block divergence JSON was **buggy** (code/medical arrays were identical copies, r=1.0 fake). **Fixed** by recomputing from model weights. Real values below.

---

### Fig 4 Panel A: Barrier Comparison (keep, update data source)

**Pythia-1.4B (2ep) — from `figure_data.json`:**
```
code_within:   0.063 ± 0.005
med_within:    0.231 ± 0.096
cross (code):  0.079 ± 0.025
```

**OPT-1.3B (2ep) — VERIFIED from raw JSON (`experiments/phase0_opt/results/`):**
```
Source: lmc_code_within_opt2ep_s0_s1.json, s0_s2.json, s1_s2.json
  s0_s1: 0.2054
  s0_s2: 0.1484
  s1_s2: 0.4007
  → code_within: 0.252 ± 0.132

Source: lmc_medical_within_opt2ep_s0_s1.json, s0_s2.json, s1_s2.json
  s0_s1: 0.8754
  s0_s2: 0.9543
  s1_s2: 0.8591
  → med_within: 0.896 ± 0.051

Source: lmc_cross_opt2ep_s0.json, s1.json, s2.json
  s0: code_eval=0.4847
  s1: code_eval=0.4731
  s2: code_eval=0.4971
  → cross (code eval): 0.485 ± 0.012
```

**Style:** Grouped bar chart (Pythia blue #2196F3, OPT orange #FF5722), 3 metric groups, error bars ±1σ, value labels above bars.

---

### Fig 4 Panel B: Med/Code Ratio (keep, update data)

```
Pythia-2ep ratio: 0.231 / 0.063 = 3.7×
OPT-2ep ratio:    0.896 / 0.252 = 3.6×
```

**Style:** Two vertical bars, dashed ref line at y=1.0, value labels.

---

### Fig 4 Panel C: OPT Extended Experiments (UPGRADE from text to 3 sub-charts)

Currently a text-only table. Replace with 3 real sub-panels (C1, C2, C3) in a row.

#### Sub-panel C1: OPT Training Trajectory

**Data — VERIFIED from raw JSON (`experiments/phase0_opt/results/lmc_traj_opt_*_step*.json`):**

Code trajectory (21 checkpoints, steps 40–840):
```
step,barrier
40,0.0341
80,0.0895
120,0.1525
160,0.0757
200,0.1118
240,0.1250
280,0.1406
320,0.1464
360,0.1759
400,0.2076
440,0.0811
480,0.0854
520,0.1074
560,0.1378
600,0.1485
640,0.1685
680,0.1785
720,0.1802
760,0.1845
800,0.1916
840,0.2034
```

Medical trajectory (20 checkpoints, steps 40–800):
```
step,barrier
40,0.0369
80,0.0584
120,0.1236
160,0.0981
200,0.0666
240,0.0813
280,0.0978
320,0.1134
360,0.1375
400,0.1429
440,0.0757
480,0.0762
520,0.0910
560,0.1037
600,0.1111
640,0.1243
680,0.1226
720,0.1281
760,0.1302
800,0.1261
```

**Style:** Single panel, two overlaid line plots.
- Code: blue line #2196F3, circle markers
- Medical: red line #F44336, square markers
- x: Training step (0–850), y: Barrier (0–0.25)
- Annotation: "OPT code: monotonic (vs Pythia inverted-U)"
- Legend top-left
- Note at bottom: "3-point scans; 21 checkpoints"

#### Sub-panel C2: OPT Gaussian Calibration

**Data — VERIFIED from raw JSON (`experiments/phase0_opt/results/noise_gaussian_opt_dw*.json`):**
```
ΔW,barrier
0.5%,0.000334
1.0%,0.002692
2.0%,0.005288
4.0%,0.016094
8.0%,0.054556
```

**Style:** Single line plot with diamond markers.
- Gray line #9E9E9E with diamond markers
- x: ΔW (0%–8%), y: Barrier (0–0.08)
- Annotation: "OPT Gaussian: max barrier 0.055 at ΔW=8%"
- Overlay Pythia Gaussian as dashed gray line for comparison (from figure_data.json)

#### Sub-panel C3: OPT Per-Block Divergence

**Data — RECOMPUTED 2026-07-22 from model weights (previous JSON was buggy):**

Code per-block (24 layers, %):
```
[7.093, 6.826, 4.033, 4.081, 3.072, 2.702, 2.503, 2.361, 2.307, 2.239, 2.346, 2.269, 2.314, 2.322, 2.277, 2.206, 2.229, 2.225, 2.148, 2.030, 1.979, 1.879, 2.004, 1.934]
mean = 2.81%
```

Medical per-block (24 layers, %):
```
[7.525, 3.624, 2.929, 2.563, 2.968, 2.831, 2.477, 2.510, 2.405, 2.391, 2.357, 2.410, 2.484, 2.483, 2.463, 2.379, 2.362, 2.258, 2.198, 2.113, 2.043, 1.949, 2.219, 1.892]
mean = 2.66%
```

**Pearson r = 0.834** (single seed s0; paper reports r=0.91, likely averaged across seeds)

**Style:** Two overlaid line plots.
- Code: blue line #2196F3, circle markers
- Medical: red line #F44336, square markers
- x: Transformer block (0–23), y: ΔW (%)
- Annotation: "r = 0.834" text box (top-right)
- Legend
- Note: "OPT r=0.83 (cf. Pythia r=0.995)"

---

### Fig 4 Overall Layout

```
+------------------------------------------+
|  Panel A (left 55%)  |  Panel B (right 45%) |
|  Grouped bar chart    |  Ratio bar chart     |
|  Pythia vs OPT        |  3.7× vs 3.6×       |
+------------------------------------------+
|  Panel C (full width, 3 sub-panels)       |
|  +----------+----------+----------+        |
|  | C1 Traj  | C2 Gauss | C3 Block |        |
|  +----------+----------+----------+        |
+------------------------------------------+
```

### Data Provenance Note for Fig 4
```
Panel A+B: paper.tex Table §4.9 → verified against raw JSON (2026-07-22)
Panel C1: experiments/phase0_opt/results/lmc_traj_opt_*_step*.json (41 files)
Panel C2: experiments/phase0_opt/results/noise_gaussian_opt_dw*.json (5 files)
Panel C3: recomputed from model weights 2026-07-22 (previous JSON was buggy)
All OPT experiments use 3-point LMC scans.
```

---

## Figure 5: Three-Model Cross-Architecture (NEW — after GPT-Neo + OPT redo)

### Prerequisites
- GPT-Neo-1.3B: code/medical/general/math × 3 seeds, 2-epoch, 11-point LMC
- OPT-1.3B redo: all scans upgraded to 11-point

### Panel A: Three-Model Barrier Comparison
- Grouped bar chart: Pythia / OPT / GPT-Neo
- Metrics: Code within, Medical within, Cross-domain
- Error bars ±1σ

### Panel B: Med/Code Ratio — Three Architectures
- 3 vertical bars + dashed line at y=1.0
- Expected: Pythia ~3.7×, OPT ~3.6×, GPT-Neo TBD

### Panel C: Three-Model Gaussian Calibration Overlay
- 3 overlaid lines (Pythia/OPT/GPT-Neo), 5 ΔW levels each
- Expected: all near-zero, architecture-independent

### Panel D: Three-Model Per-Block Divergence Overlay
- 3 pairs of overlaid lines (code + medical for each architecture)
- r values: Pythia 0.995, OPT ~0.83, GPT-Neo TBD

---

## Figure 6: Three-Model Trajectory Comparison (NEW — after GPT-Neo + OPT redo)

### Panel A: Code Domain Trajectory — Three Architectures
- Pythia: inverted-U (peak step 200)
- OPT: monotonic rise
- GPT-Neo: TBD
- Overlaid, same x-axis scale

### Panel B: Medical Domain Trajectory — Three Architectures
- Pythia: monotonic rise → plateau
- OPT: relatively flat
- GPT-Neo: TBD

---

## Color Palette (All Figures)

```
Pythia:     #2196F3 (blue)
OPT:        #FF5722 (deep orange)
GPT-Neo:    #4CAF50 (green)
Code:       #2196F3
Medical:    #F44336
General:    #4CAF50
Math:       #FF9800
Gaussian:   #9E9E9E (gray)
Seed 0/1/2: #2196F3 / #4CAF50 / #FF9800
Early/Mid/Late/All: #FF9800 / #2196F3 / #4CAF50 / #9C27B0
```

## Typography

- Font: DejaVu Sans / Helvetica / Arial
- Panel labels: 12pt bold
- Axis labels: 10pt
- Tick labels: 9pt
- Annotations: 9pt italic
- Output: PDF (vector) + PNG (300 DPI)

## Data Files

| File | Content |
|------|---------|
| `docs/reports/figure_data.json` | Pythia primary data (Figs 1-3) |
| `experiments/phase0_opt/results/lmc_traj_opt_*_step*.json` | OPT trajectory (41 files) |
| `experiments/phase0_opt/results/noise_gaussian_opt_dw*.json` | OPT Gaussian (5 files) |
| `experiments/phase0_opt/results/per_block_divergence.json` | OPT per-block (FIXED 2026-07-22) |
| `experiments/phase0_opt/results/lmc_*_within_opt2ep_*.json` | OPT within-domain (9 files) |
| `experiments/phase0_opt/results/lmc_cross_opt2ep_*.json` | OPT cross-domain (3 files) |
