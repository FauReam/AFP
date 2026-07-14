# Figure Specification for AFP LMC Paper

> All data available in `docs/reports/figure_data.json`
> Target: NeurIPS/ICML publication quality, vector format (PDF/SVG), 300 DPI

---

## Figure 1: LMC Barrier Curves and Comparative Analysis (6 panels, full-width)

**Purpose:** Overview of all primary LMC measurements.

### Panel A: Standard Divergence — Code Evaluation
- **Data:** `fig1_lmc_curves.standard.seed_0/1/2` → `alpha` vs `loss_code`
- **Style:** 3 colored lines (blue/green/orange), circle markers, thin lines
- **Axis:** x=α (0 to 1), y=BCE Loss
- **Legend:** "seed 0", "seed 1", "seed 2"

### Panel B: Standard Divergence — Medical Evaluation
- **Data:** Same as A but `loss_med`
- **Style:** 3 colored lines, square markers

### Panel C: High Divergence — Code Evaluation
- **Data:** `fig1_lmc_curves.high.seed_0/1/2` → `alpha` vs `loss_code`
- **Style:** 3 lines (pink/purple/deep purple), circle markers
- **Note:** Wider y-range than A/B

### Panel D: High Divergence — Medical Evaluation
- **Data:** Same as C but `loss_med`
- **Style:** Square markers

### Panel E: Within vs Cross-Domain Barrier Comparison
- **Data:** `within_vs_cross`
- **Style:** Grouped bar chart with error bars
- **Bars:** Code Within (0.048), Code Cross (0.053), General Within (0.071), Math Within (0.087), Medical Within (0.147), Medical Cross (0.051)
- **Colors:** Blue shades for code/general, red shades for math/medical
- **Error bars:** ±1σ

### Panel F: Per-Block Weight Divergence
- **Data:** `fig2_per_block.code`, `fig2_per_block.medical`
- **Style:** Overlaid line plots, block 0-23 on x-axis, ΔW% on y-axis
- **Colors:** Blue for code, red for medical
- **Annotation:** "r = 0.995" text box

---

## Figure 2: Training Trajectory and Mechanistic Analysis (4 panels)

**Purpose:** The core mechanistic findings.

### Panel A: Code Trajectory — Inverted-U
- **Data:** `fig3_trajectory.code` → `step` vs `bar_code`
- **Style:** Solid blue line with circle markers, vertical dashed line at step 200
- **Annotation:** "Peak: step 200 (bar=0.043)" at peak
- **Note:** Show both endpoints: step 40 (0.029) → peak (0.043) → step 400 (0.033)

### Panel B: Medical Trajectory — Monotonic Growth
- **Data:** `fig3_trajectory.medical` → `step` vs `bar_med`
- **Style:** Solid red line with square markers
- **Annotation:** "Plateaus at ~0.21" near step 400
- **Note:** Same x-axis scale as Panel A for direct comparison

### Panel C: Structured vs Unstructured Weight Displacement
- **Data:** `gaussian_calibration` (Gaussian), plus training points
- **Style:** Gray diamond line for Gaussian noise (ΔW=0.5-8%), two large green dots for training
- **Data points:** Gaussian at [0.5, 1.0, 2.0, 4.0, 8.0] → [0.003, 0.004, 0.005, 0.011, 0.014]
- **Training points:** ΔW=1.5% at bar=0.053, ΔW=8.0% at bar=0.118
- **Annotation:** "9× difference" arrow

### Panel D: Layer-Selective Barrier Decomposition
- **Data:** `layer_selective`
- **Style:** Horizontal bar chart or grouped bars
- **Bars:** Early(0-7)=0.040, Mid(8-15)=0.003, Late(16-23)=0.000, All=0.053
- **Colors:** Orange for early, blue for mid, green for late, purple for all
- **Annotation:** "75% of barrier in first 8 layers", "Late layers: ~zero penalty"

---

## Figure 3: Seed-Pair Compatibility and Domain Stability (optional, can be supplement)

### Panel A: Medical High-Divergence Seed-Pair Matrix
- **Data:** `seed_pair_medical_high`
- **Style:** Heatmap or network graph showing 5 seeds, edge width = barrier magnitude
- **Highlight:** s1↔s4 (1.213) in red, s2↔s4 (0.080) in green
- **Annotation:** "Same seed s4: 15× range depending on partner"

### Panel B: 4-Domain Stability Spectrum
- **Data:** `within_vs_cross`
- **Style:** Ordered bar chart
- **Order:** Code (0.048) < General (0.071) < Math (0.087) < Medical (0.147)
- **Reference line:** Cross-domain average (0.05)
- **Annotation:** "Cross-domain barrier ≈ constant (~0.05) regardless of domain pair"

---

## Color Palette

```
Code domain:     #2196F3 (blue)
Medical domain:  #F44336 (red)
General domain:  #4CAF50 (green)
Math domain:     #FF9800 (orange)
Gaussian/noise:  #9E9E9E (gray)
High divergence: #9C27B0 (purple)
```

## Typography

- Font: Helvetica or similar sans-serif
- Panel labels: 12pt bold (A, B, C, D, E, F)
- Axis labels: 10pt
- Tick labels: 9pt
- Annotations: 9pt italic

## Output

- Format: PDF (vector) + PNG (300 DPI raster)
- Size: Full page width (190mm) for Figure 1, half page (90mm) for Figures 2-3
- All data available in `docs/reports/figure_data.json`
