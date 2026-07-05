---
name: academic-figure-generator
description: Generate publication-quality academic figures from experiment data. Use when the user says "画图", "figure", "plot", "生成图表", "论文图", or when writing paper results sections. Enforces NeurIPS/ICML figure standards.
allowed-tools:
  - Read
  - Write
  - Bash
  - Edit
---

# Academic Figure Generator

Generate publication-quality figures from experiment data. Enforce journal standards.

## Standards (NeurIPS/ICML)

- **DPI**: 150+ for raster, vector (PDF/SVG) preferred
- **Font**: sans-serif, 10-12pt labels, 13-14pt titles
- **Colors**: colorblind-friendly (avoid red-green alone). Use `#2196F3`, `#FF9800`, `#4CAF50`, `#9C27B0`, `#F44336`
- **Grid**: light alpha (0.2-0.3), behind data
- **Legend**: inside plot when possible, 8-10pt font
- **Axis**: labeled with units, no unnecessary spines
- **Size**: single column ~3.5in wide, full page ~7in wide
- **Export**: PNG for drafts, PDF for final submission

## Figure Types

### LMC Barrier / Interpolation Curves
- X-axis: interpolation coefficient α
- Y-axis: loss or accuracy
- Multiple curves: different training intensities
- Annotate optimal α and barrier values

### Bar Charts (Per-block divergence, comparisons)
- Paired bars for two models
- Sorted by magnitude
- Error bars when available (3+ seeds)

### Heatmaps (Per-block × Domain comparisons)
- Blocks on one axis, domains on other
- Color = divergence or importance

### Scatter (Divergence vs. Performance)
- Each point = one model checkpoint
- Regression line for trend

## Workflow

1. Read experiment data from `experiments/phase0_ivn/results/*.json`
2. Use matplotlib with `matplotlib.use('Agg')`
3. Generate figure, save to `docs/reports/`
4. Output the file path and a one-line caption for the paper

## Color Palette

```python
COLORS = {
    'code': '#2196F3',      # blue
    'medical': '#4CAF50',   # green
    'base': '#9E9E9E',      # gray
    'c1m1': '#2196F3',
    'c1m3': '#FF9800',      # orange
    'c1m5': '#4CAF50',
    'lr_low': '#9E9E9E',
    'lr_high': '#F44336',   # red
}
```
