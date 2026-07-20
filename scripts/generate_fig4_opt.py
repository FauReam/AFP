#!/usr/bin/env python3
"""Generate Fig 4: Cross-architecture replication on OPT-1.3B.
Panel A: Tri-model barrier comparison (Pythia-2ep vs OPT-2ep)
Panel B: Med/code barrier ratio consistency across architectures
Output: PDF vector + PNG for paper inclusion.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT / 'docs' / 'reports'

# ============================================================
# Global style — match existing figures
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 8,
    'axes.titlesize': 9.5,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.5,
    'errorbar.capsize': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C = {
    'pythia':   '#2196F3',  # blue
    'opt':      '#FF5722',  # deep orange
    'code':     '#2196F3',
    'medical':  '#F44336',
    'cross':    '#9C27B0',  # purple
    'grid':     '#E0E0E0',
    'ratio_line':'#757575',
}

# ============================================================
# DATA — from paper.tex Table in §4.9
# ============================================================
# Tri-model comparison (2-epoch, fair comparison)
models = ['Pythia-1.4B', 'OPT-1.3B']
metrics = ['Code within', 'Medical within', 'Cross-domain']

pythia_means  = [0.063, 0.231, 0.079]
pythia_stds   = [0.005, 0.096, 0.025]
opt_means     = [0.251, 0.896, 0.485]
opt_stds      = [0.108, 0.042, 0.010]

# Med/code ratio
pythia_ratio = 3.7  # 0.231/0.063
opt_ratio    = 3.6  # 0.896/0.251

# OPT qualitative findings (for annotation panel)
opt_findings = [
    ("Gaussian perturbation", "0.0003–0.055 (all ΔW)", "Replicates Pythia: unstructured\nnoise negligible"),
    ("Trajectory (21 ckpts)", "Code 0.14→0.20 monotonic\nMedical 0.09→0.14 flat", "Inverted-U Pythia-specific;\ntrajectory shape arch-dependent"),
    ("Per-block divergence", "r = 0.91 (Pythia: r = 0.995)", "GPT-2 architecture produces\ndifferent layer-wise profile"),
]

# ============================================================
# FIGURE
# ============================================================
fig = plt.figure(figsize=(7.2, 4.6))

# ---- Panel A: Barrier comparison (grouped bar) ----
ax_a = fig.add_axes([0.07, 0.38, 0.52, 0.58])

x = np.arange(len(metrics))
width = 0.35

bars_p = ax_a.bar(x - width/2, pythia_means, width,
                   yerr=pythia_stds, capsize=3,
                   color=C['pythia'], edgecolor='white', linewidth=0.5,
                   label='Pythia-1.4B (2ep)', zorder=3)
bars_o = ax_a.bar(x + width/2, opt_means, width,
                   yerr=opt_stds, capsize=3,
                   color=C['opt'], edgecolor='white', linewidth=0.5,
                   label='OPT-1.3B (2ep)', zorder=3)

# Value labels on bars
for bar, val in zip(bars_p, pythia_means):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pythia_stds) + 0.02,
              f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold', color=C['pythia'])
for bar, val in zip(bars_o, opt_means):
    offset = max(opt_stds) + 0.03
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
              f'{val:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold', color=C['opt'])

ax_a.set_xticks(x)
ax_a.set_xticklabels(metrics)
ax_a.set_ylabel('LMC Barrier (BCE loss)', fontweight='bold')
ax_a.set_title('A  Barrier comparison: Pythia vs OPT (2-epoch)', fontweight='bold', loc='left', fontsize=9)
ax_a.legend(frameon=True, fancybox=False, edgecolor='#CCCCCC', loc='upper left')
ax_a.set_ylim(0, 1.15)
ax_a.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_a.grid(axis='y', color=C['grid'], linewidth=0.4, zorder=0)
ax_a.set_axisbelow(True)

# ---- Panel B: Med/code ratio ----
ax_b = fig.add_axes([0.66, 0.38, 0.31, 0.58])

ratio_x = np.arange(2)
ratio_vals = [pythia_ratio, opt_ratio]
ratio_colors = [C['pythia'], C['opt']]
bars_r = ax_b.bar(ratio_x, ratio_vals, 0.45, color=ratio_colors,
                   edgecolor='white', linewidth=0.5, zorder=3)

for bar, val in zip(bars_r, ratio_vals):
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
              f'{val:.1f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax_b.axhline(y=1.0, color=C['ratio_line'], linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
ax_b.text(1.4, 1.03, 'ratio = 1.0 (no asymmetry)', fontsize=6, color=C['ratio_line'], ha='right')

ax_b.set_xticks(ratio_x)
ax_b.set_xticklabels(['Pythia-1.4B\n(2ep)', 'OPT-1.3B\n(2ep)'], fontsize=7.5)
ax_b.set_ylabel('Med/Code Within-Domain Barrier Ratio', fontweight='bold', fontsize=7.5)
ax_b.set_title('B  Med/code barrier ratio', fontweight='bold', loc='left', fontsize=9)
ax_b.set_ylim(0, 4.8)
ax_b.grid(axis='y', color=C['grid'], linewidth=0.4, zorder=0)
ax_b.set_axisbelow(True)

# ---- Panel C: Annotation — OPT extended experiments ----
ax_c = fig.add_axes([0.07, 0.04, 0.90, 0.28])
ax_c.axis('off')

# Title
ax_c.text(0.0, 0.95, 'C  OPT extended experiments — qualitative summary',
          fontweight='bold', fontsize=9, transform=ax_c.transAxes, va='top')

# Three-column layout
col_width = 0.32
col_positions = [0.0, 0.34, 0.67]
col_headers = ['Experiment', 'Key result', 'Interpretation']

# Header row
for i, (pos, header) in enumerate(zip(col_positions, col_headers)):
    ax_c.text(pos, 0.78, header, fontweight='bold', fontsize=7.5,
              transform=ax_c.transAxes, va='top', color='#424242')

# Data rows
for row_idx, (exp, result, interp) in enumerate(opt_findings):
    y = 0.60 - row_idx * 0.20
    texts = [exp, result, interp]
    for col_idx, (pos, txt) in enumerate(zip(col_positions, texts)):
        ax_c.text(pos, y, txt, fontsize=7, transform=ax_c.transAxes, va='top',
                  linespacing=1.3)

# Separator lines (use plot with transform_coords for data coords + Axes x-range)
for row_idx in range(2):
    y = 0.42 - row_idx * 0.20
    ax_c.plot([0.01, 0.99], [y, y], color='#E0E0E0', linewidth=0.4,
              transform=ax_c.transAxes, clip_on=False)

# Caution note
ax_c.text(0.5, -0.02,
          'Note: OPT experiments use 3-point LMC scans; 30+ scans vs 70+ for Pythia. Results are preliminary.',
          fontsize=6.5, fontstyle='italic', color='#757575',
          transform=ax_c.transAxes, ha='center', va='top')

# ============================================================
# Save
# ============================================================
fig.savefig(FIG_DIR / 'fig4_opt_replication.pdf', format='pdf')
fig.savefig(FIG_DIR / 'fig4_opt_replication.png', format='png')
print(f"Saved: {FIG_DIR / 'fig4_opt_replication.pdf'}")
print(f"Saved: {FIG_DIR / 'fig4_opt_replication.png'}")
plt.close()
