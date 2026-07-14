#!/usr/bin/env python3
"""Generate publication-quality figures for AFP LMC paper.
Output: PDF (vector) + PNG (300 DPI) for all three figures.
Usage: python scripts/generate_figures.py
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT / 'docs' / 'reports'
DATA_FILE = FIG_DIR / 'figure_data.json'

with open(DATA_FILE) as f:
    data = json.load(f)

# ============================================================
# Global style — NeurIPS/ICML publication ready
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.5,
    'errorbar.capsize': 2.5,
    'pdf.fonttype': 42,   # embed as TrueType (editable in Illustrator)
    'ps.fonttype': 42,
})

# Color palette from spec
C = {
    'code':      '#2196F3',
    'medical':   '#F44336',
    'general':   '#4CAF50',
    'math':      '#FF9800',
    'noise':     '#9E9E9E',
    'purple':    '#9C27B0',
    'seed0':     '#2196F3',
    'seed1':     '#4CAF50',
    'seed2':     '#FF9800',
    'early':     '#FF9800',
    'mid':       '#2196F3',
    'late':      '#4CAF50',
    'all':       '#9C27B0',
}

MARKERS = ['o', 's', 'D', '^', 'v']

def mm2inch(mm):
    return mm / 25.4

# ============================================================
# Figure 1 — 6 panels, full-width (190mm)
# ============================================================
def make_figure1():
    fig = plt.figure(figsize=(mm2inch(190), mm2inch(135)), layout='constrained')

    # --- Panel A: Standard Div — Code Eval ---
    ax_a = fig.add_subplot(2, 3, 1)
    for i, seed_key in enumerate(['seed_0', 'seed_1', 'seed_2']):
        pts = data['fig1_lmc_curves']['standard'][seed_key]
        alphas = [p['alpha'] for p in pts]
        losses = [p['loss_code'] for p in pts]
        ax_a.plot(alphas, losses, color=C[f'seed{i}'], marker=MARKERS[i],
                  markersize=3, linewidth=1.1, label=f'seed {i}')
    ax_a.set_xlabel(r'$\alpha$')
    ax_a.set_ylabel('BCE Loss (code)')
    ax_a.set_title('A  Standard — Code eval', fontweight='bold', fontsize=8, loc='left')
    ax_a.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
                handlelength=1.5, borderpad=0.3, labelspacing=0.2).get_frame().set_linewidth(0.5)

    # --- Panel B: Standard Div — Medical Eval ---
    ax_b = fig.add_subplot(2, 3, 2)
    for i, seed_key in enumerate(['seed_0', 'seed_1', 'seed_2']):
        pts = data['fig1_lmc_curves']['standard'][seed_key]
        alphas = [p['alpha'] for p in pts]
        losses = [p['loss_med'] for p in pts]
        ax_b.plot(alphas, losses, color=C[f'seed{i}'], marker=MARKERS[i],
                  markersize=3, linewidth=1.1, label=f'seed {i}')
    ax_b.set_xlabel(r'$\alpha$')
    ax_b.set_ylabel('BCE Loss (medical)')
    ax_b.set_title('B  Standard — Medical eval', fontweight='bold', fontsize=8, loc='left')
    ax_b.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
                handlelength=1.5, borderpad=0.3, labelspacing=0.2).get_frame().set_linewidth(0.5)

    # --- Panel C: High Div — Code Eval ---
    ax_c = fig.add_subplot(2, 3, 4)
    for i, seed_key in enumerate(['seed_0', 'seed_1', 'seed_2']):
        pts = data['fig1_lmc_curves']['high'][seed_key]
        alphas = [p['alpha'] for p in pts]
        losses = [p['loss_code'] for p in pts]
        ax_c.plot(alphas, losses, color=C[f'seed{i}'], marker=MARKERS[i],
                  markersize=3, linewidth=1.1, label=f'seed {i}')
    ax_c.set_xlabel(r'$\alpha$')
    ax_c.set_ylabel('BCE Loss (code)')
    ax_c.set_title('C  High divergence — Code eval', fontweight='bold', fontsize=8, loc='left')

    # --- Panel D: High Div — Medical Eval ---
    ax_d = fig.add_subplot(2, 3, 5)
    for i, seed_key in enumerate(['seed_0', 'seed_1', 'seed_2']):
        pts = data['fig1_lmc_curves']['high'][seed_key]
        alphas = [p['alpha'] for p in pts]
        losses = [p['loss_med'] for p in pts]
        ax_d.plot(alphas, losses, color=C[f'seed{i}'], marker=MARKERS[i],
                  markersize=3, linewidth=1.1, label=f'seed {i}')
    ax_d.set_xlabel(r'$\alpha$')
    ax_d.set_ylabel('BCE Loss (medical)')
    ax_d.set_title('D  High divergence — Medical eval', fontweight='bold', fontsize=8, loc='left')

    # --- Panel E: Within vs Cross-Domain Bar Chart ---
    ax_e = fig.add_subplot(2, 3, 3)
    wc = data['within_vs_cross']
    labels = ['Code\nwithin', 'Code\ncross', 'General\nwithin', 'Math\nwithin', 'Medical\nwithin', 'Medical\ncross']
    means = [wc['code_within']['mean'], wc['code_cross_std']['mean'],
             wc['general_within']['mean'], wc['math_within']['mean'],
             wc['medical_within']['mean'], wc['medical_cross_std']['mean']]
    stds  = [wc['code_within']['std'], wc['code_cross_std']['std'],
             wc['general_within']['std'], wc['math_within']['std'],
             wc['medical_within']['std'], wc['medical_cross_std']['std']]
    colors_e = [C['code'], C['code'], C['general'], C['math'], C['medical'], C['medical']]
    alphas_e = [0.75, 0.45, 0.75, 0.75, 0.75, 0.45]  # cross-domain bars use lighter alpha
    x = np.arange(len(labels))
    bars = ax_e.bar(x, means, yerr=stds, color=colors_e,
                    edgecolor='white', linewidth=0.3, width=0.6, capsize=2)
    for bar, alpha in zip(bars, alphas_e):
        bar.set_alpha(alpha)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(labels, fontsize=6.5)
    ax_e.set_ylabel('LMC Barrier')
    ax_e.set_title('E  Within vs. cross-domain barrier', fontweight='bold', fontsize=8, loc='left')
    ax_e.axhline(y=0.051, color='#888888', linestyle='--', linewidth=0.7, alpha=0.6)
    ax_e.text(5.3, 0.053, '~0.05', fontsize=6, color='#888888', va='bottom')
    # highlight medical within
    ax_e.annotate('3×', xy=(4, means[4]), xytext=(4.5, means[4] + 0.04),
                  fontsize=7, fontweight='bold', color=C['medical'],
                  arrowprops=dict(arrowstyle='->', color=C['medical'], lw=0.8),
                  ha='center')

    # --- Panel F: Per-Block Weight Divergence ---
    ax_f = fig.add_subplot(2, 3, 6)
    blocks = np.arange(24)
    ax_f.plot(blocks, data['fig2_per_block']['code'], color=C['code'],
              marker='o', markersize=2.5, linewidth=1.1, label='Code')
    ax_f.plot(blocks, data['fig2_per_block']['medical'], color=C['medical'],
              marker='s', markersize=2.5, linewidth=1.1, label='Medical')
    ax_f.set_xlabel('Transformer block')
    ax_f.set_ylabel(r'$\Delta W$ (%)')
    ax_f.set_title('F  Per-block weight divergence', fontweight='bold', fontsize=8, loc='left')
    ax_f.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
                handlelength=1.5, borderpad=0.3, labelspacing=0.2).get_frame().set_linewidth(0.5)
    # r = 0.995 annotation
    ax_f.text(0.95, 0.92, r'$r = 0.995$', transform=ax_f.transAxes,
              fontsize=7.5, fontweight='bold', ha='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.85, linewidth=0.5))

    fig.savefig(FIG_DIR / 'fig1_lmc_overview.pdf')
    fig.savefig(FIG_DIR / 'fig1_lmc_overview.png')
    plt.close(fig)
    print('Figure 1 saved: fig1_lmc_overview.{pdf,png}')

# ============================================================
# Figure 2 — 4 panels, half-width (90mm)
# ============================================================
def make_figure2():
    fig = plt.figure(figsize=(mm2inch(190), mm2inch(95)), layout='constrained')

    # --- Panel A: Code Trajectory ---
    ax_a = fig.add_subplot(2, 2, 1)
    traj_c = data['fig3_trajectory']['code']
    steps = [p['step'] for p in traj_c]
    bars  = [p['bar_code'] for p in traj_c]
    ax_a.plot(steps, bars, color=C['code'], marker='o', markersize=4, linewidth=1.3)
    ax_a.axvline(x=200, color='#888888', linestyle='--', linewidth=0.7)
    ax_a.annotate('Peak: step 200\n(bar = 0.043)', xy=(200, 0.043),
                  xytext=(270, 0.044), fontsize=7, color=C['code'],
                  arrowprops=dict(arrowstyle='->', color='#555555', lw=0.7))
    ax_a.set_xlabel('Training step')
    ax_a.set_ylabel('Barrier (code)')
    ax_a.set_title('A  Code trajectory — Inverted-U', fontweight='bold', fontsize=8, loc='left')
    ax_a.set_ylim(bottom=0.020, top=0.050)

    # --- Panel B: Medical Trajectory ---
    ax_b = fig.add_subplot(2, 2, 2)
    traj_m = data['fig3_trajectory']['medical']
    steps_m = [p['step'] for p in traj_m]
    bars_m  = [p['bar_med'] for p in traj_m]
    ax_b.plot(steps_m, bars_m, color=C['medical'], marker='s', markersize=4, linewidth=1.3)
    ax_b.annotate('Plateaus at ~0.21', xy=(360, 0.208),
                  xytext=(240, 0.195), fontsize=7, color=C['medical'],
                  arrowprops=dict(arrowstyle='->', color='#555555', lw=0.7))
    ax_b.set_xlabel('Training step')
    ax_b.set_ylabel('Barrier (medical)')
    ax_b.set_title('B  Medical trajectory — Monotonic growth', fontweight='bold', fontsize=8, loc='left')
    ax_b.set_ylim(bottom=0.140, top=0.230)

    # --- Panel C: Gaussian vs Training ---
    ax_c = fig.add_subplot(2, 2, 3)
    gc = data['gaussian_calibration']
    dws = [0.5, 1.0, 2.0, 4.0, 8.0]
    gbars = [gc[f'dw_{d}'] if f'dw_{d}' in gc else gc.get(f'dw_{d:.1f}', 0) for d in dws]
    # Fix key lookup
    gbars = [gc['dw_0.5'], gc['dw_1.0'], gc['dw_2.0'], gc['dw_4.0'], gc['dw_8.0']]
    ax_c.plot(dws, gbars, color=C['noise'], marker='D', markersize=4.5,
              linewidth=1.2, label='Gaussian noise')
    # Training points
    ax_c.scatter([1.5], [0.053], color=C['code'], s=55, zorder=5, marker='o', label='Training (code std)')
    ax_c.scatter([8.0], [0.118], color=C['code'], s=55, zorder=5, marker='s', label='Training (code high)')
    # 9x arrow
    ax_c.annotate('', xy=(8.0, 0.014), xytext=(8.0, 0.118),
                  arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0))
    ax_c.text(8.3, 0.06, r'$\mathbf{\approx 9\times}$', fontsize=8, fontweight='bold', va='center')
    ax_c.set_xlabel(r'$\Delta W$ (%)')
    ax_c.set_ylabel('LMC Barrier')
    ax_c.set_title('C  Structured vs. unstructured displacement', fontweight='bold', fontsize=8, loc='left')
    ax_c.legend(frameon=True, fancybox=False, edgecolor='#cccccc', fontsize=6.5,
                handlelength=1.5, borderpad=0.3, labelspacing=0.2).get_frame().set_linewidth(0.5)

    # --- Panel D: Layer-Selective ---
    ax_d = fig.add_subplot(2, 2, 4)
    ls = data['layer_selective']
    layers = ['Early\n(0–7)', 'Mid\n(8–15)', 'Late\n(16–23)', 'All\n(0–23)']
    bar_vals = [ls['early']['bar_code'], ls['mid']['bar_code'],
                ls['late']['bar_code'], 0.053]  # all = from paper
    lcolors = [C['early'], C['mid'], C['late'], C['all']]
    xd = np.arange(len(layers))
    bars = ax_d.bar(xd, bar_vals, color=lcolors, edgecolor='white', linewidth=0.3, width=0.55)
    # Annotations
    ax_d.text(0, bar_vals[0] + 0.002, '75%', fontsize=7, fontweight='bold',
              ha='center', color=C['early'])
    ax_d.text(2, bar_vals[2] + 0.002, '~0', fontsize=7, fontweight='bold',
              ha='center', color=C['late'])
    ax_d.text(3, bar_vals[3] + 0.002, '100%', fontsize=7, ha='center', color=C['all'])
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(layers, fontsize=7)
    ax_d.set_ylabel('LMC Barrier (code)')
    ax_d.set_title('D  Layer-selective barrier decomposition', fontweight='bold', fontsize=8, loc='left')
    # % annotations on bars
    for i, (xpos, val) in enumerate(zip(xd, bar_vals)):
        pct = [75, 6, 0, 100][i]
        ax_d.text(xpos, val / 2, f'{pct}%', ha='center', va='center',
                  fontsize=7, fontweight='bold', color='white' if pct > 10 else '#555555')

    fig.savefig(FIG_DIR / 'fig2_trajectory_analysis.pdf')
    fig.savefig(FIG_DIR / 'fig2_trajectory_analysis.png')
    plt.close(fig)
    print('Figure 2 saved: fig2_trajectory_analysis.{pdf,png}')

# ============================================================
# Figure 3 — Seed-pair matrix + Stability spectrum
# ============================================================
def make_figure3():
    fig = plt.figure(figsize=(mm2inch(190), mm2inch(70)), layout='constrained')

    # --- Panel A: Seed-Pair Heatmap (pure vector via Rectangle patches) ---
    ax_a = fig.add_subplot(1, 2, 1)
    sp = data['seed_pair_medical_high']
    seeds = ['s0', 's1', 's2', 's3', 's4']
    n = len(seeds)
    mat = np.zeros((n, n))
    for k, v in sp.items():
        a, b = k.split('_')
        i, j = int(a[1]), int(b[1])
        mat[i, j] = v
        mat[j, i] = v
    np.fill_diagonal(mat, np.nan)

    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    colors_list = ['#2196F3', '#B3E5FC', '#FFFFFF', '#FFCDD2', '#D32F2F']
    cmap = LinearSegmentedColormap.from_list('barrier_cmap', colors_list, N=256)
    norm = Normalize(vmin=0, vmax=1.3)

    # Draw each cell as a Rectangle (pure vector, no raster)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if np.isnan(val):
                color = '#f5f5f5'  # diagonal: light gray
            else:
                color = cmap(norm(val))
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color,
                                 edgecolor='white', linewidth=1.5)
            ax_a.add_patch(rect)

    ax_a.set_xlim(-0.5, n - 0.5)
    ax_a.set_ylim(n - 0.5, -0.5)
    ax_a.set_xticks(range(n)); ax_a.set_xticklabels(seeds, fontsize=8)
    ax_a.set_yticks(range(n)); ax_a.set_yticklabels(seeds, fontsize=8)
    ax_a.set_aspect('equal')
    ax_a.set_title('A  Medical high-div seed-pair barrier matrix', fontweight='bold', fontsize=8, loc='left')

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax_a.text(j, i, '—', ha='center', va='center', fontsize=8, color='#aaaaaa')
            else:
                val = mat[i, j]
                text_color = 'white' if val > 0.8 else ('#333333' if val < 0.3 else 'black')
                weight = 'bold' if val > 0.8 else 'normal'
                ax_a.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=6.5,
                          color=text_color, fontweight=weight)

    # Highlight catastrophic pair
    ax_a.add_patch(plt.Rectangle((3.5, 0.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, linestyle='-'))
    ax_a.add_patch(plt.Rectangle((0.5, 3.5), 1, 1, fill=False, edgecolor='red', linewidth=2.5, linestyle='-'))
    ax_a.annotate('Catastrophic\n(1.213)', xy=(4, 1), xytext=(5.3, 1),
                  fontsize=6.5, color='red', fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color='red', lw=1.0),
                  ha='left', va='center')

    # Colorbar — pure vector: render as stacked Rectangles
    cax = ax_a.inset_axes([1.08, 0.08, 0.04, 0.84])
    n_strips = 128
    for k in range(n_strips):
        frac = k / (n_strips - 1)
        val = norm.vmin + frac * (norm.vmax - norm.vmin)
        rgba = cmap(norm(val))
        cax.add_patch(plt.Rectangle((0, frac), 1, 1/(n_strips-1),
                      facecolor=rgba, edgecolor='none', linewidth=0))
    cax.set_xlim(0, 1); cax.set_ylim(0, 1)
    cax.set_xticks([])
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    cax.set_yticks([0, 0.5, 1.0])
    cax.set_yticklabels(['0.0', '0.65', '1.3'], fontsize=6)
    cax.set_ylabel('Barrier', fontsize=7)

    # --- Panel B: 4-Domain Stability Spectrum ---
    ax_b = fig.add_subplot(1, 2, 2)
    wc = data['within_vs_cross']
    domains = ['Code', 'General', 'Math', 'Medical']
    means_b = [wc['code_within']['mean'], wc['general_within']['mean'],
               wc['math_within']['mean'], wc['medical_within']['mean']]
    stds_b = [wc['code_within']['std'], wc['general_within']['std'],
              wc['math_within']['std'], wc['medical_within']['std']]
    colors_b = [C['code'], C['general'], C['math'], C['medical']]
    xb = np.arange(len(domains))
    ax_b.bar(xb, means_b, yerr=stds_b, color=colors_b, edgecolor='white',
             linewidth=0.3, width=0.55, capsize=3)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(domains, fontsize=8)
    ax_b.set_ylabel('Within-domain LMC Barrier')
    ax_b.set_title('B  Domain stability spectrum', fontweight='bold', fontsize=8, loc='left')
    # reference line
    ax_b.axhline(y=0.051, color='#888888', linestyle='--', linewidth=0.8)
    ax_b.text(3.3, 0.056, 'Cross-domain\nmean (~0.05)', fontsize=6.5, color='#888888', ha='left')

    # Value labels
    for xi, m, s in zip(xb, means_b, stds_b):
        ax_b.text(xi, m + s + 0.006, f'{m:.3f}', ha='center', fontsize=7, fontweight='bold')

    fig.savefig(FIG_DIR / 'fig3_seedpair_stability.pdf')
    fig.savefig(FIG_DIR / 'fig3_seedpair_stability.png')
    plt.close(fig)
    print('Figure 3 saved: fig3_seedpair_stability.{pdf,png}')

# ============================================================
if __name__ == '__main__':
    print(f'Data: {len(json.dumps(data)):,} bytes from {DATA_FILE.name}')
    make_figure1()
    make_figure2()
    make_figure3()
    print(f'Done → {FIG_DIR}/')
