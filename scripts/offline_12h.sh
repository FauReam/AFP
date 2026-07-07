#!/bin/bash
# ===========================================================================
# 12-hour offline training queue — no symlinks, independent directories
#
# LR spectrum: 1e-4, 2e-4, 3e-4, 5e-4 × 3 seeds
# Then LMC scans on all valid pairs
#
# All models saved to: trained_models/{domain}_lr{lr}_s{seed}/
# ===========================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

VENV=venv/bin/python3
MODELS=experiments/trained_models
LOGDIR=experiments/phase0_training
RESULTS=experiments/phase0_ivn/results
mkdir -p "$MODELS" "$LOGDIR" "$RESULTS"

TS=$(date +%Y%m%d_%H%M%S)
echo "============================================"
echo " Offline 12h Queue — $TS"
echo "============================================"
echo ""

# ===========================================================================
# Phase 1: Train all models (~10h)
# ===========================================================================
echo "=== Phase 1: Training ==="

LRS="1e-4 2e-4 3e-4 5e-4"
SEEDS="0 1 2"
DOMAINS="code medical"

for seed in $SEEDS; do
  for lr in $LRS; do
    for domain in $DOMAINS; do
      OUTDIR="$MODELS/${domain}_lr${lr}_s${seed}"
      if [ -f "$OUTDIR/W_${domain}_final.pt" ]; then
        echo "[skip] $OUTDIR exists"
        continue
      fi

      echo "[train] $domain lr=$lr seed=$seed -> $OUTDIR"
      mkdir -p "$OUTDIR"

      $VENV -u scripts/train_agent.py \
        --domain "$domain" \
        --lr "$lr" \
        --epochs 1 \
        --save-every-n-epochs 0 \
        >> "$LOGDIR/train_${domain}_lr${lr}_s${seed}_${TS}.log" 2>&1

      # Copy from default save location (train_agent saves to OUT_DIR/domain)
      if [ -f "$MODELS/$domain/W_${domain}_final.pt" ]; then
        cp "$MODELS/$domain/W_${domain}_final.pt" "$OUTDIR/"
        cp "$MODELS/$domain/W_${domain}_head_final.pt" "$OUTDIR/" 2>/dev/null || true
        echo "  [ok] saved $(du -h "$OUTDIR/W_${domain}_final.pt" | cut -f1)"
      else
        echo "  [FAIL] training did not produce output"
        exit 1
      fi
    done
  done
done

echo "=== Phase 1 done ==="
echo ""

# ===========================================================================
# Phase 2: Verify all models (~10min)
# ===========================================================================
echo "=== Phase 2: Verify weight divergence ==="

for seed in $SEEDS; do
  for lr in $LRS; do
    echo -n "  code_lr${lr}_s${seed} vs med_lr${lr}_s${seed}: "
    $VENV -c "
import torch
c = torch.load('$MODELS/code_lr${lr}_s${seed}/W_code_final.pt', map_location='cpu', weights_only=True)
m = torch.load('$MODELS/medical_lr${lr}_s${seed}/W_medical_final.pt', map_location='cpu', weights_only=True)
d = sum((c[k]-m[k]).float().norm().item()**2 for k in c if k in m)**0.5
n = sum(m[k].float().norm().item()**2 for k in m if k in c)**0.5
pct = d/(n+1e-8)*100
print(f'{pct:.1f}%', end='')
if pct < 0.1: print(' BUG - identical to reference!')
else: print(' ok')
"
  done
done
echo ""

# ===========================================================================
# Phase 3: LMC scans on all matching LR pairs (~2h for 12 scans)
# ===========================================================================
echo "=== Phase 3: LMC scans ==="

for seed in $SEEDS; do
  for lr in $LRS; do
    CODE_DIR="$MODELS/code_lr${lr}_s${seed}"
    MED_DIR="$MODELS/medical_lr${lr}_s${seed}"
    OUT="$RESULTS/lmc_lr${lr}_s${seed}.json"

    if [ -f "$OUT" ]; then
      echo "[skip] $OUT exists"
      continue
    fi

    echo "[lmc] lr=$lr seed=$seed"
    # Symlinks for the scan script (safe: created fresh each time)
    ln -sfn "$CODE_DIR" "$MODELS/code_e1"
    ln -sfn "$MED_DIR" "$MODELS/medical_e1"

    $VENV -u scripts/lmc_barrier_scan.py \
      >> "$LOGDIR/lmc_lr${lr}_s${seed}_${TS}.log" 2>&1 || true

    # The scan saves to lmc_barrier_c1m1 — copy immediately
    if [ -f "$RESULTS/lmc_barrier_c1m1.json" ]; then
      cp "$RESULTS/lmc_barrier_c1m1.json" "$OUT"
      # Quick summary
      $VENV -c "
import json
d=json.load(open('$OUT'))
r=d['results']; c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']
print(f'  code_bar={cm-c0:.4f} (frankle={cm-(c0+c1)/2:.4f})  med_bar={mm-m0:.4f} (frankle={mm-(m0+m1)/2:.4f})')
" || echo "  [warn] no result"
    fi
  done
done

# Cleanup
rm -f "$MODELS/code_e1" "$MODELS/medical_e1"

# ===========================================================================
# Phase 4: Summary table
# ===========================================================================
echo ""
echo "============================================"
echo " FINAL SUMMARY"
echo "============================================"
printf "%-15s %8s %10s %10s %10s %10s\n" "Model" "ΔW%" "c_bar_abs" "c_bar_f" "m_bar_abs" "m_bar_f"
echo "-----------------------------------------------------------------"

for seed in $SEEDS; do
  for lr in $LRS; do
    OUT="$RESULTS/lmc_lr${lr}_s${seed}.json"
    if [ -f "$OUT" ]; then
      $VENV -c "
import json
d=json.load(open('$OUT'))
r=d['results']; c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']
print(f'lr={lr} s={seed:<7} {cm-c0:10.4f} {cm-(c0+c1)/2:10.4f} {mm-m0:10.4f} {mm-(m0+m1)/2:10.4f}')
"
    else
      echo "lr=$lr s=$seed  [missing]"
    fi
  done
done

echo ""
echo "Queue complete — $(date)"
