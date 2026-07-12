#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results; LOGDIR=experiments/phase0_training
TS=$(date +%Y%m%d_%H%M%S)
LRS="1e-4 2e-4 3e-4 5e-4"
SEEDS="0 1 2"

echo "=== LMC Scans ==="
for seed in $SEEDS; do
  for lr in $LRS; do
    CODE_DIR="$MODELS/code_lr${lr}_s${seed}"
    MED_DIR="$MODELS/medical_lr${lr}_s${seed}"
    OUT="$RESULTS/lmc_lr${lr}_s${seed}.json"
    if [ -f "$OUT" ]; then echo "[skip] $OUT exists"; continue; fi
    if [ ! -f "$CODE_DIR/W_code_final.pt" ] || [ ! -f "$MED_DIR/W_medical_final.pt" ]; then echo "[skip] $lr s$seed missing"; continue; fi
    echo "[lmc] lr=$lr seed=$seed"
    rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
    ln -sfn "$(realpath "$CODE_DIR")" "$MODELS/code_e1"
    ln -sfn "$(realpath "$MED_DIR")" "$MODELS/medical_e1"
    $VENV -u scripts/lmc_barrier_scan.py >> "$LOGDIR/lmc_lr${lr}_s${seed}_${TS}.log" 2>&1 || true
    [ -f "$RESULTS/lmc_barrier_c1m1.json" ] && cp "$RESULTS/lmc_barrier_c1m1.json" "$OUT"
    $VENV -c "import json; d=json.load(open('$OUT')); r=d['results']; c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']; m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']; print(f'  bar_c={cm-c0:.3f}({cm-(c0+c1)/2:.3f}) bar_m={mm-m0:.3f}({mm-(m0+m1)/2:.3f})')" || true
  done
done
rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
echo "=== DONE ==="
