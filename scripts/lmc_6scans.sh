#!/bin/bash
set -euo pipefail
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results; LOGDIR=experiments/phase0_training
TS=$(date +%Y%m%d_%H%M%S)
LRS="1e-4 5e-4"
SEEDS="0 1 2"

echo "=== 6 LMC Scans: lr=1e-4, 5e-4 × 3 seeds ==="
echo ""

for lr in $LRS; do
  for seed in $SEEDS; do
    CODE="$MODELS/code_lr${lr}_s${seed}"
    MED="$MODELS/medical_lr${lr}_s${seed}"
    OUT="$RESULTS/lmc_lr${lr}_s${seed}.json"

    echo "--- lr=$lr seed=$seed ---"
    rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
    # Use relative symlinks: code_e1 -> code_lr1e-4_s0 (not absolute path)
    (cd "$MODELS" && ln -sfn "code_lr${lr}_s${seed}" code_e1)
    (cd "$MODELS" && ln -sfn "medical_lr${lr}_s${seed}" medical_e1)

    $VENV -u scripts/lmc_barrier_scan.py >> "$LOGDIR/lmc_lr${lr}_s${seed}_${TS}.log" 2>&1

    cp "$RESULTS/lmc_barrier_c1m1.json" "$OUT"
    $VENV -c "
import json
d=json.load(open('$OUT')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']
print(f'  code: {c0:.4f}→{cm:.4f} bar={cm-c0:.4f}/{cm-(c0+c1)/2:.4f}  med: {m0:.4f}→{mm:.4f} bar={mm-m0:.4f}/{mm-(m0+m1)/2:.4f}')
" || true
    echo ""
  done
done
rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
echo "=== DONE ==="
