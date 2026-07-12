#!/bin/bash
set -euo pipefail
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results; LOGDIR=experiments/phase0_training
TS=$(date +%Y%m%d_%H%M%S)

echo "=== Within-Domain LMC: 6 scans ==="

for domain in code medical; do
  for pair in "0 1" "0 2" "1 2"; do
    s1=$(echo $pair | cut -d' ' -f1); s2=$(echo $pair | cut -d' ' -f2)
    DIR1="${MODELS}/${domain}_lr1e-4_s${s1}"
    DIR2="${MODELS}/${domain}_lr1e-4_s${s2}"
    OUT="$RESULTS/lmc_${domain}_s${s1}_s${s2}.json"
    if [ -f "$OUT" ]; then echo "[skip] $OUT"; continue; fi
    echo "[lmc] $domain s$s1 x s$s2"

    # lmc_barrier_scan.py expects: code_e1/W_code_final.pt + medical_e1/W_medical_final.pt
    # For within-domain, both models are same domain so we copy+rename into tmp dirs
    rm -rf "$MODELS/_a" "$MODELS/_b"
    mkdir -p "$MODELS/_a" "$MODELS/_b"
    cp "$DIR1/W_${domain}_final.pt" "$MODELS/_a/W_code_final.pt"
    cp "$DIR1/W_${domain}_head_final.pt" "$MODELS/_a/W_code_head_final.pt" 2>/dev/null || true
    cp "$DIR2/W_${domain}_final.pt" "$MODELS/_b/W_medical_final.pt"
    cp "$DIR2/W_${domain}_head_final.pt" "$MODELS/_b/W_medical_head_final.pt" 2>/dev/null || true

    rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
    (cd "$MODELS" && ln -sfn _a code_e1 && ln -sfn _b medical_e1)
    $VENV -u scripts/lmc_barrier_scan.py >> "$LOGDIR/lmc_${domain}_s${s1}_s${s2}_$TS.log" 2>&1

    [ -f "$RESULTS/lmc_barrier_c1m1.json" ] && cp "$RESULTS/lmc_barrier_c1m1.json" "$OUT"
    $VENV -c "
import json; d=json.load(open('$OUT')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
print(f'  bar={cm-(c0+c1)/2:.4f}')" || true
  done
done
rm -rf "$MODELS/_a" "$MODELS/_b" "$MODELS/code_e1" "$MODELS/medical_e1"
echo "=== DONE ==="
