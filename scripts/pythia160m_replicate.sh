#!/bin/bash
set -euo pipefail
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results; LOGDIR=experiments/phase0_training
TS=$(date +%Y%m%d_%H%M%S)
MODEL="EleutherAI/pythia-160m"

echo "=== Pythia-160M Replication ==="

# Train code
echo "[train] code 160M"
$VENV -u scripts/train_agent.py --domain code --lr 1e-4 --model-id "$MODEL" --save-every-n-epochs 0 \
  >> "$LOGDIR/train_160m_code_$TS.log" 2>&1
mkdir -p "$MODELS/code_160m_s0"
cp "$MODELS/code/W_code_final.pt" "$MODELS/code_160m_s0/"
cp "$MODELS/code/W_code_head_final.pt" "$MODELS/code_160m_s0/" 2>/dev/null || true
echo "  code saved"

# Train medical
echo "[train] medical 160M"
$VENV -u scripts/train_agent.py --domain medical --lr 1e-4 --model-id "$MODEL" --save-every-n-epochs 0 \
  >> "$LOGDIR/train_160m_med_$TS.log" 2>&1
mkdir -p "$MODELS/medical_160m_s0"
cp "$MODELS/medical/W_medical_final.pt" "$MODELS/medical_160m_s0/"
cp "$MODELS/medical/W_medical_head_final.pt" "$MODELS/medical_160m_s0/" 2>/dev/null || true
echo "  medical saved"

# LMC scan
echo "[lmc] 160M code x medical"
rm -rf "$MODELS/_a" "$MODELS/_b"
mkdir -p "$MODELS/_a" "$MODELS/_b"
cp "$MODELS/code_160m_s0/W_code_final.pt" "$MODELS/_a/W_code_final.pt"
cp "$MODELS/code_160m_s0/W_code_head_final.pt" "$MODELS/_a/W_code_head_final.pt" 2>/dev/null || true
cp "$MODELS/medical_160m_s0/W_medical_final.pt" "$MODELS/_b/W_medical_final.pt"
cp "$MODELS/medical_160m_s0/W_medical_head_final.pt" "$MODELS/_b/W_medical_head_final.pt" 2>/dev/null || true
rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
(cd "$MODELS" && ln -sfn _a code_e1 && ln -sfn _b medical_e1)
$VENV -u scripts/lmc_barrier_scan.py --model-id "$MODEL" >> "$LOGDIR/lmc_160m_$TS.log" 2>&1
cp "$RESULTS/lmc_barrier_c1m1.json" "$RESULTS/lmc_160m.json"
$VENV -c "
import json; d=json.load(open('$RESULTS/lmc_160m.json')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']
print(f'  160M: code_bar={cm-(c0+c1)/2:.4f}  med_bar={mm-(m0+m1)/2:.4f}')" || true

rm -rf "$MODELS/_a" "$MODELS/_b" "$MODELS/code_e1" "$MODELS/medical_e1"
echo "=== DONE ==="
