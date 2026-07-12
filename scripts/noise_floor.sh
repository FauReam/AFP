#!/bin/bash
set -euo pipefail
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results
TS=$(date +%Y%m%d_%H%M%S)

echo "=== Noise Floor Calibration ==="

# Test 1: pretrained vs pretrained (two copies — should be barrier≈0)
echo "--- Copy-identical (barrier≈0) ---"
rm -rf "$MODELS/_p1" "$MODELS/_p2" "$MODELS/code_e1" "$MODELS/medical_e1"
cp -r "$MODELS/code_lr1e-4_s0" "$MODELS/_p1"
cp "$MODELS/_p1/W_code_final.pt" "$MODELS/_p1/W_medical_final.pt" 2>/dev/null || true
cp "$MODELS/_p1/W_code_head_final.pt" "$MODELS/_p1/W_medical_head_final.pt" 2>/dev/null || true
cp -r "$MODELS/_p1" "$MODELS/_p2"
(cd "$MODELS" && ln -sfn _p1 code_e1 && ln -sfn _p2 medical_e1)
$VENV -u scripts/lmc_barrier_scan.py > /dev/null 2>&1 || true
cp "$RESULTS/lmc_barrier_c1m1.json" "$RESULTS/noise_identical.json"
$VENV -c "
import json; d=json.load(open('$RESULTS/noise_identical.json')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
print(f'  bar(identical)={cm-(c0+c1)/2:.6f}')" || true

# Test 2: pretrained vs random init Pythia (should be barrier>>0)
echo "--- Random-init (barrier upper bound) ---"
$VENV -c "
import torch, json, numpy as np
from transformers import AutoModel
from pathlib import Path

# Load pretrained model
pretrained = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
pretrained_sd = {k: v.detach().cpu() for k, v in pretrained.state_dict().items()}
del pretrained; torch.cuda.empty_cache()

# Create random init model (same architecture, random weights)
random_model = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
for p in random_model.parameters():
    if len(p.shape) >= 2:
        torch.nn.init.xavier_uniform_(p)
    else:
        torch.nn.init.uniform_(p, -0.1, 0.1)
random_sd = {k: v.detach().cpu() for k, v in random_model.state_dict().items()}
del random_model; torch.cuda.empty_cache()

# Save both to temp dirs
import os
os.makedirs('$MODELS/_p1', exist_ok=True)
os.makedirs('$MODELS/_p2', exist_ok=True)
torch.save(pretrained_sd, '$MODELS/_p1/W_code_final.pt')
torch.save(random_sd, '$MODELS/_p2/W_medical_final.pt')
print('Models saved for random-init test')
"
rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
(cd "$MODELS" && ln -sfn _p1 code_e1 && ln -sfn _p2 medical_e1)
$VENV -u scripts/lmc_barrier_scan.py > /dev/null 2>&1 || true
cp "$RESULTS/lmc_barrier_c1m1.json" "$RESULTS/noise_random.json"
$VENV -c "
import json; d=json.load(open('$RESULTS/noise_random.json')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
print(f'  bar(random)={cm-(c0+c1)/2:.4f}')" || true

rm -rf "$MODELS/_p1" "$MODELS/_p2" "$MODELS/code_e1" "$MODELS/medical_e1"
echo "=== DONE ==="
