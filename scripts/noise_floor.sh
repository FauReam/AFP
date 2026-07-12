#!/bin/bash
set -euo pipefail
cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
VENV=venv/bin/python3; MODELS=experiments/trained_models
RESULTS=experiments/phase0_ivn/results; TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=experiments/phase0_training

echo "=== Noise Floor Calibration ==="
echo "Started: $(date)"

# ---- Pre-clean: nuke all stale symlinks + temp dirs ----
rm -rf "$MODELS/_p1" "$MODELS/_p2" "$MODELS/_a" "$MODELS/_b"
rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
rm -f "$RESULTS/lmc_barrier_c1m1.json"

# ===================================================================
# Test 1: Identical copy — same model loaded twice → barrier ≈ 0
# ===================================================================
echo ""
echo "--- [1/2] Identical copy (barrier≈0) ---"

# Copy code_lr1e-4_s0 into _p1, then duplicate W_code → W_medical
cp -r "$MODELS/code_lr1e-4_s0" "$MODELS/_p1"
cp "$MODELS/_p1/W_code_final.pt"      "$MODELS/_p1/W_medical_final.pt"
cp "$MODELS/_p1/W_code_head_final.pt"  "$MODELS/_p1/W_medical_head_final.pt"

# _p2 is an exact copy of _p1
cp -r "$MODELS/_p1" "$MODELS/_p2"

# Symlinks: code_e1 → _p1, medical_e1 → _p2 (both contain identical weights)
(cd "$MODELS" && ln -sfn _p1 code_e1 && ln -sfn _p2 medical_e1)

# Run LMC scan
rm -f "$RESULTS/lmc_barrier_c1m1.json"
$VENV -u scripts/lmc_barrier_scan.py >> "$LOGDIR/noise_identical_$TS.log" 2>&1

if [ -f "$RESULTS/lmc_barrier_c1m1.json" ]; then
    cp "$RESULTS/lmc_barrier_c1m1.json" "$RESULTS/noise_identical.json"
    $VENV -c "
import json; d=json.load(open('$RESULTS/noise_identical.json')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']
bc = cm-(c0+c1)/2; bm = mm-(m0+m1)/2
print(f'  bar_code={bc:.6f}  bar_med={bm:.6f}')
# Verify: barrier should be basically 0
if abs(bc) > 0.001 or abs(bm) > 0.001:
    print(f'  WARNING: identical copy barrier unexpectedly > 0.001!')
" || echo "  ERROR parsing result"
else
    echo "  ERROR: lmc_barrier_c1m1.json not created — LMC scan failed"
fi

# ---- Clean up between tests ----
rm -rf "$MODELS/_p1" "$MODELS/_p2" "$MODELS/code_e1" "$MODELS/medical_e1"

# ===================================================================
# Test 2: Random init vs pretrained → barrier upper bound
# ===================================================================
echo ""
echo "--- [2/2] Random-init (barrier upper bound) ---"

$VENV -c "
import torch, os
from transformers import AutoModel

# Load pretrained Pythia-1.4B
pretrained = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
pretrained_sd = {k: v.detach().cpu() for k, v in pretrained.state_dict().items()}
del pretrained; torch.cuda.empty_cache()

# Create random init model (same architecture, fresh weights)
import torch.nn as nn
random_model = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
for p in random_model.parameters():
    if len(p.shape) >= 2:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p, -0.1, 0.1)
random_sd = {k: v.detach().cpu() for k, v in random_model.state_dict().items()}
del random_model; torch.cuda.empty_cache()

# Save to temp dirs
os.makedirs('$MODELS/_p1', exist_ok=True)
os.makedirs('$MODELS/_p2', exist_ok=True)
torch.save(pretrained_sd, '$MODELS/_p1/W_code_final.pt')
torch.save(random_sd,      '$MODELS/_p2/W_medical_final.pt')
print('Random-init models saved')
" 2>&1

# Symlinks
rm -f "$MODELS/code_e1" "$MODELS/medical_e1"
(cd "$MODELS" && ln -sfn _p1 code_e1 && ln -sfn _p2 medical_e1)

# Run LMC scan
rm -f "$RESULTS/lmc_barrier_c1m1.json"
$VENV -u scripts/lmc_barrier_scan.py >> "$LOGDIR/noise_random_$TS.log" 2>&1

if [ -f "$RESULTS/lmc_barrier_c1m1.json" ]; then
    cp "$RESULTS/lmc_barrier_c1m1.json" "$RESULTS/noise_random.json"
    $VENV -c "
import json; d=json.load(open('$RESULTS/noise_random.json')); r=d['results']
c0=r[0]['loss_code']; cm=max(x['loss_code'] for x in r); c1=r[-1]['loss_code']
m0=r[0]['loss_med']; mm=max(x['loss_med'] for x in r); m1=r[-1]['loss_med']
bc = cm-(c0+c1)/2; bm = mm-(m0+m1)/2
print(f'  bar_code={bc:.4f}  bar_med={bm:.4f}')
" || echo "  ERROR parsing result"
else
    echo "  ERROR: lmc_barrier_c1m1.json not created — LMC scan failed"
fi

# ---- Final cleanup ----
rm -rf "$MODELS/_p1" "$MODELS/_p2" "$MODELS/code_e1" "$MODELS/medical_e1"
rm -f "$RESULTS/lmc_barrier_c1m1.json"

echo ""
echo "=== Noise floor done: $(date) ==="
