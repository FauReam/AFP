#!/bin/bash
# Fixup: B noisy code LMC scans + C Hessian eigenvalues
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=venv/bin/python3; M=experiments/trained_models; R=experiments/phase0_ivn/results
L=experiments/phase0_training; TS=$(date +%Y%m%d_%H%M%S)
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- B fixup: noisy code within-domain LMC ----
log "=== B fixup: noisy code within-domain ==="
for p in "10 11" "10 12" "11 12"; do
    s1=$(echo $p|cut -d' ' -f1); s2=$(echo $p|cut -d' ' -f2)
    out="$R/lmc_code_noisy15_s${s1}_s${s2}.json"
    [ -f "$out" ] && { log "  [skip] s${s1}_s${s2}"; continue; }
    log "  s$s1 ↔ s$s2"
    $V -u scripts/lmc_3pt_scan.py \
        --model-a "$M/code_noisy15_lr1e-4_s${s1}" --model-b "$M/code_noisy15_lr1e-4_s${s2}" \
        --domain-a code --domain-b code --output "$out" >> "$L/lmc_noisy_code_s${s1}_s${s2}_$TS.log" 2>&1
    if [ -f "$out" ]; then
        $V -c "import json;d=json.load(open('$out'));r=d['results'];c0=r[0]['loss_code'];c1=r[-1]['loss_code'];cm=max(x['loss_code'] for x in r);print(f'  => bar_code={cm-(c0+c1)/2:.4f}')"
    else
        log "  ERROR"
    fi
done

# ---- C fixup: Hessian eigenvalues ----
log ""
log "=== C fixup: Hessian eigenvalues ==="
$V << 'PYEOF'
import torch, json, numpy as np
from transformers import AutoModel
from pathlib import Path
PROJECT = Path('.').resolve()
import sys; sys.path.insert(0, str(PROJECT/'src'))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

device = torch.device('cuda')

models = {}
for name, path, domain in [
    ('code_std', 'experiments/trained_models/code_lr1e-4_s0', 'code'),
    ('med_std', 'experiments/trained_models/medical_lr1e-4_s0', 'medical'),
]:
    agent = AFPAgent(domain, str(device), model_id='EleutherAI/pythia-1.4b')
    agent.load(PROJECT/path); agent.to_device(); agent.eval_mode()
    models[name] = agent

data = load_data('code', 'EleutherAI/pythia-1.4b')
n_samples = 200
inp = data['input_ids'][:n_samples].to(device)
msk = data['attention_mask'][:n_samples].to(device)
labs = data['labels'][:n_samples].to(device, dtype=torch.float32)

def loss_fn(agent):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        h = agent.backbone(input_ids=inp, attention_mask=msk).last_hidden_state
        logits = agent.head(h)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labs)

eps = 1e-3
results_hess = {}
for name, agent in models.items():
    print(f'Hessian for {name}...', flush=True)
    eig_ests = []
    for _ in range(10):
        grads = []
        agent.backbone.zero_grad()
        l = loss_fn(agent); l.backward()
        for p in agent.parameters():
            if p.grad is not None: grads.append(p.grad.detach().clone())
        norm = sum(g.norm().item()**2 for g in grads)**0.5
        v = [g / norm for g in grads]

        for p, dv in zip(agent.parameters(), v):
            if p.grad is not None: p.data.add_(dv, alpha=eps)
        agent.backbone.zero_grad(); lp = loss_fn(agent); lp.backward()
        gp = [p.grad.detach().clone() for p in agent.parameters() if p.grad is not None]

        for p, dv in zip(agent.parameters(), v):
            if p.grad is not None: p.data.add_(dv, alpha=-2*eps)
        agent.backbone.zero_grad(); lm = loss_fn(agent); lm.backward()
        gm = [p.grad.detach().clone() for p in agent.parameters() if p.grad is not None]

        for p, dv in zip(agent.parameters(), v):
            if p.grad is not None: p.data.add_(dv, alpha=eps)

        hv = [(gp[i] - gm[i]) / (2*eps) for i in range(len(gp))]
        rayleigh = sum((v[i] * hv[i]).sum().item() for i in range(len(v)))
        eig_ests.append(float(rayleigh))

    results_hess[name] = {'top_eigenvalue': float(np.mean(eig_ests)), 'std': float(np.std(eig_ests))}
    print(f'  {name}: λ_max ≈ {np.mean(eig_ests):.4f} ± {np.std(eig_ests):.4f}', flush=True)

with open('experiments/phase0_ivn/results/hessian_eigenvalues.json', 'w') as f:
    json.dump(results_hess, f, indent=2)
print('Hessian done', flush=True)
PYEOF

log "=== BC fixup DONE ==="
