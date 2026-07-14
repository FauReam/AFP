#!/bin/bash
# Theory-verification experiments A+B+C for Pythia-1.4B (~7 GPU-hours)
set -uo pipefail; cd /home/jiayu/AFP
export HF_ENDPOINT=https://hf-mirror.com HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
V=venv/bin/python3; M=experiments/trained_models; R=experiments/phase0_ivn/results
L=experiments/phase0_training; TS=$(date +%Y%m%d_%H%M%S)
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ===========================================================================
# Experiment A: Extended putt for medical (2 epochs, ~3h)
# Theory: longer convergence → wider basin → lower within-domain barrier
# ===========================================================================
log "=== A: Extended putt — medical 2 epochs (3 seeds) ==="

for seed in 10 11 12; do
    outdir="$M/medical_lr1e-4_2ep_s${seed}"
    if [ -f "$outdir/W_medical_final.pt" ]; then
        log "  [skip] $outdir"
        continue
    fi
    log "  Training $outdir (2 epochs)..."
    mkdir -p "$outdir"
    $V -u scripts/train_agent.py --domain medical --lr 1e-4 --output-dir "$outdir" --epochs 2 \
        >> "$L/train_medical_2ep_s${seed}_$TS.log" 2>&1
    # ΔW verify
    $V -c "
import torch; from transformers import AutoModel
base = AutoModel.from_pretrained('EleutherAI/pythia-1.4b', local_files_only=True, torch_dtype=torch.bfloat16)
base_sd = {k:v.detach().cpu() for k,v in base.state_dict().items()}; del base
m = torch.load('$outdir/W_medical_final.pt', map_location='cpu', weights_only=True)
dw = sum((m[k]-base_sd[k]).float().norm().item()**2 for k in m if k in base_sd)**0.5
n = sum(base_sd[k].float().norm().item()**2 for k in base_sd if k in m)**0.5
print(f'  $outdir ΔW={dw/(n+1e-8)*100:.2f}%')
" || log "  WARN: ΔW fail"
done

# LMC scans: within-domain for 2-epoch models
log ""
log "=== A scans: within-domain for 2-epoch medical ==="
for p in "10 11" "10 12" "11 12"; do
    s1=$(echo $p|cut -d' ' -f1); s2=$(echo $p|cut -d' ' -f2)
    out="$R/lmc_medical_2ep_s${s1}_s${s2}.json"
    [ -f "$out" ] && { log "  [skip] s${s1}_s${s2}"; continue; }
    log "  Scanning s$s1 ↔ s$s2"
    rm -rf "$M/_a" "$M/_b"; mkdir -p "$M/_a" "$M/_b"
    cp "$M/medical_lr1e-4_2ep_s${s1}/W_medical_final.pt" "$M/_a/W_code_final.pt"
    cp "$M/medical_lr1e-4_2ep_s${s1}/W_medical_head_final.pt" "$M/_a/W_code_head_final.pt"
    cp "$M/medical_lr1e-4_2ep_s${s2}/W_medical_final.pt" "$M/_b/W_medical_final.pt"
    cp "$M/medical_lr1e-4_2ep_s${s2}/W_medical_head_final.pt" "$M/_b/W_medical_head_final.pt"
    rm -f "$M/code_e1" "$M/medical_e1" "$R/lmc_barrier_c1m1.json"
    (cd "$M" && ln -sfn _a code_e1 && ln -sfn _b medical_e1)
    $V -u scripts/lmc_3pt_scan.py --model-a "$M/medical_lr1e-4_2ep_s${s1}" --model-b "$M/medical_lr1e-4_2ep_s${s2}" --domain-a medical --domain-b medical \
        --output "$out" >> "$L/lmc_med_2ep_s${s1}_s${s2}_$TS.log" 2>&1
    [ -f "$out" ] && $V -c "import json;d=json.load(open('$out'));r=d['results'];m0=r[0]['loss_med'];m1=r[-1]['loss_med'];mm=max(x['loss_med'] for x in r);print(f'  => bar_med={mm-(m0+m1)/2:.4f}')" || log "  ERROR"
done

# ===========================================================================
# Experiment B: Label noise on code (15% flip, 3 seeds, ~2h)
# Theory: noise → large Σ → small Pe → higher barrier (medical-like)
# ===========================================================================
log ""
log "=== B: Label-noise code (15% flip, 3 seeds) ==="

$V << 'PYEOF'
import torch, json, numpy as np, sys, time
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

PROJECT = Path('.').resolve()
sys.path.insert(0, str(PROJECT / 'src'))
from AFP.protocol import AFPAgent

# Load original code data, create noisy version
orig = torch.load(PROJECT / 'data/versaprm/train_code_pythia_L256.pt', map_location='cpu', weights_only=True)
labels = orig['labels'].clone()
n = len(labels)
np.random.seed(42)
flip_idx = np.random.choice(n, int(n*0.15), replace=False)
labels[flip_idx] = 1.0 - labels[flip_idx]
print(f'Flipped {len(flip_idx)}/{n} labels (15.0%)')

# Load val data (clean — fair comparison)
val_data = torch.load(PROJECT / 'data/versaprm/val_code_pythia_L256.pt', map_location='cpu', weights_only=True)

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

for seed in [10, 11, 12]:
    outdir = PROJECT / f'experiments/trained_models/code_noisy15_lr1e-4_s{seed}'
    if (outdir / 'W_code_final.pt').exists():
        print(f'[skip] {outdir}')
        continue
    print(f'Training {outdir}...')
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    agent = AFPAgent('code', str(device), model_id='EleutherAI/pythia-1.4b')
    agent.to_device(); agent.train_mode()
    init_sd = {k: v.detach().cpu().clone() for k, v in agent.backbone.state_dict().items()}

    train_ds = TensorDataset(orig['input_ids'], orig['attention_mask'], labels)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch  # 1 epoch

    opt = AdamW(agent.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999))
    # Drive-putt schedule
    drive_steps = int(total_steps * 0.70)
    import math
    def lr_sched(step):
        if step < drive_steps: return 1.0
        progress = (step - drive_steps) / max(total_steps - drive_steps, 1)
        return (3e-6/1e-4) + 0.5*(1 - 3e-6/1e-4)*(1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_sched)
    loss_fn = nn.BCEWithLogitsLoss()

    global_step = 0; t0 = time.time()
    for inp, msk, lab in train_dl:
        inp, msk, lab = inp.to(device), msk.to(device), lab.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp, attention_mask=msk).last_hidden_state
            logits = agent.head(h)
            loss = loss_fn(logits, lab)
        loss.backward(); opt.step(); scheduler.step()
        global_step += 1
        if global_step % 50 == 0:
            print(f'  step {global_step}/{total_steps} loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}', flush=True)

    # ΔW verify
    changed = sum((agent.backbone.state_dict()[k].detach().cpu() - init_sd[k]).float().norm().item() for k in init_sd if k in agent.backbone.state_dict())
    if changed > 1e-3:
        agent.save(outdir)
        print(f'  Saved: {outdir} Δ={changed:.1f}', flush=True)
    else:
        print(f'  WARN: no change detected for {outdir}', flush=True)
    del agent; torch.cuda.empty_cache()

print('All noisy code models trained')
PYEOF

# LMC scans
log ""
log "=== B scans: within-domain noisy code ==="
for p in "10 11" "10 12" "11 12"; do
    s1=$(echo $p|cut -d' ' -f1); s2=$(echo $p|cut -d' ' -f2)
    out="$R/lmc_code_noisy15_s${s1}_s${s2}.json"
    [ -f "$out" ] && { log "  [skip] s${s1}_s${s2}"; continue; }
    log "  Scanning s$s1 ↔ s$s2"
    rm -rf "$M/_a" "$M/_b"; mkdir -p "$M/_a" "$M/_b"
    cp "$M/code_noisy15_lr1e-4_s${s1}/W_code_final.pt" "$M/_a/W_code_final.pt"
    cp "$M/code_noisy15_lr1e-4_s${s1}/W_code_head_final.pt" "$M/_a/W_code_head_final.pt"
    cp "$M/code_noisy15_lr1e-4_s${s2}/W_code_final.pt" "$M/_b/W_medical_final.pt"
    cp "$M/code_noisy15_lr1e-4_s${s2}/W_code_head_final.pt" "$M/_b/W_medical_head_final.pt"
    rm -f "$M/code_e1" "$M/medical_e1" "$R/lmc_barrier_c1m1.json"
    (cd "$M" && ln -sfn _a code_e1 && ln -sfn _b medical_e1)
    $V -u scripts/lmc_3pt_scan.py --model-a "$M/medical_lr1e-4_2ep_s${s1}" --model-b "$M/medical_lr1e-4_2ep_s${s2}" --domain-a medical --domain-b medical \
        --output "$out" >> "$L/lmc_code_noisy15_s${s1}_s${s2}_$TS.log" 2>&1
    [ -f "$out" ] && $V -c "import json;d=json.load(open('$out'));r=d['results'];c0=r[0]['loss_code'];c1=r[-1]['loss_code'];cm=max(x['loss_code'] for x in r);print(f'  => bar_code={cm-(c0+c1)/2:.4f}')" || log "  ERROR"
done

# ===========================================================================
# Experiment C: Hessian eigenvalue estimation (~1h)
# ===========================================================================
log ""
log "=== C: Hessian eigenvalue estimation ==="

$V -c "
import torch, json, numpy as np
from transformers import AutoModel
from pathlib import Path
PROJECT = Path('.').resolve()
import sys; sys.path.insert(0, str(PROJECT/'src'))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

device = torch.device('cuda')

# Load models
models = {}
for name, path, domain in [
    ('code_std', 'experiments/trained_models/code_lr1e-4_s0', 'code'),
    ('med_std', 'experiments/trained_models/medical_lr1e-4_s0', 'medical'),
]:
    agent = AFPAgent(domain, str(device), model_id='EleutherAI/pythia-1.4b')
    agent.load(PROJECT / path); agent.to_device(); agent.eval_mode()
    models[name] = agent

# Estimate top Hessian eigenvalues via power iteration on random directions
# Hv ≈ (∇L(θ+εv) - ∇L(θ-εv)) / (2ε)
data = load_data('code', 'EleutherAI/pythia-1.4b')  # use code data for Hessian
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
    print(f'Estimating Hessian for {name}...', flush=True)
    # Compute top eigenvalue via power iteration on 10 random directions
    eig_ests = []
    for _ in range(10):
        # Random direction
        grads = []
        agent.backbone.zero_grad()
        l = loss_fn(agent); l.backward()
        for p in agent.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().clone())
        # Normalize
        norm = sum(g.norm().item()**2 for g in grads)**0.5
        v = [g / norm for g in grads]

        # Hv via finite difference
        # θ + εv
        for p, dv in zip(agent.parameters(), v):
            if p.grad is not None: p.data.add_(dv, alpha=eps)
        agent.backbone.zero_grad(); lp = loss_fn(agent); lp.backward()
        gp = [p.grad.detach().clone() for p in agent.parameters() if p.grad is not None]

        # θ - εv
        for p, dv in zip(agent.parameters(), v):
            if p.grad is not None: p.data.add_(dv, alpha=-2*eps)
        agent.backbone.zero_grad(); lm = loss_fn(agent); lm.backward()
        gm = [p.grad.detach().clone() for p in agent.parameters() if p.grad is not None]

        # Restore θ
        for p, dv in zip(agent.parameters(), v):
            if p.grad is not None: p.data.add_(dv, alpha=eps)

        # Hv ≈ (gp - gm) / (2ε)
        hv = [(gp[i] - gm[i]) / (2*eps) for i in range(len(gp))]
        # Rayleigh quotient: vᵀHv / vᵀv = vᵀHv
        rayleigh = sum((v[i] * hv[i]).sum().item() for i in range(len(v)))
        eig_ests.append(rayleigh)

    results_hess[name] = {
        'top_eigenvalue_estimate': float(np.mean(eig_ests)),
        'std': float(np.std(eig_ests)),
        'individual': [float(x) for x in eig_ests]
    }
    print(f'  {name}: λ_max ≈ {np.mean(eig_ests):.4f} ± {np.std(eig_ests):.4f}', flush=True)

with open('experiments/phase0_ivn/results/hessian_eigenvalues.json', 'w') as f:
    json.dump(results_hess, f, indent=2)
print('Hessian results saved', flush=True)
" >> "$L/hessian_eigenvalues_$TS.log" 2>&1

log ""
log "=== ALL DONE ==="
