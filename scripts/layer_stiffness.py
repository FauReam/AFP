#!/usr/bin/env python3
"""Layer-wise stiffness: per-layer loss sensitivity as Hessian curvature proxy.
Theory: barrier ∝ Δθ^T H Δθ. H = diag(H_0, ..., H_L). We measure ||H_ℓ|| via finite perturbation.
Medical should show higher per-layer stiffness than code."""
import torch, json, numpy as np, time
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src')); sys.path.insert(0, str(PROJECT))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

device = torch.device('cuda')
data_code = load_data('code', 'EleutherAI/pythia-1.4b')
data_med  = load_data('medical', 'EleutherAI/pythia-1.4b')

@torch.no_grad()
def eval_loss(agent, data, n_samples=200):
    agent.eval_mode()
    n = min(n_samples, data['n'])
    inp = data['input_ids'][:n].to(device)
    msk = data['attention_mask'][:n].to(device)
    labs = data['labels'][:n].to(device, dtype=torch.float32)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        h = agent.backbone(input_ids=inp, attention_mask=msk).last_hidden_state
        logits = agent.head(h)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labs).item()

eps = 1e-2  # relative perturbation magnitude
results = {}

for name, path, domain in [
    ('code', 'experiments/trained_models/code_lr1e-4_s0', 'code'),
    ('medical', 'experiments/trained_models/medical_lr1e-4_s0', 'medical'),
]:
    print(f'\n=== {name} ===', flush=True)
    agent = AFPAgent(domain, str(device), model_id='EleutherAI/pythia-1.4b')
    agent.load(PROJECT / path); agent.to_device(); agent.eval_mode()

    # Baseline loss
    L0_code = eval_loss(agent, data_code)
    L0_med  = eval_loss(agent, data_med)
    print(f'  Baseline: L_code={L0_code:.4f} L_med={L0_med:.4f}', flush=True)

    stiffness_code = []
    stiffness_med  = []

    for blk in range(24):
        # Get layer keys
        blk_keys = [k for k, p in agent.backbone.named_parameters() if f'layers.{blk}.' in k and 'weight' in k]
        if not blk_keys: continue

        # Save original weights
        orig = {k: agent.backbone.state_dict()[k].clone() for k in blk_keys}

        # Add perturbation
        for k in blk_keys:
            w = agent.backbone.state_dict()[k]
            noise = torch.randn_like(w.float()) * w.float().norm() * eps
            agent.backbone.state_dict()[k].data.copy_((w.float() + noise).to(w.dtype))

        # Measure
        Lp_code = eval_loss(agent, data_code)
        Lp_med  = eval_loss(agent, data_med)

        # Restore
        for k in blk_keys:
            agent.backbone.state_dict()[k].data.copy_(orig[k])

        # Stiffness = ΔL / ε² (normalized by weight norm)
        s_code = (Lp_code - L0_code) / (eps * eps)
        s_med  = (Lp_med - L0_med) / (eps * eps)
        stiffness_code.append(s_code)
        stiffness_med.append(s_med)

        if blk % 4 == 0:
            print(f'  layer {blk}: s_code={s_code:.4f} s_med={s_med:.4f}', flush=True)

    results[name] = {
        'stiffness_code': stiffness_code,
        'stiffness_med': stiffness_med,
        'mean_stiffness_code': float(np.mean(stiffness_code)),
        'mean_stiffness_med': float(np.mean(stiffness_med)),
        'ratio': float(np.mean(stiffness_med) / (np.mean(stiffness_code) + 1e-8)),
        'early_layers_code': float(np.mean(stiffness_code[:8])),
        'early_layers_med': float(np.mean(stiffness_med[:8])),
    }
    print(f'  Mean stiffness: code={np.mean(stiffness_code):.4f} med={np.mean(stiffness_med):.4f} ratio={results[name]["ratio"]:.2f}x', flush=True)
    del agent; torch.cuda.empty_cache()

with open('experiments/phase0_ivn/results/layer_stiffness.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved', flush=True)
