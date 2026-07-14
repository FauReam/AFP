#!/usr/bin/env python3
"""Per-layer Fisher Information diagonal as Hessian curvature proxy.
At minima, H ≈ F = E[∇ℓ ∇ℓ^T]. Per-layer trace: sum of squared gradients in each layer.
Theory predicts medical > code curvature."""
import torch, json, numpy as np, time
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src')); sys.path.insert(0, str(PROJECT))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

device = torch.device('cuda')
n_samples = 1000
batch_size = 32  # small batch for per-sample gradient accumulation

results = {}

for name, model_path, eval_domain in [
    ('code', 'experiments/trained_models_opt/code_lr1e-4_s0', 'code'),
    ('medical', 'experiments/trained_models_opt/medical_lr1e-4_s0', 'medical'),
]:
    print(f'\n=== {name} (evaluated on {eval_domain} data) ===', flush=True)
    agent = AFPAgent(name, str(device), model_id='facebook/opt-1.3b')
    agent.load(PROJECT / model_path); agent.to_device(); agent.eval_mode()
    data = load_data(eval_domain, 'facebook/opt-1.3b')

    # Per-layer accumulated squared gradients
    layer_names = {}
    for n, p in agent.backbone.named_parameters():
        if 'layers.' in n and p.requires_grad:
            blk = int(n.split('layers.')[1].split('.')[0])
            layer_names.setdefault(blk, []).append(n)

    fisher_per_layer = {blk: 0.0 for blk in range(24)}
    param_count = {blk: 0 for blk in range(24)}
    total_samples = 0

    n = min(n_samples, len(data['input_ids']))
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        inp = data['input_ids'][i:end].to(device)
        msk = data['attention_mask'][i:end].to(device)
        labs = data['labels'][i:end].to(device, dtype=torch.float32)

        agent.backbone.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp, attention_mask=msk).last_hidden_state
            logits = agent.head(h)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labs, reduction='sum')
        loss.backward()

        # Accumulate squared gradients per layer
        for n, p in agent.backbone.named_parameters():
            if 'layers.' in n and p.grad is not None:
                blk = int(n.split('layers.')[1].split('.')[0])
                fisher_per_layer[blk] += (p.grad.detach() ** 2).sum().item()
                param_count[blk] += p.numel()

        total_samples += (end - i)
        if i % (batch_size * 10) == 0:
            print(f'  {i}/{n} samples', flush=True)

    # Normalize: average per sample, then per parameter
    norm_fisher = {}
    for blk in range(24):
        norm_fisher[blk] = fisher_per_layer[blk] / total_samples / max(param_count[blk], 1)

    results[name] = {
        'per_layer_fisher': {str(k): float(v) for k, v in norm_fisher.items()},
        'mean_fisher': float(np.mean(list(norm_fisher.values()))),
        'early_mean': float(np.mean([norm_fisher[b] for b in range(8)])),
        'late_mean': float(np.mean([norm_fisher[b] for b in range(16, 24)])),
    }
    print(f'  Mean Fisher: {results[name]["mean_fisher"]:.2e}', flush=True)
    print(f'  Early (0-7): {results[name]["early_mean"]:.2e}', flush=True)
    print(f'  Late (16-23): {results[name]["late_mean"]:.2e}', flush=True)

    del agent; torch.cuda.empty_cache()

# Compare
ratio = results['medical']['mean_fisher'] / (results['code']['mean_fisher'] + 1e-20)
results['comparison'] = {
    'code_mean': results['code']['mean_fisher'],
    'medical_mean': results['medical']['mean_fisher'],
    'ratio': float(ratio),
    'code_early': results['code']['early_mean'],
    'medical_early': results['medical']['early_mean'],
    'early_ratio': float(results['medical']['early_mean'] / (results['code']['early_mean'] + 1e-20)),
}

print(f'\n=== RESULT ===', flush=True)
print(f'  Medical/Code Fisher ratio: {ratio:.2f}x', flush=True)
print(f'  Early layer ratio: {results["comparison"]["early_ratio"]:.2f}x', flush=True)

with open('experiments/phase0_opt/results/fisher_curvature.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved', flush=True)
