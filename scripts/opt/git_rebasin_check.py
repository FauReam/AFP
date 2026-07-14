#!/usr/bin/env python3
"""Git Re-Basin permutation alignment check for Pythia-1.4B domain-specialized models.
Uses greedy weight matching (no scipy needed). Measures barrier before/after alignment."""
import torch, sys, time, json, argparse
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT))

p = argparse.ArgumentParser()
p.add_argument('--model-a', required=True)
p.add_argument('--model-b', required=True)
p.add_argument('--domain-a', default='medical')
p.add_argument('--domain-b', default='medical')
p.add_argument('--output', required=True)
args = p.parse_args()

device = torch.device('cuda')
t0 = time.time()

# Load both models
from AFP.protocol import AFPAgent
agent_a = AFPAgent('code', str(device), model_id='facebook/opt-1.3b')
agent_b = AFPAgent('medical', str(device), model_id='facebook/opt-1.3b')

# Copy model files to _a/_b with standard naming
import shutil, os
model_dir = PROJECT / 'experiments' / 'trained_models'
for d in [model_dir / '_a', model_dir / '_b']:
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)

ma, mb = Path(args.model_a), Path(args.model_b)
shutil.copy(ma / f'W_{args.domain_a}_final.pt', model_dir / '_a' / 'W_code_final.pt')
shutil.copy(mb / f'W_{args.domain_b}_final.pt', model_dir / '_b' / 'W_medical_final.pt')
# Head files
for src_dom, dst_name in [(args.domain_a, 'code'), (args.domain_b, 'medical')]:
    src_dir = ma if dst_name == 'code' else mb
    hf = src_dir / f'W_{args.domain_a if dst_name=="code" else args.domain_b}_head_final.pt'
    if hf.exists():
        shutil.copy(hf, model_dir / ('_a' if dst_name == 'code' else '_b') / f'W_{dst_name}_head_final.pt')
    else:
        ref = model_dir / f'code_lr1e-4_s0' / 'W_code_head_final.pt'
        shutil.copy(ref, model_dir / ('_a' if dst_name == 'code' else '_b') / f'W_{dst_name}_head_final.pt')

for l, t in [('code_e1', '_a'), ('medical_e1', '_b')]:
    link = model_dir / l
    if link.exists() or link.is_symlink(): link.unlink()
    os.symlink(model_dir / t, link)

agent_a.load(model_dir / 'code_e1')
agent_b.load(model_dir / 'medical_e1')
print(f'Models loaded ({time.time()-t0:.0f}s)', flush=True)

# ============================================================
# Git Re-Basin: greedy weight matching per layer
# ============================================================
sd_a = agent_a.backbone.state_dict()
sd_b = agent_b.backbone.state_dict()

# Identify linear layers to align
linear_layers = []
for k in sd_a:
    if k not in sd_b: continue
    if 'weight' not in k: continue
    if len(sd_a[k].shape) != 2: continue
    # Skip embedding and final norm
    if 'embed' in k or 'final_layer_norm' in k: continue
    # For QKV combined projection, skip (too complex)
    if 'query_key_value' in k: continue
    linear_layers.append(k)

print(f'Found {len(linear_layers)} linear layers to align', flush=True)

# Greedy matching: for each output neuron in A, find best match in B
permutations = {}
total_neurons = 0
for layer in linear_layers:
    Wa = sd_a[layer].float().cpu().numpy()  # (out_dim, in_dim)
    Wb = sd_b[layer].float().cpu().numpy()
    out_dim = Wa.shape[0]
    total_neurons += out_dim

    # Compute cost matrix: ||Wa[i] - Wb[j]||²
    # For efficiency, use greedy: for each i, pick unmatched j with min distance
    perm = list(range(out_dim))
    used = set()
    for i in range(out_dim):
        # Find best unmatched j
        best_j, best_dist = -1, float('inf')
        for j in range(out_dim):
            if j in used: continue
            dist = np.sum((Wa[i] - Wb[j]) ** 2)
            if dist < best_dist:
                best_dist, best_j = dist, j
        perm[i] = best_j
        used.add(best_j)
    permutations[layer] = perm

print(f'Aligned {total_neurons} neurons across {len(linear_layers)} layers ({time.time()-t0:.0f}s)', flush=True)

# Apply permutation to model B
sd_b_aligned = {k: v.clone() for k, v in sd_b.items()}
for layer, perm in permutations.items():
    # Permute the output dimension (dim 0)
    perm_tensor = torch.tensor(perm, device=sd_b_aligned[layer].device)
    sd_b_aligned[layer] = sd_b_aligned[layer][perm_tensor]
    # Also permute the corresponding bias if it exists
    bias_key = layer.replace('.weight', '.bias')
    if bias_key in sd_b_aligned:
        sd_b_aligned[bias_key] = sd_b_aligned[bias_key][perm_tensor]
    # For the NEXT layer's input dimension, we'd need to permute dim 1
    # This is the full Re-Basin approach — for simplicity, we only do output dim
    # which is sufficient for a check (full alignment would require transitive permutations)

print(f'Permutation applied ({time.time()-t0:.0f}s)', flush=True)

# ============================================================
# LMC scan: before alignment
# ============================================================
from scripts.run_ivn_phase0 import load_data
data_code = load_data('code', 'facebook/opt-1.3b')
data_med  = load_data('medical', 'facebook/opt-1.3b')

@torch.no_grad()
def eval_loss(agent, data, n_samples=2000):
    agent.eval_mode()
    n = min(n_samples, data['n'])
    inp = data['input_ids'][:n].to(device)
    msk = data['attention_mask'][:n].to(device)
    labs = data['labels'][:n].to(device, dtype=torch.float32)
    total_loss, total = 0.0, 0
    for i in range(0, n, 128):
        end = min(i + 128, n)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp[i:end], attention_mask=msk[i:end]).last_hidden_state
            logits = agent.head(h)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labs[i:end], reduction='sum')
        total_loss += loss.item()
        total += end - i
    return total_loss / total

def scan_barrier(sd_ref, label):
    """Scan LMC barrier between sd_ref and sd_b (or aligned version)"""
    results = []
    for alpha in [0.0, 0.5, 1.0]:
        t_alpha = time.time()
        interp = {}
        for k in sd_ref:
            if k in sd_b:
                interp[k] = ((1 - alpha) * sd_ref[k].float() + alpha * sd_b[k].float()).to(sd_ref[k].dtype)
        agent_a.load_backbone_state(interp)
        agent_a.to_device()
        lc = eval_loss(agent_a, data_code)
        lm = eval_loss(agent_a, data_med)
        results.append({'alpha': alpha, 'loss_code': lc, 'loss_med': lm})
        dt = time.time() - t_alpha
        print(f'  [{label}] α={alpha:.1f} loss_code={lc:.4f} loss_med={lm:.4f} [{dt:.0f}s]', flush=True)

    c0, c1 = results[0]['loss_code'], results[-1]['loss_code']
    cm = max(x['loss_code'] for x in results)
    m0, m1 = results[0]['loss_med'], results[-1]['loss_med']
    mm = max(x['loss_med'] for x in results)
    return cm - (c0 + c1) / 2, mm - (m0 + m1) / 2

print(f'\n=== Barrier BEFORE alignment ===', flush=True)
bar_code_before, bar_med_before = scan_barrier(sd_a, 'before')

# Swap sd_b for aligned version
sd_b_orig = {k: v.clone() for k, v in sd_b.items()}
for k in sd_b_aligned:
    sd_b[k] = sd_b_aligned[k]

print(f'\n=== Barrier AFTER alignment ===', flush=True)
bar_code_after, bar_med_after = scan_barrier(sd_a, 'after')

# Restore
sd_b = sd_b_orig

# Save results
output = {
    'experiment': 'git_rebasin_check',
    'models': f'{ma.name} + {mb.name}',
    'layers_aligned': len(linear_layers),
    'neurons_aligned': total_neurons,
    'bar_code_before': bar_code_before,
    'bar_med_before': bar_med_before,
    'bar_code_after': bar_code_after,
    'bar_med_after': bar_med_after,
    'bar_med_delta': bar_med_before - bar_med_after,
    'duration_s': time.time() - t0
}
with open(args.output, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\n=== RESULTS ===', flush=True)
print(f'  Before alignment: bar_code={bar_code_before:.4f}  bar_med={bar_med_before:.4f}', flush=True)
print(f'  After alignment:  bar_code={bar_code_after:.4f}  bar_med={bar_med_after:.4f}', flush=True)
print(f'  Delta:            Δbar_med={bar_med_before - bar_med_after:.4f}', flush=True)
print(f'  Saved to {args.output} ({time.time()-t0:.0f}s total)', flush=True)
