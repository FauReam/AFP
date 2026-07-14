#!/usr/bin/env python3
"""Accuracy-based LMC barrier evaluation for key model pairs."""
import torch, sys, time, json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src')); sys.path.insert(0, str(PROJECT))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

device = torch.device('cuda')
t0 = time.time()

@torch.no_grad()
def eval_acc(agent, data, n_samples=2000):
    agent.eval_mode()
    n = min(n_samples, data['n'])
    inp = data['input_ids'][:n].to(device)
    msk = data['attention_mask'][:n].to(device)
    labs = data['labels'][:n].to(device, dtype=torch.float32)
    total_loss, correct, total = 0.0, 0, 0
    for i in range(0, n, 128):
        end = min(i + 128, n)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp[i:end], attention_mask=msk[i:end]).last_hidden_state
            logits = agent.head(h)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labs[i:end], reduction='sum')
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labs[i:end]).sum().item()
        total += end - i
    return total_loss / total, correct / total

data_code = load_data('code', 'facebook/opt-1.3b')
data_med  = load_data('medical', 'facebook/opt-1.3b')

results_all = {}

# ----- Pair 1: code↔medical standard (seed 0) -----
print("=== Pair 1: code_s0 ↔ medical_s0 (standard) ===", flush=True)
agent_c = AFPAgent('code', str(device), model_id='facebook/opt-1.3b')
agent_m = AFPAgent('medical', str(device), model_id='facebook/opt-1.3b')
agent_c.load(PROJECT / 'experiments/trained_models_opt/code_lr1e-4_s0')
agent_m.load(PROJECT / 'experiments/trained_models_opt/medical_lr1e-4_s0')
sd_c = {k: v.detach().cpu() for k, v in agent_c.backbone.state_dict().items()}
sd_m = {k: v.detach().cpu() for k, v in agent_m.backbone.state_dict().items()}
head_c = {k: v.detach().cpu() for k, v in agent_c.head.state_dict().items()}
head_m = {k: v.detach().cpu() for k, v in agent_m.head.state_dict().items()}

pair_results = []
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    t_a = time.time()
    interp_w = {k: ((1-alpha)*sd_c[k].float() + alpha*sd_m[k].float()).to(sd_c[k].dtype) for k in sd_c if k in sd_m}
    interp_h = {k: ((1-alpha)*head_c[k].float() + alpha*head_m[k].float()).to(head_c[k].dtype) for k in head_c if k in head_m}
    agent_c.backbone.load_state_dict(interp_w)
    agent_c.head.load_state_dict(interp_h)
    agent_c.to_device()
    loss_c, acc_c = eval_acc(agent_c, data_code)
    loss_m, acc_m = eval_acc(agent_c, data_med)
    pair_results.append({'alpha': alpha, 'loss_code': loss_c, 'acc_code': acc_c, 'loss_med': loss_m, 'acc_med': acc_m})
    print(f"  α={alpha:.1f}  acc_code={acc_c:.4f} acc_med={acc_m:.4f} [{time.time()-t_a:.0f}s]", flush=True)
results_all['code_med_standard'] = pair_results

# ----- Pair 2: code within-domain (s0↔s1) -----
print("\n=== Pair 2: code_s0 ↔ code_s1 (within-domain) ===", flush=True)
sd_c1 = torch.load(PROJECT / 'experiments/trained_models_opt/code_lr1e-4_s1/W_code_final.pt', map_location='cpu', weights_only=True)
pair_results = []
for alpha in [0.0, 0.5, 1.0]:
    interp_w = {k: ((1-alpha)*sd_c[k].float() + alpha*sd_c1[k].float()).to(sd_c[k].dtype) for k in sd_c if k in sd_c1}
    agent_c.backbone.load_state_dict(interp_w)
    agent_c.to_device()
    loss_c, acc_c = eval_acc(agent_c, data_code)
    pair_results.append({'alpha': alpha, 'loss_code': loss_c, 'acc_code': acc_c})
    print(f"  α={alpha:.1f}  acc_code={acc_c:.4f}", flush=True)
results_all['code_within_s0s1'] = pair_results

# ----- Pair 3: medical within-domain (s0↔s1) -----
print("\n=== Pair 3: medical_s0 ↔ medical_s1 (within-domain) ===", flush=True)
sd_m1 = torch.load(PROJECT / 'experiments/trained_models_opt/medical_lr1e-4_s1/W_medical_final.pt', map_location='cpu', weights_only=True)
pair_results = []
for alpha in [0.0, 0.5, 1.0]:
    interp_w = {k: ((1-alpha)*sd_m[k].float() + alpha*sd_m1[k].float()).to(sd_m[k].dtype) for k in sd_m if k in sd_m1}
    agent_m.backbone.load_state_dict(interp_w)
    agent_m.to_device()
    loss_m, acc_m = eval_acc(agent_m, data_med)
    pair_results.append({'alpha': alpha, 'loss_med': loss_m, 'acc_med': acc_m})
    print(f"  α={alpha:.1f}  acc_med={acc_m:.4f}", flush=True)
results_all['medical_within_s0s1'] = pair_results

# Summary
print("\n" + "="*60, flush=True)
for name, results in results_all.items():
    print(f"\n{name}:", flush=True)
    for r in results:
        keys = [k for k in r if k != 'alpha']
        print(f"  α={r['alpha']:.1f}: " + " ".join(f"{k}={r[k]:.4f}" for k in keys), flush=True)

# Compute accuracy barriers
print("\n=== ACCURACY BARRIERS ===", flush=True)
for name, results in results_all.items():
    keys = [k for k in results[0] if k.startswith('acc_')]
    for key in keys:
        vals = [r[key] for r in results]
        # Accuracy barrier: how much does accuracy drop at midpoint?
        mid_acc = vals[len(vals)//2]
        endpoint_mean = (vals[0] + vals[-1]) / 2
        acc_barrier = endpoint_mean - mid_acc  # positive = accuracy drops at midpoint
        print(f"  {name} {key}: endpoints={endpoint_mean:.4f} midpoint={mid_acc:.4f} barrier={acc_barrier:.4f}", flush=True)

with open('experiments/phase0_opt/results/accuracy_barriers.json', 'w') as f:
    json.dump(results_all, f, indent=2)
print(f"\nSaved ({time.time()-t0:.0f}s)", flush=True)
