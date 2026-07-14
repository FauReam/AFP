#!/usr/bin/env python3
"""Comprehensive model merging benchmark: TIES, Task Arithmetic, Layer-Selective."""
import torch, sys, time, json, numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src')); sys.path.insert(0, str(PROJECT))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

device = torch.device('cuda')
t0 = time.time()

@torch.no_grad()
def evaluate(agent, data, n_samples=2000):
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

def load_model(domain, path):
    agent = AFPAgent(domain, str(device), model_id='facebook/opt-1.3b')
    agent.load(path); agent.to_device()
    return agent

def get_sd(agent):
    return {k: v.detach().cpu() for k, v in agent.backbone.state_dict().items()}

def get_task_vector(sd, base_sd):
    return {k: (sd[k].float() - base_sd[k].float()) for k in sd if k in base_sd}

def ties_merge(task_vectors, base_sd, k=0.2):
    """TIES-Merging: Trim → Elect Sign → Disjoint Merge.
    task_vectors: list of {key: tensor} task vectors
    base_sd: base model state_dict
    k: fraction of parameters to retain after trimming"""
    merged = {k: v.float().clone() for k, v in base_sd.items()}

    for key in task_vectors[0]:
        if key not in base_sd: continue
        vecs = torch.stack([tv[key] for tv in task_vectors])  # (n_models, *shape)

        # Step 1: Trim — keep top-k by magnitude for each model
        flat = vecs.reshape(len(task_vectors), -1)
        magnitudes = flat.abs()
        k_thresh = int(flat.shape[1] * k)
        topk_vals, _ = torch.topk(magnitudes, k_thresh, dim=1)
        threshold = topk_vals[:, -1].unsqueeze(1)  # per-model threshold
        mask = magnitudes >= threshold
        trimmed = flat * mask.float()
        trimmed = trimmed.reshape_as(vecs)

        # Step 2: Elect sign — resolve conflicts by majority vote
        signs = trimmed.sign()
        sign_sum = signs.sum(dim=0)  # sum of signs across models
        elected_sign = sign_sum.sign()  # majority sign
        elected_sign[elected_sign == 0] = 1  # tie → positive

        # Step 3: Disjoint merge — average only params with elected sign
        agree_mask = (signs == elected_sign.unsqueeze(0))
        merged_vec = (trimmed * agree_mask.float()).sum(dim=0)
        count = agree_mask.float().sum(dim=0).clamp(min=1)

        merged[key] += merged_vec.reshape(base_sd[key].shape) / count.reshape(base_sd[key].shape)

    return merged

def task_arithmetic_merge(task_vectors, base_sd, lambdas):
    """Task Arithmetic: W = W_base + sum_i lambda_i * tau_i"""
    merged = {k: v.float().clone() for k, v in base_sd.items()}
    for lam, tv in zip(lambdas, task_vectors):
        for k in tv:
            if k in merged:
                merged[k] += lam * tv[k]
    return merged

# Load data and base model
print("Loading base model and data...", flush=True)
data_code = load_data('code', 'facebook/opt-1.3b')
data_med  = load_data('medical', 'facebook/opt-1.3b')
from transformers import AutoModel
base_model = AutoModel.from_pretrained('facebook/opt-1.3b', local_files_only=True, torch_dtype=torch.bfloat16)
base_sd = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
del base_model; torch.cuda.empty_cache()

results = []

# Load all standard-div models (3 seeds each, code + medical)
models = {}
for domain in ['code', 'medical']:
    for s in ['0', '1', '2']:
        path = PROJECT / f'experiments/trained_models_opt/{domain}_lr1e-4_s{s}'
        models[f'{domain}_s{s}'] = load_model(domain, path)

# ---- 1. Individual model baselines ----
print("\n=== Individual Models ===", flush=True)
for name, agent in models.items():
    lc, ac = evaluate(agent, data_code)
    lm, am = evaluate(agent, data_med)
    results.append({'method': 'individual', 'name': name, 'loss_code': lc, 'acc_code': ac, 'loss_med': lm, 'acc_med': am})
    print(f"  {name}: code_acc={ac:.4f} med_acc={am:.4f}", flush=True)

# ---- 2. Pure Averaging (multiple α, all 3 seed pairs) ----
print("\n=== Pure Averaging ===", flush=True)
for s in ['0', '1', '2']:
    sd_c = get_sd(models[f'code_s{s}'])
    sd_m = get_sd(models[f'medical_s{s}'])
    head_c = {k: v.detach().cpu() for k, v in models[f'code_s{s}'].head.state_dict().items()}
    head_m = {k: v.detach().cpu() for k, v in models[f'medical_s{s}'].head.state_dict().items()}
    for alpha in [0.3, 0.5, 0.7]:
        merged_w = {k: ((1-alpha)*sd_c[k].float() + alpha*sd_m[k].float()).to(sd_c[k].dtype) for k in sd_c if k in sd_m}
        merged_h = {k: ((1-alpha)*head_c[k].float() + alpha*head_m[k].float()).to(head_c[k].dtype) for k in head_c if k in head_m}
        agent = models['code_s0']
        agent.backbone.load_state_dict(merged_w); agent.head.load_state_dict(merged_h)
        agent.to_device()
        lc, ac = evaluate(agent, data_code)
        lm, am = evaluate(agent, data_med)
        results.append({'method': 'average', 'seed': s, 'alpha': alpha, 'loss_code': lc, 'acc_code': ac, 'loss_med': lm, 'acc_med': am})
        print(f"  avg α={alpha} s{s}: code_acc={ac:.4f} med_acc={am:.4f}", flush=True)

# ---- 3. TIES-Merging (multiple k, all 3 seeds) ----
print("\n=== TIES-Merging ===", flush=True)
for s in ['0', '1', '2']:
    sd_c = get_sd(models[f'code_s{s}']); sd_m = get_sd(models[f'medical_s{s}'])
    tv_c = get_task_vector(sd_c, base_sd); tv_m = get_task_vector(sd_m, base_sd)
    for k in [0.1, 0.2, 0.3, 0.5]:
        merged_w = ties_merge([tv_c, tv_m], base_sd, k=k)
        agent = models['code_s0']
        agent.backbone.load_state_dict({kk: vv.to(sd_c[kk].dtype) for kk, vv in merged_w.items() if kk in sd_c})
        agent.to_device()
        lc, ac = evaluate(agent, data_code)
        lm, am = evaluate(agent, data_med)
        results.append({'method': 'TIES', 'seed': s, 'k': k, 'loss_code': lc, 'acc_code': ac, 'loss_med': lm, 'acc_med': am})
        print(f"  TIES k={k} s{s}: code_acc={ac:.4f} med_acc={am:.4f}", flush=True)

# ---- 4. Task Arithmetic (multiple λ, all 3 seeds) ----
print("\n=== Task Arithmetic ===", flush=True)
for s in ['0', '1', '2']:
    sd_c = get_sd(models[f'code_s{s}']); sd_m = get_sd(models[f'medical_s{s}'])
    tv_c = get_task_vector(sd_c, base_sd); tv_m = get_task_vector(sd_m, base_sd)
    for lam in [0.1, 0.3, 0.5, 0.7]:
        merged_w = task_arithmetic_merge([tv_c, tv_m], base_sd, [lam, lam])
        agent = models['code_s0']
        agent.backbone.load_state_dict({kk: vv.to(sd_c[kk].dtype) for kk, vv in merged_w.items() if kk in sd_c})
        agent.to_device()
        lc, ac = evaluate(agent, data_code)
        lm, am = evaluate(agent, data_med)
        results.append({'method': 'TaskArith', 'seed': s, 'lambda': lam, 'loss_code': lc, 'acc_code': ac, 'loss_med': lm, 'acc_med': am})
        print(f"  TA λ={lam} s{s}: code_acc={ac:.4f} med_acc={am:.4f}", flush=True)

# ---- 5. Layer-Selective Merge (TIES on late layers only) ----
print("\n=== Layer-Selective Merge ===", flush=True)
for s in ['0', '1', '2']:
    sd_c = get_sd(models[f'code_s{s}']); sd_m = get_sd(models[f'medical_s{s}'])
    # Merge only layers 16-23 with TIES, keep code model for layers 0-15
    merged_w = {k: v.float().clone() for k, v in sd_c.items()}
    for k in sd_c:
        layer_match = any(f'layers.{l}.' in k for l in range(0, 16))
        if layer_match:
            merged_w[k] = sd_c[k].float()  # keep code
        else:
            # TIES for deep layers only
            if k in sd_m:
                tv_c_k = sd_c[k].float() - base_sd[k].float()
                tv_m_k = sd_m[k].float() - base_sd[k].float()
                vecs = torch.stack([tv_c_k, tv_m_k])
                signs = vecs.sign(); elected = signs.sum(0).sign(); elected[elected==0] = 1
                agree = (signs == elected.unsqueeze(0))
                merged_w[k] = base_sd[k].float()
                merged_w[k] += (vecs * agree.float()).sum(0) / agree.float().sum(0).clamp(min=1)
    agent = models['code_s0']
    agent.backbone.load_state_dict({kk: vv.to(sd_c[kk].dtype) for kk, vv in merged_w.items() if kk in sd_c})
    agent.to_device()
    lc, ac = evaluate(agent, data_code)
    lm, am = evaluate(agent, data_med)
    results.append({'method': 'LayerSelect', 'seed': s, 'loss_code': lc, 'acc_code': ac, 'loss_med': lm, 'acc_med': am})
    print(f"  LayerSelect s{s}: code_acc={ac:.4f} med_acc={am:.4f}", flush=True)

# ---- 6. Within-domain merge (code s0+s1) ----
print("\n=== Within-Domain Merge ===", flush=True)
sd_c0 = get_sd(models['code_s0']); sd_c1 = get_sd(models['code_s1'])
merged_w = {k: 0.5*sd_c0[k].float() + 0.5*sd_c1[k].float() for k in sd_c0 if k in sd_c1}
agent = models['code_s0']
agent.backbone.load_state_dict({kk: vv.to(sd_c0[kk].dtype) for kk, vv in merged_w.items() if kk in sd_c0})
agent.to_device()
lc, ac = evaluate(agent, data_code)
results.append({'method': 'within_code_avg', 'loss_code': lc, 'acc_code': ac})
print(f"  Code within s0+s1 avg: acc={ac:.4f}", flush=True)

sd_m0 = get_sd(models['medical_s0']); sd_m1 = get_sd(models['medical_s1'])
merged_w = {k: 0.5*sd_m0[k].float() + 0.5*sd_m1[k].float() for k in sd_m0 if k in sd_m1}
agent_m = models['medical_s0']
agent_m.backbone.load_state_dict({kk: vv.to(sd_m0[kk].dtype) for kk, vv in merged_w.items() if kk in sd_m0})
agent_m.to_device()
lm, am = evaluate(agent_m, data_med)
results.append({'method': 'within_med_avg', 'loss_med': lm, 'acc_med': am})
print(f"  Med within s0+s1 avg: acc={am:.4f}", flush=True)

# Save
with open('experiments/phase0_opt/results/merge_benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} evaluations ({time.time()-t0:.0f}s)", flush=True)

# Summary: best method for each domain
print("\n=== BEST RESULTS ===", flush=True)
best_code = max((r for r in results if 'acc_code' in r), key=lambda r: r['acc_code'])
best_med  = max((r for r in results if 'acc_med' in r), key=lambda r: r['acc_med'])
print(f"  Best code acc: {best_code.get('method','?')} {best_code['acc_code']:.4f}", flush=True)
print(f"  Best med acc:  {best_med.get('method','?')} {best_med['acc_med']:.4f}", flush=True)
print(f"  Baseline code (individual): {max(r['acc_code'] for r in results if r.get('name','').startswith('code')):.4f}", flush=True)
print(f"  Baseline med (individual):  {max(r['acc_med'] for r in results if r.get('name','').startswith('medical')):.4f}", flush=True)
