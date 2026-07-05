#!/usr/bin/env python3
"""LMC barrier scan: loss along linear interpolation between two models."""
import torch, sys, json, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

device = torch.device('cuda')
print(f'Loading models...', flush=True)
t0 = time.time()

agent_code = AFPAgent('code', str(device), model_id='EleutherAI/pythia-1.4b')
agent_med = AFPAgent('medical', str(device), model_id='EleutherAI/pythia-1.4b')
agent_code.load(PROJECT / 'experiments/trained_models/code_e1')
agent_med.load(PROJECT / 'experiments/trained_models/medical_e1')
agent_code.to_device()

sd_code = {k: v.detach().cpu() for k, v in agent_code.backbone.state_dict().items()}
sd_med = {k: v.detach().cpu() for k, v in agent_med.backbone.state_dict().items()}
print(f'  Models loaded ({time.time()-t0:.0f}s)', flush=True)

# Load eval data (small subset for speed)
t0_data = time.time()
data_code = load_data('code', 'EleutherAI/pythia-1.4b')
data_med = load_data('medical', 'EleutherAI/pythia-1.4b')
print(f'  Data loaded ({time.time()-t0_data:.0f}s)', flush=True)

@torch.no_grad()
def eval_loss(agent, data, n_samples=2000):
    agent.eval_mode()
    n = min(n_samples, data['n'])
    inp = data['input_ids'][:n].to(device)
    msk = data['attention_mask'][:n].to(device)
    labs = data['labels'][:n].to(device, dtype=torch.float32)
    total_loss, total = 0.0, 0
    batch = 128
    for i in range(0, n, batch):
        end = min(i + batch, n)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp[i:end], attention_mask=msk[i:end]).last_hidden_state
            logits = agent.head(h)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labs[i:end], reduction='sum')
        total_loss += loss.item()
        total += end - i
    return total_loss / total

# LMC scan
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

print(f'\nLMC scan: {len(alphas)} α points × 2 domains', flush=True)
for alpha in alphas:
    t_alpha = time.time()
    interp = {k: (1 - alpha) * sd_code[k] + alpha * sd_med[k] for k in sd_code}
    agent_code.load_backbone_state(interp)
    agent_code.to_device()

    loss_code = eval_loss(agent_code, data_code)
    loss_med = eval_loss(agent_code, data_med)
    results.append({'alpha': alpha, 'loss_code': loss_code, 'loss_med': loss_med})
    dt = time.time() - t_alpha
    print(f'  α={alpha:.2f}  loss_code={loss_code:.4f}  loss_med={loss_med:.4f}  [{dt:.0f}s]', flush=True)

# Save
output = {
    'experiment': 'LMC_barrier_scan',
    'models': 'code_e1 + medical_e1',
    'method': 'linear interpolation of backbone weights',
    'results': results,
    'duration_s': time.time() - t0
}
out_path = PROJECT / 'experiments/phase0_ivn/results/lmc_barrier.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

# Summary
losses_code = [r['loss_code'] for r in results]
losses_med = [r['loss_med'] for r in results]
barrier_code = max(losses_code) - losses_code[0]
barrier_med = max(losses_med) - losses_med[0]
print(f'\n--- LMC Barrier ---', flush=True)
print(f'  Code domain:  L(0)={losses_code[0]:.4f}  max L={max(losses_code):.4f}  barrier={barrier_code:.4f}', flush=True)
print(f'  Med  domain:  L(0)={losses_med[0]:.4f}  max L={max(losses_med):.4f}  barrier={barrier_med:.4f}', flush=True)
print(f'  Saved to {out_path} ({time.time()-t0:.0f}s total)', flush=True)
