#!/usr/bin/env python3
"""LMC barrier scan: loss along linear interpolation between two models.

Usage:
  python scripts/lmc_barrier_scan.py                           # code_e1 + medical_e1
  python scripts/lmc_barrier_scan.py --med-e 3                  # code_e1 + medical_e3
  python scripts/lmc_barrier_scan.py --code-e 1 --med-e 5       # code_e1 + medical_e5
"""
import torch, sys, json, time, argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

p = argparse.ArgumentParser()
p.add_argument('--code-e', type=int, default=1, help='code model epoch')
p.add_argument('--med-e', type=int, default=1, help='medical model epoch')
p.add_argument('--model-id', type=str, default='facebook/opt-1.3b', help='HF model ID')
args = p.parse_args()

device = torch.device('cuda')
t0 = time.time()

code_dir = PROJECT / f'experiments/trained_models_opt/code_e{args.code_e}'
med_dir = PROJECT / f'experiments/trained_models_opt/medical_e{args.med_e}'

print(f'Loading: {code_dir.name} + {med_dir.name}', flush=True)
agent_code = AFPAgent('code', str(device), model_id=args.model_id)
agent_med = AFPAgent('medical', str(device), model_id=args.model_id)
agent_code.load(code_dir)
agent_med.load(med_dir)

output_name = f'lmc_barrier_c{args.code_e}m{args.med_e}'
agent_code.to_device()

sd_code = {k: v.detach().cpu() for k, v in agent_code.backbone.state_dict().items()}
sd_med = {k: v.detach().cpu() for k, v in agent_med.backbone.state_dict().items()}
print(f'  Models loaded ({time.time()-t0:.0f}s)', flush=True)

# Load eval data (small subset for speed)
t0_data = time.time()
# Note: barrier formula uses Frankle definition: max(L) - (L(0)+L(1))/2
# Both the JSON output and printed summary use this definition (verified at bottom of script)
data_code = load_data('code', 'facebook/opt-1.3b')
data_med = load_data('medical', 'facebook/opt-1.3b')
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
    'models': f'{code_dir.name} + {med_dir.name}',
    'method': 'linear interpolation of backbone weights',
    'results': results,
    'duration_s': time.time() - t0
}
out_path = PROJECT / f'experiments/phase0_opt/results/{output_name}.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

# Summary
losses_code = [r['loss_code'] for r in results]
losses_med = [r['loss_med'] for r in results]
barrier_code = max(losses_code) - (losses_code[0] + losses_code[-1]) / 2
barrier_med = max(losses_med) - (losses_med[0] + losses_med[-1]) / 2
print(f'\n--- LMC Barrier ---', flush=True)
print(f'  Code domain:  L(0)={losses_code[0]:.4f}  max L={max(losses_code):.4f}  barrier={barrier_code:.4f}', flush=True)
print(f'  Med  domain:  L(0)={losses_med[0]:.4f}  max L={max(losses_med):.4f}  barrier={barrier_med:.4f}', flush=True)
# Verify: printed barrier must match Frankle definition in JSON
barrier_from_json_code = max(x['loss_code'] for x in results) - (results[0]['loss_code'] + results[-1]['loss_code']) / 2
barrier_from_json_med = max(x['loss_med'] for x in results) - (results[0]['loss_med'] + results[-1]['loss_med']) / 2
assert abs(barrier_code - barrier_from_json_code) < 1e-6, f'barrier_code mismatch: {barrier_code} vs {barrier_from_json_code}'
assert abs(barrier_med - barrier_from_json_med) < 1e-6, f'barrier_med mismatch: {barrier_med} vs {barrier_from_json_med}'
print(f'  Saved to {out_path} ({time.time()-t0:.0f}s total)', flush=True)
