#!/usr/bin/env python3
"""3-point LMC barrier scan (α=0.0, 0.5, 1.0). Fast alternative for domains where
11-point scan hangs at intermediate alphas (known issue with math/general on DGX)."""
import torch, sys, time, json, argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
sys.path.insert(0, str(PROJECT))
from AFP.protocol import AFPAgent
from scripts.run_ivn_phase0 import load_data

p = argparse.ArgumentParser()
p.add_argument('--model-a', required=True, help='path to model A dir (mapped to code_e1)')
p.add_argument('--model-b', required=True, help='path to model B dir (mapped to medical_e1)')
p.add_argument('--domain-a', default='code', help='domain name for model A file prefix')
p.add_argument('--domain-b', default='medical', help='domain name for model B file prefix')
p.add_argument('--output', required=True, help='output JSON path')
args = p.parse_args()

device = torch.device('cuda')
t0 = time.time()

# Setup symlinked dirs
model_dir = PROJECT / 'experiments' / 'trained_models'
a_dir = model_dir / '_a'
b_dir = model_dir / '_b'
code_link = model_dir / 'code_e1'
med_link = model_dir / 'medical_e1'

# Clean and setup
import shutil, os
for d in [a_dir, b_dir]:
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
for l in [code_link, med_link]:
    if l.exists() or l.is_symlink(): l.unlink(missing_ok=True)

# Copy model files with code/medical naming convention
model_a = Path(args.model_a)
model_b = Path(args.model_b)
da, db = args.domain_a, args.domain_b

shutil.copy(model_a / f'W_{da}_final.pt', a_dir / 'W_code_final.pt')
hf_a = model_a / f'W_{da}_head_final.pt'
if hf_a.exists(): shutil.copy(hf_a, a_dir / 'W_code_head_final.pt')
else: shutil.copy(model_dir / 'code_lr1e-4_s0' / 'W_code_head_final.pt', a_dir / 'W_code_head_final.pt')

shutil.copy(model_b / f'W_{db}_final.pt', b_dir / 'W_medical_final.pt')
hf_b = model_b / f'W_{db}_head_final.pt'
if hf_b.exists(): shutil.copy(hf_b, b_dir / 'W_medical_head_final.pt')
else: shutil.copy(model_dir / 'medical_lr1e-4_s0' / 'W_medical_head_final.pt', b_dir / 'W_medical_head_final.pt')

os.symlink(a_dir, code_link)
os.symlink(b_dir, med_link)

print(f'Loading: {model_a.name} + {model_b.name}', flush=True)
agent_code = AFPAgent('code', str(device), model_id='EleutherAI/pythia-1.4b')
agent_med  = AFPAgent('medical', str(device), model_id='EleutherAI/pythia-1.4b')
agent_code.load(code_link); agent_med.load(med_link)
agent_code.to_device()

sd_code = {k: v.detach().cpu() for k, v in agent_code.backbone.state_dict().items()}
sd_med  = {k: v.detach().cpu() for k, v in agent_med.backbone.state_dict().items()}
print(f'  Models loaded ({time.time()-t0:.0f}s)', flush=True)

data_code = load_data('code', 'EleutherAI/pythia-1.4b')
data_med  = load_data('medical', 'EleutherAI/pythia-1.4b')
print(f'  Data loaded ({time.time()-t0:.0f}s)', flush=True)

@torch.no_grad()
def ev(agent, data, n_samples=2000):
    agent.eval_mode()
    n = min(n_samples, data['n'])
    inp = data['input_ids'][:n].to(device)
    msk = data['attention_mask'][:n].to(device)
    labs = data['labels'][:n].to(device, dtype=torch.float32)
    total_loss, total = 0.0, 0
    batch = 512  # bumped from 128 for speed
    for i in range(0, n, batch):
        end = min(i + batch, n)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp[i:end], attention_mask=msk[i:end]).last_hidden_state
            logits = agent.head(h)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labs[i:end], reduction='sum')
        total_loss += loss.item()
        total += end - i
    return total_loss / total

results = []
for alpha in [0.0, 0.5, 1.0]:
    t_alpha = time.time()
    interp = {}
    for k in sd_code:
        if k in sd_med:
            interp[k] = ((1 - alpha) * sd_code[k].float() + alpha * sd_med[k].float()).to(sd_code[k].dtype)
    agent_code.load_backbone_state(interp)
    agent_code.to_device()
    lc = ev(agent_code, data_code)
    lm = ev(agent_code, data_med)
    results.append({'alpha': alpha, 'loss_code': lc, 'loss_med': lm})
    dt = time.time() - t_alpha
    print(f'  α={alpha:.1f}  loss_code={lc:.4f}  loss_med={lm:.4f}  [{dt:.0f}s]', flush=True)

output = {
    'experiment': 'LMC_barrier_3pt',
    'models': f'{model_a.name} + {model_b.name}',
    'method': '3-point linear interpolation (α=0.0, 0.5, 1.0)',
    'results': results,
    'duration_s': time.time() - t0
}
with open(args.output, 'w') as f:
    json.dump(output, f, indent=2)

c0, c1 = results[0]['loss_code'], results[-1]['loss_code']
cm = max(x['loss_code'] for x in results)
m0, m1 = results[0]['loss_med'], results[-1]['loss_med']
mm = max(x['loss_med'] for x in results)
print(f'\n  bar_code={cm-(c0+c1)/2:.4f}  bar_med={mm-(m0+m1)/2:.4f}', flush=True)
print(f'  Saved to {args.output} ({time.time()-t0:.0f}s total)', flush=True)
