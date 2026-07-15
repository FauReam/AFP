#!/usr/bin/env python3
"""Verify OPT training quality: self-domain loss, ΔW from base, compare to Pythia."""
import torch, json, numpy as np
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / 'src')); sys.path.insert(0, str(PROJECT))

# === ΔW from base ===
print("=== ΔW from pretrained base ===")
for base_id, base_name, model_dir, domain_key in [
    ('facebook/opt-1.3b', 'OPT', 'experiments/trained_models_opt', ''),
    ('EleutherAI/pythia-1.4b', 'Pythia', 'experiments/trained_models', ''),
]:
    from transformers import AutoModel
    base = AutoModel.from_pretrained(base_id, local_files_only=True, torch_dtype=torch.bfloat16)
    base_sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    base_norm = sum(v.float().norm().item()**2 for v in base_sd.values())**0.5
    del base; torch.cuda.empty_cache()

    for domain in ['code', 'medical']:
        for s in ['0', '1', '2']:
            if base_name == 'OPT':
                fp = f'{model_dir}/{domain}_lr1e-4_s{s}/W_{domain}_final.pt'
            else:
                fp = f'{model_dir}/{domain}_lr1e-4_s{s}/W_{domain}_final.pt'
            try:
                sd = torch.load(fp, map_location='cpu', weights_only=True)
                dw = sum((sd[k] - base_sd[k]).float().norm().item()**2 for k in sd if k in base_sd)**0.5
                pct = dw/(base_norm+1e-8)*100
                print(f'  {base_name} {domain}_s{s}: ΔW={pct:.2f}%')
            except Exception as e:
                print(f'  {base_name} {domain}_s{s}: ERROR {e}')

# === Self-domain loss from LMC data ===
print("\n=== Self-domain BCE Loss (α=0 for code, α=1 for medical) ===")
for model_name, result_dir, pattern in [
    ('Pythia', 'experiments/phase0_ivn/results', 'lmc_lr1e-4'),
    ('OPT', 'experiments/phase0_opt/results', 'lmc_code_med_opt'),
]:
    for s in ['0', '1', '2']:
        d = json.load(open(f'{result_dir}/{pattern}_s{s}.json'))
        r = d['results']
        l_code = r[0]['loss_code']  # code model at α=0
        l_med  = r[-1]['loss_med']  # medical model at α=1
        print(f'  {model_name} s{s}: code_loss={l_code:.4f} med_loss={l_med:.4f}')

# === Barrier comparison ===
print("\n=== Barrier Summary ===")
for model_name, result_dir, pattern in [
    ('Pythia', 'experiments/phase0_ivn/results', 'lmc_lr1e-4'),
    ('OPT', 'experiments/phase0_opt/results', 'lmc_code_med_opt'),
]:
    bc, bm = [], []
    for s in ['0', '1', '2']:
        d = json.load(open(f'{result_dir}/{pattern}_s{s}.json'))
        r = d['results']
        c0, c1 = r[0]['loss_code'], r[-1]['loss_code']
        cm = max(x['loss_code'] for x in r)
        m0, m1 = r[0]['loss_med'], r[-1]['loss_med']
        mm = max(x['loss_med'] for x in r)
        bc.append(cm - (c0+c1)/2)
        bm.append(mm - (m0+m1)/2)
    print(f'  {model_name}: bar_code={np.mean(bc):.3f}±{np.std(bc):.3f} bar_med={np.mean(bm):.3f}±{np.std(bm):.3f}')

# === Within-domain summary ===
print("\n=== Within-Domain Barrier Summary ===")
for model_name, result_dir, fpattern in [
    ('Pythia', 'experiments/phase0_ivn/results', 'lmc_{domain}_s{s1}_s{s2}'),
    ('OPT', 'experiments/phase0_opt/results', 'lmc_{domain}_within_opt_s{s1}_s{s2}'),
]:
    for domain in ['code', 'medical']:
        bars = []
        for s1, s2 in [('0','1'), ('0','2'), ('1','2')]:
            fn = fpattern.format(domain=domain, s1=s1, s2=s2)
            try:
                d = json.load(open(f'{result_dir}/{fn}.json'))
                r = d['results']
                if domain == 'code':
                    c0, c1 = r[0]['loss_code'], r[-1]['loss_code']
                    cm = max(x['loss_code'] for x in r)
                    bars.append(cm - (c0+c1)/2)
                else:
                    m0, m1 = r[0]['loss_med'], r[-1]['loss_med']
                    mm = max(x['loss_med'] for x in r)
                    bars.append(mm - (m0+m1)/2)
            except: pass
        if bars:
            print(f'  {model_name} {domain} within: {np.mean(bars):.3f}±{np.std(bars):.3f} (n={len(bars)})')

print("\nDone.")
