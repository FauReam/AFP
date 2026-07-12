#!/usr/bin/env python3
"""Model weight divergence verification — loads state_dicts directly from .pt files."""
import torch, json, sys
from pathlib import Path
import numpy as np

PROJECT = Path("/home/jiayu/AFP")
sys.path.insert(0, str(PROJECT / "src"))
from transformers import AutoModel
from AFP.protocol import AFPAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = PROJECT / "experiments" / "trained_models"

def load_pt_state(model_dir, domain):
    """Load backbone state_dict directly from .pt file."""
    pt_path = model_dir / f"W_{domain}_final.pt"
    return torch.load(pt_path, map_location="cpu", weights_only=True)

print("Loading base model...")
base_backbone = AutoModel.from_pretrained(
    "EleutherAI/pythia-1.4b", trust_remote_code=True,
    local_files_only=True).to(dtype=torch.bfloat16)
sd_base = {k: v.detach().cpu() for k, v in base_backbone.state_dict().items()}
base_norm = sum(v.float().norm().item() ** 2 for v in sd_base.values()) ** 0.5
print(f"  Base norm: {base_norm:.4f}")

def div_pct(sd_trained, sd_base):
    """Compute ||W_trained - W_base|| / ||W_base|| * 100."""
    diff2 = sum((sd_trained[k].float() - sd_base[k].float()).norm().item() ** 2 for k in sd_base)
    return (diff2 ** 0.5) / base_norm * 100

def cross_pct(sd_a, sd_b):
    """Compute ||W_a - W_b|| / ||W_base|| * 100."""
    diff2 = sum((sd_a[k].float() - sd_b[k].float()).norm().item() ** 2 for k in sd_a if k in sd_b)
    return (diff2 ** 0.5) / base_norm * 100

def equal_tensors(sd_trained, sd_base):
    n_eq = sum(1 for k in sd_base if (sd_trained[k] - sd_base[k]).abs().max().item() < 1e-8)
    n_total = len(sd_base)
    return n_eq, n_total

configs = [
    ("code_lr1e-4_s0", "medical_lr1e-4_s0", "standard_s0"),
    ("code_lr1e-4_s1", "medical_lr1e-4_s1", "standard_s1"),
    ("code_lr1e-4_s2", "medical_lr1e-4_s2", "standard_s2"),
    ("code_lr5e-4_s0", "medical_lr5e-4_s0", "high_s0"),
    ("code_lr5e-4_s1", "medical_lr5e-4_s1", "high_s1"),
    ("code_lr5e-4_s2", "medical_lr5e-4_s2", "high_s2"),
    ("code_lr2e-4_s0", "medical_lr2e-4_s0", "lr2e4_s0"),
    ("code_lr3e-4_s0", "medical_lr3e-4_s0", "lr3e4_s0"),
    ("code_lr2e-4_s1", "medical_lr2e-4_s1", "lr2e4_s1"),
    ("code_lr2e-4_s2", "medical_lr2e-4_s2", "lr2e4_s2"),
    ("code_lr3e-4_s1", "medical_lr3e-4_s1", "lr3e4_s1"),
    ("code_lr3e-4_s2", "medical_lr3e-4_s2", "lr3e4_s2"),
    # 160M scaling
    ("code_160m_s0", "medical_160m_s0", "160m"),
]

print("\n" + "=" * 85)
print("WEIGHT DIVERGENCE FROM BASE MODEL")
print("=" * 85)

summary = []

for code_cfg, med_cfg, label in configs:
    code_dir = OUT_DIR / code_cfg
    med_dir = OUT_DIR / med_cfg

    try:
        sd_code = load_pt_state(code_dir, "code")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  [{label}] SKIP code: {e}")
        continue
    try:
        sd_med = load_pt_state(med_dir, "medical")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  [{label}] SKIP med: {e}")
        continue

    # Skip if architecture mismatch (e.g. 160M vs 1.4B)
    first_key = next(iter(sd_base))
    if first_key not in sd_code or first_key not in sd_med:
        print(f"  [{label}] SKIP: architecture mismatch with base model")
        continue
    if sd_code[first_key].shape != sd_base[first_key].shape:
        print(f"  [{label}] SKIP: shape mismatch ({sd_code[first_key].shape} vs {sd_base[first_key].shape})")
        continue

    c_pct = div_pct(sd_code, sd_base)
    m_pct = div_pct(sd_med, sd_base)
    x_pct = cross_pct(sd_code, sd_med)
    c_eq, c_tot = equal_tensors(sd_code, sd_base)
    m_eq, m_tot = equal_tensors(sd_med, sd_base)

    print(f"  [{label}] code_ΔW={c_pct:.2f}%  med_ΔW={m_pct:.2f}%  cross={x_pct:.2f}%  "
          f"code_eq={c_eq}/{c_tot}  med_eq={m_eq}/{m_tot}  "
          f"code_clean={'YES' if c_eq < c_tot else 'WARNING:ALL EQUAL'}  "
          f"med_clean={'YES' if m_eq < m_tot else 'WARNING:ALL EQUAL'}")

    summary.append({
        "label": label, "code_dW": round(c_pct, 3), "med_dW": round(m_pct, 3),
        "cross_dW": round(x_pct, 3),
        "code_clean": c_eq < c_tot, "med_clean": m_eq < m_tot,
        "code_eq": f"{c_eq}/{c_tot}", "med_eq": f"{m_eq}/{m_tot}"
    })

# Within-domain cross-seed divergences
print("\n" + "=" * 85)
print("WITHIN-DOMAIN CROSS-SEED DIVERGENCES")
print("=" * 85)
for domain in ["code", "medical"]:
    for lr in ["lr1e-4", "lr5e-4"]:
        sds = {}
        for s in range(3):
            mdir = OUT_DIR / f"{domain}_{lr}_s{s}"
            try:
                sds[s] = load_pt_state(mdir, domain)
            except:
                pass
        if len(sds) >= 2:
            for i in range(3):
                for j in range(i+1, 3):
                    if i in sds and j in sds:
                        c = cross_pct(sds[i], sds[j])
                        print(f"  {domain} {lr} seed{i}↔seed{j}: {c:.2f}%")
    print()

# Barriers from raw data
print("=" * 85)
print("BARRIER COMPUTATION (Frankle formula: max(L) - (L(0)+L(1))/2)")
print("=" * 85)

def frankle_barrier(results):
    L = [r["loss_code"] for r in results]
    bc = max(L) - (L[0] + L[-1]) / 2
    L = [r["loss_med"] for r in results]
    bm = max(L) - (L[0] + L[-1]) / 2
    return bc, bm

for cond, files in [
    ("standard (lr=1e-4)", ["lmc_lr1e-4_s0.json", "lmc_lr1e-4_s1.json", "lmc_lr1e-4_s2.json"]),
    ("high (lr=5e-4)", ["lmc_lr5e-4_s0.json", "lmc_lr5e-4_s1.json", "lmc_lr5e-4_s2.json"]),
]:
    print(f"\n{cond}:")
    bc_list, bm_list = [], []
    for fname in files:
        f = PROJECT / f"experiments/phase0_ivn/results/{fname}"
        if f.exists():
            data = json.load(open(f))
            bc, bm = frankle_barrier(data["results"])
            bc_list.append(bc)
            bm_list.append(bm)
            print(f"  {fname}: code_barrier={bc:.6f}, med_barrier={bm:.6f}")
    if len(bc_list) >= 2:
        print(f"  MEAN: code={np.mean(bc_list):.4f}±{np.std(bc_list):.4f}, "
              f"med={np.mean(bm_list):.4f}±{np.std(bm_list):.4f}")

print("\nWithin-domain:")
for label, fname in [
    ("code s0↔s1", "lmc_code_s0_s1.json"), ("code s0↔s2", "lmc_code_s0_s2.json"),
    ("code s1↔s2", "lmc_code_s1_s2.json"), ("med s0↔s1", "lmc_medical_s0_s1.json"),
    ("med s0↔s2", "lmc_medical_s0_s2.json"), ("med s1↔s2", "lmc_medical_s1_s2.json"),
]:
    f = PROJECT / f"experiments/phase0_ivn/results/{fname}"
    if f.exists():
        data = json.load(open(f))
        bc, bm = frankle_barrier(data["results"])
        print(f"  {label}: code_barrier={bc:.6f}, med_barrier={bm:.6f}")

print("\nNoise floor:")
for name in ["noise_identical", "noise_random"]:
    f = PROJECT / f"experiments/phase0_ivn/results/{name}.json"
    if f.exists():
        data = json.load(open(f))
        bc, bm = frankle_barrier(data["results"])
        print(f"  {name}: code_barrier={bc:.6f}, med_barrier={bm:.6f}")

# Check lr2e-4 and lr3e-4 md5 identity
print("\n" + "=" * 85)
print("lr=2e-4 vs lr=3e-4 FILE IDENTITY CHECK")
print("=" * 85)
import hashlib
for s in range(3):
    for lr in ["lr2e-4", "lr3e-4"]:
        f = PROJECT / f"experiments/phase0_ivn/results/lmc_{lr}_s{s}.json"
        if f.exists():
            h = hashlib.md5(f.read_bytes()).hexdigest()
            print(f"  lmc_{lr}_s{s}.json  md5={h}")

# Check if lmc scripts use correct formula
print("\n" + "=" * 85)
print("SCRIPT BARRIER FORMULA AUDIT")
print("=" * 85)
print("  lmc_barrier_scan.py line 101-102:")
print("    barrier_code = max(losses_code) - losses_code[0]")
print("    barrier_med = max(losses_med) - losses_med[0]")
print("  WRONG: Should be max(L) - (L(0)+L(1))/2 per Frankle et al. (2020)")
print("  Paper correctly cites Frankle formula but script uses different formula.")
print("  The printed barriers in the scan output are therefore inflated (by ~L(1)/2).")
print("  However, the JSON files contain raw losses, so correct barriers can be")
print("  recomputed — as done above.")

print("\nDone.")
