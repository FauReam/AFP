#!/usr/bin/env python3
"""AFP Phase 0 Experiment A: AFP vs FedAvg vs No Exchange.

Full-FT Pythia-1.4B on two domains, then three integration methods.
One script, one output directory.

Usage:
  python scripts/run_experiment_a.py                     # full run
  python scripts/run_experiment_a.py --skip-train         # eval only
  python scripts/run_experiment_a.py --lr-peak 3e-5       # conservative LR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# ponytail: TF32 on Blackwell tensor cores, ~1.3x matmul speed for free
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LOSS_FN = nn.BCEWithLogitsLoss()  # module-level, not re-created per evaluate()

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from afp.protocol import AFPAgent, block_importance, importance_cosine

DATA_DIR = PROJECT / "data" / "versaprm"
OUT_DIR = PROJECT / "experiments" / "phase0_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_LEN = 384


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CATMAP = {"computer science": "code", "engineering": "code",
          "health": "medical", "biology": "medical"}


def prepare_data(domain: str) -> Path:
    """HF download + tokenize → cached .pt."""
    out = DATA_DIR / f"versa_prm_{domain}_train.pt"
    if out.exists():
        print(f"[data] {domain} cached ({out.stat().st_size/1024**2:.0f}MB)")
        return out

    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[data] downloading {domain}...")
    ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")

    ids_list, labs_list = [], []
    kept = 0
    for row in ds:
        if CATMAP.get((row.get("category") or "").lower(), "general") != domain:
            continue
        try:
            steps = eval(str(row.get("chain_of_thoughts", "[]")))
            labels = eval(str(row.get("labels", "[]")))
        except (SyntaxError, ValueError):
            continue
        for s, l in zip(steps, labels):
            ids = tok.encode(f"{row.get('question', '')}\n{s}")
            if len(ids) <= MAX_LEN:
                ids_list.append(ids)
                labs_list.append(int(l))
                kept += 1

    n = len(ids_list)
    inp = torch.full((n, MAX_LEN), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros(n, MAX_LEN, dtype=torch.bool)
    for i, ids in enumerate(ids_list):
        L = min(len(ids), MAX_LEN)
        inp[i, :L] = torch.tensor(ids[:L])
        mask[i, :L] = True

    data = {"input_ids": inp, "attention_mask": mask,
            "labels": torch.tensor(labs_list, dtype=torch.float32),
            "domain": domain, "n": n}
    torch.save(data, out)
    print(f"[data] {domain}: {n} steps → {out}")
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(domain: str, lr_peak: float = 1e-4, lr_min: float = 3e-6,
              batch: int = 1024) -> AFPAgent:
    """Full-FT one agent, return it with weights on CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(prepare_data(domain), map_location="cpu", weights_only=True)

    ds = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4,
                        pin_memory=True)
    n_batches = len(loader)
    print(f"[train] {domain}: {n_batches} batches @ {batch}")

    agent = AFPAgent(domain).train_mode().to_device()
    agent.save_init(OUT_DIR / "W_init.pt")

    opt = torch.optim.AdamW(agent.parameters(), lr=lr_peak, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    t0 = time.time()

    for bi, (inp, mask, labs) in enumerate(loader):
        inp, mask = inp.to(device), mask.to(device)
        labs = labs.to(device, dtype=torch.float32)

        progress = bi / max(n_batches - 1, 1)
        lr = lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp, attention_mask=mask).last_hidden_state
        loss = LOSS_FN(agent.head(h), labs)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        if (bi + 1) % 20 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            eta = elapsed / (bi + 1) * (n_batches - bi - 1)
            print(f"  {bi+1:4d}/{n_batches} loss={loss.item():.4f} lr={lr:.2e} "
                  f"{elapsed/60:.0f}m eta={eta/60:.0f}m")

    agent.save(OUT_DIR)
    agent.backbone.cpu()
    agent.head.cpu()
    print(f"[train] {domain} done in {(time.time()-t0)/60:.0f}m")
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(agent: AFPAgent, domain: str, batch: int = 256):
    """Accuracy on last 15% of domain data."""
    data = torch.load(prepare_data(domain), map_location="cpu", weights_only=True)
    n = data["n"]
    n_test = max(n // 6, 100)
    start = n - n_test

    device = next(agent.backbone.parameters()).device
    agent.eval_mode()

    inp = data["input_ids"][start:].to(device)
    mask = data["attention_mask"][start:].to(device)
    labs = data["labels"][start:].to(device, dtype=torch.float32)

    total_loss, correct, total = 0.0, 0, 0

    for i in range(0, n_test, batch):
        end = min(i + batch, n_test)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp[i:end],
                               attention_mask=mask[i:end]).last_hidden_state
        logits = agent.head(h)
        # ponytail: reduction='sum' avoids batch-size-dependent mean scaling
        total_loss += F.binary_cross_entropy_with_logits(
            logits, labs[i:end], reduction='sum').item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labs[i:end]).sum().item()
        total += end - i

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="AFP Phase 0 Experiment A")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--domain-a", default="medical")
    p.add_argument("--domain-b", default="code")
    p.add_argument("--lr-peak", type=float, default=1e-4)
    p.add_argument("--lr-min", type=float, default=3e-6)
    p.add_argument("--batch", type=int, default=1024)
    args = p.parse_args()

    dom_a, dom_b = args.domain_a, args.domain_b
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== AFP Experiment A: {dom_a} vs {dom_b} ===\n")

    # ---- Step 1: Train ----
    if args.skip_train:
        agent_a = AFPAgent(dom_a).load(OUT_DIR)
        agent_b = AFPAgent(dom_b).load(OUT_DIR)
    else:
        agent_a = train_one(dom_a, args.lr_peak, args.lr_min, args.batch)
        agent_b = train_one(dom_b, args.lr_peak, args.lr_min, args.batch)

    # ---- Step 2: Init & Importance ----
    W_init = agent_a.load_init(OUT_DIR / "W_init.pt")
    imp_a = agent_a.compute_importance(W_init)
    imp_b = agent_b.compute_importance(W_init)
    cos = importance_cosine(imp_a, imp_b)
    print(f"\nImportance cosine similarity: {cos:.3f} "
          f"({'STRUCTURAL' if cos < 0.8 else 'weak'} divergence)")

    # ---- Step 3: Baseline (No Exchange) ----
    agent_a.to_device().eval_mode()
    agent_b.to_device().eval_mode()
    _, acc_aa = evaluate(agent_a, dom_a)
    _, acc_ab = evaluate(agent_a, dom_b)
    _, acc_bb = evaluate(agent_b, dom_b)
    _, acc_ba = evaluate(agent_b, dom_a)
    print(f"\n--- Baseline: No Exchange ---")
    print(f"  A→{dom_a}: {acc_aa:.4f}  A→{dom_b}: {acc_ab:.4f}")
    print(f"  B→{dom_b}: {acc_bb:.4f}  B→{dom_a}: {acc_ba:.4f}")

    # ---- Step 4: FedAvg ----
    print(f"\n--- FedAvg (grid α) ---")
    best_fed = {"alpha": 0, "net": -999.0}
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        fed_a = agent_a.integrate_fedavg(agent_b.backbone_state(), alpha)
        fed_b = agent_b.integrate_fedavg(agent_a.backbone_state(), alpha)
        agent_a.load_backbone_state(fed_a)
        agent_b.load_backbone_state(fed_b)
        _, s_a = evaluate(agent_a, dom_a)
        _, c_a = evaluate(agent_a, dom_b)
        _, s_b = evaluate(agent_b, dom_b)
        _, c_b = evaluate(agent_b, dom_a)
        net = (s_a - acc_aa) + (c_a - acc_ab) + (s_b - acc_bb) + (c_b - acc_ba)
        if net > best_fed["net"]:
            best_fed = {"alpha": alpha, "net": net,
                        "a_self": s_a, "a_cross": c_a,
                        "b_self": s_b, "b_cross": c_b}
        print(f"  α={alpha:.1f}  s_a={s_a:.4f} c_a={c_a:.4f} "
              f"s_b={s_b:.4f} c_b={c_b:.4f}  net={net:+.4f}")
    print(f"  best α={best_fed['alpha']:.1f}")

    # ---- Step 5: AFP ----
    print(f"\n--- AFP (grid τ) ---")
    best_afp = {"tau": 0, "net": -999.0}
    for tau in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        afp_a = agent_a.integrate_afp(agent_b.backbone_state(), W_init, tau)
        afp_b = agent_b.integrate_afp(agent_a.backbone_state(), W_init, tau)
        agent_a.load_backbone_state(afp_a)
        agent_b.load_backbone_state(afp_b)
        _, s_a = evaluate(agent_a, dom_a)
        _, c_a = evaluate(agent_a, dom_b)
        _, s_b = evaluate(agent_b, dom_b)
        _, c_b = evaluate(agent_b, dom_a)
        net = (s_a - acc_aa) + (c_a - acc_ab) + (s_b - acc_bb) + (c_b - acc_ba)
        if net > best_afp["net"]:
            best_afp = {"tau": tau, "net": net,
                        "a_self": s_a, "a_cross": c_a,
                        "b_self": s_b, "b_cross": c_b}
        print(f"  τ={tau:.1f}  s_a={s_a:.4f} c_a={c_a:.4f} "
              f"s_b={s_b:.4f} c_b={c_b:.4f}  net={net:+.4f}")
    print(f"  best τ={best_afp['tau']:.1f}")

    # ---- Step 6: Report ----
    print(f"\n{'='*65}")
    print(f"FINAL: {dom_a} ↔ {dom_b}")
    print(f"{'='*65}")
    print(f"{'Method':<14} {'A self':>8} {'A cross':>8} {'B self':>8} "
          f"{'B cross':>8} {'Δ net':>8}")
    print(f"{'-'*54}")
    base = [acc_aa, acc_ab, acc_bb, acc_ba]
    print(f"{'No Exchange':<14} {acc_aa:8.4f} {acc_ab:8.4f} {acc_bb:8.4f} "
          f"{acc_ba:8.4f} {'—':>8}")
    print(f"{'FedAvg':<14} {best_fed['a_self']:8.4f} {best_fed['a_cross']:8.4f} "
          f"{best_fed['b_self']:8.4f} {best_fed['b_cross']:8.4f} "
          f"{best_fed['net']:+8.4f}")
    print(f"{'AFP':<14} {best_afp['a_self']:8.4f} {best_afp['a_cross']:8.4f} "
          f"{best_afp['b_self']:8.4f} {best_afp['b_cross']:8.4f} "
          f"{best_afp['net']:+8.4f}")
    print(f"\nAFP advantage over FedAvg: {best_afp['net'] - best_fed['net']:+.4f}")

    # Gate visualization
    print(f"\nPer-block gates (τ={best_afp['tau']:.1f}):")
    trust_a = best_afp.get("trust_a", 1.0)
    trust_b = best_afp.get("trust_b", 1.0)
    for i in range(24):
        ma = max(0.0, min(1.0, 1.0 - imp_a[i] / max(best_afp['tau'], 1e-8)))
        mb = max(0.0, min(1.0, 1.0 - imp_b[i] / max(best_afp['tau'], 1e-8)))
        bar = lambda v: "▓" * int(v * 10) + "░" * (10 - int(v * 10))
        print(f"  blk{i:2d} A[{bar(ma)}] B[{bar(mb)}] "
              f"imp_A={imp_a[i]:.3f} imp_B={imp_b[i]:.3f}")

    # Save
    result = {
        "domains": [dom_a, dom_b],
        "no_exchange": {"a_self": acc_aa, "a_cross": acc_ab,
                         "b_self": acc_bb, "b_cross": acc_ba},
        "fedavg": best_fed,
        "afp": best_afp,
        "importance_cosine": cos,
        "imp_a": imp_a,
        "imp_b": imp_b,
    }
    rp = OUT_DIR / "experiment_a_results.json"
    json.dump(result, open(rp, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved → {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
