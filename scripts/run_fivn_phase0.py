#!/usr/bin/env python3
"""F-IVN Phase 0: Function-space Iterative Virtual-Negotiation.

Cross-architecture: models negotiate in prediction space, distill back to weights.

Default: Qwen2.5-Math-1.5B (teacher) ⇄ Pythia-1.4B (student) — heterogeneous.

ponytail: PRM heads must be trained before negotiation. Phase 0 trains heads
on VersaPRM data if no checkpoint found, then runs F-IVN.

Usage:
  python scripts/run_fivn_phase0.py
  python scripts/run_fivn_phase0.py --teacher Qwen/Qwen2.5-Math-1.5B-Instruct --student EleutherAI/pythia-1.4b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LOSS_FN = nn.BCEWithLogitsLoss()

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from AFP.protocol import guess_hidden

OUT_DIR = PROJECT / "experiments" / "phase0_fivn"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_LEN = 384
N_REF = 500

CATMAP = {"computer science": "code", "engineering": "code",
          "health": "medical", "biology": "medical",
          "math": "math", "mathematics": "math"}


# ===========================================================================
# Data (same as IVN, tokenizer-aware)
# ===========================================================================

def load_data(domain: str, model_id: str) -> dict:
    safe_name = model_id.replace("/", "__")
    cache = PROJECT / "data" / "versaprm" / f"versa_prm_{domain}_{safe_name}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)

    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[data] downloading {domain} for {model_id}...")
    ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")

    ids_list, labs_list = [], []
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
    torch.save(data, cache)
    print(f"[data] {domain}: {n} steps -> {cache}")
    return data


# ===========================================================================
# Model wrapper (architecture-agnostic)
# ===========================================================================

class FAgent:
    """Architecture-agnostic agent: backbone + trained PRM head."""
    def __init__(self, model_id: str, domain: str, device: str = "cuda"):
        self.model_id = model_id
        self.domain = domain
        self.device = device
        print(f"[load] {model_id} as {domain}")
        self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.backbone.to(device).to(torch.bfloat16)
        self.hidden = guess_hidden(model_id, self.backbone)
        self.head = nn.Sequential(
            nn.Linear(self.hidden, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Prediction probabilities ∈ [0,1] for each sample."""
        self.backbone.eval(); self.head.eval()
        probs = []
        for i in range(0, input_ids.shape[0], 64):
            inp = input_ids[i:i+64].to(self.device)
            msk = attention_mask[i:i+64].to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = self.backbone(input_ids=inp, attention_mask=msk).last_hidden_state
            probs.append(torch.sigmoid(self.head(h)).cpu())
        return torch.cat(probs)

    def train_mode(self): self.backbone.train(); self.head.train(); return self
    def eval_mode(self): self.backbone.eval(); self.head.eval(); return self

    def parameters(self):
        from itertools import chain
        return chain(self.backbone.parameters(), self.head.parameters())

    def save_head(self, path: Path):
        torch.save({k: v.cpu().clone() for k, v in self.head.state_dict().items()}, path)

    def load_head(self, path: Path):
        self.head.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return self


# ===========================================================================
# Head training (Phase 0: train if no checkpoint)
# ===========================================================================

def train_head(agent: FAgent, domain: str, epochs: int = 1, batch: int = 128):
    """Train PRM head on domain data. Backbone frozen."""
    from torch.utils.data import DataLoader, TensorDataset

    ckpt = OUT_DIR / f"head_{agent.model_id.replace('/', '__')}_{domain}.pt"
    if ckpt.exists():
        print(f"[head] loading cached {ckpt}")
        return agent.load_head(ckpt)

    data = load_data(domain, agent.model_id)
    device = agent.device
    ds = TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)

    agent.backbone.eval()
    agent.head.train()
    # ponytail: frozen backbone with no_grad (Bug 3 fix)
    opt = torch.optim.AdamW(agent.head.parameters(), lr=1e-4, weight_decay=0.01)
    t0 = time.time()
    n_batches = len(loader)

    for ep in range(epochs):
        for bi, (inp, msk, labs) in enumerate(loader):
            inp, msk = inp.to(device), msk.to(device)
            labs = labs.to(device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h = agent.backbone(input_ids=inp, attention_mask=msk).last_hidden_state
            # ponytail: .float() for bf16→fp32 (Bug 4 fix)
            logits = agent.head(h.float())
            loss = LOSS_FN(logits, labs)
            loss.backward()
            opt.step()
            if (bi + 1) % 50 == 0:
                print(f"  head-train ep{ep+1} {bi+1}/{n_batches} loss={loss.item():.4f}")

    agent.save_head(ckpt)
    agent.eval_mode()
    print(f"[head] trained {domain} in {(time.time()-t0)/60:.0f}m -> {ckpt}")
    return agent


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(agent: FAgent, domain: str, batch: int = 256) -> tuple[float, float]:
    """Accuracy on last 15% of domain data."""
    data = load_data(domain, agent.model_id)
    n_test = max(data["n"] // 6, 100)
    start = data["n"] - n_test
    device = agent.device
    agent.eval_mode()

    inp = data["input_ids"][start:].to(device)
    msk = data["attention_mask"][start:].to(device)
    labs = data["labels"][start:].to(device, dtype=torch.float32)

    total_loss, correct, total = 0.0, 0, 0
    for i in range(0, n_test, batch):
        end = min(i + batch, n_test)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp[i:end],
                               attention_mask=msk[i:end]).last_hidden_state
        logits = agent.head(h)
        # ponytail: reduction='sum' avoids batch-size-dependent mean scaling
        total_loss += F.binary_cross_entropy_with_logits(
            logits, labs[i:end], reduction='sum').item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labs[i:end]).sum().item()
        total += end - i

    return total_loss / total, correct / total


# ===========================================================================
# Reference set (shared, unlabeled)
# ===========================================================================

def build_reference_set(n: int = N_REF) -> tuple[torch.Tensor, torch.Tensor]:
    """Build unlabeled reference set from both domains. Tokenizer-agnostic."""
    ref_cache = OUT_DIR / "reference_set.pt"
    if ref_cache.exists():
        data = torch.load(ref_cache, map_location="cpu", weights_only=True)
        return data["input_ids"], data["attention_mask"]

    # ponytail: use Pythia tokenizer for reference set (it's the student)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from datasets import load_dataset
    ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")

    ids_list = []
    for row in ds:
        cat = (row.get("category") or "").lower()
        if CATMAP.get(cat, "general") not in ("code", "math"):
            continue
        try:
            steps = eval(str(row.get("chain_of_thoughts", "[]")))
        except (SyntaxError, ValueError):
            continue
        for s in steps[:2]:  # ponytail: first 2 steps only, avoid token bloat
            ids = tok.encode(f"{row.get('question', '')}\n{s}")
            if len(ids) <= MAX_LEN:
                ids_list.append(ids)
        if len(ids_list) >= n:
            break

    inp = torch.full((n, MAX_LEN), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros(n, MAX_LEN, dtype=torch.bool)
    for i, ids in enumerate(ids_list[:n]):
        L = min(len(ids), MAX_LEN)
        inp[i, :L] = torch.tensor(ids[:L])
        mask[i, :L] = True

    torch.save({"input_ids": inp, "attention_mask": mask}, ref_cache)
    print(f"[ref] {n} samples cached")
    return inp, mask


# ===========================================================================
# Function-space negotiation
# ===========================================================================

def negotiate_fspace(agent_a: FAgent, agent_b: FAgent,
                     X_ref: torch.Tensor, mask_ref: torch.Tensor,
                     tau: float = 0.5, lr: float = 0.1,
                     max_rounds: int = 30, tol: float = 1e-4) -> tuple[torch.Tensor, list[float]]:
    """Function-space negotiation: V_{t+1} = V_t + M_A⊙d_B + M_B⊙d_A."""
    N = X_ref.shape[0]
    V = torch.full((N,), 0.5)
    trajectory = []

    for t in range(max_rounds):
        P_A = agent_a.predict(X_ref, mask_ref)
        P_B = agent_b.predict(X_ref, mask_ref)

        d_A = -lr * (V - P_A)
        d_B = -lr * (V - P_B)

        imp_A = (P_A - 0.5).abs() * 2
        imp_B = (P_B - 0.5).abs() * 2
        M_A = (1 - imp_A / max(tau, 1e-8)).clamp(0, 1)
        M_B = (1 - imp_B / max(tau, 1e-8)).clamp(0, 1)

        V_prev = V.clone()
        V = (V + M_A * d_B + M_B * d_A).clamp(0.001, 0.999)
        delta = (V - V_prev).abs().mean().item()
        trajectory.append(delta)

        if (t + 1) % 10 == 0 or t == 0 or delta < tol:
            print(f"  round {t+1:3d}  ΔV={delta:.6f}  "
                  f"|P_A-V|={F.l1_loss(P_A, V).item():.4f}  "
                  f"|P_B-V|={F.l1_loss(P_B, V).item():.4f}")
        if delta < tol:
            print(f"  converged at round {t+1}")
            break

    return V, trajectory


# ===========================================================================
# Distillation
# ===========================================================================

def distill(agent: FAgent, V_T: torch.Tensor, X_ref: torch.Tensor,
            mask_ref: torch.Tensor, own_domain: str,
            alpha: float = 0.5, steps: int = 100) -> FAgent:
    """Distill V_T into model: Loss = α·BCE(P_model, V_T) + (1-α)·BCE(P_model, own_labels)."""
    own_data = load_data(own_domain, agent.model_id)
    device = agent.device
    agent.train_mode()

    n_own = min(N_REF, own_data["n"])
    own_idx = torch.randperm(own_data["n"])[:n_own]
    own_inp, own_msk = own_data["input_ids"][own_idx], own_data["attention_mask"][own_idx]
    own_labs = own_data["labels"][own_idx]

    opt = torch.optim.AdamW(agent.parameters(), lr=1e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")

    for _ in range(steps):
        ri = torch.randperm(N_REF)[:32]
        inp_r, msk_r = X_ref[ri].to(device), mask_ref[ri].to(device)
        v_r = V_T[ri].to(device)

        oi = torch.randperm(n_own)[:32]
        inp_o = own_inp[oi].to(device); msk_o = own_msk[oi].to(device)
        labs_o = own_labs[oi].to(device, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h_r = agent.backbone(input_ids=inp_r, attention_mask=msk_r).last_hidden_state
            loss_ref = F.binary_cross_entropy_with_logits(agent.head(h_r), v_r.float())
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h_o = agent.backbone(input_ids=inp_o, attention_mask=msk_o).last_hidden_state
            loss_own = LOSS_FN(agent.head(h_o), labs_o)

        scaler.scale(alpha * loss_ref + (1 - alpha) * loss_own).backward()
        scaler.step(opt)
        scaler.update()

    agent.eval_mode()
    return agent


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="F-IVN Phase 0 — cross-architecture")
    p.add_argument("--teacher", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--student", default="EleutherAI/pythia-1.4b")
    p.add_argument("--teacher-domain", default="math")
    p.add_argument("--student-domain", default="code")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--max-rounds", type=int, default=30)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dom_t, dom_s = args.teacher_domain, args.student_domain
    print(f"=== F-IVN: {args.teacher} ⇄ {args.student} ===\n")

    # ---- Load + train heads ----
    agent_a = FAgent(args.student, dom_s, device)
    agent_b = FAgent(args.teacher, dom_t, device)
    agent_a = train_head(agent_a, dom_s)
    agent_b = train_head(agent_b, dom_t)

    # ---- Reference set ----
    X_ref, mask_ref = build_reference_set(N_REF)

    # ---- Baseline ----
    _, acc_aa = evaluate(agent_a, dom_s); _, acc_ab = evaluate(agent_a, dom_t)
    _, acc_bb = evaluate(agent_b, dom_t); _, acc_ba = evaluate(agent_b, dom_s)
    print(f"\n--- Baseline ---")
    print(f"  Student→{dom_s}: {acc_aa:.4f}  Student→{dom_t}: {acc_ab:.4f}")
    print(f"  Teacher→{dom_t}: {acc_bb:.4f}  Teacher→{dom_s}: {acc_ba:.4f}")

    # ---- F-IVN ----
    print(f"\n--- F-IVN negotiation (τ={args.tau}) ---")
    t0 = time.time()
    V_T, trajectory = negotiate_fspace(
        agent_a, agent_b, X_ref, mask_ref,
        tau=args.tau, lr=args.lr, max_rounds=args.max_rounds)

    # ---- Distill ----
    print(f"\n--- Distilling V_T into student ---")
    student_ivn = distill(
        FAgent(args.student, dom_s, device).load_head(
            OUT_DIR / f"head_{args.student.replace('/', '__')}_{dom_s}.pt"),
        V_T, X_ref, mask_ref, dom_s)
    _, fivn_aa = evaluate(student_ivn, dom_s); _, fivn_ab = evaluate(student_ivn, dom_t)

    print(f"\n--- Distilling V_T into teacher ---")
    teacher_ivn = distill(
        FAgent(args.teacher, dom_t, device).load_head(
            OUT_DIR / f"head_{args.teacher.replace('/', '__')}_{dom_t}.pt"),
        V_T, X_ref, mask_ref, dom_t)
    _, fivn_bb = evaluate(teacher_ivn, dom_t); _, fivn_ba = evaluate(teacher_ivn, dom_s)

    d_s = fivn_aa - acc_aa; d_c = fivn_ab - acc_ab
    d_t = fivn_bb - acc_bb; d_tc = fivn_ba - acc_ba
    net = d_s + d_c + d_t + d_tc

    # ---- Random baseline ----
    V_rand = torch.rand(N_REF)
    student_rand = distill(
        FAgent(args.student, dom_s, device).load_head(
            OUT_DIR / f"head_{args.student.replace('/', '__')}_{dom_s}.pt"),
        V_rand, X_ref, mask_ref, dom_s)
    _, rand_aa = evaluate(student_rand, dom_s); _, rand_ab = evaluate(student_rand, dom_t)

    # ---- Report ----
    print(f"\n{'='*60}")
    print(f"F-IVN: {args.student} ⇄ {args.teacher}  ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    print(f"{'Method':<18} {'S self':>8} {'S cross':>8} {'T self':>8} {'T cross':>8} {'Δ net':>8}")
    print(f"{'-'*58}")
    print(f"{'No Exchange':<18} {acc_aa:8.4f} {acc_ab:8.4f} {acc_bb:8.4f} {acc_ba:8.4f} {'—':>8}")
    print(f"{'F-IVN':<18} {fivn_aa:8.4f} {fivn_ab:8.4f} {fivn_bb:8.4f} {fivn_ba:8.4f} {net:+8.4f}")
    print(f"{'Random distill':<18} {rand_aa:8.4f} {rand_ab:8.4f} {'—':>8} {'—':>8} {'—':>8}")
    print(f"\n  Negotiation: {len(trajectory)} rounds, final ΔV={trajectory[-1]:.6f}")

    result = {
        "teacher": args.teacher, "student": args.student,
        "teacher_domain": dom_t, "student_domain": dom_s,
        "no_exchange": {"s_self": acc_aa, "s_cross": acc_ab, "t_self": acc_bb, "t_cross": acc_ba},
        "fivn": {"s_self": fivn_aa, "s_cross": fivn_ab, "t_self": fivn_bb, "t_cross": fivn_ba,
                 "d_s": d_s, "d_c": d_c, "d_t": d_t, "d_tc": d_tc, "net": net,
                 "rounds": len(trajectory), "trajectory": trajectory},
        "random_distill": {"s_self": rand_aa, "s_cross": rand_ab},
    }
    json.dump(result, open(OUT_DIR / "fivn_results.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_DIR / 'fivn_results.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        crash = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "error_type": type(sys.exc_info()[1]).__name__,
                 "error_message": str(sys.exc_info()[1]),
                 "traceback": traceback.format_exc()}
        crash_dir = OUT_DIR / "crashes"
        crash_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        json.dump(crash, open(crash_dir / f"crash_{ts}.json", "w"), indent=2)
        print(f"\nCRASH -> {crash_dir / f'crash_{ts}.json'}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
