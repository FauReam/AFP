#!/usr/bin/env python3
"""IVN Phase 0: Iterative Virtual-Negotiation Protocol.

Two same-architecture models negotiate over a virtual weight V through
multiple rounds of gradient proposals + per-block gated review.

Default: Qwen2.5-Coder-1.5B ⇄ Qwen2.5-Math-1.5B

Usage:
  python scripts/run_ivn_phase0.py
  python scripts/run_ivn_phase0.py --teacher Qwen/Qwen2.5-Coder-1.5B-Instruct --student Qwen/Qwen2.5-Math-1.5B-Instruct
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LOSS_FN = nn.BCEWithLogitsLoss()

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from AFP.protocol import (AFPAgent, block_importance, gate_linear, gate_rational,
                             guess_hidden, importance_cosine, mas_importance)

OUT_DIR = PROJECT / "experiments" / "phase0_ivn"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_LEN = 384
N_BLOCKS = 32  # safe upper bound (Pythia=24, Qwen2.5=28)

CATMAP = {"computer science": "code", "engineering": "code",
          "health": "medical", "biology": "medical",
          "math": "math", "mathematics": "math"}


# ===========================================================================
# Data — tokenizer-aware, cached per model+domain
# ===========================================================================

def load_data(domain: str, model_id: str) -> dict:
    """Load or download+tokenize VersaPRM data, cached per model+domain."""
    safe_name = model_id.replace("/", "__")
    cache = PROJECT / "data" / "versaprm" / f"versa_prm_{domain}_{safe_name}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True)

    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[data] downloading {domain} for {model_id}...")
    ds = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")

    ids_list, labs_list = [], []
    for row in ds:
        cat = (row.get("category") or "").lower()
        if CATMAP.get(cat, "general") != domain:
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
# Evaluation — tokenizer-aware
# ===========================================================================

@torch.no_grad()
def evaluate(agent: AFPAgent, domain: str, batch: int = 256) -> tuple[float, float]:
    """Accuracy on last 15% of domain data. Uses model's own tokenizer."""
    data = load_data(domain, agent.model_id)
    n_test = max(data["n"] // 6, 100)
    start = data["n"] - n_test
    device = next(agent.backbone.parameters()).device
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
# Model loading
# ===========================================================================


def load_agent(model_id: str, domain: str, device: str = "cuda") -> AFPAgent:
    """Load any HuggingFace model as AFPAgent."""
    print(f"[load] {model_id} as {domain}")
    # ponytail: guess hidden dim from model_id (all Qwen2.5 are 1536, Pythia 2048).
    # Avoid double-loading — AFPAgent.__init__ loads the model.
    hidden = guess_hidden(model_id)
    return AFPAgent(domain, device, model_id=model_id, hidden=hidden)


def load_base_weights(model_id: str, device: torch.device) -> dict:
    """Load base model weights for importance computation."""
    print(f"[base] {model_id}")
    base = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    base.to(device).to(torch.bfloat16)
    sd = {k: v.detach() for k, v in base.state_dict().items()}
    del base; torch.cuda.empty_cache()
    return sd


def compute_importance_from_models(agent: AFPAgent, base_sd: dict) -> list[float]:
    """Per-block importance: compare agent weights to base weights."""
    device = next(agent.backbone.parameters()).device
    trained = {k: v.detach() for k, v in agent.backbone.state_dict().items()}
    base_on_dev = {k: v.to(device) for k, v in base_sd.items() if k in trained}
    return block_importance(trained, base_on_dev)


# ===========================================================================
# Negotiation engine
# ===========================================================================

def negotiate(agent_a: AFPAgent, agent_b: AFPAgent,
              w_init: dict, imp_a: list[float], imp_b: list[float],
              tau: float = 0.5, lr: float = 1e-5,
              max_rounds: int = 30, tol: float = 1e-4,
              gate: str = "rational") -> tuple[dict, list[float]]:
    """Multi-round negotiation: V_{t+1} = V_t + M_A⊙d_B + M_B⊙d_A."""
    gate_fn = gate_rational if gate == "rational" else gate_linear
    gates_a = gate_fn(imp_a, tau)
    gates_b = gate_fn(imp_b, tau)
    device = next(agent_a.backbone.parameters()).device

    V = {k: v.clone().to(device).detach() for k, v in w_init.items()
         if "layers." in k}

    data_a = load_data(agent_a.domain, agent_a.model_id)
    data_b = load_data(agent_b.domain, agent_b.model_id)

    neg_batch = 64
    n_batches = min(data_a["n"], data_b["n"]) // neg_batch
    idx_a = torch.randperm(data_a["n"])[:neg_batch * n_batches]
    idx_b = torch.randperm(data_b["n"])[:neg_batch * n_batches]
    trajectory = []

    for t in range(max_rounds):
        bi = t % n_batches
        ba = idx_a[bi * neg_batch: (bi + 1) * neg_batch]
        bb = idx_b[bi * neg_batch: (bi + 1) * neg_batch]

        d_a = _compute_proposal(agent_a, V, data_a, ba, lr, device)
        d_b = _compute_proposal(agent_b, V, data_b, bb, lr, device)

        V_prev = {k: v.clone() for k, v in V.items()}
        for k in V:
            blk = _block_index(k)
            if blk is None or blk >= N_BLOCKS:
                continue
            ma = gates_a[blk]
            mb = gates_b[blk]
            V[k] = V[k] + ma * d_b[k] + mb * d_a[k]

        delta = math.sqrt(sum((V[k] - V_prev[k]).norm().item() ** 2 for k in V))
        trajectory.append(delta)

        if (t + 1) % 10 == 0 or t == 0 or delta < tol:
            print(f"  round {t+1:3d}  ΔV={delta:.6f}")
        if delta < tol:
            print(f"  converged at round {t+1}")
            break

    return V, trajectory


def _compute_proposal(agent: AFPAgent, V: dict, data: dict,
                      indices: torch.Tensor, lr: float, device: torch.device) -> dict:
    """d = -lr * ∇L(V) for one agent."""
    inp = data["input_ids"][indices].to(device)
    msk = data["attention_mask"][indices].to(device)
    labs = data["labels"][indices].to(device, dtype=torch.float32)

    sd = {k: V[k] for k in V if k in agent.backbone.state_dict()}
    agent.backbone.load_state_dict(sd, strict=False)
    agent.train_mode()
    for pg in agent.backbone.parameters():
        pg.requires_grad = True
    # ponytail: zero grads before backward — load_state_dict doesn't touch .grad
    agent.backbone.zero_grad(set_to_none=True)
    agent.head.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = agent.backbone(input_ids=inp, attention_mask=msk).last_hidden_state
    LOSS_FN(agent.head(h), labs).backward()

    d = {}
    for name, param in agent.backbone.named_parameters():
        if name in V:
            d[name] = -lr * param.grad.detach().clone() if param.grad is not None \
                      else torch.zeros_like(V[name])
    return d


def learn_from_negotiation(agent: AFPAgent, V_final: dict, w_init: dict,
                           imp: list[float], tau: float = 0.5,
                           gate: str = "rational") -> AFPAgent:
    """W' = W + M⊙(V_final - W_init)."""
    gate_fn = gate_rational if gate == "rational" else gate_linear
    gates = gate_fn(imp, tau)
    state = agent.backbone_state()
    device = next(iter(state.values())).device
    for k in state:
        blk = _block_index(k)
        if blk is None or blk >= N_BLOCKS or k not in V_final:
            continue
        m = gates[blk]
        vk = V_final[k].to(device)
        wk = w_init[k].to(device) if k in w_init else state[k]
        state[k] = state[k] + m * (vk - wk)
    agent.load_backbone_state(state)
    return agent


def _block_index(key: str) -> int | None:
    if "layers." not in key:
        return None
    try:
        idx = int(key.split("layers.")[1].split(".")[0])
    except (ValueError, IndexError):
        return None
    return idx if 0 <= idx < N_BLOCKS else None


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="IVN Phase 0")
    p.add_argument("--teacher", type=str,
                   default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--student", type=str,
                   default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--base-model", type=str,
                   default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--teacher-domain", default="code")
    p.add_argument("--student-domain", default="math")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-rounds", type=int, default=30)
    p.add_argument("--importance", choices=["mas", "magnitude"], default="mas",
                   help="MAS (principled, needs data) or magnitude (fast, TIES-style)")
    p.add_argument("--mas-samples", type=int, default=500,
                   help="samples for MAS importance estimate")
    p.add_argument("--gate", choices=["rational", "linear"], default="rational",
                   help="rational (EWC-derived, default) or linear (ablation)")
    args = p.parse_args()

    dom_t, dom_s = args.teacher_domain, args.student_domain
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== IVN Phase 0: {args.teacher} ⇄ {args.student} ===\n")

    # ---- Load models ----
    agent_a = load_agent(args.student, dom_s, str(device))
    agent_b = load_agent(args.teacher, dom_t, str(device))

    # ---- Importance ----
    if args.importance == "mas":
        print(f"[importance] MAS (n={args.mas_samples})")
        data_a = load_data(dom_s, agent_a.model_id)
        data_b = load_data(dom_t, agent_b.model_id)
        imp_a = agent_a.compute_mas_importance(
            data_a["input_ids"], data_a["attention_mask"],
            n_samples=args.mas_samples)
        imp_b = agent_b.compute_mas_importance(
            data_b["input_ids"], data_b["attention_mask"],
            n_samples=args.mas_samples)
    else:
        print("[importance] magnitude (TIES-Merging style)")
        base_sd_imp = load_base_weights(args.base_model, device)
        imp_a = compute_importance_from_models(agent_a, base_sd_imp)
        imp_b = compute_importance_from_models(agent_b, base_sd_imp)
        del base_sd_imp; torch.cuda.empty_cache()

    cos = importance_cosine(imp_a, imp_b)
    print(f"Importance ({args.importance}): {[f'{v:.3f}' for v in imp_a]}")
    print(f"Importance ({args.importance}): {[f'{v:.3f}' for v in imp_b]}")
    print(f"Importance cosine: {cos:.3f} "
          f"({'COMPLEMENTARY' if cos < 0.5 else 'overlapping' if cos < 0.8 else 'too similar'})\n")

    # ---- Base weights (always needed for W_init + AFP baseline + negotiation) ----
    base_sd = load_base_weights(args.base_model, device)

    # ---- Baseline ----
    agent_a.to_device().eval_mode()
    agent_b.to_device().eval_mode()
    _, acc_aa = evaluate(agent_a, dom_s)
    _, acc_ab = evaluate(agent_a, dom_t)
    _, acc_bb = evaluate(agent_b, dom_t)
    _, acc_ba = evaluate(agent_b, dom_s)
    print(f"--- Baseline ---")
    print(f"  A→{dom_s}: {acc_aa:.4f}  A→{dom_t}: {acc_ab:.4f}")
    print(f"  B→{dom_t}: {acc_bb:.4f}  B→{dom_s}: {acc_ba:.4f}")

    orig_a = agent_a.backbone_state()
    orig_b = agent_b.backbone_state()

    # ---- Noise control ----
    # ponytail: rules out "any weight perturbation improves zero-shot accuracy".
    # Compute peer delta magnitude per parameter, inject Gaussian noise of
    # same std, apply same gate. If noise Δ ≈ 0 but IVN Δ > 0 → real signal.
    gate_fn_main = gate_rational if args.gate == "rational" else gate_linear
    gates_a_noise = gate_fn_main(imp_a, args.tau)
    gates_b_noise = gate_fn_main(imp_b, args.tau)
    best_noise = {"net": -999.0}
    noise_scale_a = {}
    noise_scale_b = {}
    for k in orig_a:
        if k in orig_b and k in base_sd:
            da = (orig_a[k] - base_sd[k].to(device)).float()
            db = (orig_b[k] - base_sd[k].to(device)).float()
            noise_scale_a[k] = da.std().item()  # A's delta magnitude per param
            noise_scale_b[k] = db.std().item()
    for _ in range(3):  # 3 random seeds
        agent_a.load_backbone_state(orig_a)
        agent_b.load_backbone_state(orig_b)
        # Inject peer-magnitude noise through gate
        noisy_a = {}
        noisy_b = {}
        for k in orig_a:
            blk = _block_index(k)
            blk_a = blk if blk is not None and blk < len(gates_a_noise) else None
            blk_b = blk if blk is not None and blk < len(gates_b_noise) else None
            if blk_a is not None and k in noise_scale_b and noise_scale_b[k] > 0:
                noise_a = torch.randn_like(orig_a[k]) * noise_scale_b[k]
                noisy_a[k] = orig_a[k] + gates_a_noise[blk_a] * noise_a
            else:
                noisy_a[k] = orig_a[k].clone()
            if blk_b is not None and k in noise_scale_a and noise_scale_a[k] > 0:
                noise_b = torch.randn_like(orig_b[k]) * noise_scale_a[k]
                noisy_b[k] = orig_b[k] + gates_b_noise[blk_b] * noise_b
            else:
                noisy_b[k] = orig_b[k].clone()
        agent_a.load_backbone_state(noisy_a)
        agent_b.load_backbone_state(noisy_b)
        _, s_a = evaluate(agent_a, dom_s); _, c_a = evaluate(agent_a, dom_t)
        _, s_b = evaluate(agent_b, dom_t); _, c_b = evaluate(agent_b, dom_s)
        net = (s_a - acc_aa) + (c_a - acc_ab) + (s_b - acc_bb) + (c_b - acc_ba)
        if net > best_noise["net"]:
            best_noise = {"net": net, "a_self": s_a, "a_cross": c_a,
                          "b_self": s_b, "b_cross": c_b}
    agent_a.load_backbone_state(orig_a)
    agent_b.load_backbone_state(orig_b)

    # ---- FedAvg grid search ----
    best_fed = {"net": -999.0}
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        agent_a.load_backbone_state(orig_a)
        agent_b.load_backbone_state(orig_b)
        fed_a = agent_a.integrate_fedavg(agent_b.backbone_state(), alpha)
        fed_b = agent_b.integrate_fedavg(agent_a.backbone_state(), alpha)
        agent_a.load_backbone_state(fed_a)
        agent_b.load_backbone_state(fed_b)
        _, s_a = evaluate(agent_a, dom_s); _, c_a = evaluate(agent_a, dom_t)
        _, s_b = evaluate(agent_b, dom_t); _, c_b = evaluate(agent_b, dom_s)
        net = (s_a - acc_aa) + (c_a - acc_ab) + (s_b - acc_bb) + (c_b - acc_ba)
        if net > best_fed["net"]:
            best_fed = {"alpha": alpha, "net": net,
                        "a_self": s_a, "a_cross": c_a, "b_self": s_b, "b_cross": c_b}

    # ---- AFP one-shot grid search ----
    best_afp = {"net": -999.0}
    # ponytail: reuse base_sd (loaded above) instead of re-loading base model
    for tau in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        agent_a.load_backbone_state(orig_a)
        agent_b.load_backbone_state(orig_b)
        afp_a = agent_a.integrate_afp(agent_b.backbone_state(), base_sd, tau, gate=args.gate)
        afp_b = agent_b.integrate_afp(agent_a.backbone_state(), base_sd, tau, gate=args.gate)
        agent_a.load_backbone_state(afp_a)
        agent_b.load_backbone_state(afp_b)
        _, s_a = evaluate(agent_a, dom_s); _, c_a = evaluate(agent_a, dom_t)
        _, s_b = evaluate(agent_b, dom_t); _, c_b = evaluate(agent_b, dom_s)
        net = (s_a - acc_aa) + (c_a - acc_ab) + (s_b - acc_bb) + (c_b - acc_ba)
        if net > best_afp["net"]:
            best_afp = {"tau": tau, "net": net,
                        "a_self": s_a, "a_cross": c_a, "b_self": s_b, "b_cross": c_b}

    # ---- IVN ----
    agent_a.load_backbone_state(orig_a)
    agent_b.load_backbone_state(orig_b)
    print(f"--- IVN (τ={args.tau}, rounds≤{args.max_rounds}) ---")
    t0 = time.time()
    V_final, trajectory = negotiate(
        agent_a, agent_b, base_sd, imp_a, imp_b,
        tau=args.tau, lr=args.lr, max_rounds=args.max_rounds, gate=args.gate)

    agent_a.load_backbone_state(orig_a)
    agent_b.load_backbone_state(orig_b)
    agent_a_ivn = learn_from_negotiation(agent_a, V_final, base_sd, imp_a, args.tau, gate=args.gate)
    agent_b_ivn = learn_from_negotiation(agent_b, V_final, base_sd, imp_b, args.tau, gate=args.gate)

    _, ivn_aa = evaluate(agent_a_ivn, dom_s); _, ivn_ab = evaluate(agent_a_ivn, dom_t)
    _, ivn_bb = evaluate(agent_b_ivn, dom_t); _, ivn_ba = evaluate(agent_b_ivn, dom_s)
    ivn_net = (ivn_aa - acc_aa) + (ivn_ab - acc_ab) + (ivn_bb - acc_bb) + (ivn_ba - acc_ba)

    # ---- Report ----
    print(f"\n{'='*60}")
    print(f"FINAL  ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    print(f"{'Method':<16} {'A self':>8} {'A cross':>8} {'B self':>8} {'B cross':>8} {'Δ net':>8}")
    print(f"{'-'*54}")
    print(f"{'No Exchange':<16} {acc_aa:8.4f} {acc_ab:8.4f} {acc_bb:8.4f} {acc_ba:8.4f} {'—':>8}")
    print(f"{'Noise ctl':<16} {best_noise['a_self']:8.4f} {best_noise['a_cross']:8.4f} "
          f"{best_noise['b_self']:8.4f} {best_noise['b_cross']:8.4f} {best_noise['net']:+8.4f}")
    print(f"{'FedAvg':<16} {best_fed['a_self']:8.4f} {best_fed['a_cross']:8.4f} "
          f"{best_fed['b_self']:8.4f} {best_fed['b_cross']:8.4f} {best_fed['net']:+8.4f}")
    print(f"{'AFP 1-shot':<16} {best_afp['a_self']:8.4f} {best_afp['a_cross']:8.4f} "
          f"{best_afp['b_self']:8.4f} {best_afp['b_cross']:8.4f} {best_afp['net']:+8.4f}")
    print(f"{'IVN':<16} {ivn_aa:8.4f} {ivn_ab:8.4f} {ivn_bb:8.4f} {ivn_ba:8.4f} {ivn_net:+8.4f}")
    print(f"\n  Signal check: IVN net {ivn_net:+.4f} vs Noise net {best_noise['net']:+.4f} "
          f"({'SIGNAL' if abs(ivn_net - best_noise['net']) > 0.01 else 'NOISE'})")

    print(f"\n  Convergence: {len(trajectory)} rounds, final ΔV={trajectory[-1]:.6f}")
    for i, d in enumerate(trajectory[:5]):
        print(f"    r{i+1}: ΔV={d:.6f}")
    if len(trajectory) > 5:
        print(f"    ...  r{len(trajectory)}: ΔV={trajectory[-1]:.6f}")

    result = {
        "teacher": args.teacher, "student": args.student,
        "domains": [dom_s, dom_t],
        "importance_cosine": cos,
        "imp_a": imp_a, "imp_b": imp_b,
        "no_exchange": {"a_self": acc_aa, "a_cross": acc_ab, "b_self": acc_bb, "b_cross": acc_ba},
        "noise_control": best_noise,
        "fedavg": best_fed, "afp_oneshot": best_afp,
        "ivn": {"tau": args.tau, "lr": args.lr, "rounds": len(trajectory),
                "trajectory": trajectory,
                "a_self": ivn_aa, "a_cross": ivn_ab,
                "b_self": ivn_bb, "b_cross": ivn_ba, "net": ivn_net},
    }
    json.dump(result, open(OUT_DIR / "ivn_results.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_DIR / 'ivn_results.json'}")
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
