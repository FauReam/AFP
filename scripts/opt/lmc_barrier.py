#!/usr/bin/env python3
"""LMC Barrier Measurement — Phase 0.5

Measure loss barrier along linear interpolation path between two trained models.
W(λ) = (1-λ) · W_A + λ · W_B, λ ∈ [0,1]

Usage:
  python scripts/lmc_barrier.py \
      --model-a experiments/trained_models_opt/code \
      --model-b experiments/trained_models_opt/medical \
      --domain-a code --domain-b medical \
      --base-model facebook/opt-1.3b \
      --n-lambda 41
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from AFP.protocol import AFPAgent, PRMHead

PROJECT = Path(__file__).resolve().parent.parent.parent
MAX_LEN = 384


def load_data(domain: str, model_id: str) -> dict:
    safe_name = model_id.replace("/", "__")
    cache = PROJECT / "data" / "versaprm" / f"versa_prm_{domain}_{safe_name}.pt"
    if cache.exists():
        data = torch.load(cache, map_location="cpu", weights_only=True)
        data["labels"] = (data["labels"] > 0).float()
        return data
    raise FileNotFoundError(f"No cached data for {domain}")


def load_checkpoint(dir_path, domain_hint=None):
    """Load backbone + head from checkpoint dir. Auto-detects filename pattern."""
    path = Path(dir_path)
    # Train saves as W_{domain}_final.pt; epoch ckpt saves as same pattern in subdir
    files = list(path.glob("W_*_final.pt"))
    head_files = list(path.glob("W_*_head_final.pt"))
    if not files:
        raise FileNotFoundError(f"No W_*_final.pt found in {path}")
    w = torch.load(files[0], map_location="cpu", weights_only=True)
    h = torch.load(head_files[0], map_location="cpu", weights_only=True) if head_files else None
    return w, h


@torch.no_grad()
def evaluate_state(state: dict, head_state: dict, domain: str,
                   model_id: str = "facebook/opt-1.3b",
                   batch: int = 256, max_samples: int = 500,
                   device: torch.device | None = None) -> tuple[float, float]:
    """Evaluate a weight state (not an agent) on domain data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(domain, model_id)
    n_test = min(max(data["n"] // 6, 100), max_samples)
    start = data["n"] - n_test

    inp = data["input_ids"][start:].to(device)
    msk = data["attention_mask"][start:].to(device)
    labs = data["labels"][start:].to(device, dtype=torch.float32)

    # Create temporary backbone + head
    backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                          local_files_only=True).to(device).to(torch.bfloat16)
    backbone.load_state_dict(state, strict=False)
    hidden = backbone.config.hidden_size
    head = PRMHead(hidden).to(device).to(torch.bfloat16)
    if head_state:
        head.load_state_dict(head_state)

    backbone.eval(); head.eval()
    total_loss, correct, total = 0.0, 0, 0
    for i in range(0, n_test, batch):
        end = min(i + batch, n_test)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = backbone(input_ids=inp[i:end], attention_mask=msk[i:end]).last_hidden_state
        logits = head(h[:, -1, :].float()).squeeze(-1)
        total_loss += F.binary_cross_entropy_with_logits(
            logits, labs[i:end], reduction='sum').item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labs[i:end]).sum().item()
        total += end - i

    del backbone, head; torch.cuda.empty_cache()
    return total_loss / total, correct / total


def interpolate_weights(w_a: dict, w_b: dict, lam: float) -> dict:
    """W(λ) = (1-λ)·W_A + λ·W_B"""
    return {k: (1 - lam) * w_a[k].float() + lam * w_b[k].float()
            for k in w_a if k in w_b}


def main():
    p = argparse.ArgumentParser(description="LMC Barrier Measurement")
    p.add_argument("--model-a", required=True, help="path to model A checkpoint dir")
    p.add_argument("--model-b", required=True, help="path to model B checkpoint dir")
    p.add_argument("--domain-a", required=True)
    p.add_argument("--domain-b", required=True)
    p.add_argument("--base-model", default="facebook/opt-1.3b")
    p.add_argument("--n-lambda", type=int, default=41, help="interpolation points")
    p.add_argument("--max-eval-samples", type=int, default=500)
    p.add_argument("--output", default=None, help="output JSON path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== LMC Barrier: {args.model_a} ⇄ {args.model_b} ===\n")

    # Load weights
    print("[1] Loading models ...")
    t0 = time.time()
    w_a, head_a = load_checkpoint(args.model_a)
    w_b, head_b = load_checkpoint(args.model_b)
    print(f"    done ({time.time()-t0:.0f}s)")

    # Evaluate endpoints
    print("[2] Evaluating endpoints ...")
    t0 = time.time()
    loss_a_on_a, acc_a_on_a = evaluate_state(w_a, None, args.domain_a, args.base_model, max_samples=args.max_eval_samples)
    loss_a_on_b, acc_a_on_b = evaluate_state(w_a, None, args.domain_b, args.base_model, max_samples=args.max_eval_samples)
    loss_b_on_b, acc_b_on_b = evaluate_state(w_b, None, args.domain_b, args.base_model, max_samples=args.max_eval_samples)
    loss_b_on_a, acc_b_on_a = evaluate_state(w_b, None, args.domain_a, args.base_model, max_samples=args.max_eval_samples)
    print(f"    A on {args.domain_a}: loss={loss_a_on_a:.4f} acc={acc_a_on_a:.4f}")
    print(f"    A on {args.domain_b}: loss={loss_a_on_b:.4f} acc={acc_a_on_b:.4f}")
    print(f"    B on {args.domain_b}: loss={loss_b_on_b:.4f} acc={acc_b_on_b:.4f}")
    print(f"    B on {args.domain_a}: loss={loss_b_on_a:.4f} acc={acc_b_on_a:.4f}")
    print(f"    done ({time.time()-t0:.0f}s)")

    # Interpolation scan
    print(f"[3] Scanning {args.n_lambda} interpolation points ...")
    t0 = time.time()
    results = []

    for i, lam in enumerate(torch.linspace(0, 1, args.n_lambda).tolist()):
        w_interp = interpolate_weights(w_a, w_b, lam)
        loss_a, acc_a = evaluate_state(w_interp, None, args.domain_a, args.base_model, max_samples=args.max_eval_samples)
        loss_b, acc_b = evaluate_state(w_interp, None, args.domain_b, args.base_model, max_samples=args.max_eval_samples)
        results.append({
            "lambda": lam,
            "loss_A": loss_a, "acc_A": acc_a,
            "loss_B": loss_b, "acc_B": acc_b,
        })
        print(f"    λ={lam:.3f}  loss_A={loss_a:.4f}  loss_B={loss_b:.4f}  "
              f"acc_A={acc_a:.4f}  acc_B={acc_b:.4f}")

    # Compute barrier (Frankle definition)
    loss_a0 = results[0]["loss_A"];  loss_a1 = results[-1]["loss_A"]
    loss_b0 = results[0]["loss_B"];  loss_b1 = results[-1]["loss_B"]
    max_loss_a = max(r["loss_A"] for r in results)
    max_loss_b = max(r["loss_B"] for r in results)
    barrier_a = max_loss_a - (loss_a0 + loss_a1) / 2
    barrier_b = max_loss_b - (loss_b0 + loss_b1) / 2

    print(f"\n=== RESULTS ({time.time()-t0:.0f}s) ===")
    print(f"  Barrier (domain {args.domain_a}): {barrier_a:.6f}  (max={max_loss_a:.4f}, L(0)={loss_a0:.4f}, L(1)={loss_a1:.4f})")
    print(f"  Barrier (domain {args.domain_b}): {barrier_b:.6f}  (max={max_loss_b:.4f}, L(0)={loss_b0:.4f}, L(1)={loss_b1:.4f})")

    # Functional divergence
    gap_a = abs(acc_a_on_a - acc_b_on_a)
    gap_b = abs(acc_b_on_b - acc_a_on_b)
    div = (gap_a + gap_b) / 2
    print(f"  Functional divergence: {div:.4f}")

    output = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "domain_a": args.domain_a,
        "domain_b": args.domain_b,
        "barrier_A": barrier_a,
        "barrier_B": barrier_b,
        "divergence": div,
        "endpoints": {
            "A_on_A": {"loss": loss_a_on_a, "acc": acc_a_on_a},
            "A_on_B": {"loss": loss_a_on_b, "acc": acc_a_on_b},
            "B_on_B": {"loss": loss_b_on_b, "acc": acc_b_on_b},
            "B_on_A": {"loss": loss_b_on_a, "acc": acc_b_on_a},
        },
        "interpolation": results,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(output, open(args.output, "w"), indent=2)
        print(f"  Saved -> {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
