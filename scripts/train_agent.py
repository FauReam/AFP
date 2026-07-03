#!/usr/bin/env python3
"""Train AFPAgent (Pythia-1.4B + PRM head) full-FT on a single domain.

Phase 0 model preparation: trains two Pythia-1.4B agents — one on code,
one on medical — to create domain-specialized peers with orthogonal
importance patterns for IVN experiments.

Training recipe (adapted from CLAUDE.md v2):
  full-FT all backbone + PRM head, batch=1024, cosine LR 1e-4→3e-6,
  AdamW (β=0.9, 0.999), weight_decay=0.1, bf16 autocast + fp32 head,
  max_len=384, 3 epochs, early stopping on val loss.

Usage:
  python scripts/train_agent.py --domain code
  python scripts/train_agent.py --domain medical
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))
from AFP.protocol import AFPAgent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_LEN = 256          # 256 keeps ~86% of data; reduces FLOPs ~33% vs 384
BATCH_SIZE = 128
LR_MAX = 1e-4          # Claude.md v2 spec
LR_MIN = 3e-6          # Claude.md v2 spec
WEIGHT_DECAY = 0.1
EPOCHS = 1             # Claude.md v2 spec
VAL_SPLIT = 0.15
PATIENCE = 5  # early stopping (epochs without improvement)
MODEL_ID = "EleutherAI/pythia-1.4b"
OUT_DIR = PROJECT / "experiments" / "trained_models"


# ---------------------------------------------------------------------------
# Data — tokenize VersaPRM, filter by length, split train/val, cache
# ---------------------------------------------------------------------------

def prepare_data(domain: str, tokenizer, max_samples: int = 0) -> tuple[dict, dict]:
    """Tokenize VersaPRM for domain → train + val dicts. Cache to disk.

    Cache filename includes MAX_LEN so different sequence-length configs
    don't silently reuse stale caches.
    """
    cache_train = PROJECT / "data" / "versaprm" / f"train_{domain}_pythia_L{MAX_LEN}.pt"
    cache_val = PROJECT / "data" / "versaprm" / f"val_{domain}_pythia_L{MAX_LEN}.pt"

    if cache_train.exists() and cache_val.exists():
        print(f"[data] loading cached: {cache_train}")
        print(f"[data] loading cached: {cache_val}")
        train = torch.load(cache_train, map_location="cpu", weights_only=True)
        val = torch.load(cache_val, map_location="cpu", weights_only=True)
        return train, val

    jsonl = PROJECT / "data" / "versaprm" / "versa_prm.jsonl"
    input_ids_list, labels_list = [], []
    n_samples, n_filtered = 0, 0
    lens = []

    print(f"[data] tokenizing {domain}...")
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            if d.get("domain") != domain:
                continue
            n_samples += 1
            question = d.get("question", "")
            for step, label in zip(d["steps"], d["labels"]):
                text = f"{question}\n{step}"
                ids = tokenizer.encode(text)
                if len(ids) <= MAX_LEN:
                    input_ids_list.append(ids)
                    # VersaPRM labels are -1/1 → convert to 0/1 for BCEWithLogitsLoss
                    labels_list.append(1.0 if int(label) == 1 else 0.0)
                    lens.append(len(ids))
                else:
                    n_filtered += 1

    n = len(input_ids_list)
    print(f"[data] {domain}: {n} valid steps from {n_samples} samples "
          f"(filtered {n_filtered} > {MAX_LEN} tok)")

    if lens:
        lens.sort()
        for p in [10, 50, 90, 95, 99]:
            print(f"[data]   p{p}: {lens[n * p // 100]} tok")

    # Build tensors
    pad_id = tokenizer.pad_token_id or 0
    inp = torch.full((n, MAX_LEN), pad_id, dtype=torch.long)
    mask = torch.zeros(n, MAX_LEN, dtype=torch.bool)
    for i, ids in enumerate(input_ids_list):
        L = len(ids)
        inp[i, :L] = torch.tensor(ids[:L])
        mask[i, :L] = True
    labs = torch.tensor(labels_list, dtype=torch.float32)

    # Shuffle + split 85/15
    perm = torch.randperm(n)
    n_val = int(n * VAL_SPLIT)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train = {"input_ids": inp[train_idx], "attention_mask": mask[train_idx],
             "labels": labs[train_idx], "n": len(train_idx)}
    val = {"input_ids": inp[val_idx], "attention_mask": mask[val_idx],
           "labels": labs[val_idx], "n": len(val_idx)}

    # Cache
    cache_train.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train, cache_train)
    torch.save(val, cache_val)
    print(f"[data] train: {train['n']}, val: {val['n']}")

    # Optional: subsample training data for faster iteration
    if max_samples and max_samples < train["n"]:
        idx = torch.randperm(train["n"])[:max_samples]
        for key in ("input_ids", "attention_mask", "labels"):
            train[key] = train[key][idx]
        train["n"] = max_samples
        print(f"[data] subsampled train -> {max_samples}")

    return train, val


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(agent: AFPAgent, val_data: dict, batch_size: int, device: torch.device) -> float:
    """Compute val loss (no metric yet — BCE loss is fine for PRM training)."""
    agent.eval_mode()
    inp = val_data["input_ids"]
    msk = val_data["attention_mask"]
    labs = val_data["labels"]
    n = val_data["n"]

    total_loss, total = 0.0, 0
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp[i:end].to(device),
                               attention_mask=msk[i:end].to(device)).last_hidden_state
            logits = agent.head(h)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labs[i:end].to(device), reduction="sum")
        total_loss += loss.item()
        total += end - i

    agent.train_mode()
    # No gradient checkpointing: batch=1024 full-FT Pythia-1.4B uses ~66GB/121GB.
# Checkpointing causes pathological slowdown on ARM64 CUDA 13.0 first pass.
    return total_loss / total


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Free perf on fixed-shape inputs: cuDNN auto-tuner finds best kernel.
    # TF32 tensor cores on Blackwell SM 12.1 are faster than FP32 fallback.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"=== Train AFPAgent: {MODEL_ID} on {args.domain} ===\n")

    # Tokenizer (Pythia = GPT-NeoX)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Data
    t0_data = time.time()
    train_data, val_data = prepare_data(args.domain, tok, max_samples=args.max_samples)
    print(f"[data] ready ({time.time() - t0_data:.0f}s)\n")

    # Agent
    agent = AFPAgent(args.domain, str(device), model_id=MODEL_ID)
    agent.to_device()
    agent.train_mode()
    # No gradient checkpointing: batch=1024 full-FT Pythia-1.4B uses ~66GB/121GB.
# Checkpointing causes pathological slowdown on ARM64 CUDA 13.0 first pass.
    n_params = sum(p.numel() for p in agent.parameters())
    torch.cuda.empty_cache()
    mem_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    mem_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    print(f"[model] {n_params:,} params ({n_params * 2 / 1e9:.1f} GB bf16)")
    print(f"[gpu]   allocated={mem_used:.1f} GB, reserved={mem_reserved:.1f} GB\n")

    # DataLoaders
    train_ds = TensorDataset(
        train_data["input_ids"], train_data["attention_mask"], train_data["labels"])
    # No pin_memory: unified memory (GB10) shares CPU/GPU phys RAM,
    # pinning is a no-op that wastes CUDA driver overhead.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=False)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    print(f"[train] {steps_per_epoch} batches/epoch × {args.epochs} epochs "
          f"= {total_steps} steps")

    # Optimizer & scheduler
    params = list(agent.parameters())
    optimizer = AdamW(params, lr=LR_MAX, weight_decay=WEIGHT_DECAY,
                      betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR_MIN)

    # State
    loss_fn = nn.BCEWithLogitsLoss()
    # No GradScaler: bf16 has fp32 dynamic range; GradScaler._unscale_grads_
    # foreach CUDA kernel not implemented for BFloat16 on ARM64 CUDA 13.0.
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        agent.train_mode()
        # No gradient checkpointing: batch=1024 full-FT Pythia-1.4B uses ~66GB/121GB.
# Checkpointing causes pathological slowdown on ARM64 CUDA 13.0 first pass.
        running_loss = 0.0
        t_epoch = time.time()

        for inp, mask, labs in train_dl:
            # No non_blocking: unified memory is zero-copy;
            # async xfer adds stream-sync overhead with no benefit.
            inp = inp.to(device)
            mask = mask.to(device)
            labs = labs.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = agent.backbone(input_ids=inp,
                                   attention_mask=mask).last_hidden_state
                logits = agent.head(h)
                loss = loss_fn(logits, labs)

            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()

            if global_step % 10 == 0:
                elapsed = time.time() - t0
                avg = running_loss / 10
                print(f"  [e{epoch}] step {global_step:4d}/{total_steps} | "
                      f"loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{elapsed:.0f}s")
                running_loss = 0.0

        train_loss = running_loss / max(1, steps_per_epoch % 10)
        train_time = time.time() - t_epoch

        # ---- Validate ----
        val_loss = validate(agent, val_data, args.batch_size, device)
        print(f"  [e{epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"time={train_time:.0f}s | lr={scheduler.get_last_lr()[0]:.2e}")

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best
            save_dir = OUT_DIR / args.domain
            agent.save(save_dir)
            print(f"  [e{epoch}] ✓ best model saved -> {save_dir}")
        else:
            patience_counter += 1
            print(f"  [e{epoch}]   no improvement (patience {patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"  [e{epoch}] early stopping (best val_loss={best_val_loss:.4f} at e{best_epoch})")
                break

        # GPU memory report
        if torch.cuda.is_available():
            used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [e{epoch}] GPU peak: {used:.1f} GB")

    elapsed = time.time() - t0
    print(f"\n[done] {global_step} steps in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  best val_loss={best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  model saved to {OUT_DIR / args.domain}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global MAX_LEN  # allow CLI override before any use in default= or add_argument

    p = argparse.ArgumentParser(
        description="Train AFPAgent full-FT on a domain (Phase 0 prep)")
    p.add_argument("--domain", required=True, choices=["code", "medical"],
                   help="domain to train on")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR_MAX)
    p.add_argument("--max-len", type=int, default=MAX_LEN,
                   help="max token length (affects cache filename)")
    p.add_argument("--max-samples", type=int, default=0,
                   help="limit training samples (0 = use all)")
    args = p.parse_args()
    # Propagate CLI override to module-level constants used by prepare_data
    MAX_LEN = args.max_len

    try:
        return train(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
