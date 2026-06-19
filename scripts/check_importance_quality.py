#!/usr/bin/env python3
"""Check whether per-block importance has genuine structural variation.

Answers: is the max-normalization inflating noise, or is there real
per-block sensitivity variation?

Usage:
  python scripts/check_importance_quality.py
  python scripts/check_importance_quality.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --domain code --base Qwen/Qwen2.5-1.5B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from afp.protocol import block_importance, guess_hidden
# ponytail: import helpers from the IVN script via exec (scripts aren't packages)
import importlib.util as _iu
_ivn_path = PROJECT / "scripts" / "run_ivn_phase0.py"
_ivn_spec = _iu.spec_from_file_location("run_ivn_phase0", _ivn_path)
_ivn = _iu.module_from_spec(_ivn_spec)
_ivn_spec.loader.exec_module(_ivn)
load_data = _ivn.load_data
load_agent = _ivn.load_agent
load_base_weights = _ivn.load_base_weights

LOSS_FN = nn.BCEWithLogitsLoss()
MAX_LEN = 384


def main():
    p = argparse.ArgumentParser(description="Check importance quality")
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--domain", default="code")
    p.add_argument("--base", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--mas-samples", type=int, default=500)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Importance Quality Check: {args.model} ({args.domain}) ===\n")

    # Load model
    agent = load_agent(args.model, args.domain, str(device))

    # 1. Magnitude importance
    base_sd = load_base_weights(args.base, device)
    trained = {k: v.detach() for k, v in agent.backbone.state_dict().items()}
    base_on_dev = {k: v.to(device) for k, v in base_sd.items() if k in trained}
    imp_mag_raw = block_importance(trained, base_on_dev)

    # 2. MAS importance (raw, before normalization)
    data = load_data(args.domain, agent.model_id)
    # Call compute_mas_importance which normalizes internally —
    # we need raw values. Re-implement inline.
    agent.to_device().eval_mode()
    n = min(args.mas_samples, data["input_ids"].shape[0])
    idx = torch.randperm(data["input_ids"].shape[0])[:n]

    raw_imp = [0.0] * 24
    raw_cnt = [0] * 24
    batch_size = 16

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        bi = idx[i:end]
        inp = data["input_ids"][bi].to(device)
        msk = data["attention_mask"][bi].to(device)

        agent.backbone.zero_grad(set_to_none=True)
        agent.head.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = agent.backbone(input_ids=inp, attention_mask=msk).last_hidden_state
        logits = agent.head(h)
        s = (logits ** 2).sum()
        s.backward()

        for name, param in agent.backbone.named_parameters():
            if param.grad is None:
                continue
            if "layers." not in name:
                continue
            try:
                blk = int(name.split("layers.")[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if 0 <= blk < 24:
                raw_imp[blk] += param.grad.float().abs().mean().item()
                raw_cnt[blk] += 1

    imp_mas_raw = [raw_imp[j] / max(raw_cnt[j], 1) for j in range(24)]

    # ---- Report ----
    print(f"{'Block':<8} {'Mag (norm)':>12} {'MAS (raw)':>12} {'MAS (norm)':>12}")
    print(f"{'-'*48}")

    mas_max = max(imp_mas_raw)
    mag_max = max(imp_mag_raw)

    for j in range(24):
        mas_norm = imp_mas_raw[j] / mas_max if mas_max > 0 else 0
        mag_norm = imp_mag_raw[j] / mag_max if mag_max > 0 else 0
        bar_mas = "▓" * int(mas_norm * 20) + "░" * (20 - int(mas_norm * 20))
        bar_mag = "▓" * int(mag_norm * 20) + "░" * (20 - int(mag_norm * 20))
        print(f"  blk{j:2d}  {mag_norm:12.6f}  {imp_mas_raw[j]:12.6e}  {mas_norm:12.6f}  "
              f"M:{bar_mag}  Ω:{bar_mas}")

    # ---- Diagnostic ----
    mas_range = mas_max / (min(imp_mas_raw) + 1e-12)
    mag_range = mag_max / (min(imp_mag_raw) + 1e-12)
    mas_std = torch.tensor(imp_mas_raw).std().item() / (torch.tensor(imp_mas_raw).mean().item() + 1e-12)
    mag_std = torch.tensor(imp_mag_raw).std().item() / (torch.tensor(imp_mag_raw).mean().item() + 1e-12)

    print(f"\n--- Diagnostics ---")
    print(f"  Magnitude:  max/min={mag_range:.1f}x  CV={mag_std:.3f}")
    print(f"  MAS:        max/min={mas_range:.1f}x  CV={mas_std:.3f}")
    print()

    if mas_range < 2.0:
        print("  ⚠️  MAS range < 2x — all blocks near-equally sensitive.")
        print("     Gate signal is noise. Consider per-layer τ or skip importance entirely.")
    elif mas_range < 10:
        print("  ⚠️  MAS range 2-10x — weak but real signal.")
        print("     Gate works but τ is critical.")
    else:
        print("  ✅ MAS range > 10x — genuine structural variation.")
        print("     Gate has real signal. Max normalization is fine.")

    print(f"\n  CV interpretation: > 0.5 = strong block differentiation, < 0.2 = weak.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
