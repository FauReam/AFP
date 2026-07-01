"""AFP trust & importance — Phase 0.

Trust: Phase 0 defaults to trust=1.0 (both agents honest).
       compute_trust() reserved for Experiment C (robustness).

Importance: two methods —
  magnitude (TIES-Merging style): per-block mean |W - W_init|, normalized [0,1].
  MAS (Memory Aware Synapses): per-block E[|∂F(x)²/∂θ|], principled functional importance.

Gate: two forms —
  rational (EWC-derived, default): M[j] = τ / (τ + Ω[j]).
    Derivation: EWC Lagrangian dual → B's gradient scaled by 1/(1+λ·F).
    Replace Fisher F with MAS importance Ω, set τ = 1/λ → M = τ/(τ+Ω).
  linear (legacy, for ablation): M[j] = clamp(1 - Ω[j]/τ, 0, 1).
"""

import torch
import torch.nn.functional as F

N_BLOCKS = 32  # safe upper bound (Pythia=24, Qwen2.5=28)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_index(key: str) -> int | None:
    """Extract block index from parameter name. Returns None if not a layer param."""
    if "layers." not in key:
        return None
    try:
        idx = int(key.split("layers.")[1].split(".")[0])
    except (ValueError, IndexError):
        return None
    return idx if 0 <= idx < N_BLOCKS else None


# ---------------------------------------------------------------------------
# Magnitude-based importance (TIES-Merging style)
# ---------------------------------------------------------------------------

def block_importance(trained: dict, init: dict) -> list[float]:
    """Per-block mean(|W - W_init|), normalized [0,1]. GPU-native, no copies.

    Reference: Yadav et al. (NeurIPS 2023) — TIES-Merging uses |τ| = |θ - θ_init|
    as the signal for trim-elect-sign. This is the same metric, soft-aggregated
    per block and normalized continuously.

    Limitation: measures "how much changed", not "how important for function".
    A block with small |ΔW| may encode critical universal knowledge (e.g. early
    layers) and be under-protected. Use mas_importance() for principled measure.
    """
    imp = [0.0] * N_BLOCKS
    cnt = [0] * N_BLOCKS
    for k, v in trained.items():
        blk = _block_index(k)
        if blk is None or k not in init:
            continue
        # ponytail: compute on-device, single .item() per param tensor
        imp[blk] += (v.float() - init[k].float()).abs().mean().item()
        cnt[blk] += 1
    imp = [imp[i] / max(cnt[i], 1) for i in range(N_BLOCKS)]
    mx = max(imp)
    return [v / mx for v in imp] if mx > 0 else imp


# ---------------------------------------------------------------------------
# MAS-based importance (Memory Aware Synapses)
# ---------------------------------------------------------------------------

@torch.enable_grad()
def mas_importance(backbone, head, input_ids: torch.Tensor,
                   attention_mask: torch.Tensor,
                   n_samples: int = 500, batch_size: int = 16,
                   device: torch.device | None = None) -> list[float]:
    """Per-block MAS importance: E[|∂F(x)²/∂θ|] per block, normalized [0,1].

    Reference: Aljundi et al. (ECCV 2018) — "Memory Aware Synapses: Learning
    what (not) to forget."

    For each parameter p:  Ω_p = E_x[ |∂[F(x)²]/∂θ_p| ]
    For block j:           Ω_j = mean_{p∈block_j} Ω_p

    Interpretation: "if I perturb parameters in block j, how much does my
    model's output change, on average over my data?"

    This directly measures functional importance — what the gate actually
    needs to know. Contrast with magnitude-based which only measures change
    magnitude, not functional impact.

    Args:
        backbone: HuggingFace model (frozen params, grad computed per sample).
        head: PRM head (scalar logit output).
        input_ids, attention_mask: agent's private data (no labels needed).
        n_samples: number of samples to estimate expectation.
        batch_size: samples per backward pass.
        device: if None, inferred from backbone.

    Returns:
        Per-block importance list, length N_BLOCKS, normalized to [0,1].
    """
    if device is None:
        device = next(backbone.parameters()).device

    was_training = backbone.training
    backbone.eval()  # no dropout, no BN update — just gradient computation
    head.eval()

    n = min(n_samples, input_ids.shape[0])
    idx = torch.randperm(input_ids.shape[0])[:n]

    # GPU-side accumulation: one sum + count tensor per block, sync once at end.
    # ponytail: Bug 11 fix — per-parameter .item() caused 10,416 GPU syncs
    # (336 params × 31 batches), each ~1.7s on GB10 → 5+ hours.
    # Now: accumulate on GPU, 1 sync at end → ~1 minute total.
    imp_sum = torch.zeros(N_BLOCKS, device=device, dtype=torch.float64)
    imp_cnt = torch.zeros(N_BLOCKS, device=device, dtype=torch.int32)

    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        bi = idx[i:end]
        inp = input_ids[bi].to(device)
        msk = attention_mask[bi].to(device)

        backbone.zero_grad(set_to_none=True)
        head.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = backbone(input_ids=inp, attention_mask=msk).last_hidden_state
        # ponytail: head does .float() internally (Bug 4 fix), logits are fp32
        logits = head(h)                     # [B]
        s = (logits ** 2).sum()              # scalar: ||F(x)||²
        s.backward()

        for name, param in backbone.named_parameters():
            if param.grad is None:
                continue
            blk = _block_index(name)
            if blk is not None:
                # Accumulate on GPU — no .item() sync
                imp_sum[blk] += param.grad.float().abs().mean()
                imp_cnt[blk] += 1

    if was_training:
        backbone.train()
        head.train()

    # Single GPU→CPU sync — O(N_BLOCKS) instead of O(params × batches)
    imp_cpu = imp_sum.cpu()
    cnt_cpu = imp_cnt.cpu()
    imp = [float(imp_cpu[j]) / max(int(cnt_cpu[j]), 1) for j in range(N_BLOCKS)]
    mx = max(imp)
    return [v / mx for v in imp] if mx > 0 else imp


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------

def gate_rational(importance: list[float], tau: float = 0.5) -> list[float]:
    """EWC-derived rational gate: M[j] = τ / (τ + Ω[j]).

    Derivation from Kirkpatrick et al. (PNAS 2017):
      EWC-regularized loss: L(θ) = L_new(θ) + (λ/2)·Σ F_i·(θ_i - θ*_i)²
      At optimum: ∇L_new = -λ·F⊙(θ-θ*) → θ-θ* = -(1/λ)·F^(-1)⊙∇L_new
      Effective gate on peer's gradient: M_i = 1/(1 + λ·F_i)
      Substitute τ = 1/λ, Ω = F → M[j] = τ/(τ + Ω[j]).

    Properties:
      Ω=0 → M=1 (fully open)
      Ω=τ → M=0.5
      Ω→∞ → M→0 (vanishes, never hard-zero)
      Smooth, differentiable everywhere. No hard clip.

    Args:
        importance: per-block importance, normalized [0,1].
        tau: temperature. Higher → more permissive.
    Returns:
        Per-block gate values in (0, 1].
    """
    eps = max(tau, 1e-8)
    return [eps / (eps + max(v, 0.0)) for v in importance]


def gate_linear(importance: list[float], tau: float = 0.5) -> list[float]:
    """Legacy linear-clamp gate: M[j] = clamp(1 - Ω[j]/τ, 0, 1).

    Simple, interpretable. Kept for ablation comparison against rational gate.
    Behavior: Ω≥τ → M=0 (hard zero — block fully protected).
    """
    eps = max(tau, 1e-8)
    return [max(0.0, min(1.0, 1.0 - v / eps)) for v in importance]


# ---------------------------------------------------------------------------
# Trust
# ---------------------------------------------------------------------------

def compute_trust(w_other: dict, w_init: dict, w_self: dict,
                  lam: float = 1.0) -> float:
    """Trust = exp(-λ · ||W_other - W_init|| / ||W_self - W_init||).

    Used in Experiment C (robustness). Phase 0 defaults to trust=1.0.
    """
    d_other = 0.0
    d_self = 0.0
    for k in w_init:
        if k in w_other and k in w_self:
            d_other += (w_other[k].float() - w_init[k].float()).norm().item() ** 2
            d_self += (w_self[k].float() - w_init[k].float()).norm().item() ** 2
    ratio = (d_other / max(d_self, 1e-12)) ** 0.5
    return float(torch.exp(torch.tensor(-lam * ratio)).item())


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------

def importance_cosine(imp_a: list[float], imp_b: list[float]) -> float:
    """Cosine similarity of two importance vectors. 1.0=identical, 0.0=orthogonal."""
    a, b = torch.tensor(imp_a), torch.tensor(imp_b)
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))
