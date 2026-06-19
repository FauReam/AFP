"""AFP trust & importance — Phase 0.

Trust: Phase 0 defaults to trust=1.0 (both agents honest).
       compute_trust() reserved for Experiment C (robustness).
Importance: per-block mean absolute deviation from init, computed on GPU.
"""

import torch

N_BLOCKS = 24


def block_importance(trained: dict, init: dict) -> list[float]:
    """Per-block mean(|W - W_init|), normalized [0,1]. GPU-native, no copies."""
    imp = [0.0] * N_BLOCKS
    cnt = [0] * N_BLOCKS
    for k, v in trained.items():
        if "layers." not in k or k not in init:
            continue
        try:
            idx = int(k.split("layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            continue
        if 0 <= idx < N_BLOCKS:
            # ponytail: compute on-device, single .item() per block
            imp[idx] += (v.float() - init[k].float()).abs().mean().item()
            cnt[idx] += 1
    imp = [imp[i] / max(cnt[i], 1) for i in range(N_BLOCKS)]
    mx = max(imp)
    return [v / mx for v in imp] if mx > 0 else imp


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


def importance_cosine(imp_a: list[float], imp_b: list[float]) -> float:
    """Cosine similarity of two importance vectors. 1.0=identical, 0.0=orthogonal."""
    a, b = torch.tensor(imp_a), torch.tensor(imp_b)
    return float((a @ b) / (a.norm() * b.norm() + 1e-8))
