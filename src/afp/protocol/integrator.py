"""AFP integration operators — Phase 0.

FedAvg:  W' = (1-α)W + α·W_other
AFP:     W' = W + M⊙(W_other - W), M[j] = trust · clamp(1 - imp[j]/τ, 0, 1)
"""

N_BLOCKS = 24


def fedavg(w_self: dict, w_other: dict, alpha: float) -> dict:
    """Linear interpolation: W' = (1-α)·W_self + α·W_other."""
    return {k: (1 - alpha) * w_self[k] + alpha * w_other.get(k, w_self[k])
            for k in w_self}


def afp_integrate(w_self: dict, w_other: dict, importance: list[float],
                  trust: float = 1.0, tau: float = 0.5) -> dict:
    """Selective per-block update: W'[j] = W[j] + M[j]·(W_other[j] - W[j]).

    M[j] = trust · clamp(1 - importance[j]/τ, 0, 1)
    High importance → low M → protected. Low importance → high M → open.
    """
    result = {}
    for k in w_self:
        if "layers." not in k or k not in w_other:
            result[k] = w_self[k].clone()
            continue
        try:
            idx = int(k.split("layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            result[k] = w_self[k].clone()
            continue
        if 0 <= idx < N_BLOCKS:
            m = trust * max(0.0, min(1.0, 1.0 - importance[idx] / max(tau, 1e-8)))
            result[k] = w_self[k] + m * (w_other[k] - w_self[k])
        else:
            result[k] = w_self[k].clone()
    return result
