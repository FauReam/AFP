"""AFP integration operators — Phase 0.

FedAvg:  W' = (1-α)W + α·W_other
AFP:     W' = W + M⊙(W_other - W), M[j] = gate(imp[j], τ)

Gate functions: gate_rational (EWC-derived, default), gate_linear (ablation).
"""

from .trust import gate_rational, gate_linear, _block_index

N_BLOCKS = 32  # safe upper bound (Pythia=24, Qwen2.5=28)


def fedavg(w_self: dict, w_other: dict, alpha: float) -> dict:
    """Linear interpolation: W' = (1-α)·W_self + α·W_other."""
    return {k: (1 - alpha) * w_self[k] + alpha * w_other.get(k, w_self[k])
            for k in w_self}


def afp_integrate(w_self: dict, w_other: dict, importance: list[float],
                  trust: float = 1.0, tau: float = 0.5,
                  gate: str = "rational") -> dict:
    """Selective per-block update: W'[j] = W[j] + M[j]·(W_other[j] - W[j]).

    M[j] = trust · gate(imp[j], τ).
    Gate functions (see trust.py):
      "rational" — EWC-derived: τ/(τ+Ω). Smooth, principled (default).
      "linear"   — Legacy: clamp(1-Ω/τ, 0, 1). Kept for ablation.

    High importance → low M → protected. Low importance → high M → open.
    """
    gate_fn = gate_rational if gate == "rational" else gate_linear
    gates = [trust * g for g in gate_fn(importance, tau)]

    result = {}
    for k in w_self:
        blk = _block_index(k)
        if blk is None or k not in w_other:
            result[k] = w_self[k].clone()
            continue
        m = gates[blk]
        result[k] = w_self[k] + m * (w_other[k] - w_self[k])
    return result
