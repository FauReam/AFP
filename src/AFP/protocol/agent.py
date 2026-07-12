"""AFP Agent — Phase 0.

Wraps any HuggingFace AutoModel + PRM head. Handles:
- Loading/saving weights
- Computing per-block importance
- Integration (AFP / FedAvg)

ponytail: backbone_state() returns GPU tensors (no .cpu() copy).
Phase 0 trusts all peers (trust=1.0, no compute_trust overhead).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

from .trust import block_importance, mas_importance
from .integrator import afp_integrate, fedavg


class PRMHead(nn.Module):
    """hidden → 256 → 1, ReLU. Handles bf16→fp32 internally."""
    def __init__(self, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, h):
        # ponytail: .float() fixes Bug 4 (bf16→fp32 dtype mismatch)
        return self.net(h[:, -1, :].float() if h.dim() == 3 else h.float()).squeeze(-1)


class AFPAgent:
    """One AFP agent: backbone + PRM head + importance profile."""

    def __init__(self, domain: str, device: str = "cuda",
                 model_id: str = "EleutherAI/pythia-1.4b", hidden: int = 0):
        self.domain = domain
        self.device = device
        self.model_id = model_id
        self.backbone = AutoModel.from_pretrained(
            model_id, trust_remote_code=True,
            local_files_only=True).to(dtype=torch.bfloat16)
        # Auto-detect hidden size from model config if not specified
        if hidden <= 0:
            hidden = self.backbone.config.hidden_size
        self.hidden = hidden
        self.head = PRMHead(hidden)
        self._importance: list[float] | None = None

    # ---- Weights I/O ----

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu().clone() for k, v in self.backbone.state_dict().items()},
                   out_dir / f"W_{self.domain}_final.pt")
        torch.save({k: v.cpu().clone() for k, v in self.head.state_dict().items()},
                   out_dir / f"W_{self.domain}_head_final.pt")

    def load(self, out_dir: Path):
        self.backbone.load_state_dict(
            torch.load(out_dir / f"W_{self.domain}_final.pt",
                       map_location="cpu", weights_only=True))
        self.head.load_state_dict(
            torch.load(out_dir / f"W_{self.domain}_head_final.pt",
                       map_location="cpu", weights_only=True))
        return self

    def backbone_state(self) -> dict[str, torch.Tensor]:
        """Returns GPU-resident state dict. No CPU copy — used in grid search loop."""
        return {k: v.detach() for k, v in self.backbone.state_dict().items()}

    def load_backbone_state(self, state: dict):
        self.backbone.load_state_dict(state)

    # ---- Init snapshot ----

    def save_init(self, path: Path):
        if not path.exists():
            torch.save({k: v.cpu().clone() for k, v in self.backbone.state_dict().items()}, path)

    def load_init(self, path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=True)

    # ---- Importance ----

    def compute_importance(self, init: dict) -> list[float]:
        """Magnitude-based per-block importance (TIES-Merging style, L1 mean).

        Simple, fast, no data needed. Good default baseline.
        Prefer compute_importance_l2() for better domain differentiation.
        """
        trained = {k: v.detach() for k, v in self.backbone.state_dict().items()}
        device = next(iter(trained.values())).device
        init_on_device = {k: v.to(device) for k, v in init.items() if k in trained}
        self._importance = block_importance(trained, init_on_device)
        return self._importance

    def compute_importance_l2(self, init: dict) -> list[float]:
        """Relative L2 per-block importance: ||ΔW||/||W_base||, joint-normalized.

        Best domain differentiation (cosine=0.9912). Training-free, data-free.
        """
        trained = {k: v.detach() for k, v in self.backbone.state_dict().items()}
        device = next(iter(trained.values())).device
        init_on_device = {k: v.to(device) for k, v in init.items() if k in trained}
        self._importance = block_importance_l2(trained, init_on_device)
        return self._importance

    def compute_sta_importance(self, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor,
                                delta_w: dict,
                                n_samples: int = 500,
                                batch_size: int = 16) -> list[float]:
        """STA-based per-block importance (Tian et al., arXiv 2411.16139).

        STA = |∂L/∂θ · (θ_trained - θ_base)| — first-order Taylor expansion
        of loss change. Filters out architectural gradient artifacts by
        multiplying by actual weight deltas. Learning rate invariant.

        Args:
            input_ids, attention_mask: agent's private data tensors.
            delta_w: dict of W_trained - W_base per parameter.
        """
        from .trust import sta_importance as _sta
        device = next(self.backbone.parameters()).device
        self._importance = _sta(
            self.backbone, self.head,
            input_ids, attention_mask, delta_w,
            n_samples=n_samples, batch_size=batch_size, device=device)
        return self._importance

    def compute_mas_importance(self, input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               n_samples: int = 500,
                               batch_size: int = 16) -> list[float]:
        """MAS-based per-block importance (Aljundi et al., ECCV 2018).

        Principled: measures functional sensitivity — how much does perturbing
        block j change the model's output? Uses agent's private data (no labels).

        Args:
            input_ids, attention_mask: agent's private data tensors.
            n_samples: number of samples for expectation estimate.
            batch_size: samples per backward pass.
        """
        device = next(self.backbone.parameters()).device
        self._importance = mas_importance(
            self.backbone, self.head,
            input_ids, attention_mask,
            n_samples=n_samples, batch_size=batch_size, device=device)
        return self._importance

    @property
    def importance(self) -> list[float] | None:
        return self._importance

    # ---- Integration ----

    def integrate_afp(self, peer_state: dict, init: dict,
                      tau: float = 0.5, gate: str = "rational") -> dict:
        """AFP selective update. Phase 0: trust=1.0 (both agents honest).

        gate: "rational" (EWC-derived, default) or "linear" (ablation).
        """
        if self._importance is None:
            self.compute_importance(init)
        return afp_integrate(self.backbone_state(), peer_state,
                             self._importance, trust=1.0, tau=tau, gate=gate)

    def integrate_fedavg(self, peer_state: dict, alpha: float) -> dict:
        return fedavg(self.backbone_state(), peer_state, alpha)

    # ---- Training ----

    def train_mode(self):
        self.backbone.train().requires_grad_(True)
        self.head.train()
        return self

    def eval_mode(self):
        self.backbone.eval()
        self.head.eval()
        return self

    def to_device(self):
        self.backbone.to(self.device)
        self.head.to(self.device)
        return self

    def parameters(self):
        from itertools import chain
        return chain(self.backbone.parameters(), self.head.parameters())
