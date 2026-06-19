"""AFP Agent — Phase 0.

Wraps a Pythia-1.4B backbone + PRM head. Handles:
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

from .trust import block_importance
from .integrator import afp_integrate, fedavg

HIDDEN = 2048


class PRMHead(nn.Module):
    """2048 → 256 → 1, ReLU. Handles bf16→fp32 internally."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, h):
        # ponytail: .float() fixes Bug 4 (bf16→fp32 dtype mismatch)
        return self.net(h[:, -1, :].float() if h.dim() == 3 else h.float()).squeeze(-1)


class AFPAgent:
    """One AFP agent: backbone + PRM head + importance profile."""

    def __init__(self, domain: str, device: str = "cuda"):
        self.domain = domain
        self.device = device
        self.backbone = AutoModel.from_pretrained(
            "EleutherAI/pythia-1.4b").to(dtype=torch.bfloat16)
        self.head = PRMHead()
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

    def head_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach() for k, v in self.head.state_dict().items()}

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
        """Compute per-block importance. Runs on GPU, no copies."""
        trained = {k: v.detach() for k, v in self.backbone.state_dict().items()}
        # Move init to same device as trained for on-device computation
        device = next(iter(trained.values())).device
        init_on_device = {k: v.to(device) for k, v in init.items() if k in trained}
        self._importance = block_importance(trained, init_on_device)
        return self._importance

    @property
    def importance(self) -> list[float] | None:
        return self._importance

    # ---- Integration ----

    def integrate_afp(self, peer_state: dict, init: dict,
                      tau: float = 0.5) -> dict:
        """AFP selective update. Phase 0: trust=1.0 (both agents honest)."""
        if self._importance is None:
            self.compute_importance(init)
        # ponytail: trust=1.0 in Phase 0, skip compute_trust() scan.
        # compute_trust() used in Experiment C (robustness).
        return afp_integrate(self.backbone_state(), peer_state,
                             self._importance, trust=1.0, tau=tau)

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
