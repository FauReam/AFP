"""
AFP 交互协议核心。

一次交互回合（Agent A 收到 Agent B 的权重 W_B）：

1. Analysis  — 子空间分析：B 的知识主要在哪些参数子空间？
2. Trust     — 信任评估：B 值得信任吗？（trust.py）
3. Decision  — 自主决策：学不学？学什么？学多少？（agent.py）
4. Integrate — 选择性整合：W' = W_A + M⊙(W_B - W_A)（integrator.py）
5. Feedback  — 反馈对话：告诉 B 哪些被接受了（Phase 1+）
"""

import torch.nn as nn

from .agent import AFPAgent, PRMHead
from .integrator import afp_integrate, fedavg
from .trust import block_importance, compute_trust, importance_cosine, mas_importance

# ponytail: one lookup dict, shared across all scripts
_HIDDEN_MAP = {"pythia": 2048, "qwen2": 1536, "qwen2.5": 1536,
               "starcoder2": 2048, "llama": 2048, "tinyllama": 2048}


def guess_hidden(model_id: str, backbone: nn.Module | None = None) -> int:
    """Guess hidden dim from model_id string. Falls back to probing backbone."""
    mid = model_id.lower()
    for key, dim in _HIDDEN_MAP.items():
        if key in mid:
            return dim
    if backbone is not None:
        for m in backbone.modules():
            if isinstance(m, nn.Linear):
                return m.out_features
    return 2048
