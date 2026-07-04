"""Detector network with Monte-Carlo-dropout uncertainty.

A compact MLP over the ~100-dim structure/style/predictability embedding.
Dropout is kept active at inference and the input is passed through the
network many times; the spread of the resulting probabilities is an
uncertainty estimate that drives the abstention ("inconclusive") verdict —
the mechanism that keeps the false-accusation rate low.
"""

from __future__ import annotations

import torch
from torch import nn


class DetectorNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def mc_predict(
    model: DetectorNet,
    x: torch.Tensor,
    temperature: float = 1.0,
    passes: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mean_p_ai, std_p_ai) over `passes` stochastic forward passes.

    x: (n, d) float tensor. Model is put in train() mode so dropout stays on.
    """
    was_training = model.training
    model.train()
    probs = []
    for _ in range(passes):
        logits = model(x) / temperature
        probs.append(torch.softmax(logits, dim=-1)[:, 1])
    model.train(was_training)
    stacked = torch.stack(probs)
    return stacked.mean(dim=0), stacked.std(dim=0)
