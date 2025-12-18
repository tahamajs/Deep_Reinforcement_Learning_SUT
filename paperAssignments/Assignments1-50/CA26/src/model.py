from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """Simple fully connected MLP for small regression tasks."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation: str = "relu"):
        super().__init__()
        dims = [int(input_dim), *[int(h) for h in hidden_dims], int(output_dim)]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(get_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
