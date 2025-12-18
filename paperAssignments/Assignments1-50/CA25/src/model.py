from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple configurable MLP with ReLU activations and optional output activation."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, output_activation: nn.Module | None = None):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
