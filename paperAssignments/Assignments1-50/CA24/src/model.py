from typing import Sequence

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """A small, configurable MLP used for demonstration and unit tests.

    The model is intentionally simple so it's easy to extend for experiments.
    """

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, input_dim)
        Returns:
            Tensor of shape (batch, output_dim)
        """
        return self.net(x)
