from __future__ import annotations
from typing import Optional
import numpy as np


class BaseModel:
    """A minimal model with a numpy-first implementation and an optional Torch backend.

    This keeps the package import-safe when Torch is not installed.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, backend: Optional[str] = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.backend = backend or "numpy"

        if self.backend == "torch":
            try:
                import torch
                import torch.nn as nn

                class MLP(nn.Module):
                    def __init__(self, in_dim, h_dim, out_dim):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(in_dim, h_dim),
                            nn.ReLU(),
                            nn.Linear(h_dim, out_dim),
                        )

                    def forward(self, x):
                        return self.net(x)

                self.model = MLP(input_dim, hidden_dim, output_dim)
                self._forward = self._forward_torch
            except Exception:
                # fallback to numpy implementation
                self.backend = "numpy"
                self._init_numpy_weights()
                self._forward = self._forward_numpy
        else:
            self._init_numpy_weights()
            self._forward = self._forward_numpy

    def _init_numpy_weights(self):
        # deterministic small random weights; real experiments should use better initialization
        rng = np.random.RandomState(0)
        self.W1 = rng.normal(scale=0.1, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((self.hidden_dim,))
        self.W2 = rng.normal(scale=0.1, size=(self.hidden_dim, self.output_dim))
        self.b2 = np.zeros((self.output_dim,))

    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        h = x.dot(self.W1) + self.b1
        h = np.maximum(h, 0.0)  # ReLU
        out = h.dot(self.W2) + self.b2
        return out

    def _forward_torch(self, x):
        import torch

        return self.model(x)

    def forward(self, x):
        return self._forward(x)
