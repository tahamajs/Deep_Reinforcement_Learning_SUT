from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
from .utils import make_rng


class BaseModel:
    """A minimal numpy MLP with deterministic initialization.

    Parameters
    ----------
    input_dim: int
    hidden_dim: int
    output_dim: int
    seed: Optional[int] - if provided, used for deterministic init
    backend: str - "numpy" or "torch" (torch optional)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: Optional[int] = None, backend: Optional[str] = None):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.backend = backend or "numpy"
        self.seed = seed

        # Prefer torch backend only if explicitly requested
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

                self.model = MLP(self.input_dim, self.hidden_dim, self.output_dim)
                self._forward = self._forward_torch
            except Exception:
                # torch not available; fallback to numpy
                self.backend = "numpy"
                self._init_numpy_weights()
                self._forward = self._forward_numpy
        else:
            self._init_numpy_weights()
            self._forward = self._forward_numpy

    def _init_numpy_weights(self):
        # deterministic small random weights; use provided seed when present
        rng = make_rng(self.seed)
        self.W1 = rng.normal(scale=0.1, size=(self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((self.hidden_dim,))
        self.W2 = rng.normal(scale=0.1, size=(self.hidden_dim, self.output_dim))
        self.b2 = np.zeros((self.output_dim,))

    def _forward_numpy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        assert x.ndim == 2 and x.shape[1] == self.input_dim, "Input must be (batch, input_dim)"
        h = x.dot(self.W1) + self.b1
        h = np.maximum(h, 0.0)  # ReLU
        out = h.dot(self.W2) + self.b2
        return out

    def _forward_torch(self, x):
        import torch

        with torch.no_grad():
            return self.model(torch.as_tensor(x))

    def forward(self, x):
        return self._forward(x)
