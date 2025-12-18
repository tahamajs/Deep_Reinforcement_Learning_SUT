"""
Loss wrappers for MaxSink.
Provides Sinkhorn loss wrapper with a safe fallback to an MMD-like kernel loss
if geomloss is not installed.
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from geomloss import SamplesLoss  # type: ignore
except Exception:
    SamplesLoss = None  # type: ignore


class SinkhornWrapper(nn.Module):
    """
    Wrapper around geomloss.SamplesLoss("sinkhorn", ...) if available.
    Fallback: gaussian-kernel MMD loss (not OT, but stable).
    Expects inputs x,y of shape [B, N, d] (float).
    """

    def __init__(
        self, blur: float = 0.01, scaling: float = 0.9, p: int = 2, debias: bool = True
    ):
        super().__init__()
        self.blur = blur
        self.scaling = scaling
        self.p = p
        self.debias = debias
        if SamplesLoss is not None:
            self._use_geomloss = True
            self._loss = SamplesLoss(
                "sinkhorn", p=p, blur=blur, scaling=scaling, debias=debias
            )
        else:
            self._use_geomloss = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Sinkhorn loss per batch element and return shape [B].
        """
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError("x and y must be [B, N, d]")
        if self._use_geomloss:
            # geomloss returns scalar loss; compute per-batch by looping to avoid reduction across batch
            losses = []
            for xi, yi in zip(x, y):
                # SamplesLoss expects shape [N, d]
                losses.append(self._loss(xi, yi))
            return torch.stack([l.reshape(()) for l in losses])
        else:
            # fallback: simple Gaussian MMD with bandwidth = blur
            # compute pairwise kernels
            B, N, d = x.shape
            x_flat = x.view(B, N, d)
            y_flat = y.view(B, N, d)
            sigma = max(self.blur, 1e-6)

            def kernel(a, b):
                # a,b: [B, N, d]
                diff = a.unsqueeze(2) - b.unsqueeze(1)  # [B, N, N, d]
                dist2 = (diff**2).sum(-1)  # [B, N, N]
                return torch.exp(-dist2 / (2 * sigma**2))

            k_xx = kernel(x_flat, x_flat).mean(dim=(1, 2))
            k_yy = kernel(y_flat, y_flat).mean(dim=(1, 2))
            k_xy = kernel(x_flat, y_flat).mean(dim=(1, 2))
            mmd = k_xx + k_yy - 2.0 * k_xy
            return mmd
