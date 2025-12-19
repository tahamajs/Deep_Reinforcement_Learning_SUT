import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    """Mean squared error with an optional weight per-output dimension.

    The derivative is identical to MSE but scaled by weights.
    This is provided as an example of a custom loss term.
    """

    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (pred - target) ** 2
        if self.weight is not None:
            diff = diff * self.weight
        return diff.mean()
