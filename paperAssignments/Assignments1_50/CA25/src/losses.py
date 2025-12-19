from __future__ import annotations

import torch
import torch.nn.functional as F


def regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred.squeeze(-1), target.float())


def classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target.long())
