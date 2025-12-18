"""Value network conditioned on morphology latent."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, latent_dim: int, morph_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + morph_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, z_m], -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.out(x).squeeze(-1)



