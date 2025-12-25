from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import CFG


class C51Network(nn.Module):
    """
    C51 categorical distributional network.
    Outputs probabilities over atoms per action.
    """
    def __init__(self, state_dim: int, action_dim: int, num_atoms: int = CFG.c51_num_atoms,
                 v_min: float = CFG.c51_v_min, v_max: float = CFG.c51_v_max, hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim * num_atoms)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        logits = logits.view(-1, self.action_dim, self.num_atoms)
        probs = F.softmax(logits, dim=-1)
        return probs

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward(x)
        q = (probs * self.support).sum(dim=-1)
        return q


def project_distribution(next_dist: torch.Tensor,
                         rewards: torch.Tensor,
                         dones: torch.Tensor,
                         gamma: float,
                         support: torch.Tensor,
                         v_min: float,
                         v_max: float) -> torch.Tensor:
    """
    Vectorized projection of the Bellman-updated categorical distribution onto fixed support.
    next_dist: [B, num_atoms] (probabilities for the chosen next-action)
    rewards: [B]
    dones: [B] binary
    support: [num_atoms]
    Returns: [B, num_atoms]
    """
    batch_size = rewards.size(0)
    num_atoms = support.size(0)

    tz = rewards.unsqueeze(1) + gamma * (1.0 - dones.unsqueeze(1)) * support.unsqueeze(0)
    tz = tz.clamp(v_min, v_max)

    b = (tz - v_min) / ((v_max - v_min) / (num_atoms - 1))
    l = b.floor().long()
    u = b.ceil().long()

    projected = torch.zeros_like(next_dist)

    # Distribute probabilities
    offset = torch.arange(0, batch_size, device=next_dist.device).unsqueeze(1) * num_atoms
    l_idx = (l + offset).view(-1)
    u_idx = (u + offset).view(-1)

    m = (u.float() - b)
    m_u = (b - l.float())

    next_flat = next_dist.view(-1)

    # accumulate using scatter_add
    projected_flat = torch.zeros(batch_size * num_atoms, device=next_dist.device)
    projected_flat.index_add_(0, l_idx, next_flat * m.view(-1))
    projected_flat.index_add_(0, u_idx, next_flat * m_u.view(-1))

    projected = projected_flat.view(batch_size, num_atoms)
    return projected







