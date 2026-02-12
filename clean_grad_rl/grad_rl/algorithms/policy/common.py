from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grad_rl.core.networks import CategoricalActor, ValueNet, mlp


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128)):
        super().__init__()
        self.backbone = mlp(obs_dim, hidden, hidden[-1])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        h = self.backbone(obs)
        mu = self.mu(h)
        std = torch.exp(self.log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logp_old: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


def compute_gae(rewards: List[float], values: List[float], dones: List[float], gamma: float, lam: float):
    adv = []
    gae = 0.0
    vals = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * vals[t + 1] * (1.0 - dones[t]) - vals[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv.append(gae)
    adv.reverse()
    returns = [a + v for a, v in zip(adv, values)]
    return np.array(returns, dtype=np.float32), np.array(adv, dtype=np.float32)


def minibatches(size: int, batch_size: int):
    idx = np.arange(size)
    np.random.shuffle(idx)
    for start in range(0, size, batch_size):
        yield idx[start : start + batch_size]
