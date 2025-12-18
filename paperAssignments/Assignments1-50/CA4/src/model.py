from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileMLP(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, n_q: int = 50, hid: int = 512):
        super().__init__()
        self.n_q = n_q
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, n_q)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))

class TanhGaussianPolicy(nn.Module):
    """
    Gaussian policy with tanh squashing and reparameterization.
    Returns action, log_prob (unsquashed) and mean for eval.
    """
    def __init__(self, s_dim: int, a_dim: int, hid: int = 256, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(s_dim, hid),
            nn.Mish(),
            nn.Linear(hid, hid),
            nn.Mish()
        )
        self.mean_head = nn.Linear(hid, a_dim)
        self.log_std_head = nn.Linear(hid, a_dim)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(s)
        mu = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        # reparameterize
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        a = torch.tanh(pre_tanh)
        # compute logprob of squashed action (approximate)
        # Note: for stability we compute Gaussian logprob then adjust by tanh correction
        pre_sum = -0.5 * (((pre_tanh - mu) / (std + 1e-8)) ** 2 + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))
        log_prob = pre_sum.sum(dim=-1, keepdim=True)
        # tanh correction
        log_prob = log_prob - torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return a, log_prob, mu

class SCASReg(nn.Module):
    """
    Residual dynamics model for SCAS: predicts s' = s + f(s,a)
    """
    def __init__(self, s_dim: int, a_dim: int, hid: int = 256):
        super().__init__()
        self.dyn = nn.Sequential(
            nn.Linear(s_dim + a_dim, hid),
            nn.Mish(),
            nn.Linear(hid, hid),
            nn.Mish(),
            nn.Linear(hid, s_dim)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return s + self.dyn(torch.cat([s, a], dim=-1))

    def loss(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        pred = self.forward(s, a)
        return F.mse_loss(pred, s_next)

