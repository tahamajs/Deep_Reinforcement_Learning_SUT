from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .c51 import C51Network, project_distribution
from .config import CFG


class NoisyLinear(nn.Module):
    """
    NoisyNet linear layer with factorized gaussian noise (simplicity: per original paper).
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / (self.in_features ** 0.5)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.sigma_init / (self.in_features ** 0.5))
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.bias_sigma, self.sigma_init / (self.in_features ** 0.5))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features).to(self.weight_epsilon.device)
        epsilon_out = self._scale_noise(self.out_features).to(self.weight_epsilon.device)
        self.weight_epsilon.copy_(epsilon_out.unsqueeze(1) * epsilon_in.unsqueeze(0))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """
    Minimal Rainbow DQN skeleton combining noisy nets, dueling and distributional C51.
    This is a modular implementation suitable for HW9 examples.
    """
    def __init__(self, state_dim: int, action_dim: int,
                 num_atoms: int = CFG.rainbow_num_atoms, v_min: float = CFG.rainbow_v_min,
                 v_max: float = CFG.rainbow_v_max, hidden: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        self.feature = nn.Sequential(
            NoisyLinear(state_dim, hidden),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(hidden, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, action_dim * num_atoms)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.action_dim, self.num_atoms)
        q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
        q_dist = F.softmax(q_atoms, dim=-1)
        return q_dist

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.forward(x)
        return (dist * self.support).sum(dim=-1)






