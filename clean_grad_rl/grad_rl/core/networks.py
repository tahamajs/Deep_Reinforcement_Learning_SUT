from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        self.weight_eps.normal_()
        self.bias_eps.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


def mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int, noisy: bool = False):
    layers = []
    last = in_dim
    linear = NoisyLinear if noisy else nn.Linear
    for h in hidden:
        layers += [linear(last, h), nn.ReLU()]
        last = h
    layers.append(linear(last, out_dim))
    return nn.Sequential(*layers)


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128), noisy: bool = False):
        super().__init__()
        linear = NoisyLinear if noisy else nn.Linear
        self.feature = nn.Sequential(linear(obs_dim, hidden[0]), nn.ReLU())
        self.value = nn.Sequential(linear(hidden[0], hidden[1]), nn.ReLU(), linear(hidden[1], 1))
        self.adv = nn.Sequential(linear(hidden[0], hidden[1]), nn.ReLU(), linear(hidden[1], act_dim))

    def forward(self, x):
        h = self.feature(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


class ActorGaussian(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.backbone = mlp(obs_dim, hidden, hidden[-1])
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std


class CriticQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


class CategoricalActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128)):
        super().__init__()
        self.net = mlp(obs_dim, hidden, act_dim)

    def forward(self, obs):
        logits = self.net(obs)
        return torch.distributions.Categorical(logits=logits)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=(128, 128)):
        super().__init__()
        self.v = mlp(obs_dim, hidden, 1)

    def forward(self, obs):
        return self.v(obs)


class MonotonicMixer(nn.Module):
    """QMIX-style mixer with positive weights via abs()."""

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, agent_qs, state):
        batch = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.hidden_dim)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.elu(hidden)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        y = torch.bmm(hidden, w2) + b2
        return y.view(batch, 1)
