"""Implicit Q-Learning implementation for offline RL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden)

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


@dataclass
class IQLConfig:
    obs_dim: int
    act_dim: int
    device: str = "cpu"
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    beta: float = 3.0
    lr: float = 3e-4


class IQLAgent:
    def __init__(self, cfg: IQLConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.policy = DeterministicPolicy(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.q = TwinQ(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.q_targ = TwinQ(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.v = MLP(cfg.obs_dim, 1).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=cfg.lr)

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, expectile: float):
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * diff.pow(2)).mean()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.policy(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()

    def update(self, batch: Tuple[torch.Tensor, ...]):
        obs, act, rew, next_obs, done = [b.to(self.device) for b in batch]

        with torch.no_grad():
            next_v = self.v(next_obs)
            target_q = rew + (1 - done) * self.cfg.discount * next_v

        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        with torch.no_grad():
            q1_det, q2_det = self.q(obs, act)
            q_min = torch.min(q1_det, q2_det)
        v_pred = self.v(obs)
        v_loss = self._expectile_loss(q_min - v_pred, self.cfg.expectile)
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        with torch.no_grad():
            adv = q_min - self.v(obs)
            weights = torch.exp(self.cfg.beta * adv).clamp(max=100.0)
        pi = self.policy(obs)
        policy_loss = (weights * (pi - act).pow(2).mean(dim=-1, keepdim=True)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_targ.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "q_loss": float(q_loss.item()),
            "v_loss": float(v_loss.item()),
            "policy_loss": float(policy_loss.item()),
        }

    def save(self, path: str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "q": self.q.state_dict(),
                "v": self.v.state_dict(),
                "cfg": self.cfg,
                "algo": "iql",
            },
            path,
        )

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.q.load_state_dict(ckpt["q"])
        self.v.load_state_dict(ckpt["v"])
        self.q_targ.load_state_dict(self.q.state_dict())
        return ckpt
