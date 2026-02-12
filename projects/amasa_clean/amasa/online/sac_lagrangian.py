"""SAC-Lagrangian for safe online fine-tuning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.amasa_clean.amasa.offline.cql import GaussianPolicy, Critic


@dataclass
class SACLagConfig:
    obs_dim: int
    act_dim: int
    device: str = "cpu"
    discount: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2


class SACLagrangianAgent:
    def __init__(self, cfg: SACLagConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = GaussianPolicy(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.rcritic = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.rcritic_targ = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.ccritic = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.ccritic_targ = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.rcritic_targ.load_state_dict(self.rcritic.state_dict())
        self.ccritic_targ.load_state_dict(self.ccritic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.r_opt = torch.optim.Adam(self.rcritic.parameters(), lr=cfg.lr)
        self.c_opt = torch.optim.Adam(self.ccritic.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        a, _ = self.actor.sample(obs_t.unsqueeze(0))
        return a.squeeze(0).cpu().numpy()

    def update(self, batch: Tuple[torch.Tensor, ...], lambda_value: float):
        obs, act, rew, next_obs, done, cost = [b.to(self.device) for b in batch]

        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            rq1_t, rq2_t = self.rcritic_targ(next_obs, next_a)
            cq1_t, cq2_t = self.ccritic_targ(next_obs, next_a)
            rq_t = torch.min(rq1_t, rq2_t) - self.cfg.alpha * next_logp
            cq_t = torch.min(cq1_t, cq2_t)
            r_backup = rew + (1 - done) * self.cfg.discount * rq_t
            c_backup = cost + (1 - done) * self.cfg.discount * cq_t

        rq1, rq2 = self.rcritic(obs, act)
        cq1, cq2 = self.ccritic(obs, act)
        r_loss = F.mse_loss(rq1, r_backup) + F.mse_loss(rq2, r_backup)
        c_loss = F.mse_loss(cq1, c_backup) + F.mse_loss(cq2, c_backup)

        self.r_opt.zero_grad()
        r_loss.backward()
        self.r_opt.step()

        self.c_opt.zero_grad()
        c_loss.backward()
        self.c_opt.step()

        pi, logp = self.actor.sample(obs)
        rq1_pi, rq2_pi = self.rcritic(obs, pi)
        cq1_pi, cq2_pi = self.ccritic(obs, pi)
        r_obj = torch.min(rq1_pi, rq2_pi)
        c_obj = torch.min(cq1_pi, cq2_pi)
        actor_loss = (self.cfg.alpha * logp - r_obj + lambda_value * c_obj).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.rcritic.parameters(), self.rcritic_targ.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, tp in zip(self.ccritic.parameters(), self.ccritic_targ.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "actor_loss": float(actor_loss.item()),
            "reward_critic_loss": float(r_loss.item()),
            "cost_critic_loss": float(c_loss.item()),
        }

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "rcritic": self.rcritic.state_dict(),
                "ccritic": self.ccritic.state_dict(),
                "cfg": self.cfg,
                "algo": "sac_lag",
            },
            path,
        )

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.rcritic.load_state_dict(ckpt["rcritic"])
        self.ccritic.load_state_dict(ckpt["ccritic"])
        self.rcritic_targ.load_state_dict(self.rcritic.state_dict())
        self.ccritic_targ.load_state_dict(self.ccritic.state_dict())
        return ckpt
