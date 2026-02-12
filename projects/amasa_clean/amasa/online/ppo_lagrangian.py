"""PPO-Lagrangian with separate reward and cost advantages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from projects.amasa_clean.amasa.offline.cql import GaussianPolicy


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=(256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


@dataclass
class PPOLagConfig:
    obs_dim: int
    act_dim: int
    device: str = "cpu"
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    train_iters: int = 10


class PPOLagrangianAgent:
    def __init__(self, cfg: PPOLagConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.actor = GaussianPolicy(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.vr = ValueNet(cfg.obs_dim).to(self.device)
        self.vc = ValueNet(cfg.obs_dim).to(self.device)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.vr.parameters()) + list(self.vc.parameters()), lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy(), float(logp.squeeze(0).cpu().numpy())

    def _gae(self, rewards, values, dones):
        adv = np.zeros_like(rewards)
        gae = 0.0
        next_v = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.discount * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.cfg.discount * self.cfg.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
            next_v = values[t]
        ret = adv + values
        return adv, ret

    def update(self, traj: Dict[str, np.ndarray], lambda_value: float):
        obs = torch.as_tensor(traj["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(traj["act"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(traj["logp"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        r_adv = torch.as_tensor(traj["r_adv"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        c_adv = torch.as_tensor(traj["c_adv"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        r_ret = torch.as_tensor(traj["r_ret"], dtype=torch.float32, device=self.device).unsqueeze(-1)
        c_ret = torch.as_tensor(traj["c_ret"], dtype=torch.float32, device=self.device).unsqueeze(-1)

        r_adv = (r_adv - r_adv.mean()) / (r_adv.std() + 1e-8)
        c_adv = (c_adv - c_adv.mean()) / (c_adv.std() + 1e-8)

        metrics = {}
        for _ in range(self.cfg.train_iters):
            dist = self.actor.forward(obs)
            raw_action = torch.atanh(act.clamp(-0.999, 0.999))
            logp = dist.log_prob(raw_action) - torch.log(1 - act.pow(2) + 1e-7)
            logp = logp.sum(dim=-1, keepdim=True)
            ratio = torch.exp(logp - old_logp)

            lag_adv = r_adv - lambda_value * c_adv
            s1 = ratio * lag_adv
            s2 = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * lag_adv
            policy_loss = -torch.min(s1, s2).mean()

            vr_loss = F.mse_loss(self.vr(obs), r_ret)
            vc_loss = F.mse_loss(self.vc(obs), c_ret)
            loss = policy_loss + vr_loss + vc_loss

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            metrics = {
                "policy_loss": float(policy_loss.item()),
                "value_reward_loss": float(vr_loss.item()),
                "value_cost_loss": float(vc_loss.item()),
            }
        return metrics

    def build_trajectory(self, buffer: Dict[str, List[float]]):
        obs = np.asarray(buffer["obs"], dtype=np.float32)
        rewards = np.asarray(buffer["rew"], dtype=np.float32)
        costs = np.asarray(buffer["cost"], dtype=np.float32)
        dones = np.asarray(buffer["done"], dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            r_vals = self.vr(obs_t).squeeze(-1).cpu().numpy()
            c_vals = self.vc(obs_t).squeeze(-1).cpu().numpy()

        r_adv, r_ret = self._gae(rewards, r_vals, dones)
        c_adv, c_ret = self._gae(costs, c_vals, dones)

        return {
            "obs": obs,
            "act": np.asarray(buffer["act"], dtype=np.float32),
            "logp": np.asarray(buffer["logp"], dtype=np.float32),
            "r_adv": r_adv.astype(np.float32),
            "c_adv": c_adv.astype(np.float32),
            "r_ret": r_ret.astype(np.float32),
            "c_ret": c_ret.astype(np.float32),
        }

    def save(self, path: str):
        payload = {
            "actor": self.actor.state_dict(),
            "vr": self.vr.state_dict(),
            "vc": self.vc.state_dict(),
            "cfg": self.cfg,
            "algo": "ppo_lag",
        }
        try:
            torch.save(payload, path)
        except RuntimeError:
            torch.save(payload, path, _use_new_zipfile_serialization=False)

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.vr.load_state_dict(ckpt["vr"])
        self.vc.load_state_dict(ckpt["vc"])
        return ckpt
