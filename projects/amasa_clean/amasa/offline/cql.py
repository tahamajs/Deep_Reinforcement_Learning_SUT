"""Conservative Q-Learning (continuous) used for Phase I offline pretraining."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- models ---------------------------------------------------------------
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


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256), log_std_bounds=(-5, 2)):
        super().__init__()
        self.trunk = MLP(obs_dim, 2 * act_dim, hidden)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs):
        mu_logstd = self.trunk(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.tanh(log_std)
        lo, hi = self.log_std_bounds
        log_std = lo + 0.5 * (hi - lo) * (log_std + 1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist

    def sample(self, obs):
        dist = self.forward(obs)
        raw = dist.rsample()
        action = torch.tanh(raw)
        logp = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-7)
        return action, logp.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


@dataclass
class CQLConfig:
    obs_dim: int
    act_dim: int
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature: float = 0.2  # entropy weight alpha (fixed)
    cql_alpha: float = 5.0
    bc_coef: float = 2.5
    device: str = "cpu"


class CQLAgent:
    def __init__(self, cfg: CQLConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.actor = GaussianPolicy(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.critic = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.target_critic = Critic(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            dist = self.actor.forward(obs_t)
            a = torch.tanh(dist.mean)
        else:
            a, _ = self.actor.sample(obs_t)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch: Tuple[torch.Tensor, ...]):
        obs, act, rew, next_obs, done = batch
        obs = obs.to(self.device)
        act = act.to(self.device)
        rew = rew.to(self.device)
        next_obs = next_obs.to(self.device)
        done = done.to(self.device)

        # critic loss with target smoothing
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(next_obs)
            q1_t, q2_t = self.target_critic(next_obs, next_a)
            q_t = torch.min(q1_t, q2_t) - self.cfg.temperature * next_logp
            backup = rew + (1.0 - done) * self.cfg.discount * q_t

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        # CQL regularizer: push down unseen actions
        rand_actions = torch.empty_like(act).uniform_(-1, 1)
        pi_actions, pi_logp = self.actor.sample(obs)
        with torch.no_grad():
            next_rand_actions = torch.empty_like(act).uniform_(-1, 1)
            targ_pi_actions, _ = self.actor.sample(next_obs)
        all_actions = torch.cat([rand_actions, pi_actions, next_rand_actions, targ_pi_actions], dim=0)
        all_obs = torch.cat([obs for _ in range(4)], dim=0)
        q1_cat, q2_cat = self.critic(all_obs, all_actions)
        logsumexp_q1 = torch.logsumexp(q1_cat, dim=0)
        logsumexp_q2 = torch.logsumexp(q2_cat, dim=0)
        cql_term = (logsumexp_q1 + logsumexp_q2).mean() - (q1.mean() + q2.mean())
        critic_loss = critic_loss + self.cfg.cql_alpha * cql_term

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # actor loss: entropy-regularized + behavior cloning pull
        pi_actions, logp = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.cfg.temperature * logp - q_pi).mean()
        bc_loss = F.mse_loss(pi_actions, act)
        actor_loss = actor_loss + self.cfg.bc_coef * bc_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # target soft update
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "bc_loss": bc_loss.item(),
            "cql_term": cql_term.item(),
        }

    def save(self, path: str):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "cfg": self.cfg,
            "algo": "cql",
        }
        try:
            torch.save(payload, path)
        except RuntimeError:
            torch.save(payload, path, _use_new_zipfile_serialization=False)

    def load(self, path: str, map_location=None):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        return checkpoint


# ---- utils ---------------------------------------------------------------
def make_minibatches(buffer: dict, batch_size: int, device: str):
    n = buffer["obs"].shape[0]
    idx = np.random.permutation(n)
    for start in range(0, n, batch_size):
        j = idx[start : start + batch_size]
        yield (
            torch.as_tensor(buffer["obs"][j], device=device),
            torch.as_tensor(buffer["actions"][j], device=device),
            torch.as_tensor(buffer["rewards"][j], device=device).unsqueeze(-1),
            torch.as_tensor(buffer["next_obs"][j], device=device),
            torch.as_tensor(buffer["dones"][j], device=device).unsqueeze(-1),
        )
