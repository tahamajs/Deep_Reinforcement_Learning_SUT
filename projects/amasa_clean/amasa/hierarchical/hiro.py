"""HIRO-style hierarchical controller for suturing.

Two levels:
- Meta policy π_hi outputs a relative goal g in needle space every H steps.
- Low-level policy π_lo acts conditioned on current goal (obs || g).
Off-policy correction relabels stored transitions so that past meta actions
(older goals) remain useful as π_lo changes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

from projects.amasa_clean.amasa.offline.cql import GaussianPolicy, Critic, MLP


@dataclass
class HIROConfig:
    obs_dim: int
    goal_dim: int = 3  # operate in needle xyz space
    act_dim: int = 7
    horizon: int = 20
    discount: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    temperature: float = 0.2
    device: str = "cpu"


class GoalPolicy(GaussianPolicy):
    """Meta policy outputs relative goal in [-0.02, 0.02]^goal_dim."""

    def sample(self, obs):
        action, logp = super().sample(obs)
        action = 0.02 * action  # scale down to reasonable spatial offsets
        return action, logp


class HIROAgent:
    def __init__(self, cfg: HIROConfig):
        self.cfg = cfg
        dev = torch.device(cfg.device)
        self.meta_actor = GoalPolicy(cfg.obs_dim, cfg.goal_dim).to(dev)
        self.meta_critic = Critic(cfg.obs_dim, cfg.goal_dim).to(dev)
        self.meta_targ = Critic(cfg.obs_dim, cfg.goal_dim).to(dev)
        self.meta_targ.load_state_dict(self.meta_critic.state_dict())

        # low-level conditioned on goal
        lo_obs_dim = cfg.obs_dim + cfg.goal_dim
        self.lo_actor = GaussianPolicy(lo_obs_dim, cfg.act_dim).to(dev)
        self.lo_critic = Critic(lo_obs_dim, cfg.act_dim).to(dev)
        self.lo_targ = Critic(lo_obs_dim, cfg.act_dim).to(dev)
        self.lo_targ.load_state_dict(self.lo_critic.state_dict())

        self.meta_opt = torch.optim.Adam(
            list(self.meta_actor.parameters()) + list(self.meta_critic.parameters()), lr=cfg.lr
        )
        self.lo_opt = torch.optim.Adam(
            list(self.lo_actor.parameters()) + list(self.lo_critic.parameters()), lr=cfg.lr
        )
        self.dev = dev

    @torch.no_grad()
    def act(self, obs, goal):
        obs_t = torch.as_tensor(np.concatenate([obs, goal], axis=-1), device=self.dev, dtype=torch.float32)
        a, _ = self.lo_actor.sample(obs_t.unsqueeze(0))
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def propose_goal(self, obs):
        obs_t = torch.as_tensor(obs, device=self.dev, dtype=torch.float32)
        g, _ = self.meta_actor.sample(obs_t.unsqueeze(0))
        return g.squeeze(0).cpu().numpy()

    def _soft_update(self, src, tgt):
        with torch.no_grad():
            for p, tp in zip(src.parameters(), tgt.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def update_low(self, batch: Tuple[torch.Tensor, ...]):
        obs, goal, act, rew, next_obs, done = [b.to(self.dev) for b in batch]
        # critic
        with torch.no_grad():
            next_in = torch.cat([next_obs, goal], dim=-1)
            next_a, next_logp = self.lo_actor.sample(next_in)
            q1_t, q2_t = self.lo_targ(next_in, next_a)
            target = rew + (1 - done) * self.cfg.discount * (torch.min(q1_t, q2_t) - self.cfg.temperature * next_logp)
        obs_in = torch.cat([obs, goal], dim=-1)
        q1, q2 = self.lo_critic(obs_in, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        # actor
        pi_a, logp = self.lo_actor.sample(obs_in)
        q1_pi, q2_pi = self.lo_critic(obs_in, pi_a)
        actor_loss = (self.cfg.temperature * logp - torch.min(q1_pi, q2_pi)).mean()

        loss = critic_loss + actor_loss
        self.lo_opt.zero_grad(); loss.backward(); self.lo_opt.step()
        self._soft_update(self.lo_critic, self.lo_targ)
        return {"lo_actor_loss": actor_loss.item(), "lo_critic_loss": critic_loss.item()}

    def update_meta(self, batch: Tuple[torch.Tensor, ...]):
        obs, goal, ret, next_obs, done = [b.to(self.dev) for b in batch]
        # critic
        with torch.no_grad():
            next_goal, next_logp = self.meta_actor.sample(next_obs)
            q1_t, q2_t = self.meta_targ(next_obs, next_goal)
            target = ret + (1 - done) * self.cfg.discount * (torch.min(q1_t, q2_t) - self.cfg.temperature * next_logp)
        q1, q2 = self.meta_critic(obs, goal)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        # actor
        g, logp = self.meta_actor.sample(obs)
        q1_pi, q2_pi = self.meta_critic(obs, g)
        actor_loss = (self.cfg.temperature * logp - torch.min(q1_pi, q2_pi)).mean()

        loss = critic_loss + actor_loss
        self.meta_opt.zero_grad(); loss.backward(); self.meta_opt.step()
        self._soft_update(self.meta_critic, self.meta_targ)
        return {"meta_actor_loss": actor_loss.item(), "meta_critic_loss": critic_loss.item()}


class TrajectoryBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, goal_dim: int, device: str = "cpu"):
        self.size = size
        self.ptr = 0
        self.full = False
        self.device = device
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.goal = np.zeros((size, goal_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

    def add(self, obs, goal, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.goal[self.ptr] = goal
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size: int):
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.goal[idx], device=self.device),
            torch.as_tensor(self.act[idx], device=self.device),
            torch.as_tensor(self.rew[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
        )


class MetaBuffer:
    """Stores meta transitions every H steps with off-policy relabeling."""

    def __init__(self, size: int, obs_dim: int, goal_dim: int, device: str = "cpu"):
        self.size = size
        self.ptr = 0
        self.full = False
        self.device = device
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.goal = np.zeros((size, goal_dim), dtype=np.float32)
        self.ret = np.zeros((size, 1), dtype=np.float32)
        self.next_obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

    def add(self, obs, goal, ret, next_obs, done):
        self.obs[self.ptr] = obs
        self.goal[self.ptr] = goal
        self.ret[self.ptr] = ret
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def relabel_goal(self, obs, next_obs, raw_goal):
        # HIRO relabel: set goal to next_obs_needle - obs_needle so that stored
        # transition matches current low-level behavior.
        o_n = obs[..., 14:17]
        n_n = next_obs[..., 14:17]
        return n_n - o_n

    def sample(self, batch_size: int):
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch_size)
        obs = self.obs[idx]
        next_obs = self.next_obs[idx]
        goal = self.goal[idx]
        goal = self.relabel_goal(obs, next_obs, goal)
        return (
            torch.as_tensor(obs, device=self.device),
            torch.as_tensor(goal, device=self.device),
            torch.as_tensor(self.ret[idx], device=self.device),
            torch.as_tensor(next_obs, device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
        )
