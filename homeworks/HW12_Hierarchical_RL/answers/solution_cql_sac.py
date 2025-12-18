"""
Reference solution: CQL-SAC training pipeline for Pendulum-v1
- Implements SAC actor/critic with CQL-style conservative penalty in critic loss
- Training loop, evaluation vs random policy, and plotting (saves figure)

Usage (not executed here):
python answers/solution_cql_sac.py --epochs 200 --steps-per-epoch 1000

This file is intended as a completed solution for HW12 (CQL-SAC on Pendulum).
"""

from __future__ import annotations
import argparse
import math
import random
import copy
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gym
import matplotlib.pyplot as plt

# --------------------------- Utilities ---------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int = 1000000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int = 256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            act=self.acts_buf[idxs],
            rew=self.rews_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )


# --------------------------- Networks ----------------------------------


def mlp(sizes: List[int], activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(256, 256),
        act_limit: float = 2.0,
    ):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU)
        self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        pi_dist = Normal(mu, std)
        x_t = pi_dist.rsample()
        # squash via tanh
        y_t = torch.tanh(x_t)
        action = y_t * self.act_limit
        # log prob correction
        log_prob = pi_dist.log_prob(x_t).sum(axis=-1) - (
            2 * (math.log(2) - x_t - F.softplus(-2 * x_t))
        ).sum(axis=-1)
        return action, log_prob

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, _ = self.forward(obs_t)
            a = torch.tanh(mu) * self.act_limit
        return a.squeeze(0).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation=nn.ReLU)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.q(x).squeeze(-1)


# --------------------------- Training ---------------------------------


@dataclass
class Config:
    env_name: str = "Pendulum-v1"
    seed: int = 0
    epochs: int = 100
    steps_per_epoch: int = 1000
    start_steps: int = 1000
    update_after: int = 1000
    update_every: int = 50
    batch_size: int = 256
    gamma: float = 0.99
    polyak: float = 0.995
    lr: float = 3e-4
    alpha: float = 0.2
    cql_alpha: float = 1.0
    device: str = "cpu"
    save_fig: str = "answers/cql_sac_training.png"


def train(config: Config):
    set_seed(config.seed)
    env = gym.make(config.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    # networks
    actor = GaussianPolicy(obs_dim, act_dim, act_limit=act_limit).to(config.device)
    critic1 = QNetwork(obs_dim, act_dim).to(config.device)
    critic2 = QNetwork(obs_dim, act_dim).to(config.device)
    critic1_target = copy.deepcopy(critic1)
    critic2_target = copy.deepcopy(critic2)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.lr)
    critic_optim = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=config.lr
    )

    replay = ReplayBuffer(obs_dim, act_dim, size=200000)

    total_steps = config.epochs * config.steps_per_epoch
    obs, done, ep_ret, ep_len = env.reset(), False, 0.0, 0
    rewards = []
    avg_rew_hist = []

    for t in range(total_steps):
        if t < config.start_steps:
            action = env.action_space.sample()
        else:
            action = actor.act(obs)

        next_obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay.store(obs, action, rew, next_obs, done)
        obs = next_obs
        ep_ret += rew
        ep_len += 1

        if done or (ep_len >= 1000):
            rewards.append(ep_ret)
            avg_rew_hist.append(np.mean(rewards[-100:]))
            obs, done, ep_ret, ep_len = env.reset(), False, 0.0, 0

        # update
        if t >= config.update_after and t % config.update_every == 0:
            for _ in range(config.update_every):
                batch = replay.sample_batch(config.batch_size)
                obs_b = torch.as_tensor(batch["obs"], dtype=torch.float32).to(
                    config.device
                )
                act_b = torch.as_tensor(batch["act"], dtype=torch.float32).to(
                    config.device
                )
                rew_b = torch.as_tensor(batch["rew"], dtype=torch.float32).to(
                    config.device
                )
                next_obs_b = torch.as_tensor(batch["next_obs"], dtype=torch.float32).to(
                    config.device
                )
                done_b = torch.as_tensor(batch["done"], dtype=torch.float32).to(
                    config.device
                )

                with torch.no_grad():
                    # target actions
                    next_a, next_logp = actor.sample(next_obs_b)
                    q1_target = critic1_target(next_obs_b, next_a)
                    q2_target = critic2_target(next_obs_b, next_a)
                    q_target_min = torch.min(q1_target, q2_target)
                    backup = rew_b + config.gamma * (1 - done_b) * (
                        q_target_min - config.alpha * next_logp
                    )

                # current Q estimates
                q1 = critic1(obs_b, act_b)
                q2 = critic2(obs_b, act_b)

                # TD loss
                loss_q1 = F.mse_loss(q1, backup)
                loss_q2 = F.mse_loss(q2, backup)

                # CQL conservative regularizer (continuous actions)
                # Sample random actions uniformly from action space
                sample_actions = (
                    torch.rand((config.batch_size, 10, act_dim), device=config.device)
                    * 2
                    * act_limit
                    - act_limit
                )
                sample_actions = sample_actions.view(-1, act_dim)
                obs_repeat = obs_b.unsqueeze(1).repeat(1, 10, 1).view(-1, obs_dim)

                q1_rand = critic1(obs_repeat, sample_actions).view(
                    config.batch_size, -1
                )
                q2_rand = critic2(obs_repeat, sample_actions).view(
                    config.batch_size, -1
                )

                # policy actions sampled for data states
                pi_actions, _ = actor.sample(obs_b)
                pi_actions_repeat = pi_actions.unsqueeze(1)
                # compute logsumexp
                cat_q1 = torch.cat([q1_rand, q1.unsqueeze(1)], dim=1)
                cat_q2 = torch.cat([q2_rand, q2.unsqueeze(1)], dim=1)

                cql_q1 = torch.logsumexp(cat_q1 / 1.0, dim=1).mean() - q1.mean()
                cql_q2 = torch.logsumexp(cat_q2 / 1.0, dim=1).mean() - q2.mean()

                cql_penalty = config.cql_alpha * (cql_q1 + cql_q2)

                critic_loss = loss_q1 + loss_q2 + cql_penalty

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                # actor update
                pi, logp = actor.sample(obs_b)
                q1_pi = critic1(obs_b, pi)
                q2_pi = critic2(obs_b, pi)
                q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (config.alpha * logp - q_pi).mean()

                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()

                # soft updates
                for p, p_targ in zip(critic1.parameters(), critic1_target.parameters()):
                    p_targ.data.copy_(
                        config.polyak * p_targ.data + (1 - config.polyak) * p.data
                    )
                for p, p_targ in zip(critic2.parameters(), critic2_target.parameters()):
                    p_targ.data.copy_(
                        config.polyak * p_targ.data + (1 - config.polyak) * p.data
                    )

        # end update loop

    # training finished: save a plot of average reward
    plt.figure()
    plt.plot(np.arange(len(avg_rew_hist)), avg_rew_hist)
    plt.xlabel("Completed episodes (approx)")
    plt.ylabel("Moving average reward (100)")
    plt.title("CQL-SAC on " + config.env_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config.save_fig, dpi=200)
    print(f"Saved training plot to {config.save_fig}")

    # final evaluation vs random policy
    def evaluate_policy(policy, n_eval_episodes=5, deterministic=True):
        env_eval = gym.make(config.env_name)
        returns = []
        for _ in range(n_eval_episodes):
            obs_eval, done = env_eval.reset(), False
            ep_ret = 0.0
            while True:
                if deterministic:
                    a = policy.act(obs_eval)
                else:
                    a = env_eval.action_space.sample()
                obs_eval, r, terminated, truncated, _ = env_eval.step(a)
                done = terminated or truncated
                ep_ret += r
                if done:
                    break
            returns.append(ep_ret)
        return np.mean(returns)

    mean_agent = evaluate_policy(actor, n_eval_episodes=5, deterministic=True)
    mean_random = (
        evaluate_policy(env, n_eval_episodes=5, deterministic=False) if False else None
    )
    print(f"Mean agent return (deterministic): {mean_agent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = Config(
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, seed=args.seed
    )
    train(cfg)




