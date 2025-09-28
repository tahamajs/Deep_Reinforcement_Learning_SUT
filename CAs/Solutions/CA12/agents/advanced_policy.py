import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal, kl_divergence
import torch.multiprocessing as mp
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, deque, namedtuple
import random
import pickle
import json
import copy
import time
import threading
from typing import Tuple, List, Dict, Optional, Union, NamedTuple, Any
import warnings
from dataclasses import dataclass, field
import math
from tqdm import tqdm
from abc import ABC, abstractmethod
import itertools

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class PPONetwork(nn.Module):
    """Combined actor-critic network for PPO."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64, discrete=True):
        super(PPONetwork, self).__init__()
        self.discrete = discrete

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if discrete:
            self.actor = nn.Linear(hidden_dim, action_dim)
        else:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        shared_features = self.shared(obs)
        value = self.critic(shared_features)

        if self.discrete:
            action_logits = self.actor(shared_features)
            return action_logits, value
        else:
            action_mean = self.actor_mean(shared_features)
            action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
            return (action_mean, action_std), value

    def get_action_and_value(self, obs, action=None):
        if self.discrete:
            logits, value = self.forward(obs)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action), probs.entropy(), value
        else:
            (mean, std), value = self.forward(obs)
            probs = Normal(mean, std)
            if action is None:
                action = probs.sample()
            return (
                action,
                probs.log_prob(action).sum(-1),
                probs.entropy().sum(-1),
                value,
            )


class PPOAgent:
    """Proximal Policy Optimization agent."""

    def __init__(self, obs_dim, action_dim, lr=3e-4, discrete=True):
        self.network = PPONetwork(obs_dim, action_dim, discrete=discrete).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.discrete = discrete

        self.clip_coef = 0.2
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = 0.01

    def get_action_and_value(self, obs, action=None):
        return self.network.get_action_and_value(obs, action)

    def update(self, rollouts, n_epochs=10, minibatch_size=64):
        """Update PPO using clipped objective."""
        obs, actions, logprobs, returns, values, advantages = rollouts

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clipfracs = []
        total_losses = []

        for epoch in range(n_epochs):
            indices = torch.randperm(len(obs))

            for start in range(0, len(obs), minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_logprobs = logprobs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_values = values[mb_indices]
                mb_advantages = advantages[mb_indices]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(
                    mb_obs, mb_actions
                )

                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(newvalue.squeeze(), mb_returns)

                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_losses.append(loss.item())

            if approx_kl > self.target_kl:
                break

        return {
            "total_loss": np.mean(total_losses),
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
        }


class SACAgent:
    """Soft Actor-Critic agent."""

    def __init__(self, obs_dim, action_dim, lr=3e-4, alpha=0.2, tau=0.005):
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        ).to(device)

        self.actor_mean = nn.Linear(256, action_dim).to(device)
        self.actor_logstd = nn.Linear(256, action_dim).to(device)

        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters())
            + list(self.actor_mean.parameters())
            + list(self.actor_logstd.parameters()),
            lr=lr,
        )
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.alpha = alpha
        self.tau = tau
        self.gamma = 0.99

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        obs = torch.FloatTensor(obs).to(device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        features = self.actor(obs)
        mean = self.actor_mean(features)
        log_std = self.actor_logstd(features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = Normal(mean, std)
            x = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x)

            log_prob = normal.log_prob(x)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)

        return action.cpu().data.numpy(), log_prob if not deterministic else None

    def update(self, batch):
        """Update SAC networks."""
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        with torch.no_grad():
            next_actions, next_log_probs = self.get_action(next_states)
            next_actions = torch.FloatTensor(next_actions).to(device)

            target_q1 = self.target_q1(torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.target_q2(torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones.float()) * target_q

        current_q1 = self.q1(torch.cat([states, actions], dim=1))
        current_q2 = self.q2(torch.cat([states, actions], dim=1))

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_probs = self.get_action(states)
        new_actions = torch.FloatTensor(new_actions).to(device)

        q1_new = self.q1(torch.cat([states, new_actions], dim=1))
        q2_new = self.q2(torch.cat([states, new_actions], dim=1))
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (
            -self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        self.soft_update()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
        }

    def soft_update(self):
        """Soft update target networks."""
        for target_param, param in zip(
            self.target_q1.parameters(), self.q1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_q2.parameters(), self.q2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class GAEBuffer:
    """Buffer for collecting trajectories and computing GAE."""

    def __init__(self, size, obs_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        self.size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.logprobs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.max_size = size

    def store(self, obs, action, reward, value, logprob, done):
        """Store a single transition."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.logprobs[self.ptr] = logprob
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size

    def compute_gae(self, last_value=0):
        """Compute GAE advantages and returns."""
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.rewards)

        last_gae = 0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_nonterminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_nonterminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_nonterminal
                - self.values[t]
            )
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            )

        returns = advantages + self.values
        return advantages, returns

    def get_batch(self):
        """Get all stored data as tensors."""
        return {
            "obs": torch.FloatTensor(self.obs).to(device),
            "actions": torch.FloatTensor(self.actions).to(device),
            "rewards": torch.FloatTensor(self.rewards).to(device),
            "values": torch.FloatTensor(self.values).to(device),
            "logprobs": torch.FloatTensor(self.logprobs).to(device),
            "dones": torch.FloatTensor(self.dones).to(device),
        }
