
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Array
import queue
import threading
from threading import Lock
import time

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
from agents.advanced_policy import PPONetwork, PPOAgent

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParameterServer:
    """Parameter server for distributed RL."""

    def __init__(self, model_state_dict):
        self.params = {
            k: v.clone().share_memory_() for k, v in model_state_dict.items()
        }
        self.lock = Lock()
        self.version = Value("i", 0)
        self.update_count = Value("i", 0)

    def get_parameters(self):
        """Get current parameters."""
        with self.lock:
            return {k: v.clone() for k, v in self.params}, self.version.value

    def update_parameters(self, gradients, lr=1e-4):
        """Update parameters with gradients."""
        with self.lock:
            for key, grad in gradients.items():
                if key in self.params:
                    self.params[key] -= lr * grad

            self.version.value += 1
            self.update_count.value += 1

    def get_stats(self):
        """Get server statistics."""
        return {"version": self.version.value, "updates": self.update_count.value}


class A3CWorker:
    """A3C worker for distributed training."""

    def __init__(
        self, worker_id, global_model, local_model, env_fn, gamma=0.99, n_steps=5
    ):
        self.worker_id = worker_id
        self.global_model = global_model
        self.local_model = local_model
        self.env = env_fn()
        self.gamma = gamma
        self.n_steps = n_steps
        self.optimizer = optim.Adam(global_model.parameters(), lr=1e-4)

    def compute_n_step_returns(self, rewards, values, next_value, dones):
        """Compute n-step returns."""
        returns = []
        R = next_value

        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R * (1 - dones[i])
            returns.insert(0, R)

        return returns

    def train_step(self):
        """Single training step for A3C worker."""
        self.local_model.load_state_dict(self.global_model.state_dict())

        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        state = self.env.reset()
        for _ in range(self.n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                logits, value = self.local_model(state_tensor)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, done, _ = self.env.step(action.item())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob)
            dones.append(done)

            state = next_state if not done else self.env.reset()

            if done:
                break

        with torch.no_grad():
            if done:
                next_value = 0
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                _, next_value = self.local_model(state_tensor)
                next_value = next_value.item()

        returns = self.compute_n_step_returns(rewards, values, next_value, dones)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(values)
        log_probs = torch.stack(log_probs)

        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()

        critic_loss = F.mse_loss(values, returns)

        logits, _ = self.local_model(states)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 40)

        for global_param, local_param in zip(
            self.global_model.parameters(), self.local_model.parameters()
        ):
            if global_param.grad is not None:
                global_param.grad = local_param.grad
            else:
                global_param.grad = local_param.grad.clone()

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }


class IMPALALearner:
    """IMPALA learner with V-trace correction."""

    def __init__(self, model, lr=1e-4, rho_bar=1.0, c_bar=1.0):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.rho_bar = rho_bar  # Importance sampling clipping for policy gradient
        self.c_bar = c_bar  # Importance sampling clipping for value function

    def vtrace(
        self,
        rewards,
        values,
        behavior_log_probs,
        target_log_probs,
        bootstrap_value,
        gamma=0.99,
    ):
        """Compute V-trace targets."""
        rhos = torch.exp(target_log_probs - behavior_log_probs)
        clipped_rhos = torch.clamp(rhos, max=self.rho_bar)
        clipped_cs = torch.clamp(rhos, max=self.c_bar)

        values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)])
        deltas = clipped_rhos * (rewards + gamma * values_t_plus_1 - values)

        vs = []
        v_s = values[-1] + deltas[-1]
        vs.append(v_s)

        for i in reversed(range(len(deltas) - 1)):
            v_s = (
                values[i]
                + deltas[i]
                + gamma * clipped_cs[i] * (v_s - values_t_plus_1[i])
            )
            vs.append(v_s)

        vs.reverse()
        return torch.stack(vs)

    def update(self, batch):
        """Update IMPALA learner."""
        states, actions, rewards, behavior_log_probs, bootstrap_value = batch

        logits, values = self.model(states)

        target_log_probs = (
            F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )

        vtrace_targets = self.vtrace(
            rewards,
            values.squeeze(),
            behavior_log_probs,
            target_log_probs,
            bootstrap_value,
        )

        advantages = vtrace_targets - values.squeeze()

        policy_loss = -(target_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values.squeeze(), vtrace_targets.detach())

        entropy = (
            -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        )

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }


class DistributedPPOCoordinator:
    """Coordinator for distributed PPO training."""

    def __init__(self, n_workers, obs_dim, action_dim, lr=3e-4):
        self.n_workers = n_workers
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.global_model = PPONetwork(obs_dim, action_dim, discrete=True)
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=lr)

        self.task_queues = [Queue() for _ in range(n_workers)]
        self.result_queue = Queue()

        self.episode_rewards = []
        self.losses = []

    def collect_rollouts(self, n_steps=128):
        """Coordinate rollout collection across workers."""
        for i in range(self.n_workers):
            self.task_queues[i].put(("collect", n_steps))

        all_rollouts = []
        for _ in range(self.n_workers):
            rollouts = self.result_queue.get()
            all_rollouts.append(rollouts)

        return all_rollouts

    def aggregate_rollouts(self, rollouts_list):
        """Aggregate rollouts from all workers."""
        aggregated = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "advantages": [],
            "returns": [],
        }

        for rollouts in rollouts_list:
            for key in aggregated:
                aggregated[key].extend(rollouts[key])

        for key in aggregated:
            aggregated[key] = torch.FloatTensor(aggregated[key])

        return aggregated

    def update_global_model(self, rollouts):
        """Update global model using aggregated rollouts."""
        ppo_agent = PPOAgent(self.obs_dim, self.action_dim)
        ppo_agent.network = self.global_model
        ppo_agent.optimizer = self.optimizer

        obs = rollouts["obs"]
        actions = rollouts["actions"]
        log_probs = rollouts["log_probs"]
        returns = rollouts["returns"]
        values = rollouts["values"]
        advantages = rollouts["advantages"]

        ppo_rollouts = (obs, actions, log_probs, returns, values, advantages)
        losses = ppo_agent.update(ppo_rollouts)

        return losses

    def broadcast_parameters(self):
        """Send updated parameters to all workers."""
        state_dict = self.global_model.state_dict()
        for i in range(self.n_workers):
            self.task_queues[i].put(("update_params", state_dict))


class EvolutionaryStrategy:
    """Simple evolutionary strategy for RL."""

    def __init__(self, model, population_size=50, sigma=0.1, lr=0.01):
        self.model = model
        self.population_size = population_size
        self.sigma = sigma
        self.lr = lr

        self.param_shapes = []
        self.param_sizes = []
        for param in model.parameters():
            self.param_shapes.append(param.shape)
            self.param_sizes.append(param.numel())

        self.total_params = sum(self.param_sizes)

    def generate_population(self):
        """Generate population of parameter perturbations."""
        return [np.random.randn(self.total_params) for _ in range(self.population_size)]

    def set_parameters(self, flat_params):
        """Set model parameters from flattened array."""
        idx = 0
        with torch.no_grad():
            for param, size, shape in zip(
                self.model.parameters(), self.param_sizes, self.param_shapes
            ):
                param_values = flat_params[idx : idx + size].reshape(shape)
                param.copy_(torch.FloatTensor(param_values))
                idx += size

    def get_parameters(self):
        """Get flattened model parameters."""
        params = []
        for param in self.model.parameters():
            params.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(params)

    def update(self, rewards, perturbations):
        """Update parameters using ES."""
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        current_params = self.get_parameters()
        param_update = np.zeros_like(current_params)

        for reward, perturbation in zip(rewards, perturbations):
            param_update += reward * perturbation

        param_update = self.lr * param_update / (self.population_size * self.sigma)

        new_params = current_params + param_update
        self.set_parameters(new_params)

        return param_update


def demonstrate_parameter_server():
    """Demonstrate parameter server functionality."""
    print("🖥️  Parameter Server Demo")

    model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))

    param_server = ParameterServer(model.state_dict())

    print(f"Initial version: {param_server.get_stats()['version']}")

    dummy_gradients = {
        name: torch.randn_like(param) for name, param in model.named_parameters()
    }
    param_server.update_parameters(dummy_gradients)

    print(f"After update: {param_server.get_stats()}")

    return param_server


def demonstrate_evolutionary_strategy():
    """Demonstrate evolutionary strategy."""
    print("\n🧬 Evolutionary Strategy Demo")

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    es = EvolutionaryStrategy(model, population_size=10, sigma=0.1)

    population = es.generate_population()
    print(f"Generated population of size: {len(population)}")
    print(f"Parameter dimensionality: {es.total_params}")

    rewards = np.random.randn(len(population))
    es.update(rewards, population)

    print("✅ ES update completed")

    return es


print("🌐 Distributed Reinforcement Learning Systems")
param_server_demo = demonstrate_parameter_server()
es_demo = demonstrate_evolutionary_strategy()

print("\n🚀 Distributed RL implementations ready!")
print("✅ Parameter server, A3C, IMPALA, and ES components implemented!")
