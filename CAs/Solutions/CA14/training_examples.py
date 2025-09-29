"""
Computer Assignment 14: Advanced Deep Reinforcement Learning
Safe RL, Multi-Agent RL, Offline RL, and Robust RL

This file contains comprehensive implementations of advanced RL algorithms including:
- Offline RL: Conservative Q-Learning (CQL), Implicit Q-Learning (IQL)
- Safe RL: Constrained Policy Optimization (CPO), Lagrangian methods
- Multi-Agent RL: MADDPG, QMIX with value function factorization
- Robust RL: Domain randomization, adversarial training, uncertainty estimation

Author: [Your Name]
Date: September 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MultivariateNormal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import deque, namedtuple
import random
import copy
import gym
from typing import List, Dict, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration dictionaries for different algorithms
OFFLINE_RL_CONFIG = {
    "batch_size": 256,
    "buffer_size": 100000,
    "conservative_weight": 1.0,
    "expectile": 0.7,
    "behavior_cloning_weight": 0.1,
}

SAFE_RL_CONFIG = {
    "constraint_threshold": 0.1,
    "lagrange_lr": 1e-3,
    "penalty_weight": 10.0,
    "safety_buffer_size": 10000,
    "trust_region_delta": 0.01,
}

MULTI_AGENT_CONFIG = {
    "num_agents": 4,
    "communication_dim": 16,
    "centralized_critic": True,
    "shared_experience": False,
    "mixing_network_hidden": 256,
}

ROBUST_RL_CONFIG = {
    "domain_randomization": True,
    "adversarial_training": True,
    "uncertainty_estimation": True,
    "robust_loss_weight": 0.5,
    "adversarial_epsilon": 0.1,
}

print("🚀 Advanced Deep RL Environment Initialized!")
print("📚 Topics: Offline RL, Safe RL, Multi-Agent RL, Robust RL")
print("🔬 Ready for advanced reinforcement learning research and implementation!")


# =============================================================================
# Section 1: Offline Reinforcement Learning
# =============================================================================


class OfflineDataset:
    """Dataset class for offline RL training with quality assessment."""

    def __init__(
        self, states, actions, rewards, next_states, dones, dataset_type="mixed"
    ):
        self.states = np.array(states)
        self.actions = np.array(actions)
        self.rewards = np.array(rewards)
        self.next_states = np.array(next_states)
        self.dones = np.array(dones)
        self.dataset_type = dataset_type
        self.size = len(states)

        # Compute statistics
        self.reward_mean = np.mean(rewards)
        self.reward_std = np.std(rewards)
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-8

        # Normalize dataset
        self.normalize_dataset()

        # Quality metrics
        self.quality_metrics = self.compute_quality_metrics()

    def normalize_dataset(self):
        """Normalize states and rewards for stable training."""
        self.states = (self.states - self.state_mean) / self.state_std
        self.next_states = (self.next_states - self.state_mean) / self.state_std
        self.rewards = (self.rewards - self.reward_mean) / (self.reward_std + 1e-8)

    def compute_quality_metrics(self):
        """Compute dataset quality metrics."""
        metrics = {}

        # Action distribution entropy
        if len(self.actions.shape) == 1:  # Discrete actions
            action_counts = np.bincount(self.actions)
            action_probs = action_counts / self.size
            metrics["action_entropy"] = -np.sum(
                action_probs * np.log(action_probs + 1e-8)
            )
        else:  # Continuous actions
            metrics["action_entropy"] = float("nan")

        # State coverage (simplified)
        state_variance = np.var(self.states, axis=0).mean()
        metrics["state_coverage"] = state_variance

        # Reward distribution
        metrics["reward_skewness"] = pd.Series(self.rewards).skew()
        metrics["reward_kurtosis"] = pd.Series(self.rewards).kurtosis()

        # Dataset diversity
        unique_states = len(np.unique(self.states.round(2), axis=0))
        metrics["state_diversity"] = unique_states / self.size

        return metrics

    def sample_batch(self, batch_size):
        """Sample random batch from dataset."""
        indices = np.random.randint(0, self.size, batch_size)

        batch_states = torch.FloatTensor(self.states[indices]).to(device)
        batch_actions = torch.LongTensor(self.actions[indices]).to(device)
        batch_rewards = torch.FloatTensor(self.rewards[indices]).to(device)
        batch_next_states = torch.FloatTensor(self.next_states[indices]).to(device)
        batch_dones = torch.BoolTensor(self.dones[indices]).to(device)

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
        )

    def get_action_distribution(self):
        """Analyze action distribution in dataset."""
        if len(self.actions.shape) == 1:  # Discrete actions
            action_counts = np.bincount(self.actions)
            return action_counts / self.size
        else:  # Continuous actions
            return np.mean(self.actions, axis=0), np.std(self.actions, axis=0)


class ConservativeQNetwork(nn.Module):
    """Q-network for Conservative Q-Learning (CQL)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        """Forward pass through Q-network."""
        q_values = self.q_network(state)
        state_value = self.value_network(state)
        return q_values, state_value

    def get_q_values(self, state):
        """Get Q-values for all actions."""
        q_values, _ = self.forward(state)
        return q_values


class ConservativeQLearning:
    """Conservative Q-Learning (CQL) for offline RL."""

    def __init__(self, state_dim, action_dim, lr=3e-4, conservative_weight=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.conservative_weight = conservative_weight

        self.q_network = ConservativeQNetwork(state_dim, action_dim).to(device)
        self.target_q_network = copy.deepcopy(self.q_network).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.gamma = 0.99
        self.tau = 0.005  # Target network update rate
        self.update_count = 0

        self.losses = []
        self.conservative_losses = []
        self.bellman_losses = []

    def compute_conservative_loss(self, states, actions):
        """Compute CQL conservative loss."""
        q_values, _ = self.q_network(states)

        logsumexp_q = torch.logsumexp(q_values, dim=1)

        behavior_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        conservative_loss = (logsumexp_q - behavior_q_values).mean()

        return conservative_loss

    def compute_bellman_loss(self, states, actions, rewards, next_states, dones):
        """Compute standard Bellman loss."""
        q_values, _ = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values, _ = self.target_q_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (~dones))

        bellman_loss = F.mse_loss(current_q_values, target_q_values)
        return bellman_loss

    def update(self, batch):
        """Update CQL agent."""
        states, actions, rewards, next_states, dones = batch

        conservative_loss = self.compute_conservative_loss(states, actions)
        bellman_loss = self.compute_bellman_loss(
            states, actions, rewards, next_states, dones
        )

        total_loss = self.conservative_weight * conservative_loss + bellman_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % 100 == 0:
            self.soft_update_target()

        self.losses.append(total_loss.item())
        self.conservative_losses.append(conservative_loss.item())
        self.bellman_losses.append(bellman_loss.item())

        return {
            "total_loss": total_loss.item(),
            "conservative_loss": conservative_loss.item(),
            "bellman_loss": bellman_loss.item(),
        }

    def soft_update_target(self):
        """Soft update of target network."""
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def get_action(self, state, epsilon=0.0):
        """Get action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network.get_q_values(state_tensor)
            return q_values.argmax().item()


class ImplicitQLearning:
    """Implicit Q-Learning (IQL) for offline RL."""

    def __init__(self, state_dim, action_dim, lr=3e-4, expectile=0.7):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expectile = expectile  # Expectile for advantage estimation

        self.q_network = ConservativeQNetwork(state_dim, action_dim).to(device)
        self.target_q_network = copy.deepcopy(self.q_network).to(device)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        ).to(device)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        self.gamma = 0.99
        self.tau = 0.005

        self.q_losses = []
        self.policy_losses = []
        self.advantages = []

    def compute_expectile_loss(self, errors, expectile):
        """Compute expectile loss (asymmetric squared loss)."""
        weights = torch.where(errors > 0, expectile, 1 - expectile)
        return (weights * errors.pow(2)).mean()

    def update_q_function(self, states, actions, rewards, next_states, dones):
        """Update Q-function using expectile regression."""
        q_values, state_values = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            _, next_state_values = self.target_q_network(next_states)
            target_q_values = rewards + (
                self.gamma * next_state_values.squeeze() * (~dones)
            )

        q_errors = target_q_values - current_q_values
        q_loss = self.compute_expectile_loss(
            q_errors, 0.5
        )  # Standard MSE for Q-function

        advantages = current_q_values.detach() - state_values.squeeze()
        value_loss = self.compute_expectile_loss(advantages, self.expectile)

        total_q_loss = q_loss + value_loss

        self.q_optimizer.zero_grad()
        total_q_loss.backward()
        self.q_optimizer.step()

        return total_q_loss.item(), advantages.mean().item()

    def update_policy(self, states, actions):
        """Update policy using advantage-weighted regression."""
        with torch.no_grad():
            q_values, state_values = self.q_network(states)
            current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            advantages = current_q_values - state_values.squeeze()
            weights = torch.exp(advantages / 3.0).clamp(max=100)  # Temperature scaling

        action_probs = self.policy_network(states)
        log_probs = torch.log(
            action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8
        )

        policy_loss = -(weights.detach() * log_probs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    def update(self, batch):
        """Update IQL agent."""
        states, actions, rewards, next_states, dones = batch

        q_loss, avg_advantage = self.update_q_function(
            states, actions, rewards, next_states, dones
        )

        policy_loss = self.update_policy(states, actions)

        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        self.q_losses.append(q_loss)
        self.policy_losses.append(policy_loss)
        self.advantages.append(avg_advantage)

        return {
            "q_loss": q_loss,
            "policy_loss": policy_loss,
            "avg_advantage": avg_advantage,
        }

    def get_action(self, state):
        """Get action from learned policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = self.policy_network(state_tensor)
            action_dist = Categorical(action_probs)
            return action_dist.sample().item()


def generate_offline_dataset(env_name="CartPole-v1", dataset_type="mixed", size=50000):
    """Generate offline dataset with different quality levels."""

    class SimpleGridWorld:
        def __init__(self, size=5):
            self.size = size
            self.state = [0, 0]
            self.goal = [size - 1, size - 1]
            self.action_space = 4  # up, down, left, right

        def reset(self):
            self.state = [0, 0]
            return np.array(self.state, dtype=np.float32)

        def step(self, action):
            if action == 0 and self.state[1] < self.size - 1:
                self.state[1] += 1
            elif action == 1 and self.state[1] > 0:
                self.state[1] -= 1
            elif action == 2 and self.state[0] > 0:
                self.state[0] -= 1
            elif action == 3 and self.state[0] < self.size - 1:
                self.state[0] += 1

            done = self.state == self.goal
            reward = 1.0 if done else -0.1

            return np.array(self.state, dtype=np.float32), reward, done, {}

    env = SimpleGridWorld(size=5)

    states, actions, rewards, next_states, dones = [], [], [], [], []

    for _ in range(size):
        state = env.reset()
        episode_done = False
        episode_length = 0

        while not episode_done and episode_length < 50:
            if dataset_type == "expert":
                if state[0] < env.goal[0]:
                    action = 3  # right
                elif state[1] < env.goal[1]:
                    action = 0  # up
                else:
                    action = np.random.randint(4)
            elif dataset_type == "random":
                action = np.random.randint(4)
            else:  # mixed
                if np.random.random() < 0.7:
                    if state[0] < env.goal[0]:
                        action = 3  # right
                    elif state[1] < env.goal[1]:
                        action = 0  # up
                    else:
                        action = np.random.randint(4)
                else:
                    action = np.random.randint(4)

            next_state, reward, done, _ = env.step(action)

            states.append(state.copy())
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state.copy())
            dones.append(done)

            state = next_state
            episode_done = done
            episode_length += 1

            if episode_done:
                break

    return OfflineDataset(states, actions, rewards, next_states, dones, dataset_type)


def train_offline_rl_algorithms():
    """Train and compare offline RL algorithms."""
    print("📚 Training Offline RL Algorithms")
    print("=" * 40)

    # Generate datasets
    datasets = {
        "expert": generate_offline_dataset(dataset_type="expert", size=10000),
        "mixed": generate_offline_dataset(dataset_type="mixed", size=15000),
        "random": generate_offline_dataset(dataset_type="random", size=8000),
    }

    for name, dataset in datasets.items():
        print(f"\n📊 {name.title()} Dataset:")
        print(f"  Size: {dataset.size}")
        print(f"  Average Reward: {dataset.reward_mean:.3f} ± {dataset.reward_std:.3f}")
        print(f"  State Dim: {dataset.states.shape[1]}")
        action_dist = dataset.get_action_distribution()
        print(f"  Action Distribution: {action_dist}")
        print(f"  Quality Metrics: {dataset.quality_metrics}")

    # Initialize algorithms
    state_dim = datasets["mixed"].states.shape[1]
    action_dim = 4  # Grid world actions

    cql_agent = ConservativeQLearning(state_dim, action_dim)
    iql_agent = ImplicitQLearning(state_dim, action_dim)

    algorithms = {"CQL": cql_agent, "IQL": iql_agent}
    results = {name: {"rewards": [], "losses": []} for name in algorithms.keys()}

    num_updates = 1000
    batch_size = OFFLINE_RL_CONFIG["batch_size"]

    print("\n🔄 Training algorithms...")

    for update in range(num_updates):
        batch = datasets["mixed"].sample_batch(batch_size)

        for alg_name, agent in algorithms.items():
            update_info = agent.update(batch)
            results[alg_name]["losses"].append(update_info)

        if update % 100 == 0:
            print(f"Update {update}:")
            for alg_name in algorithms.keys():
                recent_losses = np.mean(
                    [r[list(r.keys())[0]] for r in results[alg_name]["losses"][-10:]]
                )
                print(f"  {alg_name} Loss: {recent_losses:.4f}")

    # Evaluation
    print("\n📈 Evaluating learned policies...")

    eval_env = SimpleGridWorld(size=5)
    eval_results = {name: [] for name in algorithms.keys()}

    for alg_name, agent in algorithms.items():
        total_reward = 0
        num_episodes = 50

        for _ in range(num_episodes):
            state = eval_env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < 50:
                action = agent.get_action(state)
                next_state, reward, done, _ = eval_env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1

            total_reward += episode_reward

        avg_reward = total_reward / num_episodes
        eval_results[alg_name] = avg_reward
        print(f"  {alg_name} Average Reward: {avg_reward:.3f}")

    return results, eval_results, datasets


# =============================================================================
# Section 2: Safe Reinforcement Learning
# =============================================================================


class SafeEnvironment:
    """Environment with safety constraints for Safe RL demonstration."""

    def __init__(self, size=6, hazard_positions=None, constraint_threshold=0.1):
        self.size = size
        self.state = [0, 0]
        self.goal = [size - 1, size - 1]
        self.constraint_threshold = constraint_threshold

        if hazard_positions is None:
            self.hazards = [[2, 2], [3, 1], [1, 3], [4, 3]]
        else:
            self.hazards = hazard_positions

        self.action_space = 4  # up, down, left, right
        self.max_episode_steps = 50
        self.current_step = 0

        self.constraint_violations = 0
        self.total_constraint_cost = 0

    def reset(self):
        """Reset environment to initial state."""
        self.state = [0, 0]
        self.current_step = 0
        self.constraint_violations = 0
        self.total_constraint_cost = 0
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        """Take action in environment with safety constraints."""
        self.current_step += 1

        prev_state = self.state.copy()
        if action == 0 and self.state[1] < self.size - 1:  # up
            self.state[1] += 1
        elif action == 1 and self.state[1] > 0:  # down
            self.state[1] -= 1
        elif action == 2 and self.state[0] > 0:  # left
            self.state[0] -= 1
        elif action == 3 and self.state[0] < self.size - 1:  # right
            self.state[0] += 1

        done = self.state == self.goal
        reward = 10.0 if done else -0.1

        constraint_cost = self._compute_constraint_cost(self.state)

        episode_done = done or self.current_step >= self.max_episode_steps

        info = {
            "constraint_cost": constraint_cost,
            "constraint_violation": constraint_cost > 0,
            "total_violations": self.constraint_violations,
            "position": self.state.copy(),
        }

        return np.array(self.state, dtype=np.float32), reward, episode_done, info

    def _compute_constraint_cost(self, state):
        """Compute constraint violation cost."""
        cost = 0.0

        if state in self.hazards:
            cost += 1.0  # High cost for being in hazardous areas
            self.constraint_violations += 1

        if (
            state[0] == 0
            or state[0] == self.size - 1
            or state[1] == 0
            or state[1] == self.size - 1
        ):
            cost += 0.1  # Small cost for being near boundaries

        self.total_constraint_cost += cost
        return cost

    def is_safe_state(self, state):
        """Check if state is safe (no constraint violations)."""
        return state not in self.hazards

    def get_safe_actions(self, state):
        """Get list of safe actions from current state."""
        safe_actions = []
        for action in range(self.action_space):
            next_state = state.copy()
            if action == 0 and state[1] < self.size - 1:
                next_state[1] += 1
            elif action == 1 and state[1] > 0:
                next_state[1] -= 1
            elif action == 2 and state[0] > 0:
                next_state[0] -= 1
            elif action == 3 and state[0] < self.size - 1:
                next_state[0] += 1

            if self.is_safe_state(next_state):
                safe_actions.append(action)

        return safe_actions if safe_actions else list(range(self.action_space))


class ConstrainedPolicyOptimization:
    """Constrained Policy Optimization (CPO) for Safe RL."""

    def __init__(self, state_dim, action_dim, constraint_limit=0.1, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_limit = constraint_limit

        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        ).to(device)

        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        self.cost_value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        self.cost_optimizer = optim.Adam(self.cost_value_network.parameters(), lr=lr)

        self.gamma = 0.99
        self.lam = 0.95  # GAE parameter
        self.clip_ratio = 0.2
        self.target_kl = 0.01
        self.damping = 0.1

        self.constraint_violations = []
        self.policy_losses = []
        self.value_losses = []
        self.cost_losses = []

    def get_action(self, state):
        """Get action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = self.policy_network(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_value_step = values[step + 1]

            delta = (
                rewards[step]
                + self.gamma * next_value_step * next_non_terminal
                - values[step]
            )
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages).to(device)

    def compute_policy_loss(self, states, actions, advantages, old_log_probs):
        """Compute clipped policy loss."""
        action_probs = self.policy_network(states)
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        )

        policy_loss = -torch.min(surr1, surr2).mean()

        kl_div = (old_log_probs - new_log_probs).mean()

        return policy_loss, kl_div

    def compute_constraint_violation(
        self, states, actions, cost_advantages, old_log_probs
    ):
        """Compute expected constraint violation."""
        action_probs = self.policy_network(states)
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        constraint_violation = (ratio * cost_advantages).mean()

        return constraint_violation

    def update(self, trajectories):
        """Update CPO agent with constraint satisfaction."""
        if not trajectories:
            return None

        all_states, all_actions, all_rewards, all_costs = [], [], [], []
        all_dones, all_log_probs = [], []

        for trajectory in trajectories:
            states, actions, rewards, costs, dones, log_probs = zip(*trajectory)
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_costs.extend(costs)
            all_dones.extend(dones)
            all_log_probs.extend(log_probs)

        states = torch.FloatTensor(all_states).to(device)
        actions = torch.LongTensor(all_actions).to(device)
        rewards = torch.FloatTensor(all_rewards).to(device)
        costs = torch.FloatTensor(all_costs).to(device)
        old_log_probs = torch.FloatTensor(all_log_probs).to(device)

        values = self.value_network(states).squeeze()
        cost_values = self.cost_value_network(states).squeeze()

        with torch.no_grad():
            next_value = self.value_network(states[-1:]).squeeze()
            next_cost_value = self.cost_value_network(states[-1:]).squeeze()

        advantages = self.compute_gae(
            all_rewards, values.detach().cpu().numpy(), all_dones, next_value.item()
        )
        cost_advantages = self.compute_gae(
            all_costs,
            cost_values.detach().cpu().numpy(),
            all_dones,
            next_cost_value.item(),
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (
            cost_advantages.std() + 1e-8
        )

        returns = advantages + values.detach()
        cost_returns = cost_advantages + cost_values.detach()

        value_loss = F.mse_loss(values, returns)
        cost_loss = F.mse_loss(cost_values, cost_returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()

        constraint_violation = self.compute_constraint_violation(
            states, actions, cost_advantages, old_log_probs
        )

        policy_loss, kl_div = self.compute_policy_loss(
            states, actions, advantages, old_log_probs
        )

        if constraint_violation.item() <= self.constraint_limit:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_network.parameters(), max_norm=0.5
            )
            self.policy_optimizer.step()
        else:
            print(
                f"⚠️ Policy update skipped due to constraint violation: {constraint_violation.item():.4f}"
            )

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.cost_losses.append(cost_loss.item())
        self.constraint_violations.append(constraint_violation.item())

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "cost_loss": cost_loss.item(),
            "constraint_violation": constraint_violation.item(),
            "kl_divergence": kl_div.item(),
        }


class LagrangianSafeRL:
    """Lagrangian method for Safe RL with adaptive penalty."""

    def __init__(
        self, state_dim, action_dim, constraint_limit=0.1, lr=3e-4, lagrange_lr=1e-2
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.constraint_limit = constraint_limit

        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        ).to(device)

        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        self.lagrange_multiplier = nn.Parameter(torch.tensor(1.0, device=device))
        self.lagrange_optimizer = optim.Adam([self.lagrange_multiplier], lr=lagrange_lr)

        self.gamma = 0.99

        self.lagrange_history = []
        self.constraint_costs = []
        self.total_rewards = []

    def get_action(self, state):
        """Get action with safety consideration."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = self.policy_network(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

        return action.item(), log_prob.item()

    def update(self, trajectories):
        """Update using Lagrangian method."""
        if not trajectories:
            return None

        all_states, all_actions, all_rewards, all_costs = [], [], [], []
        all_log_probs = []

        for trajectory in trajectories:
            states, actions, rewards, costs, _, log_probs = zip(*trajectory)
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_costs.extend(costs)
            all_log_probs.extend(log_probs)

        states = torch.FloatTensor(all_states).to(device)
        actions = torch.LongTensor(all_actions).to(device)
        rewards = torch.FloatTensor(all_rewards).to(device)
        costs = torch.FloatTensor(all_costs).to(device)
        old_log_probs = torch.FloatTensor(all_log_probs).to(device)

        discounted_rewards = []
        discounted_costs = []

        for trajectory in trajectories:
            traj_rewards = [step[2] for step in trajectory]
            traj_costs = [step[3] for step in trajectory]

            reward_return = 0
            cost_return = 0
            for r, c in zip(reversed(traj_rewards), reversed(traj_costs)):
                reward_return = r + self.gamma * reward_return
                cost_return = c + self.gamma * cost_return
                discounted_rewards.insert(0, reward_return)
                discounted_costs.insert(0, cost_return)

        returns = torch.FloatTensor(discounted_rewards).to(device)
        cost_returns = torch.FloatTensor(discounted_costs).to(device)

        values = self.value_network(states).squeeze()
        advantages = returns - values.detach()
        cost_advantages = cost_returns

        action_probs = self.policy_network(states)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        policy_loss = -(
            log_probs * (advantages - self.lagrange_multiplier * cost_advantages)
        ).mean()

        value_loss = F.mse_loss(values, returns)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        avg_cost = cost_returns.mean()
        constraint_violation = avg_cost - self.constraint_limit

        lagrange_loss = -self.lagrange_multiplier * constraint_violation

        self.lagrange_optimizer.zero_grad()
        lagrange_loss.backward()
        self.lagrange_optimizer.step()

        with torch.no_grad():
            self.lagrange_multiplier.clamp_(min=0.0)

        self.lagrange_history.append(self.lagrange_multiplier.item())
        self.constraint_costs.append(avg_cost.item())
        self.total_rewards.append(returns.mean().item())

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "lagrange_multiplier": self.lagrange_multiplier.item(),
            "constraint_violation": constraint_violation.item(),
            "avg_cost": avg_cost.item(),
        }


def collect_safe_trajectory(env, agent, max_steps=50):
    """Collect trajectory with safety information."""
    trajectory = []
    state = env.reset()

    for step in range(max_steps):
        action, log_prob = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        constraint_cost = info["constraint_cost"]

        trajectory.append(
            (state.copy(), action, reward, constraint_cost, done, log_prob)
        )

        if done:
            break

        state = next_state

    return trajectory


def train_safe_rl_algorithms():
    """Train and compare safe RL algorithms."""
    print("🛡️ Training Safe RL Algorithms")
    print("=" * 35)

    env = SafeEnvironment(size=6, constraint_threshold=0.1)

    agents = {
        "CPO": ConstrainedPolicyOptimization(
            state_dim=2, action_dim=4, constraint_limit=0.1
        ),
        "Lagrangian": LagrangianSafeRL(state_dim=2, action_dim=4, constraint_limit=0.1),
    }

    results = {
        name: {"rewards": [], "constraint_violations": [], "episode_lengths": []}
        for name in agents.keys()
    }

    num_episodes = 300
    update_frequency = 10

    for episode in range(num_episodes):
        for agent_name, agent in agents.items():
            trajectories = []
            episode_rewards = []
            episode_violations = []
            episode_lengths = []

            for _ in range(update_frequency):
                trajectory = collect_safe_trajectory(env, agent)
                trajectories.append(trajectory)

                episode_reward = sum(step[2] for step in trajectory)
                episode_violation = sum(step[3] for step in trajectory)
                episode_length = len(trajectory)

                episode_rewards.append(episode_reward)
                episode_violations.append(episode_violation)
                episode_lengths.append(episode_length)

            if trajectories:
                update_info = agent.update(trajectories)

            results[agent_name]["rewards"].extend(episode_rewards)
            results[agent_name]["constraint_violations"].extend(episode_violations)
            results[agent_name]["episode_lengths"].extend(episode_lengths)

        if episode % 50 == 0:
            print(f"\nEpisode {episode}:")
            for agent_name in agents.keys():
                recent_rewards = np.mean(results[agent_name]["rewards"][-50:])
                recent_violations = np.mean(
                    results[agent_name]["constraint_violations"][-50:]
                )
                print(
                    f"  {agent_name}: Reward={recent_rewards:.2f}, Violations={recent_violations:.3f}"
                )

    return results, agents, env


# =============================================================================
# Section 3: Multi-Agent Reinforcement Learning
# =============================================================================


class MultiAgentEnvironment:
    """Multi-agent environment for MARL demonstration."""

    def __init__(self, grid_size=8, num_agents=4, num_targets=3):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.max_episode_steps = 100

        self.reset()

        self.action_space = 5
        self.observation_space = (
            2 + 2 * num_agents + 2 * num_targets
        )  # pos + other_agents + targets

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0

        self.agent_positions = []
        for _ in range(self.num_agents):
            while True:
                pos = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size),
                ]
                if pos not in self.agent_positions:
                    self.agent_positions.append(pos)
                    break

        self.target_positions = []
        for _ in range(self.num_targets):
            while True:
                pos = [
                    np.random.randint(0, self.grid_size),
                    np.random.randint(0, self.grid_size),
                ]
                if pos not in self.agent_positions and pos not in self.target_positions:
                    self.target_positions.append(pos)
                    break

        self.targets_collected = [False] * self.num_targets
        return self.get_observations()

    def get_observations(self):
        """Get observations for all agents."""
        observations = []

        for i in range(self.num_agents):
            obs = []

            obs.extend(
                [
                    self.agent_positions[i][0] / self.grid_size,
                    self.agent_positions[i][1] / self.grid_size,
                ]
            )

            for j in range(self.num_agents):
                if i != j:
                    rel_pos = [
                        (self.agent_positions[j][0] - self.agent_positions[i][0])
                        / self.grid_size,
                        (self.agent_positions[j][1] - self.agent_positions[i][1])
                        / self.grid_size,
                    ]
                    obs.extend(rel_pos)

            for k, target_pos in enumerate(self.target_positions):
                if not self.targets_collected[k]:
                    rel_pos = [
                        (target_pos[0] - self.agent_positions[i][0]) / self.grid_size,
                        (target_pos[1] - self.agent_positions[i][1]) / self.grid_size,
                    ]
                    obs.extend(rel_pos)
                else:
                    obs.extend([0.0, 0.0])  # Target collected

            observations.append(np.array(obs, dtype=np.float32))

        return observations

    def step(self, actions):
        """Execute joint action and return results."""
        self.current_step += 1
        rewards = [0.0] * self.num_agents

        new_positions = []
        for i, action in enumerate(actions):
            pos = self.agent_positions[i].copy()

            if action == 1 and pos[1] < self.grid_size - 1:  # up
                pos[1] += 1
            elif action == 2 and pos[1] > 0:  # down
                pos[1] -= 1
            elif action == 3 and pos[0] > 0:  # left
                pos[0] -= 1
            elif action == 4 and pos[0] < self.grid_size - 1:  # right
                pos[0] += 1

            new_positions.append(pos)

        collision_agents = set()
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if new_positions[i] == new_positions[j]:
                    collision_agents.add(i)
                    collision_agents.add(j)

        for i in range(self.num_agents):
            if i not in collision_agents:
                self.agent_positions[i] = new_positions[i]
            else:
                rewards[i] -= 0.5  # Collision penalty

        targets_collected_this_step = []
        for i in range(self.num_agents):
            for j, target_pos in enumerate(self.target_positions):
                if (
                    not self.targets_collected[j]
                    and self.agent_positions[i] == target_pos
                ):
                    self.targets_collected[j] = True
                    rewards[i] += 10.0  # Target collection reward
                    targets_collected_this_step.append(j)

        if targets_collected_this_step:
            team_bonus = 2.0 * len(targets_collected_this_step)
            for i in range(self.num_agents):
                rewards[i] += team_bonus / self.num_agents

        for i in range(self.num_agents):
            rewards[i] -= 0.1

        done = (
            all(self.targets_collected) or self.current_step >= self.max_episode_steps
        )

        observations = self.get_observations()
        info = {
            "targets_collected": sum(self.targets_collected),
            "total_targets": self.num_targets,
            "collisions": len(collision_agents) // 2,
        }

        return observations, rewards, done, info


class MADDPGAgent:
    """Multi-Agent Deep Deterministic Policy Gradient agent."""

    def __init__(self, obs_dim, action_dim, num_agents, agent_id, lr=1e-3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_id = agent_id

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        ).to(device)

        global_obs_dim = obs_dim * num_agents
        global_action_dim = action_dim * num_agents
        self.critic = nn.Sequential(
            nn.Linear(global_obs_dim + global_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = 0.95
        self.tau = 0.01  # Soft update rate

        self.actor_losses = []
        self.critic_losses = []

    def get_action(self, observation, exploration_noise=0.1):
        """Get action with optional exploration noise."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
            action_probs = self.actor(obs_tensor)

            if exploration_noise > 0:
                noise = torch.randn_like(action_probs) * exploration_noise
                action_probs = torch.softmax(action_probs + noise, dim=-1)

            action_dist = Categorical(action_probs)
            action = action_dist.sample()

        return action.item()

    def update(self, batch, other_agents):
        """Update MADDPG agent using centralized training."""
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards[:, self.agent_id]).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        batch_size = states.shape[0]

        states_flat = states.view(batch_size, -1)
        next_states_flat = next_states.view(batch_size, -1)

        actions_onehot = F.one_hot(actions, num_classes=self.action_dim).float()
        actions_flat = actions_onehot.view(batch_size, -1)

        next_actions = []
        with torch.no_grad():
            for i in range(self.num_agents):
                if i == self.agent_id:
                    next_action_probs = self.actor_target(next_states[:, i])
                else:
                    next_action_probs = other_agents[i].actor_target(next_states[:, i])
                next_actions.append(next_action_probs)

        next_actions_concat = torch.cat(next_actions, dim=-1)

        with torch.no_grad():
            critic_input = torch.cat([next_states_flat, next_actions_concat], dim=-1)
            target_q_values = self.critic_target(critic_input).squeeze()
            target_q_values = rewards + self.gamma * target_q_values * (~dones)

        current_q_input = torch.cat([states_flat, actions_flat], dim=-1)
        current_q_values = self.critic(current_q_input).squeeze()

        critic_loss = F.mse_loss(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        current_actions = []
        for i in range(self.num_agents):
            if i == self.agent_id:
                current_actions.append(self.actor(states[:, i]))
            else:
                with torch.no_grad():
                    current_actions.append(other_agents[i].actor(states[:, i]))

        current_actions_concat = torch.cat(current_actions, dim=-1)
        actor_critic_input = torch.cat([states_flat, current_actions_concat], dim=-1)
        actor_loss = -self.critic(actor_critic_input).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.soft_update()

        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def soft_update(self):
        """Soft update of target networks."""
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class QMIXAgent:
    """QMIX agent with value function factorization."""

    def __init__(self, obs_dim, action_dim, num_agents, state_dim, lr=1e-3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.state_dim = state_dim

        self.q_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim),
                ).to(device)
                for _ in range(num_agents)
            ]
        )

        self.mixing_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_agents * 32),  # Weights for mixing
            nn.ReLU(),
        ).to(device)

        self.final_layer = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
        ).to(device)

        self.target_q_networks = copy.deepcopy(self.q_networks)
        self.target_mixing_network = copy.deepcopy(self.mixing_network)
        self.target_final_layer = copy.deepcopy(self.final_layer)

        all_params = (
            list(self.q_networks.parameters())
            + list(self.mixing_network.parameters())
            + list(self.final_layer.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr)

        self.gamma = 0.95
        self.tau = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        self.losses = []
        self.team_rewards = []

    def get_actions(self, observations):
        """Get actions for all agents."""
        actions = []

        with torch.no_grad():
            for i, obs in enumerate(observations):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                q_values = self.q_networks[i](obs_tensor)

                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.action_dim)
                else:
                    action = q_values.argmax().item()

                actions.append(action)

        return actions

    def mixing_forward(self, individual_q_values, state):
        """Forward pass through mixing network."""
        mixing_weights = self.mixing_network(state)
        mixing_weights = mixing_weights.view(-1, self.num_agents, 32)

        mixing_weights = torch.abs(mixing_weights)

        individual_q_values = individual_q_values.unsqueeze(-1)  # [batch, agents, 1]
        mixed_values = torch.bmm(
            mixing_weights.transpose(1, 2), individual_q_values
        )  # [batch, 32, 1]
        mixed_values = mixed_values.squeeze(-1)  # [batch, 32]

        team_q_value = self.final_layer(mixed_values)

        return team_q_value

    def update(self, batch):
        """Update QMIX agent."""
        states, actions, rewards, next_states, dones = batch

        batch_size = len(states)

        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        team_rewards = torch.FloatTensor([sum(r) for r in rewards]).to(device)
        next_states_tensor = torch.FloatTensor(next_states).to(device)
        dones_tensor = torch.BoolTensor(dones).to(device)

        states_flat = states_tensor.view(batch_size, -1)
        next_states_flat = next_states_tensor.view(batch_size, -1)

        individual_q_values = []
        for i in range(self.num_agents):
            q_vals = self.q_networks[i](states_tensor[:, i])
            chosen_q_vals = q_vals.gather(
                1, actions_tensor[:, i].unsqueeze(1)
            ).squeeze()
            individual_q_values.append(chosen_q_vals)

        individual_q_values = torch.stack(individual_q_values, dim=1)  # [batch, agents]

        team_q_values = self.mixing_forward(individual_q_values, states_flat).squeeze()

        with torch.no_grad():
            next_individual_q_values = []
            for i in range(self.num_agents):
                next_q_vals = self.target_q_networks[i](next_states_tensor[:, i])
                max_next_q_vals = next_q_vals.max(1)[0]
                next_individual_q_values.append(max_next_q_vals)

            next_individual_q_values = torch.stack(next_individual_q_values, dim=1)

            target_mixing_weights = self.target_mixing_network(next_states_flat)
            target_mixing_weights = target_mixing_weights.view(-1, self.num_agents, 32)
            target_mixing_weights = torch.abs(target_mixing_weights)

            next_individual_q_values_expanded = next_individual_q_values.unsqueeze(-1)
            target_mixed_values = torch.bmm(
                target_mixing_weights.transpose(1, 2), next_individual_q_values_expanded
            ).squeeze(-1)

            target_team_q_values = self.target_final_layer(
                target_mixed_values
            ).squeeze()
            target_team_q_values = team_rewards + self.gamma * target_team_q_values * (
                ~dones_tensor
            )

        loss = F.mse_loss(team_q_values, target_team_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_networks.parameters())
            + list(self.mixing_network.parameters())
            + list(self.final_layer.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        if len(self.losses) % 100 == 0:
            self.soft_update_targets()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        self.team_rewards.append(team_rewards.mean().item())

        return {
            "loss": loss.item(),
            "team_reward": team_rewards.mean().item(),
            "epsilon": self.epsilon,
        }

    def soft_update_targets(self):
        """Soft update of target networks."""
        for target, source in zip(self.target_q_networks, self.q_networks):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        for target_param, param in zip(
            self.target_mixing_network.parameters(), self.mixing_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for target_param, param in zip(
            self.target_final_layer.parameters(), self.final_layer.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent learning."""

    def __init__(self, capacity, num_agents, obs_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Store transition in buffer."""
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.next_states.append(None)
            self.dones.append(None)

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch from buffer."""
        if self.size < batch_size:
            return None

        indices = np.random.choice(self.size, batch_size, replace=False)

        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_rewards = [self.rewards[i] for i in indices]
        batch_next_states = [self.next_states[i] for i in indices]
        batch_dones = [self.dones[i] for i in indices]

        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones,
        )


def train_multi_agent_rl():
    """Train and compare multi-agent RL algorithms."""
    print("🤝 Training Multi-Agent RL Algorithms")
    print("=" * 42)

    env = MultiAgentEnvironment(grid_size=8, num_agents=4, num_targets=3)

    obs_dim = env.observation_space
    action_dim = env.action_space
    num_agents = env.num_agents

    maddpg_agents = [
        MADDPGAgent(obs_dim, action_dim, num_agents, i) for i in range(num_agents)
    ]

    state_dim = obs_dim * num_agents  # Global state dimension
    qmix_agent = QMIXAgent(obs_dim, action_dim, num_agents, state_dim)

    maddpg_buffer = MultiAgentReplayBuffer(
        capacity=50000, num_agents=num_agents, obs_dim=obs_dim
    )
    qmix_buffer = MultiAgentReplayBuffer(
        capacity=50000, num_agents=num_agents, obs_dim=obs_dim
    )

    results = {
        "MADDPG": {"rewards": [], "targets_collected": [], "cooperation_rate": []},
        "QMIX": {"rewards": [], "targets_collected": [], "cooperation_rate": []},
    }

    num_episodes = 500
    batch_size = 32

    for episode in range(num_episodes):
        observations = env.reset()
        episode_reward = 0
        targets_collected = 0
        cooperation_events = 0

        for step in range(100):
            actions = []
            for i, agent in enumerate(maddpg_agents):
                action = agent.get_action(observations[i], exploration_noise=0.1)
                actions.append(action)

            next_observations, rewards, done, info = env.step(actions)

            maddpg_buffer.push(observations, actions, rewards, next_observations, done)

            episode_reward += sum(rewards)
            targets_collected = info["targets_collected"]
            if info["targets_collected"] > 0:
                cooperation_events += 1

            observations = next_observations

            if done:
                break

        if maddpg_buffer.size > batch_size:
            batch = maddpg_buffer.sample(batch_size)
            for agent in maddpg_agents:
                agent.update(batch, maddpg_agents)

        results["MADDPG"]["rewards"].append(episode_reward)
        results["MADDPG"]["targets_collected"].append(targets_collected)
        results["MADDPG"]["cooperation_rate"].append(
            cooperation_events / max(1, step + 1)
        )

        observations = env.reset()
        episode_reward = 0
        targets_collected = 0
        cooperation_events = 0

        for step in range(100):
            actions = qmix_agent.get_actions(observations)
            next_observations, rewards, done, info = env.step(actions)

            qmix_buffer.push(observations, actions, rewards, next_observations, done)

            episode_reward += sum(rewards)
            targets_collected = info["targets_collected"]
            if info["targets_collected"] > 0:
                cooperation_events += 1

            observations = next_observations

            if done:
                break

        if qmix_buffer.size > batch_size:
            batch = qmix_buffer.sample(batch_size)
            qmix_agent.update(batch)

        results["QMIX"]["rewards"].append(episode_reward)
        results["QMIX"]["targets_collected"].append(targets_collected)
        results["QMIX"]["cooperation_rate"].append(
            cooperation_events / max(1, step + 1)
        )

        if episode % 50 == 0:
            print(f"\nEpisode {episode}:")
            for alg_name in results.keys():
                recent_rewards = np.mean(results[alg_name]["rewards"][-50:])
                recent_targets = np.mean(results[alg_name]["targets_collected"][-50:])
                print(
                    f"  {alg_name}: Reward={recent_rewards:.2f}, Targets={recent_targets:.2f}"
                )

    return results, maddpg_agents, qmix_agent, env


# =============================================================================
# Section 4: Robust Reinforcement Learning
# =============================================================================


class RobustEnvironment:
    """Environment with uncertainty and domain shifts for robust RL."""

    def __init__(self, base_size=5, noise_level=0.1, domain_shift_prob=0.1):
        self.base_size = base_size
        self.noise_level = noise_level
        self.domain_shift_prob = domain_shift_prob

        self.current_size = base_size
        self.reset()

        self.action_space = 4  # up, down, left, right

    def reset(self):
        """Reset environment with possible domain shift."""
        if np.random.random() < self.domain_shift_prob:
            self.current_size = np.random.choice(
                [self.base_size - 1, self.base_size, self.base_size + 1]
            )
            self.current_size = max(
                3, min(8, self.current_size)
            )  # Keep within reasonable bounds

        self.state = [0, 0]
        self.goal = [self.current_size - 1, self.current_size - 1]
        self.step_count = 0
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        """Take action with added noise and uncertainty."""
        self.step_count += 1

        # Add action noise
        if np.random.random() < self.noise_level:
            action = np.random.randint(4)  # Random action with small probability

        if action == 0 and self.state[1] < self.current_size - 1:  # up
            self.state[1] += 1
        elif action == 1 and self.state[1] > 0:  # down
            self.state[1] -= 1
        elif action == 2 and self.state[0] > 0:  # left
            self.state[0] -= 1
        elif action == 3 and self.state[0] < self.current_size - 1:  # right
            self.state[0] += 1

        # Add observation noise
        noisy_state = np.array(self.state) + np.random.normal(
            0, self.noise_level, size=2
        )
        noisy_state = np.clip(noisy_state, 0, self.current_size - 1)

        done = self.state == self.goal
        reward = 10.0 if done else -0.1

        # Add reward noise
        reward += np.random.normal(0, self.noise_level)

        episode_done = done or self.step_count >= 50

        return (
            noisy_state.astype(np.float32),
            reward,
            episode_done,
            {
                "true_state": self.state.copy(),
                "noise_level": self.noise_level,
                "domain_shift": self.current_size != self.base_size,
            },
        )


class UncertaintyEstimator:
    """Uncertainty estimation module for robust RL."""

    def __init__(self, state_dim, action_dim, ensemble_size=5):
        self.ensemble_size = ensemble_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ensemble = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim),
                ).to(device)
                for _ in range(ensemble_size)
            ]
        )

        self.optimizers = [
            optim.Adam(model.parameters(), lr=1e-3) for model in self.ensemble
        ]

    def get_uncertainty(self, state):
        """Estimate uncertainty using ensemble variance."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            predictions = []

            for model in self.ensemble:
                q_values = model(state_tensor)
                predictions.append(q_values)

            predictions = torch.stack(predictions)  # [ensemble_size, 1, action_dim]
            mean_pred = predictions.mean(0)
            variance = predictions.var(0)

            return mean_pred.squeeze(), variance.squeeze()

    def update_ensemble(self, states, actions, targets):
        """Update ensemble with diverse training."""
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        targets = torch.FloatTensor(targets).to(device)

        total_loss = 0
        for i, model in enumerate(self.ensemble):
            # Add noise to targets for diversity
            noisy_targets = targets + torch.randn_like(targets) * 0.1

            predictions = model(states)
            pred_values = predictions.gather(1, actions.unsqueeze(1)).squeeze()

            loss = F.mse_loss(pred_values, noisy_targets)
            total_loss += loss.item()

            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

        return total_loss / self.ensemble_size


class RobustRLAgent:
    """Robust RL agent with uncertainty estimation and domain randomization."""

    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        ).to(device)

        self.target_q_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.uncertainty_estimator = UncertaintyEstimator(state_dim, action_dim)

        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

        self.update_count = 0
        self.losses = []
        self.uncertainties = []

    def get_action(self, state, exploration_noise=0.1):
        """Get action with uncertainty-aware exploration."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            mean_q, uncertainty = self.uncertainty_estimator.get_uncertainty(state)

            # UCB-style exploration with uncertainty bonus
            ucb_values = mean_q + exploration_noise * torch.sqrt(uncertainty + 1e-8)

            return ucb_values.argmax().item()

    def update(self, batch):
        """Update robust RL agent."""
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # Update Q-network
        q_values = self.q_network(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (~dones)

        q_loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update uncertainty estimator
        uncertainty_loss = self.uncertainty_estimator.update_ensemble(
            states.cpu().numpy(),
            actions.cpu().numpy(),
            target_q_values.detach().cpu().numpy(),
        )

        self.update_count += 1
        if self.update_count % 100 == 0:
            self.soft_update_target()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Track uncertainty
        with torch.no_grad():
            _, uncertainties = self.uncertainty_estimator.get_uncertainty(
                states[0].cpu().numpy()
            )
            avg_uncertainty = uncertainties.mean().item()
            self.uncertainties.append(avg_uncertainty)

        self.losses.append(q_loss.item())

        return {
            "q_loss": q_loss.item(),
            "uncertainty_loss": uncertainty_loss,
            "avg_uncertainty": avg_uncertainty,
            "epsilon": self.epsilon,
        }

    def soft_update_target(self):
        """Soft update of target network."""
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class AdversarialRobustAgent:
    """Adversarial training for robustness against perturbations."""

    def __init__(self, state_dim, action_dim, adversarial_epsilon=0.1, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.adversarial_epsilon = adversarial_epsilon

        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        ).to(device)

        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        self.gamma = 0.99
        self.clip_ratio = 0.2

        self.policy_losses = []
        self.value_losses = []
        self.adversarial_losses = []

    def generate_adversarial_state(self, state):
        """Generate adversarial perturbation."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        state_tensor.requires_grad_(True)

        action_probs = self.policy_network(state_tensor)
        worst_action_prob = action_probs.min()

        worst_action_prob.backward()

        perturbation = self.adversarial_epsilon * state_tensor.grad.sign()
        adversarial_state = state_tensor + perturbation

        return adversarial_state.detach().squeeze().cpu().numpy()

    def get_action(self, state):
        """Get action from policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs = self.policy_network(state_tensor)
            action_dist = Categorical(action_probs)
            return action_dist.sample().item()

    def update(self, trajectories):
        """Update with adversarial training."""
        if not trajectories:
            return None

        all_states, all_actions, all_rewards, all_log_probs = [], [], [], []

        for trajectory in trajectories:
            states, actions, rewards, _, _, log_probs = zip(*trajectory)
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)

        states = torch.FloatTensor(all_states).to(device)
        actions = torch.LongTensor(all_actions).to(device)
        rewards = torch.FloatTensor(all_rewards).to(device)
        old_log_probs = torch.FloatTensor(all_log_probs).to(device)

        # Standard policy update
        returns = []
        for trajectory in trajectories:
            traj_rewards = [step[2] for step in trajectory]
            ret = 0
            for r in reversed(traj_rewards):
                ret = r + self.gamma * ret
                returns.insert(0, ret)

        returns = torch.FloatTensor(returns).to(device)
        values = self.value_network(states).squeeze()
        advantages = returns - values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_probs = self.policy_network(states)
        action_dist = Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        )

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)

        # Adversarial loss
        adversarial_loss = 0
        for state in all_states:
            adv_state = self.generate_adversarial_state(state)
            adv_state_tensor = torch.FloatTensor(adv_state).unsqueeze(0).to(device)

            adv_action_probs = self.policy_network(adv_state_tensor)
            adv_worst_prob = adv_action_probs.min()

            adversarial_loss += adv_worst_prob

        adversarial_loss = adversarial_loss / len(all_states)

        total_policy_loss = policy_loss + 0.1 * adversarial_loss

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.adversarial_losses.append(adversarial_loss.item())

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "adversarial_loss": adversarial_loss.item(),
        }


def train_robust_rl_algorithms():
    """Train and compare robust RL algorithms."""
    print("🛡️ Training Robust RL Algorithms")
    print("=" * 35)

    env = RobustEnvironment(base_size=5, noise_level=0.1, domain_shift_prob=0.1)

    agents = {
        "Standard": RobustRLAgent(state_dim=2, action_dim=4),
        "Adversarial": AdversarialRobustAgent(state_dim=2, action_dim=4),
    }

    results = {
        name: {"rewards": [], "uncertainties": [], "episode_lengths": []}
        for name in agents.keys()
    }

    num_episodes = 300
    update_frequency = 10

    for episode in range(num_episodes):
        for agent_name, agent in agents.items():
            trajectories = []
            episode_rewards = []
            episode_uncertainties = []
            episode_lengths = []

            for _ in range(update_frequency):
                state = env.reset()
                trajectory = []
                episode_reward = 0
                steps = 0

                while steps < 50:
                    action = agent.get_action(state)
                    next_state, reward, done, info = env.step(action)

                    trajectory.append(
                        (state, action, reward, 0, done, 0)
                    )  # Simplified trajectory
                    episode_reward += reward
                    state = next_state
                    steps += 1

                    if done:
                        break

                trajectories.append(trajectory)
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)

                # Track uncertainty for robust agent
                if hasattr(agent, "uncertainty_estimator"):
                    uncertainty = agent.uncertainties[-1] if agent.uncertainties else 0
                    episode_uncertainties.append(uncertainty)

            if trajectories:
                update_info = agent.update(trajectories)

            results[agent_name]["rewards"].extend(episode_rewards)
            results[agent_name]["episode_lengths"].extend(episode_lengths)
            if episode_uncertainties:
                results[agent_name]["uncertainties"].extend(episode_uncertainties)

        if episode % 50 == 0:
            print(f"\nEpisode {episode}:")
            for agent_name in agents.keys():
                recent_rewards = np.mean(results[agent_name]["rewards"][-50:])
                print(f"  {agent_name}: Reward={recent_rewards:.2f}")

    return results, agents, env


# =============================================================================
# Main Training and Analysis Functions
# =============================================================================


def train_all_advanced_rl_algorithms():
    """Train all advanced RL algorithms and provide comprehensive analysis."""
    print("🚀 Training All Advanced RL Algorithms")
    print("=" * 50)

    # Train Offline RL
    print("\n1️⃣ Training Offline RL Algorithms...")
    offline_results, offline_eval, offline_datasets = train_offline_rl()

    # Train Safe RL
    print("\n2️⃣ Training Safe RL Algorithms...")
    safe_results, safe_agents, safe_env = train_safe_rl_algorithms()

    # Train Multi-Agent RL
    print("\n3️⃣ Training Multi-Agent RL Algorithms...")
    marl_results, maddpg_agents, qmix_agent, marl_env = train_multi_agent_rl()

    # Train Robust RL
    print("\n4️⃣ Training Robust RL Algorithms...")
    robust_results, robust_agents, robust_env = train_robust_rl_algorithms()

    return {
        "offline": (offline_results, offline_eval, offline_datasets),
        "safe": (safe_results, safe_agents, safe_env),
        "marl": (marl_results, maddpg_agents, qmix_agent, marl_env),
        "robust": (robust_results, robust_agents, robust_env),
    }


def analyze_advanced_rl_performance(results_dict):
    """Analyze performance across all advanced RL algorithms."""
    print("\n📊 Advanced RL Performance Analysis")
    print("=" * 45)

    analysis = {}

    # Offline RL Analysis
    offline_results, offline_eval, _ = results_dict["offline"]
    print("\n📚 Offline RL Results:")
    for alg, eval_reward in offline_eval.items():
        print(f"  {alg}: {eval_reward:.3f}")

    # Safe RL Analysis
    safe_results, _, _ = results_dict["safe"]
    print("\n🛡️ Safe RL Results:")
    for agent_name, data in safe_results.items():
        final_reward = np.mean(data["rewards"][-50:])
        final_violations = np.mean(data["constraint_violations"][-50:])
        print(
            f"  {agent_name}: Reward={final_reward:.2f}, Violations={final_violations:.3f}"
        )

    # Multi-Agent RL Analysis
    marl_results, _, _, _ = results_dict["marl"]
    print("\n🤝 Multi-Agent RL Results:")
    for alg_name, data in marl_results.items():
        final_reward = np.mean(data["rewards"][-50:])
        final_targets = np.mean(data["targets_collected"][-50:])
        print(f"  {alg_name}: Reward={final_reward:.2f}, Targets={final_targets:.2f}")

    # Robust RL Analysis
    robust_results, _, _ = results_dict["robust"]
    print("\n🛡️ Robust RL Results:")
    for agent_name, data in robust_results.items():
        final_reward = np.mean(data["rewards"][-50:])
        print(f"  {agent_name}: Reward={final_reward:.2f}")

    return analysis


def create_advanced_rl_visualizations(results_dict):
    """Create comprehensive visualizations for advanced RL algorithms."""
    print("\n📈 Creating Advanced RL Visualizations...")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Offline RL Learning Curves
    offline_results, _, _ = results_dict["offline"]
    ax = axes[0, 0]
    for alg_name, data in offline_results.items():
        losses = [r[list(r.keys())[0]] for r in data["losses"]]
        ax.plot(losses, label=alg_name, alpha=0.7)
    ax.set_title("Offline RL Training Losses")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Safe RL Performance
    safe_results, _, _ = results_dict["safe"]
    ax = axes[0, 1]
    for agent_name, data in safe_results.items():
        window_size = 20
        if len(data["rewards"]) >= window_size:
            smoothed = pd.Series(data["rewards"]).rolling(window_size).mean()
            ax.plot(smoothed, label=agent_name, linewidth=2)
    ax.set_title("Safe RL Learning Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Multi-Agent RL Cooperation
    marl_results, _, _, _ = results_dict["marl"]
    ax = axes[0, 2]
    for alg_name, data in marl_results.items():
        window_size = 20
        if len(data["targets_collected"]) >= window_size:
            smoothed = pd.Series(data["targets_collected"]).rolling(window_size).mean()
            ax.plot(smoothed, label=alg_name, linewidth=2)
    ax.set_title("Multi-Agent Target Collection")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Targets Collected")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Robust RL Performance
    robust_results, _, _ = results_dict["robust"]
    ax = axes[0, 3]
    for agent_name, data in robust_results.items():
        window_size = 20
        if len(data["rewards"]) >= window_size:
            smoothed = pd.Series(data["rewards"]).rolling(window_size).mean()
            ax.plot(smoothed, label=agent_name, linewidth=2)
    ax.set_title("Robust RL Learning Curves")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Constraint Violations
    ax = axes[1, 0]
    for agent_name, data in safe_results.items():
        if "constraint_violations" in data:
            window_size = 20
            if len(data["constraint_violations"]) >= window_size:
                smoothed = (
                    pd.Series(data["constraint_violations"]).rolling(window_size).mean()
                )
                ax.plot(smoothed, label=agent_name, linewidth=2)
    ax.set_title("Constraint Violations")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Violations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Multi-Agent Cooperation Rate
    ax = axes[1, 1]
    for alg_name, data in marl_results.items():
        if "cooperation_rate" in data:
            window_size = 20
            if len(data["cooperation_rate"]) >= window_size:
                smoothed = (
                    pd.Series(data["cooperation_rate"]).rolling(window_size).mean()
                )
                ax.plot(smoothed, label=alg_name, linewidth=2)
    ax.set_title("Cooperation Rate")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cooperation Events/Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Uncertainty Evolution (Robust RL)
    ax = axes[1, 2]
    for agent_name, data in robust_results.items():
        if "uncertainties" in data and data["uncertainties"]:
            ax.plot(data["uncertainties"], label=agent_name, linewidth=2)
    ax.set_title("Uncertainty Evolution")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Uncertainty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Algorithm Comparison
    ax = axes[1, 3]
    algorithms = [
        "CQL",
        "IQL",
        "CPO",
        "Lagrangian",
        "MADDPG",
        "QMIX",
        "Standard",
        "Adversarial",
    ]
    performances = []

    # Extract final performances
    offline_results, offline_eval, _ = results_dict["offline"]
    performances.extend([offline_eval["CQL"], offline_eval["IQL"]])

    safe_results, _, _ = results_dict["safe"]
    for agent_name in ["CPO", "Lagrangian"]:
        final_reward = np.mean(safe_results[agent_name]["rewards"][-50:])
        performances.append(final_reward)

    marl_results, _, _, _ = results_dict["marl"]
    for alg_name in ["MADDPG", "QMIX"]:
        final_reward = np.mean(marl_results[alg_name]["rewards"][-50:])
        performances.append(final_reward)

    robust_results, _, _ = results_dict["robust"]
    for agent_name in ["Standard", "Adversarial"]:
        final_reward = np.mean(robust_results[agent_name]["rewards"][-50:])
        performances.append(final_reward)

    bars = ax.bar(algorithms, performances, color="skyblue", alpha=0.7)
    ax.set_title("Algorithm Performance Comparison")
    ax.set_ylabel("Final Average Reward")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, perf in zip(bars, performances):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{perf:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()

    print("✅ Advanced RL visualizations created successfully!")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("🎯 Computer Assignment 14: Advanced Deep Reinforcement Learning")
    print("=" * 70)
    print("Topics: Offline RL, Safe RL, Multi-Agent RL, Robust RL")
    print(
        "Algorithms: CQL, IQL, CPO, Lagrangian, MADDPG, QMIX, Uncertainty Estimation, Adversarial Training"
    )
    print("=" * 70)

    # Train all algorithms
    results_dict = train_all_advanced_rl_algorithms()

    # Analyze performance
    analysis = analyze_advanced_rl_performance(results_dict)

    # Create visualizations
    create_advanced_rl_visualizations(results_dict)

    print("\n🎉 Advanced RL Training Complete!")
    print("📚 Learned algorithms for real-world deployment challenges")
    print("🛡️ Mastered safety, robustness, and multi-agent coordination")
    print("🔬 Ready for cutting-edge RL research and applications!")

    # Summary statistics
    print("\n📊 Final Summary:")
    print("-" * 30)

    # Offline RL summary
    offline_results, offline_eval, _ = results_dict["offline"]
    best_offline = max(offline_eval.items(), key=lambda x: x[1])
    print(f"Best Offline RL: {best_offline[0]} ({best_offline[1]:.3f})")

    # Safe RL summary
    safe_results, _, _ = results_dict["safe"]
    best_safe = max(
        [(name, np.mean(data["rewards"][-50:])) for name, data in safe_results.items()],
        key=lambda x: x[1],
    )
    print(f"Best Safe RL: {best_safe[0]} ({best_safe[1]:.3f})")

    # Multi-Agent RL summary
    marl_results, _, _, _ = results_dict["marl"]
    best_marl = max(
        [(name, np.mean(data["rewards"][-50:])) for name, data in marl_results.items()],
        key=lambda x: x[1],
    )
    print(f"Best Multi-Agent RL: {best_marl[0]} ({best_marl[1]:.3f})")

    # Robust RL summary
    robust_results, _, _ = results_dict["robust"]
    best_robust = max(
        [
            (name, np.mean(data["rewards"][-50:]))
            for name, data in robust_results.items()
        ],
        key=lambda x: x[1],
    )
    print(f"Best Robust RL: {best_robust[0]} ({best_robust[1]:.3f})")

    print("\n🚀 CA14 Complete: Advanced RL mastery achieved!")
    print("Next: CA15 - Final frontiers of deep reinforcement learning")
