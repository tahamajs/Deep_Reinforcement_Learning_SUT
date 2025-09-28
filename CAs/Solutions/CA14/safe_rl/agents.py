"""
Safe Reinforcement Learning Agents

This module implements Constrained Policy Optimization (CPO) and Lagrangian methods
for safe reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
