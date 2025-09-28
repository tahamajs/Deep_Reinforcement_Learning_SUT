import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ..utils.setup import device, Categorical
import gymnasium as gym
from collections import deque
import random


class CuriosityDrivenAgent:
    """
    Agent with intrinsic curiosity for exploration
    """

    def __init__(
        self, state_dim, action_dim, hidden_dim=128, lr=1e-3, beta=0.2, gamma=0.99
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.beta = beta
        self.gamma = gamma

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        ).to(device)

        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=lr)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=lr)

        self.buffer = deque(maxlen=10000)

        self.episode_rewards = []
        self.intrinsic_rewards = []
        self.extrinsic_rewards = []

    def select_action(self, state, return_log_prob=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.policy_net(state)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        action = dist.sample()

        if return_log_prob:
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

        return action.item()

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic curiosity reward"""

        state_action = torch.cat(
            [torch.FloatTensor(state), torch.FloatTensor([action])]
        ).to(device)

        with torch.no_grad():
            predicted_next_state = self.forward_model(state_action)

        actual_next_state = torch.FloatTensor(next_state).to(device)

        intrinsic_reward = F.mse_loss(predicted_next_state, actual_next_state).item()

        return intrinsic_reward

    def update_curiosity_models(self, batch_size=64):
        """Update forward and inverse models"""
        if len(self.buffer) < batch_size:
            return

        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        state_action = torch.cat([states, actions.unsqueeze(-1).float()], dim=1)
        predicted_next_states = self.forward_model(state_action)
        forward_loss = F.mse_loss(predicted_next_states, next_states)

        state_transitions = torch.cat([states, next_states], dim=1)
        predicted_actions = self.inverse_model(state_transitions)
        inverse_loss = F.cross_entropy(predicted_actions, actions)

        self.forward_optimizer.zero_grad()
        forward_loss.backward()
        self.forward_optimizer.step()

        self.inverse_optimizer.zero_grad()
        inverse_loss.backward()
        self.inverse_optimizer.step()

        return forward_loss.item(), inverse_loss.item()

    def update_policy(self, states, actions, rewards):
        """Update policy with combined rewards"""
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        returns = torch.FloatTensor(returns).to(device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        logits = self.policy_net(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        policy_loss = -(log_probs * returns).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    def train_episode(self, env):
        """Train episode with curiosity"""
        state, _ = env.reset()
        states, actions, extrinsic_rewards, intrinsic_rewards = [], [], [], []
        episode_extrinsic = 0
        episode_intrinsic = 0

        while True:
            action = self.select_action(state)
            next_state, extrinsic_reward, terminated, truncated, _ = env.step(action)

            intrinsic_reward = self.compute_intrinsic_reward(state, action, next_state)

            total_reward = extrinsic_reward + self.beta * intrinsic_reward

            self.buffer.append((state, action, next_state))

            states.append(state)
            actions.append(action)
            extrinsic_rewards.append(extrinsic_reward)
            intrinsic_rewards.append(intrinsic_reward)

            episode_extrinsic += extrinsic_reward
            episode_intrinsic += intrinsic_reward

            state = next_state

            if terminated or truncated:
                break

        self.update_curiosity_models()

        combined_rewards = [
            e + self.beta * i for e, i in zip(extrinsic_rewards, intrinsic_rewards)
        ]
        policy_loss = self.update_policy(states, actions, combined_rewards)

        self.episode_rewards.append(episode_extrinsic)
        self.extrinsic_rewards.append(episode_extrinsic)
        self.intrinsic_rewards.append(episode_intrinsic)

        return episode_extrinsic, episode_intrinsic, policy_loss


class MetaLearningAgent:
    """
    Agent with meta-learning capabilities for few-shot adaptation
    """

    def __init__(
        self, state_dim, action_dim, hidden_dim=128, lr=1e-3, meta_lr=1e-4, gamma=0.99
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.base_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.adaptation_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.base_policy.parameters(), lr=lr)
        self.meta_optimizer = optim.Adam(self.adaptation_net.parameters(), lr=meta_lr)

        self.adaptation_memory = []

        self.episode_rewards = []
        self.adaptation_losses = []

    def adapt_to_task(self, task_samples, adaptation_steps=5):
        """Adapt policy to new task using few samples"""
        adapted_params = {}

        for param_name, param in self.base_policy.named_parameters():
            adapted_params[param_name] = param.clone()

        for _ in range(adaptation_steps):

            states, actions, rewards = zip(
                *random.sample(task_samples, min(10, len(task_samples)))
            )

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)

            adapted_logits = self.base_policy(states)
            adapted_dist = Categorical(logits=adapted_logits)
            adapted_log_probs = adapted_dist.log_prob(actions)

            adaptation_loss = -(adapted_log_probs * rewards).mean()

            adaptation_input = torch.cat(
                [states.mean(dim=0), actions.float().mean(dim=0)]
            ).unsqueeze(0)
            adaptation_params = self.adaptation_net(adaptation_input)

            for i, (param_name, param) in enumerate(
                self.base_policy.named_parameters()
            ):
                if i < adaptation_params.shape[1]:
                    adapted_params[param_name] = (
                        param + 0.1 * adaptation_params[0, i] * param
                    )

        return adapted_params

    def select_action_adapted(self, state, adapted_params):
        """Select action using adapted parameters"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        original_params = {}
        for param_name, param in self.base_policy.named_parameters():
            original_params[param_name] = param.clone()
            param.data = adapted_params[param_name].data

        with torch.no_grad():
            logits = self.base_policy(state)
            probs = F.softmax(logits, dim=1)

        for param_name, param in self.base_policy.named_parameters():
            param.data = original_params[param_name].data

        dist = Categorical(probs)
        action = dist.sample()

        return action.item()

    def meta_update(self, task_losses):
        """Meta-learning update"""
        meta_loss = torch.mean(torch.stack(task_losses))

        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()


class HierarchicalAgent:
    """
    Hierarchical agent with high-level and low-level policies
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.high_level_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        ).to(device)

        self.low_level_policies = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim + 1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                ).to(device)
                for _ in range(4)
            ]
        )

        self.optimizer = optim.Adam(
            [
                {"params": self.high_level_policy.parameters()},
                {"params": self.low_level_policies.parameters()},
            ],
            lr=lr,
        )

        self.episode_rewards = []
        self.goal_successes = []

    def select_goal(self, state):
        """Select high-level goal"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.high_level_policy(state)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        goal = dist.sample()

        return goal.item()

    def select_action(self, state, goal):
        """Select action given goal"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal_tensor = torch.FloatTensor([goal]).unsqueeze(0).to(device)

        state_goal = torch.cat([state, goal_tensor], dim=1)

        with torch.no_grad():
            logits = self.low_level_policies[goal](state_goal)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        action = dist.sample()

        return action.item()

    def update_hierarchical(self, trajectories):
        """Update hierarchical policies"""
        total_loss = 0

        for trajectory in trajectories:
            states, goals, actions, rewards = trajectory

            for t, (state, goal, reward) in enumerate(zip(states, goals, rewards)):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                goal_tensor = torch.LongTensor([goal]).to(device)

                high_logits = self.high_level_policy(state_tensor)
                high_dist = Categorical(logits=high_logits)
                high_log_prob = high_dist.log_prob(goal_tensor)

                high_loss = -high_log_prob * reward
                total_loss += high_loss

            for t, (state, goal, action, reward) in enumerate(
                zip(states, goals, actions, rewards)
            ):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                goal_tensor = torch.FloatTensor([goal]).unsqueeze(0).to(device)
                action_tensor = torch.LongTensor([action]).to(device)

                state_goal = torch.cat([state_tensor, goal_tensor], dim=1)
                low_logits = self.low_level_policies[goal](state_goal)
                low_dist = Categorical(logits=low_logits)
                low_log_prob = low_dist.log_prob(action_tensor)

                low_loss = -low_log_prob * reward
                total_loss += low_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()


class SafeRLAgent:
    """
    Agent with safety constraints using constrained policy optimization
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        cost_limit=10.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.cost_limit = cost_limit

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        self.cost_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.cost_optimizer = optim.Adam(self.cost_net.parameters(), lr=lr)

        self.episode_rewards = []
        self.episode_costs = []
        self.constraint_violations = []

    def select_action(self, state, return_log_prob=False):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.policy_net(state)
            probs = F.softmax(logits, dim=1)

        dist = Categorical(probs)
        action = dist.sample()

        if return_log_prob:
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

        return action.item()

    def compute_cost(self, state, action, next_state):
        """Compute safety cost (placeholder - domain specific)"""

        if len(state) >= 3:
            pole_angle = abs(state[2])
            cost = 1.0 if pole_angle > 0.2 else 0.0
        else:
            cost = 0.0

        return cost

    def update_safe_policy(self, states, actions, rewards, costs, next_states, dones):
        """Update policy with safety constraints"""
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        costs = torch.FloatTensor(costs).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()

        cost_values = self.cost_net(states).squeeze()
        next_cost_values = self.cost_net(next_states).squeeze()

        td_targets = rewards + self.gamma * next_values * (1 - dones)
        cost_td_targets = costs + self.gamma * next_cost_values * (1 - dones)

        value_loss = F.mse_loss(values, td_targets.detach())
        cost_loss = F.mse_loss(cost_values, cost_td_targets.detach())

        advantages = (td_targets - values).detach()
        cost_advantages = (cost_td_targets - cost_values).detach()

        logits = self.policy_net(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        lambda_cost = 1.0
        policy_loss = -(log_probs * (advantages - lambda_cost * cost_advantages)).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item(), value_loss.item(), cost_loss.item()

    def train_episode_safe(self, env):
        """Train episode with safety constraints"""
        state, _ = env.reset()
        states, actions, rewards, costs, next_states, dones = [], [], [], [], [], []
        episode_reward = 0
        episode_cost = 0

        while True:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            cost = self.compute_cost(state, action, next_state)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            costs.append(cost)
            next_states.append(next_state)
            dones.append(terminated or truncated)

            episode_reward += reward
            episode_cost += cost

            state = next_state

            if terminated or truncated:
                break

        policy_loss, value_loss, cost_loss = self.update_safe_policy(
            states, actions, rewards, costs, next_states, dones
        )

        self.episode_rewards.append(episode_reward)
        self.episode_costs.append(episode_cost)
        self.constraint_violations.append(1 if episode_cost > self.cost_limit else 0)

        return episode_reward, episode_cost, policy_loss


def demonstrate_advanced_applications():
    """Demonstrate advanced applications"""
    print("🚀 Advanced Applications Demonstration")

    print("\n🧠 Curiosity-Driven Exploration")
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    curiosity_agent = CuriosityDrivenAgent(state_dim, action_dim, beta=0.1)

    print("Training with curiosity...")
    for episode in range(50):
        extrinsic_reward, intrinsic_reward, loss = curiosity_agent.train_episode(env)
        if (episode + 1) % 10 == 0:
            print(".2f" ".2f")

    print("\n🛡️ Safe Reinforcement Learning")
    safe_agent = SafeRLAgent(state_dim, action_dim, cost_limit=5.0)

    print("Training with safety constraints...")
    for episode in range(50):
        reward, cost, loss = safe_agent.train_episode_safe(env)
        if (episode + 1) % 10 == 0:
            print(".2f" ".2f")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(curiosity_agent.extrinsic_rewards, label="Extrinsic", alpha=0.7)
    axes[0, 0].plot(curiosity_agent.intrinsic_rewards, label="Intrinsic", alpha=0.7)
    axes[0, 0].set_title("Curiosity-Driven Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(safe_agent.episode_rewards, label="Reward", alpha=0.7)
    axes[0, 1].plot(safe_agent.episode_costs, label="Cost", alpha=0.7)
    axes[0, 1].axhline(
        y=safe_agent.cost_limit, color="red", linestyle="--", label="Cost Limit"
    )
    axes[0, 1].set_title("Safe RL: Rewards vs Costs")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    violations = np.cumsum(safe_agent.constraint_violations)
    axes[1, 0].plot(violations, label="Cumulative Violations", alpha=0.7)
    axes[1, 0].set_title("Safety Constraint Violations")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Cumulative Violations")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].text(
        0.5,
        0.5,
        "Hierarchical RL\nand Meta-Learning\nimplementations\navailable in\napplications.py",
        ha="center",
        va="center",
        transform=axes[1, 1].transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
    )
    axes[1, 1].set_title("Advanced Architectures")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    env.close()

    return {"curiosity_agent": curiosity_agent, "safe_agent": safe_agent}


if __name__ == "__main__":
    demonstrate_advanced_applications()
