"""Hierarchical reinforcement learning agents."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.distributions import Categorical


class OptionsCriticNetwork(nn.Module):
    """Options-Critic architecture for learning hierarchical policies."""

    def __init__(self, state_dim, action_dim, num_options=4, hidden_dim=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options

        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.option_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
            nn.Softmax(dim=-1),
        )

        self.intra_option_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim),
                    nn.Softmax(dim=-1),
                )
                for _ in range(num_options)
            ]
        )

        self.termination_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_options)
            ]
        )

        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_options),
        )

    def forward(self, state):
        """Forward pass through the Options-Critic architecture."""
        features = self.feature_net(state)

        option_probs = self.option_net(features)

        action_probs = torch.stack(
            [net(features) for net in self.intra_option_nets], dim=1
        )

        termination_probs = torch.stack(
            [net(features) for net in self.termination_nets], dim=1
        ).squeeze(-1)

        option_values = self.value_net(features)

        return option_probs, action_probs, termination_probs, option_values


class OptionsCriticAgent:
    """Agent using Options-Critic for hierarchical learning."""

    def __init__(self, state_dim, action_dim, num_options=4, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_options = num_options

        self.network = OptionsCriticNetwork(state_dim, action_dim, num_options)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.current_option = None
        self.option_length = 0
        self.max_option_length = 10

        self.gamma = 0.99
        self.beta_reg = 0.01  # Regularization for termination

        self.option_usage = np.zeros(num_options)
        self.option_lengths = []
        self.losses = []

    def select_option(self, state):
        """Select option using epsilon-greedy on option probabilities."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            option_probs, _, _, _ = self.network(state_tensor)
            return Categorical(option_probs).sample().item()

    def select_action(self, state, option):
        """Select action using the intra-option policy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, action_probs, _, _ = self.network(state_tensor)
            return Categorical(action_probs[0, option]).sample().item()

    def should_terminate(self, state, option):
        """Check if current option should terminate."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, _, termination_probs, _ = self.network(state_tensor)
            return np.random.random() < termination_probs[0, option].item()

    def act(self, state):
        """Full action selection with option management."""
        if (
            self.current_option is None
            or self.should_terminate(state, self.current_option)
            or self.option_length >= self.max_option_length
        ):
            self.current_option = self.select_option(state)
            self.option_usage[self.current_option] += 1
            if self.option_length > 0:
                self.option_lengths.append(self.option_length)
            self.option_length = 0

        action = self.select_action(state, self.current_option)
        self.option_length += 1

        return action, self.current_option

    def update(self, trajectory):
        """Update using Options-Critic learning algorithm."""
        if len(trajectory) < 2:
            return None

        states, actions, rewards, options = zip(*trajectory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        options = torch.LongTensor(options)

        option_probs, action_probs, termination_probs, option_values = self.network(
            states
        )

        returns = torch.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G

        current_option_values = option_values.gather(1, options.unsqueeze(1)).squeeze()
        value_loss = F.mse_loss(current_option_values, returns.detach())

        advantages = returns - current_option_values.detach()

        selected_action_probs = []
        for i in range(len(actions)):
            selected_action_probs.append(action_probs[i, options[i], actions[i]])
        selected_action_probs = torch.stack(selected_action_probs)

        policy_loss = -(torch.log(selected_action_probs) * advantages).mean()

        termination_reg = self.beta_reg * termination_probs.mean()

        total_loss = value_loss + policy_loss + termination_reg

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.losses.append(total_loss.item())

        return {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "termination_reg": termination_reg.item(),
        }


class FeudalNetwork(nn.Module):
    """Feudal Network with Manager-Worker hierarchy."""

    def __init__(
        self, state_dim, action_dim, goal_dim=8, hidden_dim=128, temporal_horizon=10
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.temporal_horizon = temporal_horizon

        self.manager_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LSTM(hidden_dim, hidden_dim),
        )
        self.manager_goal_net = nn.Linear(hidden_dim, goal_dim)

        self.worker_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        self.manager_value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self.worker_value_net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.manager_hidden = None

    def forward(self, state, goal=None):
        """Forward pass through Feudal Network."""
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        manager_features = self.manager_net[0](state)  # First layer
        manager_features = self.manager_net[1](manager_features)  # ReLU

        if self.manager_hidden is None or self.manager_hidden[0].size(1) != batch_size:
            self.manager_hidden = (
                torch.zeros(1, batch_size, self.manager_net[2].hidden_size),
                torch.zeros(1, batch_size, self.manager_net[2].hidden_size),
            )

        lstm_out, self.manager_hidden = self.manager_net[2](
            manager_features.unsqueeze(0), self.manager_hidden
        )
        manager_features = lstm_out.squeeze(0)

        goals = self.manager_goal_net(manager_features)
        goals = F.normalize(goals, p=2, dim=-1)  # Unit normalize goals

        manager_value = self.manager_value_net(manager_features)

        if goal is None:
            goal = goals

        worker_input = torch.cat([state, goal], dim=-1)
        action_probs = self.worker_net(worker_input)
        worker_value = self.worker_value_net(worker_input)

        return goals, action_probs, manager_value, worker_value

    def reset_manager_state(self):
        """Reset manager LSTM state."""
        self.manager_hidden = None


class FeudalAgent:
    """Feudal Networks agent with hierarchical learning."""

    def __init__(self, state_dim, action_dim, goal_dim=8, lr=1e-3, temporal_horizon=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.temporal_horizon = temporal_horizon

        self.network = FeudalNetwork(
            state_dim, action_dim, goal_dim, temporal_horizon=temporal_horizon
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.current_goal = None
        self.goal_step_count = 0

        self.gamma = 0.99
        self.intrinsic_reward_scale = 0.5

        self.manager_losses = []
        self.worker_losses = []
        self.goal_changes = []

    def compute_intrinsic_reward(self, state, next_state, goal):
        """Compute intrinsic reward based on goal achievement."""
        state_diff = next_state - state
        state_diff_norm = np.linalg.norm(state_diff)

        if state_diff_norm > 1e-6:
            cosine_sim = np.dot(state_diff, goal) / (
                state_diff_norm * np.linalg.norm(goal)
            )
            return self.intrinsic_reward_scale * cosine_sim * state_diff_norm
        return 0.0

    def act(self, state):
        """Select action using feudal hierarchy."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            if (
                self.current_goal is None
                or self.goal_step_count >= self.temporal_horizon
            ):
                goals, _, _, _ = self.network(state_tensor)
                self.current_goal = goals[0].numpy()
                self.goal_step_count = 0
                self.goal_changes.append(len(self.goal_changes))

            goal_tensor = torch.FloatTensor(self.current_goal).unsqueeze(0)
            _, action_probs, _, _ = self.network(state_tensor, goal_tensor)
            action = Categorical(action_probs).sample().item()

            self.goal_step_count += 1

        return action

    def update(self, trajectories):
        """Update feudal networks using hierarchical returns."""
        if not trajectories:
            return None

        total_manager_loss = 0
        total_worker_loss = 0
        num_updates = 0

        for traj in trajectories:
            if len(traj) < 2:
                continue

            states, actions, rewards, next_states = zip(*traj)
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)

            self.network.reset_manager_state()

            goals, action_probs, manager_values, worker_values = self.network(states)

            intrinsic_rewards = []
            for i in range(len(states) - 1):
                intrinsic_reward = self.compute_intrinsic_reward(
                    states[i].numpy(), next_states[i].numpy(), goals[i].numpy()
                )
                intrinsic_rewards.append(intrinsic_reward)
            intrinsic_rewards = torch.FloatTensor(intrinsic_rewards)

            manager_returns = torch.zeros_like(rewards)
            G = 0
            for t in reversed(range(len(rewards))):
                G = rewards[t] + self.gamma * G
                manager_returns[t] = G

            manager_advantages = manager_returns - manager_values.squeeze()
            manager_loss = (manager_advantages**2).mean()

            total_rewards = rewards[:-1] + intrinsic_rewards
            worker_returns = torch.zeros_like(total_rewards)
            G = 0
            for t in reversed(range(len(total_rewards))):
                G = total_rewards[t] + self.gamma * G
                worker_returns[t] = G

            worker_advantages = worker_returns - worker_values[:-1].squeeze()

            selected_action_probs = (
                action_probs[:-1].gather(1, actions[:-1].unsqueeze(1)).squeeze()
            )
            worker_policy_loss = -(
                torch.log(selected_action_probs) * worker_advantages.detach()
            ).mean()
            worker_value_loss = (worker_advantages**2).mean()
            worker_loss = worker_policy_loss + 0.5 * worker_value_loss

            total_manager_loss += manager_loss
            total_worker_loss += worker_loss
            num_updates += 1

        if num_updates == 0:
            return None

        avg_manager_loss = total_manager_loss / num_updates
        avg_worker_loss = total_worker_loss / num_updates
        total_loss = avg_manager_loss + avg_worker_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.manager_losses.append(avg_manager_loss.item())
        self.worker_losses.append(avg_worker_loss.item())

        return {
            "manager_loss": avg_manager_loss.item(),
            "worker_loss": avg_worker_loss.item(),
            "total_loss": total_loss.item(),
        }
