"""
Advanced DQN Extensions Module

This module contains advanced extensions and experimental features for DQN:
- Huber loss implementations
- Novelty-based prioritization
- Multi-objective DQN
- Additional utility classes

Author: DRL Course CA5
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, deque
import random


# =====================================================================
# Huber Loss DQN Extensions
# =====================================================================

class DoubleDQNHuberAgent:
    """
    Double DQN agent with Huber loss for more robust training.

    Combines overestimation bias reduction with outlier-robust loss function.
    """

    def __init__(self, state_size, action_size, lr=1e-3, huber_delta=1.0):
        from agents.dqn_base import DQN
        
        self.state_size = state_size
        self.action_size = action_size
        self.huber_delta = huber_delta

        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.gamma = 0.99
        self.tau = 1e-3

    def compute_double_dqn_targets(self, rewards, next_states, dones):
        """
        Compute Double DQN targets using current network for action selection
        and target network for action evaluation.
        """
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            )
            targets = rewards + (self.gamma * next_q_values * (1 - dones))

        return targets

    def huber_loss(self, td_errors, delta=1.0):
        """
        Compute Huber loss for TD errors.

        Args:
            td_errors: Temporal difference errors (predicted - target)
            delta: Threshold for switching between L2 and L1 loss

        Returns:
            Huber loss values
        """
        abs_errors = torch.abs(td_errors)
        quadratic = torch.clamp(abs_errors, max=delta)
        linear = abs_errors - quadratic

        return 0.5 * quadratic.pow(2) + delta * linear

    def train_step(self, states, actions, rewards, next_states, dones):
        """Single training step with Double DQN + Huber loss."""
        current_q_values = self.q_network(states).gather(1, actions)
        targets = self.compute_double_dqn_targets(rewards, next_states, dones)
        td_errors = current_q_values - targets
        loss = self.huber_loss(td_errors, self.huber_delta).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update_target_network()

        return loss.item()

    def soft_update_target_network(self):
        """Soft update target network parameters."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


# =====================================================================
# Novelty-Based Prioritization
# =====================================================================

class NoveltyEstimator:
    """
    Estimates state novelty using multiple methods.
    Combines count-based and neural approaches for robust novelty estimation.
    """

    def __init__(self, state_dim, method="hybrid", k_neighbors=5):
        self.state_dim = state_dim
        self.method = method
        self.k_neighbors = k_neighbors

        self.visit_counts = defaultdict(int)
        self.state_buffer = deque(maxlen=10000)

        self.density_model = self._build_density_model()
        self.density_optimizer = torch.optim.Adam(
            self.density_model.parameters(), lr=1e-3
        )

        self.knn_model = NearestNeighbors(n_neighbors=k_neighbors)
        self.knn_fitted = False

    def _build_density_model(self):
        """Simple autoencoder for density estimation."""
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.state_dim),
        )

    def _discretize_state(self, state, bins=20):
        """Convert continuous state to discrete for counting."""
        discrete_state = tuple(np.round(state * bins).astype(int))
        return discrete_state

    def update(self, state):
        """Update novelty estimator with new state."""
        discrete_state = self._discretize_state(state)
        self.visit_counts[discrete_state] += 1
        self.state_buffer.append(state)

        if len(self.state_buffer) > 100:
            self._update_density_model(state)

        if (
            len(self.state_buffer) % 100 == 0
            and len(self.state_buffer) > self.k_neighbors
        ):
            self.knn_model.fit(list(self.state_buffer))
            self.knn_fitted = True

    def _update_density_model(self, state):
        """Update density model with single state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        reconstruction = self.density_model(state_tensor)
        loss = nn.MSELoss()(reconstruction, state_tensor)

        self.density_optimizer.zero_grad()
        loss.backward()
        self.density_optimizer.step()

    def compute_novelty(self, state):
        """Compute novelty score for given state."""
        if self.method == "count":
            return self._count_based_novelty(state)
        elif self.method == "neural":
            return self._neural_novelty(state)
        elif self.method == "knn":
            return self._knn_novelty(state)
        elif self.method == "hybrid":
            return self._hybrid_novelty(state)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _count_based_novelty(self, state):
        """Count-based novelty estimation."""
        discrete_state = self._discretize_state(state)
        count = self.visit_counts.get(discrete_state, 0)
        return 1.0 / np.sqrt(count + 1)

    def _neural_novelty(self, state):
        """Neural density-based novelty."""
        if len(self.state_buffer) < 100:
            return 1.0

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            reconstruction = self.density_model(state_tensor)
            reconstruction_error = nn.MSELoss()(reconstruction, state_tensor).item()

        return np.clip(reconstruction_error, 0, 10)

    def _knn_novelty(self, state):
        """k-NN based novelty in feature space."""
        if not self.knn_fitted or len(self.state_buffer) < self.k_neighbors:
            return 1.0

        distances, _ = self.knn_model.kneighbors([state])
        mean_distance = np.mean(distances)
        return np.clip(mean_distance, 0, 10)

    def _hybrid_novelty(self, state):
        """Combine multiple novelty measures."""
        count_novelty = self._count_based_novelty(state)
        neural_novelty = self._neural_novelty(state)
        knn_novelty = self._knn_novelty(state)

        hybrid = 0.4 * count_novelty + 0.3 * neural_novelty + 0.3 * knn_novelty
        return hybrid


class NoveltyPrioritizedReplayBuffer:
    """
    Enhanced prioritized replay buffer with novelty-based priorities.

    Combines TD error with state novelty for more effective experience sampling.
    """

    def __init__(
        self,
        capacity,
        state_dim,
        alpha_td=0.6,
        alpha_novelty=0.4,
        beta=0.4,
        epsilon=1e-6,
    ):
        from agents.prioritized_replay import SumTree
        
        self.capacity = capacity
        self.alpha_td = alpha_td
        self.alpha_novelty = alpha_novelty
        self.beta = beta
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        self.novelty_estimator = NoveltyEstimator(state_dim, method="hybrid")

        self.priority_history = []
        self.td_error_history = []
        self.novelty_history = []

    def add(self, state, action, reward, next_state, done, td_error=None):
        """Add experience with hybrid priority."""
        self.novelty_estimator.update(state)
        novelty = self.novelty_estimator.compute_novelty(state)

        if td_error is not None:
            td_component = abs(td_error)
        else:
            td_component = self.max_priority

        priority = (
            self.alpha_td * td_component + self.alpha_novelty * novelty + self.epsilon
        )

        experience = (state, action, reward, next_state, done)
        self.tree.add(priority, experience)

        self.max_priority = max(self.max_priority, priority)

        self.priority_history.append(priority)
        self.td_error_history.append(td_component)
        self.novelty_history.append(novelty)

    def sample(self, batch_size):
        """Sample batch with hybrid priorities."""
        batch = []
        indices = []
        priorities = []

        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            left = segment * i
            right = segment * (i + 1)
            sample_value = np.random.uniform(left, right)

            tree_idx, priority, experience = self.tree.get_leaf(sample_value)

            batch.append(experience)
            indices.append(tree_idx)
            priorities.append(priority)

        sampling_probs = np.array(priorities) / self.tree.total_priority
        weights = np.power(self.tree.size * sampling_probs, -self.beta)
        weights = weights / weights.max()

        return batch, indices, weights

    def update_priorities(self, indices, td_errors, states):
        """Update priorities with new TD errors and current novelty."""
        for idx, td_error, state in zip(indices, td_errors, states):
            novelty = self.novelty_estimator.compute_novelty(state)

            priority = (
                self.alpha_td * abs(td_error)
                + self.alpha_novelty * novelty
                + self.epsilon
            )

            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class NoveltyPriorityDebugger:
    """Debug and analyze novelty-based prioritization."""

    def __init__(self, buffer):
        self.buffer = buffer

    def plot_priority_components(self):
        """Plot TD error vs novelty contributions to priority."""
        if len(self.buffer.priority_history) < 100:
            print("Not enough data for analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].plot(self.buffer.priority_history[-1000:])
        axes[0, 0].set_title("Priority Evolution (Last 1000)")
        axes[0, 0].set_ylabel("Priority")

        td_errors = np.array(self.buffer.td_error_history[-1000:])
        novelties = np.array(self.buffer.novelty_history[-1000:])

        axes[0, 1].scatter(td_errors, novelties, alpha=0.6)
        axes[0, 1].set_xlabel("TD Error Component")
        axes[0, 1].set_ylabel("Novelty Component")
        axes[0, 1].set_title("TD Error vs Novelty")

        axes[1, 0].hist(self.buffer.priority_history[-1000:], bins=50, alpha=0.7)
        axes[1, 0].set_xlabel("Priority")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Priority Distribution")

        td_contrib = self.buffer.alpha_td * td_errors
        novelty_contrib = self.buffer.alpha_novelty * novelties

        axes[1, 1].hist(td_contrib, bins=30, alpha=0.5, label="TD Component")
        axes[1, 1].hist(novelty_contrib, bins=30, alpha=0.5, label="Novelty Component")
        axes[1, 1].set_xlabel("Contribution to Priority")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Component Contributions")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()


# =====================================================================
# Multi-Objective DQN
# =====================================================================

class MultiObjectiveDQN(nn.Module):
    """
    Multi-Objective Deep Q-Network with separate heads for each objective.

    Learns separate Q-functions for each objective and combines them
    for action selection using various scalarization methods.
    """

    def __init__(self, state_size, action_size, num_objectives, hidden_size=128):
        super(MultiObjectiveDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_objectives = num_objectives

        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.objective_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, action_size),
                )
                for _ in range(num_objectives)
            ]
        )

    def forward(self, x):
        """
        Forward pass returning Q-values for all objectives.

        Returns:
            List of tensors, each of shape (batch_size, action_size)
        """
        shared_features = self.shared_layers(x)

        q_values_per_objective = []
        for head in self.objective_heads:
            q_values = head(shared_features)
            q_values_per_objective.append(q_values)

        return q_values_per_objective


class MultiObjectiveDQNAgent:
    """
    Multi-Objective DQN Agent with various scalarization methods.

    Supports different action selection strategies for multi-objective optimization.
    """

    def __init__(
        self,
        state_size,
        action_size,
        num_objectives,
        scalarization="linear",
        objective_weights=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.num_objectives = num_objectives
        self.scalarization = scalarization

        if objective_weights is None:
            self.objective_weights = np.ones(num_objectives) / num_objectives
        else:
            self.objective_weights = np.array(objective_weights)

        self.q_network = MultiObjectiveDQN(state_size, action_size, num_objectives)
        self.target_network = MultiObjectiveDQN(state_size, action_size, num_objectives)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.tau = 1e-3

        self.pareto_solutions = []

    def act(self, state, epsilon=0.1):
        """Select action using multi-objective Q-values."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values_list = self.q_network(state_tensor)

        if self.scalarization == "linear":
            return self._linear_scalarization_action(q_values_list)
        elif self.scalarization == "chebyshev":
            return self._chebyshev_scalarization_action(q_values_list)
        elif self.scalarization == "lexicographic":
            return self._lexicographic_action(q_values_list)
        elif self.scalarization == "pareto":
            return self._pareto_action(q_values_list)
        else:
            raise ValueError(f"Unknown scalarization: {self.scalarization}")

    def _linear_scalarization_action(self, q_values_list):
        """Linear weighted sum of Q-values."""
        q_arrays = [q.detach().numpy()[0] for q in q_values_list]

        scalarized_q = np.zeros(self.action_size)
        for i, (q_vals, weight) in enumerate(zip(q_arrays, self.objective_weights)):
            scalarized_q += weight * q_vals

        return np.argmax(scalarized_q)

    def _chebyshev_scalarization_action(self, q_values_list):
        """Chebyshev scalarization (minimize maximum weighted deviation)."""
        q_arrays = [q.detach().numpy()[0] for q in q_values_list]

        scalarized_q = np.zeros(self.action_size)
        for action in range(self.action_size):
            weighted_objectives = []
            for obj_idx, (q_vals, weight) in enumerate(
                zip(q_arrays, self.objective_weights)
            ):
                weighted_objectives.append(weight * q_vals[action])
            scalarized_q[action] = np.min(weighted_objectives)

        return np.argmax(scalarized_q)

    def _lexicographic_action(self, q_values_list):
        """Lexicographic ordering (prioritize objectives in order)."""
        q_arrays = [q.detach().numpy()[0] for q in q_values_list]

        action_scores = []
        for action in range(self.action_size):
            score_tuple = tuple(q_vals[action] for q_vals in q_arrays)
            action_scores.append((action, score_tuple))

        action_scores.sort(key=lambda x: x[1], reverse=True)

        return action_scores[0][0]

    def _pareto_action(self, q_values_list):
        """Select action from Pareto-optimal set (random choice among non-dominated)."""
        q_arrays = [q.detach().numpy()[0] for q in q_values_list]

        pareto_actions = []
        for action in range(self.action_size):
            action_objectives = [q_vals[action] for q_vals in q_arrays]
            is_dominated = False

            for other_action in range(self.action_size):
                if action == other_action:
                    continue

                other_objectives = [q_vals[other_action] for q_vals in q_arrays]

                if self._dominates(other_objectives, action_objectives):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_actions.append(action)

        if pareto_actions:
            return np.random.choice(pareto_actions)
        else:
            return np.random.randint(self.action_size)

    def _dominates(self, obj1, obj2):
        """Check if obj1 Pareto-dominates obj2."""
        better_in_all = all(o1 >= o2 for o1, o2 in zip(obj1, obj2))
        better_in_some = any(o1 > o2 for o1, o2 in zip(obj1, obj2))
        return better_in_all and better_in_some

    def train_step(self, states, actions, rewards_multi, next_states, dones):
        """Training step with multi-objective rewards."""
        current_q_lists = self.q_network(states)
        current_q_values = []

        for obj_idx, q_values in enumerate(current_q_lists):
            current_q = q_values.gather(1, actions.unsqueeze(1))
            current_q_values.append(current_q)

        with torch.no_grad():
            next_q_lists = self.target_network(next_states)
            targets = []

            for obj_idx, (q_values, rewards) in enumerate(
                zip(next_q_lists, rewards_multi)
            ):
                next_q_max = q_values.max(1)[0].unsqueeze(1)
                target = rewards + (self.gamma * next_q_max * (1 - dones))
                targets.append(target)

        total_loss = 0
        for obj_idx, (current_q, target) in enumerate(zip(current_q_values, targets)):
            loss = nn.MSELoss()(current_q, target)
            total_loss += self.objective_weights[obj_idx] * loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.soft_update_target_network()

        return total_loss.item()

    def soft_update_target_network(self):
        """Soft update target network."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def plot_pareto_front(self, states_sample):
        """Visualize Pareto front (works for 2 or 3 objectives)."""
        pareto_points = self.compute_pareto_front(states_sample)

        if self.num_objectives == 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(
                pareto_points[:, 0],
                pareto_points[:, 1],
                alpha=0.6,
                c="red",
                label="Pareto Front",
            )
            plt.xlabel("Objective 1")
            plt.ylabel("Objective 2")
            plt.title("Pareto Front Visualization")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        elif self.num_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                pareto_points[:, 0],
                pareto_points[:, 1],
                pareto_points[:, 2],
                alpha=0.6,
                c="red",
                s=50,
                label="Pareto Front",
            )
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.set_zlabel("Objective 3")
            ax.set_title("3D Pareto Front Visualization")
            ax.legend()
            plt.show()

        else:
            print(f"Visualization not implemented for {self.num_objectives} objectives")
            print(f"Pareto front contains {len(pareto_points)} points")

    def compute_pareto_front(self, states_sample):
        """Compute and visualize Pareto front for given states."""
        pareto_points = []

        with torch.no_grad():
            for state in states_sample:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values_list = self.q_network(state_tensor)

                action_objectives = []
                for action in range(self.action_size):
                    objectives = []
                    for q_values in q_values_list:
                        objectives.append(q_values[0][action].item())
                    action_objectives.append(objectives)

                pareto_actions = self._find_pareto_optimal(action_objectives)
                pareto_points.extend(pareto_actions)

        return np.array(pareto_points)

    def _find_pareto_optimal(self, objectives_list):
        """Find Pareto-optimal points from list of objective vectors."""
        pareto_optimal = []

        for i, obj1 in enumerate(objectives_list):
            is_dominated = False
            for j, obj2 in enumerate(objectives_list):
                if i != j and self._dominates(obj2, obj1):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_optimal.append(obj1)

        return pareto_optimal


class MultiObjectiveEnvironment:
    """
    Example multi-objective environment: Navigation with safety constraints.

    Objectives:
    1. Reach goal (reward)
    2. Minimize energy consumption
    3. Avoid obstacles (safety)
    """

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.state_size = 2
        self.action_size = 4

        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.obstacles = [(3, 3), (3, 4), (4, 3), (6, 7), (7, 6)]

        self.reset()

    def reset(self):
        """Reset environment to starting position."""
        self.pos = (0, 0)
        return np.array(self.pos, dtype=np.float32)

    def step(self, action):
        """Take action and return multi-objective rewards."""
        moves = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        dx, dy = moves[action]

        new_x = max(0, min(self.grid_size - 1, self.pos[0] + dx))
        new_y = max(0, min(self.grid_size - 1, self.pos[1] + dy))

        old_pos = self.pos
        self.pos = (new_x, new_y)

        rewards = self._compute_rewards(old_pos, self.pos, action)
        done = self.pos == self.goal_pos

        return np.array(self.pos, dtype=np.float32), rewards, done

    def _compute_rewards(self, old_pos, new_pos, action):
        """Compute multi-objective rewards."""
        old_dist = np.sqrt(
            (old_pos[0] - self.goal_pos[0]) ** 2 + (old_pos[1] - self.goal_pos[1]) ** 2
        )
        new_dist = np.sqrt(
            (new_pos[0] - self.goal_pos[0]) ** 2 + (new_pos[1] - self.goal_pos[1]) ** 2
        )

        goal_reward = old_dist - new_dist
        if new_pos == self.goal_pos:
            goal_reward += 10.0

        energy_reward = -0.1

        safety_reward = 0.0
        if new_pos in self.obstacles:
            safety_reward = -5.0

        return [goal_reward, energy_reward, safety_reward]


# =====================================================================
# Utility functions
# =====================================================================

def analyze_loss_functions():
    """Compare MSE vs Huber loss behavior."""
    td_errors = np.linspace(-5, 5, 1000)

    mse_loss = 0.5 * td_errors**2

    delta = 1.0
    abs_errors = np.abs(td_errors)
    huber_loss = np.where(
        abs_errors <= delta, 0.5 * td_errors**2, delta * (abs_errors - 0.5 * delta)
    )

    mse_grad = td_errors
    huber_grad = np.where(abs_errors <= delta, td_errors, delta * np.sign(td_errors))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(td_errors, mse_loss, label="MSE Loss", linewidth=2)
    ax1.plot(td_errors, huber_loss, label="Huber Loss (δ=1.0)", linewidth=2)
    ax1.set_xlabel("TD Error")
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Loss Function Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(td_errors, mse_grad, label="MSE Gradient", linewidth=2)
    ax2.plot(td_errors, huber_grad, label="Huber Gradient", linewidth=2)
    ax2.set_xlabel("TD Error")
    ax2.set_ylabel("Gradient Value")
    ax2.set_title("Gradient Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Key Observations:")
    print(f"- MSE gradient at error=3.0: {3.0}")
    print(f"- Huber gradient at error=3.0: {delta}")
    print("- Huber gradient is clipped, preventing large updates")
    print("- Transition point maintains smoothness for optimization")


if __name__ == "__main__":
    print("Advanced DQN Extensions Module Loaded!")
    print("\nAvailable classes:")
    print("- DoubleDQNHuberAgent: Double DQN with Huber loss")
    print("- NoveltyEstimator: State novelty estimation")
    print("- NoveltyPrioritizedReplayBuffer: Novelty-based prioritization")
    print("- NoveltyPriorityDebugger: Debug novelty priorities")
    print("- MultiObjectiveDQN: Multi-objective neural network")
    print("- MultiObjectiveDQNAgent: Multi-objective DQN agent")
    print("- MultiObjectiveEnvironment: Test environment")
