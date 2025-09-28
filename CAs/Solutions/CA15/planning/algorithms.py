"""
Advanced Planning and Control Algorithms

This module contains implementations of advanced planning algorithms including:
- Monte Carlo Tree Search (MCTS)
- Model-Based Value Expansion (MVE)
- Latent space planning
- World models for planning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCTSNode:
    """Node in Monte Carlo Tree Search tree."""

    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}

        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

        self.policy_priors = None
        self.value_estimate = 0.0

    def is_leaf(self):
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0

    def is_root(self):
        """Check if node is root (no parent)."""
        return self.parent is None

    def get_value(self):
        """Get average value of node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=1.4):
        """Compute UCB1 score for node selection."""
        if self.visit_count == 0:
            return float("inf")

        exploitation = self.get_value()

        if self.parent is not None:
            exploration = (
                c_puct
                * self.prior
                * math.sqrt(self.parent.visit_count)
                / (1 + self.visit_count)
            )
        else:
            exploration = 0

        return exploitation + exploration

    def select_child(self, c_puct=1.4):
        """Select child with highest UCB score."""
        if self.is_leaf():
            return None

        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))

    def expand(self, actions, priors=None):
        """Expand node by adding children for all possible actions."""
        if priors is None:
            priors = [1.0 / len(actions)] * len(actions)

        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state=None,  # State will be set during simulation
                    parent=self,
                    action=action,
                    prior=prior,
                )

    def backup(self, value):
        """Backup value through the tree."""
        self.visit_count += 1
        self.value_sum += value

        if not self.is_root():
            self.parent.backup(value)


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search for planning."""

    def __init__(self, model, value_network=None, policy_network=None):
        self.model = model
        self.value_network = value_network
        self.policy_network = policy_network
        self.c_puct = 1.4
        self.num_simulations = 100

    def search(self, root_state, num_simulations=None):
        """Perform MCTS search from root state."""
        if num_simulations is None:
            num_simulations = self.num_simulations

        root = MCTSNode(root_state)

        if self.policy_network is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(root_state).unsqueeze(0).to(device)
                policy_logits = self.policy_network(state_tensor)
                priors = F.softmax(policy_logits, dim=-1).squeeze().cpu().numpy()
                root.expand(list(range(len(priors))), priors)
        else:
            num_actions = 4  # Assume 4 actions for simplicity
            root.expand(list(range(num_actions)))

        for _ in range(num_simulations):
            self._simulate(root)

        return root

    def _simulate(self, root):
        """Single MCTS simulation."""
        current = root
        path = []

        while not current.is_leaf():
            current = current.select_child(self.c_puct)
            path.append(current)

        if current.visit_count == 0:
            value = self._evaluate_leaf(current)
        else:
            if hasattr(self.model, "get_possible_actions"):
                actions = self.model.get_possible_actions(current.state)
            else:
                actions = list(range(4))  # Default actions

            current.expand(actions)

            if current.children:
                action = np.random.choice(list(current.children.keys()))
                child = current.children[action]

                if hasattr(self.model, "predict_mean"):
                    next_state, reward = self.model.predict_mean(
                        torch.FloatTensor(current.state).to(device),
                        torch.LongTensor([action]).to(device),
                    )
                    child.state = next_state.cpu().numpy()
                else:
                    child.state = current.state  # Placeholder

                value = self._evaluate_leaf(child)
                path.append(child)
            else:
                value = self._evaluate_leaf(current)

        for node in reversed(path):
            node.backup(value)
        root.backup(value)

    def _evaluate_leaf(self, node):
        """Evaluate leaf node value."""
        if self.value_network is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(device)
                value = self.value_network(state_tensor).item()
        else:
            value = self._rollout(node.state)

        return value

    def _rollout(self, state, depth=10):
        """Random rollout for value estimation."""
        total_reward = 0
        current_state = state

        for i in range(depth):
            action = np.random.randint(4)

            if hasattr(self.model, "predict_mean"):
                next_state, reward = self.model.predict_mean(
                    torch.FloatTensor(current_state).to(device),
                    torch.LongTensor([action]).to(device),
                )
                total_reward += reward.item() * (0.99**i)
                current_state = next_state.cpu().numpy()
            else:
                reward = np.random.randn()
                total_reward += reward * (0.99**i)

        return total_reward

    def get_action_probabilities(self, root):
        """Get action probabilities from MCTS results."""
        if root.is_leaf():
            return np.ones(4) / 4  # Uniform if no children

        visits = []
        actions = []

        for action, child in root.children.items():
            actions.append(action)
            visits.append(child.visit_count)

        if sum(visits) == 0:
            return np.ones(len(actions)) / len(actions)

        visits = np.array(visits)
        probabilities = visits / visits.sum()

        full_probs = np.zeros(4)  # Assume 4 actions
        for action, prob in zip(actions, probabilities):
            if action < len(full_probs):
                full_probs[action] = prob

        return full_probs


class ModelBasedValueExpansion:
    """Model-Based Value Expansion for planning."""

    def __init__(self, model, value_function, expansion_depth=3):
        self.model = model
        self.value_function = value_function
        self.expansion_depth = expansion_depth

    def expand_value(self, state, depth=0):
        """Recursively expand value function using model."""
        if depth >= self.expansion_depth:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                return self.value_function(state_tensor).item()

        num_actions = 4  # Assume discrete action space
        action_values = []

        for action in range(num_actions):
            if hasattr(self.model, "predict_mean"):
                next_state, reward = self.model.predict_mean(
                    torch.FloatTensor(state).to(device),
                    torch.LongTensor([action]).to(device),
                )
                next_state = next_state.cpu().numpy()
                reward = reward.item()
            else:
                next_state = state
                reward = np.random.randn()

            next_value = self.expand_value(next_state, depth + 1)
            action_value = reward + 0.99 * next_value
            action_values.append(action_value)

        return max(action_values)

    def plan_action(self, state):
        """Select best action using value expansion."""
        num_actions = 4
        action_values = []

        for action in range(num_actions):
            if hasattr(self.model, "predict_mean"):
                next_state, reward = self.model.predict_mean(
                    torch.FloatTensor(state).to(device),
                    torch.LongTensor([action]).to(device),
                )
                next_state = next_state.cpu().numpy()
                reward = reward.item()
            else:
                next_state = state
                reward = np.random.randn()

            next_value = self.expand_value(next_state, depth=1)
            action_value = reward + 0.99 * next_value
            action_values.append(action_value)

        return np.argmax(action_values)


class LatentSpacePlanner:
    """Planning in learned latent representations."""

    def __init__(self, encoder, decoder, latent_dynamics, latent_dim=64):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dynamics = latent_dynamics
        self.latent_dim = latent_dim

        self.population_size = 500
        self.elite_fraction = 0.1
        self.num_iterations = 10

    def encode_state(self, state):
        """Encode state to latent representation."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            latent_state = self.encoder(state_tensor)
        return latent_state

    def decode_state(self, latent_state):
        """Decode latent state to observation space."""
        with torch.no_grad():
            decoded_state = self.decoder(latent_state)
        return decoded_state.cpu().numpy()

    def plan_in_latent_space(self, initial_state, horizon=10):
        """Plan action sequence in latent space using CEM."""
        latent_state = self.encode_state(initial_state)

        action_dim = 4  # Assume discrete actions
        action_mean = np.zeros((horizon, action_dim))
        action_std = np.ones((horizon, action_dim))

        best_actions = None
        best_reward = float("-inf")

        for iteration in range(self.num_iterations):
            action_sequences = []
            rewards = []

            for _ in range(self.population_size):
                actions = []
                for t in range(horizon):
                    action_logits = np.random.normal(action_mean[t], action_std[t])
                    action = np.argmax(action_logits)
                    actions.append(action)

                action_sequences.append(actions)

                reward = self._evaluate_latent_sequence(latent_state, actions)
                rewards.append(reward)

            elite_idx = np.argsort(rewards)[
                -int(self.elite_fraction * self.population_size) :
            ]
            elite_actions = [action_sequences[i] for i in elite_idx]

            if max(rewards) > best_reward:
                best_reward = max(rewards)
                best_actions = action_sequences[np.argmax(rewards)]

            if len(elite_actions) > 0:
                elite_array = np.array(elite_actions)
                for t in range(horizon):
                    action_counts = np.bincount(elite_array[:, t], minlength=action_dim)
                    action_probs = action_counts / len(elite_actions)

                    action_mean[t] = np.log(action_probs + 1e-8)
                    action_std[t] *= 0.9  # Reduce exploration over iterations

        return best_actions[0] if best_actions else 0  # Return first action

    def _evaluate_latent_sequence(self, initial_latent_state, actions):
        """Evaluate action sequence in latent space."""
        current_latent = initial_latent_state
        total_reward = 0

        for t, action in enumerate(actions):
            action_tensor = torch.LongTensor([action]).to(device)

            if hasattr(self.latent_dynamics, "forward"):
                with torch.no_grad():
                    next_latent, reward = self.latent_dynamics(
                        current_latent, action_tensor
                    )
                    total_reward += reward.item() * (0.99**t)
                    current_latent = next_latent
            else:
                reward = np.random.randn()
                total_reward += reward * (0.99**t)

        return total_reward


class WorldModel(nn.Module):
    """World model for latent space planning (inspired by PlaNet)."""

    def __init__(self, obs_dim, action_dim, latent_dim=64, hidden_dim=256):
        super(WorldModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # Mean and log_std
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim + 1),  # Next latent state + reward
        )

        self.rnn = nn.GRU(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim * 2)

    def encode(self, obs):
        """Encode observation to latent state."""
        encoded = self.encoder(obs)
        mean, log_std = encoded.chunk(2, dim=-1)
        return mean, log_std

    def decode(self, latent):
        """Decode latent state to observation."""
        return self.decoder(latent)

    def sample_latent(self, mean, log_std):
        """Sample from latent distribution."""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mean + eps * std

    def predict_next(self, latent_state, action):
        """Predict next latent state and reward."""
        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        input_tensor = torch.cat([latent_state, action], dim=-1)
        output = self.dynamics(input_tensor)

        next_latent = output[:, : self.latent_dim]
        reward = output[:, self.latent_dim :]

        return next_latent, reward

    def forward(self, obs_sequence, action_sequence):
        """Forward pass through world model."""
        batch_size, seq_len = obs_sequence.shape[:2]

        obs_flat = obs_sequence.view(-1, self.obs_dim)
        latent_mean, latent_log_std = self.encode(obs_flat)
        latent_mean = latent_mean.view(batch_size, seq_len, self.latent_dim)
        latent_log_std = latent_log_std.view(batch_size, seq_len, self.latent_dim)

        latent_states = self.sample_latent(latent_mean, latent_log_std)

        predicted_latents = []
        predicted_rewards = []

        for t in range(seq_len - 1):
            next_latent, reward = self.predict_next(
                latent_states[:, t], action_sequence[:, t]
            )
            predicted_latents.append(next_latent)
            predicted_rewards.append(reward)

        predicted_latents = torch.stack(predicted_latents, dim=1)
        predicted_rewards = torch.stack(predicted_rewards, dim=1)

        predicted_obs = self.decode(predicted_latents.view(-1, self.latent_dim))
        predicted_obs = predicted_obs.view(batch_size, seq_len - 1, self.obs_dim)

        return {
            "latent_mean": latent_mean,
            "latent_log_std": latent_log_std,
            "predicted_obs": predicted_obs,
            "predicted_rewards": predicted_rewards,
            "latent_states": latent_states,
        }
