"""
Advanced Planning Algorithms

This module contains advanced planning implementations including:
- AlphaZero-style MCTS with neural network guidance
- Model Predictive Control with constraints
- Latent space planning with variational autoencoders
- World models with recurrent neural networks
- Multi-step planning with value iteration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical, MultivariateNormal
import random
import copy
from collections import deque
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlphaZeroMCTS:
    """AlphaZero-style MCTS with neural network guidance."""

    def __init__(self, state_dim, action_dim, num_simulations=800, c_puct=1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        # Neural networks for AlphaZero
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)

        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=1e-3)

    def search(self, root_state, dynamics_model):
        """Perform MCTS search with neural network guidance."""
        root = MCTSNode(root_state)

        # Get initial policy and value from neural network
        with torch.no_grad():
            policy_probs = self.policy_network(
                torch.FloatTensor(root_state).unsqueeze(0)
            )
            policy_probs = F.softmax(policy_probs, dim=-1).squeeze().cpu().numpy()
            root.value_estimate = self.value_network(
                torch.FloatTensor(root_state).unsqueeze(0)
            ).item()

        # Expand root with policy priors
        root.expand_with_priors(policy_probs)

        for _ in range(self.num_simulations):
            # Selection
            node = self._select(root)

            # Expansion
            if not node.is_terminal and node.visit_count > 0:
                self._expand(node, dynamics_model)

            # Simulation
            value = self._simulate(node, dynamics_model)

            # Backpropagation
            self._backpropagate(node, value)

        return root

    def _select(self, node):
        """Select path through tree using UCB."""
        while not node.is_leaf():
            node = node.select_child(self.c_puct)
        return node

    def _expand(self, node, dynamics_model):
        """Expand node with all possible actions."""
        if node.is_terminal:
            return

        # Get policy from neural network
        with torch.no_grad():
            policy_probs = self.policy_network(
                torch.FloatTensor(node.state).unsqueeze(0)
            )
            policy_probs = F.softmax(policy_probs, dim=-1).squeeze().cpu().numpy()

        # Add children for all actions
        for action in range(self.action_dim):
            child_state = self._get_next_state(node.state, action, dynamics_model)
            child = MCTSNode(
                child_state, parent=node, action=action, prior=policy_probs[action]
            )
            node.add_child(child)

    def _simulate(self, node, dynamics_model):
        """Simulate from node to get value estimate."""
        if node.is_terminal:
            return node.get_terminal_value()

        # Use neural network value estimate
        with torch.no_grad():
            value = self.value_network(
                torch.FloatTensor(node.state).unsqueeze(0)
            ).item()

        return value

    def _backpropagate(self, node, value):
        """Backpropagate value through tree."""
        while node is not None:
            node.update(value)
            value = -value  # Alternate for alternating moves
            node = node.parent

    def _get_next_state(self, state, action, dynamics_model):
        """Get next state using dynamics model."""
        with torch.no_grad():
            next_state, _ = dynamics_model.predict_mean(
                torch.FloatTensor(state).unsqueeze(0), torch.LongTensor([action])
            )
            return next_state.squeeze().cpu().numpy()

    def get_action_probabilities(self, root):
        """Get action probabilities from MCTS results."""
        if root.is_leaf():
            return np.ones(self.action_dim) / self.action_dim

        visits = np.zeros(self.action_dim)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if visits.sum() == 0:
            return np.ones(self.action_dim) / self.action_dim

        return visits / visits.sum()

    def train_networks(self, states, action_probs, values):
        """Train policy and value networks."""
        states = torch.FloatTensor(states)
        action_probs = torch.FloatTensor(action_probs)
        values = torch.FloatTensor(values)

        # Train policy network
        policy_output = self.policy_network(states)
        policy_loss = F.cross_entropy(policy_output, action_probs.argmax(dim=-1))

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Train value network
        value_output = self.value_network(states)
        value_loss = F.mse_loss(value_output.squeeze(), values)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


class MCTSNode:
    """Node in MCTS tree."""

    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}

        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.value_estimate = 0.0

        self.is_terminal = self._is_terminal()

    def _is_terminal(self):
        """Check if state is terminal."""
        # Simple terminal condition - can be customized
        return False

    def is_leaf(self):
        """Check if node is leaf."""
        return len(self.children) == 0

    def select_child(self, c_puct):
        """Select child using UCB."""
        best_score = float("-inf")
        best_child = None

        for child in self.children.values():
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploitation = child.value_sum / child.visit_count
                exploration = (
                    c_puct
                    * child.prior
                    * math.sqrt(self.visit_count)
                    / (1 + child.visit_count)
                )
                score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand_with_priors(self, priors):
        """Expand node with action priors."""
        for action in range(len(priors)):
            if action not in self.children:
                child = MCTSNode(None, parent=self, action=action, prior=priors[action])
                self.children[action] = child

    def add_child(self, child):
        """Add child to node."""
        self.children[child.action] = child

    def update(self, value):
        """Update node statistics."""
        self.visit_count += 1
        self.value_sum += value

    def get_terminal_value(self):
        """Get terminal value."""
        # Simple terminal value - can be customized
        return 0.0


class PolicyNetwork(nn.Module):
    """Policy network for AlphaZero."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.network(state)


class ValueNetwork(nn.Module):
    """Value network for AlphaZero."""

    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Output between -1 and 1
        )

    def forward(self, state):
        return self.network(state)


class ConstrainedMPC:
    """Model Predictive Control with constraints."""

    def __init__(self, dynamics_model, action_dim, horizon=10, num_samples=1000):
        self.dynamics_model = dynamics_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples

        # Constraint parameters
        self.state_constraints = None
        self.action_constraints = None
        self.safety_constraints = None

    def set_constraints(
        self, state_constraints=None, action_constraints=None, safety_constraints=None
    ):
        """Set constraints for MPC."""
        self.state_constraints = state_constraints
        self.action_constraints = action_constraints
        self.safety_constraints = safety_constraints

    def plan(self, state, goal_state=None, num_iterations=5):
        """Plan optimal action sequence with constraints."""
        state = torch.FloatTensor(state).to(device)

        # Initialize action sequence
        action_sequence = self._initialize_action_sequence()

        for iteration in range(num_iterations):
            # Sample action sequences
            action_sequences = self._sample_action_sequences(action_sequence)

            # Evaluate sequences
            rewards = []
            valid_sequences = []

            for actions in action_sequences:
                if self._check_constraints(state, actions):
                    reward = self._evaluate_sequence(state, actions, goal_state)
                    rewards.append(reward)
                    valid_sequences.append(actions)

            if not valid_sequences:
                # No valid sequences found, relax constraints
                action_sequence = self._relax_constraints_and_plan(state, goal_state)
                break

            # Select best sequence
            best_idx = np.argmax(rewards)
            action_sequence = valid_sequences[best_idx]

            # Update action distribution
            action_sequence = self._update_action_distribution(
                action_sequence, valid_sequences
            )

        return action_sequence[0].cpu().numpy()

    def _initialize_action_sequence(self):
        """Initialize action sequence."""
        if isinstance(self.action_dim, int):  # Discrete actions
            return torch.randint(0, self.action_dim, (self.horizon,)).to(device)
        else:  # Continuous actions
            return torch.randn(self.horizon, self.action_dim).to(device)

    def _sample_action_sequences(self, action_sequence):
        """Sample action sequences around current sequence."""
        sequences = []

        for _ in range(self.num_samples):
            if isinstance(self.action_dim, int):  # Discrete actions
                # Add noise and discretize
                noise = torch.randn_like(action_sequence.float())
                noisy_actions = action_sequence.float() + 0.1 * noise
                actions = torch.clamp(noisy_actions, 0, self.action_dim - 1).long()
            else:  # Continuous actions
                noise = torch.randn_like(action_sequence)
                actions = action_sequence + 0.1 * noise

            sequences.append(actions)

        return sequences

    def _check_constraints(self, state, actions):
        """Check if action sequence satisfies constraints."""
        current_state = state

        for action in actions:
            # Check action constraints
            if self.action_constraints is not None:
                if not self.action_constraints(action):
                    return False

            # Predict next state
            next_state, _ = self.dynamics_model.predict_mean(
                current_state.unsqueeze(0), action.unsqueeze(0)
            )
            next_state = next_state.squeeze()

            # Check state constraints
            if self.state_constraints is not None:
                if not self.state_constraints(next_state):
                    return False

            # Check safety constraints
            if self.safety_constraints is not None:
                if not self.safety_constraints(current_state, action, next_state):
                    return False

            current_state = next_state

        return True

    def _evaluate_sequence(self, state, actions, goal_state):
        """Evaluate action sequence."""
        current_state = state
        total_reward = 0

        for t, action in enumerate(actions):
            next_state, reward = self.dynamics_model.predict_mean(
                current_state.unsqueeze(0), action.unsqueeze(0)
            )
            next_state = next_state.squeeze()

            # Add goal-based reward
            if goal_state is not None:
                goal_reward = -torch.norm(
                    next_state - torch.FloatTensor(goal_state).to(device)
                )
                total_reward += goal_reward * (0.99**t)
            else:
                total_reward += reward * (0.99**t)

            current_state = next_state

        return total_reward.item()

    def _relax_constraints_and_plan(self, state, goal_state):
        """Relax constraints and plan."""
        # Temporarily remove constraints and plan
        original_constraints = (
            self.state_constraints,
            self.action_constraints,
            self.safety_constraints,
        )
        self.state_constraints = None
        self.action_constraints = None
        self.safety_constraints = None

        # Plan without constraints
        action_sequence = self._initialize_action_sequence()

        # Restore constraints
        self.state_constraints, self.action_constraints, self.safety_constraints = (
            original_constraints
        )

        return action_sequence

    def _update_action_distribution(self, best_sequence, valid_sequences):
        """Update action distribution based on valid sequences."""
        # Simple update: add small random noise to best sequence
        noise = torch.randn_like(best_sequence.float()) * 0.01
        return best_sequence.float() + noise


class VariationalLatentPlanner(nn.Module):
    """Latent space planning with variational autoencoder."""

    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=256):
        super(VariationalLatentPlanner, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_std
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Latent dynamics model
        self.latent_dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_std
        )

        # Latent reward model
        self.latent_reward = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # Latent value function
        self.latent_value = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, obs):
        """Encode observation to latent space."""
        encoded = self.encoder(obs)
        mean, log_std = encoded.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        return mean, std

    def reparameterize(self, mean, std):
        """Reparameterization trick."""
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, latent):
        """Decode latent to observation."""
        return self.decoder(latent)

    def predict_next_latent(self, latent, action):
        """Predict next latent state."""
        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        input_tensor = torch.cat([latent, action], dim=-1)
        output = self.latent_dynamics(input_tensor)
        mean, log_std = output.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-20, 2))
        return mean, std

    def predict_latent_reward(self, latent):
        """Predict reward from latent state."""
        return self.latent_reward(latent)

    def get_latent_value(self, latent):
        """Get value from latent state."""
        return self.latent_value(latent)

    def plan_in_latent_space(self, initial_obs, horizon=10, num_candidates=100):
        """Plan action sequence in latent space."""
        # Encode initial observation
        latent_mean, latent_std = self.encode(initial_obs)
        initial_latent = self.reparameterize(latent_mean, latent_std)

        best_actions = None
        best_value = float("-inf")

        for _ in range(num_candidates):
            # Sample random action sequence
            actions = []
            current_latent = initial_latent
            total_value = 0

            for t in range(horizon):
                # Sample random action
                if isinstance(self.action_dim, int):
                    action = torch.randint(0, self.action_dim, (1,)).to(device)
                else:
                    action = torch.randn(1, self.action_dim).to(device)

                actions.append(action)

                # Predict next latent state
                next_latent_mean, next_latent_std = self.predict_next_latent(
                    current_latent, action
                )
                next_latent = self.reparameterize(next_latent_mean, next_latent_std)

                # Get reward and value
                reward = self.predict_latent_reward(next_latent)
                value = self.get_latent_value(next_latent)

                total_value += reward * (0.99**t) + value * (0.99**t)
                current_latent = next_latent

            if total_value > best_value:
                best_value = total_value
                best_actions = actions

        return (
            best_actions[0]
            if best_actions
            else torch.zeros(1, self.action_dim).to(device)
        )

    def train_step(self, obs, actions, next_obs, rewards):
        """Train the VAE and latent models."""
        # Encode observations
        latent_mean, latent_std = self.encode(obs)
        latent = self.reparameterize(latent_mean, latent_std)

        # Decode latent
        reconstructed_obs = self.decode(latent)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed_obs, obs)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(
            1 + torch.log(latent_std**2) - latent_mean**2 - latent_std**2
        )

        # Latent dynamics loss
        next_latent_mean, next_latent_std = self.predict_next_latent(latent, actions)
        next_latent = self.reparameterize(next_latent_mean, next_latent_std)

        # Encode next observations
        next_latent_mean_true, next_latent_std_true = self.encode(next_obs)
        next_latent_true = self.reparameterize(
            next_latent_mean_true, next_latent_std_true
        )

        dynamics_loss = F.mse_loss(next_latent, next_latent_true)

        # Reward prediction loss
        predicted_rewards = self.predict_latent_reward(next_latent)
        reward_loss = F.mse_loss(predicted_rewards.squeeze(), rewards)

        # Total loss
        total_loss = recon_loss + 0.1 * kl_loss + dynamics_loss + reward_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
