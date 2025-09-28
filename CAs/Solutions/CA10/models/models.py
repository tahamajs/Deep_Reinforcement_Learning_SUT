import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict, deque
import random
import pickle
from typing import Tuple, List, Dict, Optional, Union
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Plotting configuration
plt.style.use("default")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12

print("Environment setup complete!")
print(f"PyTorch version: {torch.__version__}")
print(f"Gymnasium version: {gym.__version__}")
print(f"NumPy version: {np.__version__}")


class TabularModel:
    """Tabular environment model using counting"""

    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        # Transition counts: N(s,a,s')
        self.transition_counts = np.zeros((num_states, num_actions, num_states))

        # State-action counts: N(s,a)
        self.sa_counts = np.zeros((num_states, num_actions))

        # Reward sums and counts
        self.reward_sums = np.zeros((num_states, num_actions))
        self.reward_counts = np.zeros((num_states, num_actions))

    def update(self, state, action, next_state, reward):
        """Update model with new transition"""
        self.transition_counts[int(state), int(action), int(next_state)] += 1
        self.sa_counts[int(state), int(action)] += 1

        self.reward_sums[int(state), int(action)] += reward
        self.reward_counts[int(state), int(action)] += 1

    def get_transition_prob(self, state, action, next_state):
        """Get transition probability P(s'|s,a)"""
        if self.sa_counts[state, action] == 0:
            return 1.0 / self.num_states  # Uniform prior
        return (
            self.transition_counts[state, action, next_state]
            / self.sa_counts[state, action]
        )

    def get_expected_reward(self, state, action):
        """Get expected reward R(s,a)"""
        if self.reward_counts[state, action] == 0:
            return 0.0  # Neutral prior
        return self.reward_sums[state, action] / self.reward_counts[state, action]

    def sample_transition(self, state, action):
        """Sample next state and reward from model"""
        # Sample next state
        if self.sa_counts[state, action] == 0:
            next_state = np.random.randint(self.num_states)
        else:
            probs = (
                self.transition_counts[state, action] / self.sa_counts[state, action]
            )
            next_state = np.random.choice(self.num_states, p=probs)

        # Get expected reward
        reward = self.get_expected_reward(state, action)

        return next_state, reward

    def get_transition_matrix(self, action):
        """Get full transition matrix P(s'|s,a) for given action"""
        P = np.zeros((self.num_states, self.num_states))

        for s in range(self.num_states):
            if self.sa_counts[s, action] == 0:
                P[s, :] = 1.0 / self.num_states  # Uniform prior
            else:
                P[s, :] = (
                    self.transition_counts[s, action, :] / self.sa_counts[s, action]
                )

        return P

    def get_reward_vector(self, action):
        """Get reward vector R(s,a) for given action"""
        R = np.zeros(self.num_states)

        for s in range(self.num_states):
            R[s] = self.get_expected_reward(s, action)

        return R


class NeuralModel:
    """Neural network environment model"""

    def __init__(self, state_dim, action_dim, hidden_dim=256, ensemble_size=1):
        # super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Create ensemble of models
        self.models = nn.ModuleList()

        for _ in range(ensemble_size):
            model = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim + 1),  # next_state + reward
            )
            self.models.append(model)

    def forward(self, state, action, model_idx=None):
        """Forward pass through model(s)"""
        # Concatenate state and action
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 0:
            action = action.unsqueeze(0)

        # Handle discrete actions
        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        x = torch.cat([state, action], dim=1)

        if model_idx is not None:
            # Use specific model
            output = self.models[model_idx](x)
        else:
            # Use ensemble average
            outputs = torch.stack([model(x) for model in self.models])
            output = outputs.mean(dim=0)

        # Split into next state and reward
        next_state = output[:, : self.state_dim]
        reward = output[:, self.state_dim]

        return next_state, reward

    def predict_with_uncertainty(self, state, action):
        """Predict with uncertainty from ensemble"""
        with torch.no_grad():
            outputs = []

            for i in range(self.ensemble_size):
                next_state, reward = self.forward(state, action, model_idx=i)
                outputs.append(torch.cat([next_state, reward.unsqueeze(1)], dim=1))

            outputs = torch.stack(outputs)  # (ensemble_size, batch_size, state_dim + 1)

            # Compute mean and uncertainty
            mean = outputs.mean(dim=0)
            uncertainty = outputs.std(dim=0)

            next_state_mean = mean[:, : self.state_dim]
            reward_mean = mean[:, self.state_dim]
            next_state_std = uncertainty[:, : self.state_dim]
            reward_std = uncertainty[:, self.state_dim]

            return next_state_mean, reward_mean, next_state_std, reward_std

    def sample_from_model(self, state, action):
        """Sample transition from one random model in ensemble"""
        model_idx = np.random.randint(self.ensemble_size)
        return self.forward(state, action, model_idx=model_idx)

    def sample_transition(self, state, action):
        """Sample next state and reward from model (alias for sample_from_model)"""
        return self.sample_from_model(state, action)


class ModelTrainer:
    """Trainer for neural environment models"""

    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_history = []

    def train_step(self, states, actions, next_states, rewards):
        """Single training step"""
        self.optimizer.zero_grad()

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = (
            torch.LongTensor(actions).to(device)
            if len(actions.shape) == 1
            else torch.FloatTensor(actions).to(device)
        )
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)

        total_loss = 0

        # Train each model in ensemble
        for i in range(self.model.ensemble_size):
            pred_next_states, pred_rewards = self.model.forward(
                states, actions, model_idx=i
            )

            # Compute loss
            state_loss = F.mse_loss(pred_next_states, next_states)
            reward_loss = F.mse_loss(pred_rewards, rewards)

            loss = state_loss + reward_loss
            total_loss += loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train_batch(self, data, epochs=10, batch_size=32):
        """Train on batch of data"""
        states, actions, next_states, rewards = data
        n_samples = len(states)

        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            # Shuffle data
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_rewards = rewards[batch_indices]

                loss = self.train_step(
                    batch_states, batch_actions, batch_next_states, batch_rewards
                )
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
