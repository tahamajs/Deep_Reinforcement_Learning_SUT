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


class NeuralModel(nn.Module):
    """Neural network environment model built as an ensemble of MLPs.

    This model predicts the next state and reward given a current state and action.
    It uses an ensemble of networks to estimate uncertainty.

    Args:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Dimensionality of the action space.
        hidden_dim (int, optional): Number of neurons in hidden layers. Defaults to 256.
        ensemble_size (int, optional): Number of neural networks in the ensemble. Defaults to 1.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, ensemble_size: int = 1
    ):
        super().__init__()

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

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        model_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through model(s) to predict next state and reward.

        Args:
            state (torch.Tensor): Current state tensor. Expected shape: [batch_size, state_dim].
            action (torch.Tensor): Action tensor. Expected shape: [batch_size, action_dim] (for continuous)
                                       or [batch_size,] (for discrete).
            model_idx (Optional[int]): If provided, uses a specific model from the ensemble.
                                           Otherwise, uses the ensemble average.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - next_state (torch.Tensor): Predicted next state tensor [batch_size, state_dim].
                - reward (torch.Tensor): Predicted reward tensor [batch_size,].
        """
        # Ensure state has batch dimension
        if state.ndim == 1:
            state = state.unsqueeze(0)
        # Ensure action has batch dimension for continuous, or is correctly shaped for discrete
        if action.ndim == 0:
            action = action.unsqueeze(0)
        if action.ndim == 1 and action.dtype == torch.long: # Discrete action, convert to one-hot
            action_one_hot = torch.zeros(action.size(0), self.action_dim, device=action.device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action_input = action_one_hot
        elif action.ndim == 1 and action.dtype == torch.float: # Continuous scalar action
            action_input = action.unsqueeze(1)
        else: # Continuous multi-dimensional action or already one-hot
            action_input = action

        x = torch.cat([state, action_input.float()], dim=1) # Ensure action input is float

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

    def predict_with_uncertainty(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state and reward with uncertainty using the ensemble.

        Args:
            state (torch.Tensor): Current state tensor [batch_size, state_dim].
            action (torch.Tensor): Action tensor [batch_size, action_dim] or [batch_size,].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - next_state_mean (torch.Tensor): Mean predicted next state [batch_size, state_dim].
                - reward_mean (torch.Tensor): Mean predicted reward [batch_size,].
                - next_state_std (torch.Tensor): Standard deviation of next state predictions [batch_size, state_dim].
                - reward_std (torch.Tensor): Standard deviation of reward predictions [batch_size,].
        """
        with torch.no_grad():
            next_states_preds = []
            rewards_preds = []

            for i in range(self.ensemble_size):
                next_state, reward = self.forward(state, action, model_idx=i)
                next_states_preds.append(next_state)
                rewards_preds.append(reward)

            next_states_preds = torch.stack(next_states_preds) # (ensemble_size, batch_size, state_dim)
            rewards_preds = torch.stack(rewards_preds)       # (ensemble_size, batch_size)

            # Compute mean and uncertainty
            next_state_mean = next_states_preds.mean(dim=0)
            reward_mean = rewards_preds.mean(dim=0)

            # Compute standard deviation for uncertainty
            next_state_std = next_states_preds.std(dim=0)
            reward_std = rewards_preds.std(dim=0)

            return next_state_mean, reward_mean, next_state_std, reward_std

    def sample_from_model(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample transition from one random model in ensemble.

        Args:
            state (torch.Tensor): Current state tensor [batch_size, state_dim].
            action (torch.Tensor): Action tensor [batch_size, action_dim] or [batch_size,].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - next_state (torch.Tensor): Predicted next state tensor [batch_size, state_dim].
                - reward (torch.Tensor): Predicted reward tensor [batch_size,].
        """
        model_idx = np.random.randint(self.ensemble_size)
        return self.forward(state, action, model_idx=model_idx)

    def sample_transition(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample next state and reward from model (alias for sample_from_model).

        Args:
            state (torch.Tensor): Current state tensor [batch_size, state_dim].
            action (torch.Tensor): Action tensor [batch_size, action_dim] or [batch_size,].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - next_state (torch.Tensor): Predicted next state tensor [batch_size, state_dim].
                - reward (torch.Tensor): Predicted reward tensor [batch_size,].
        """
        return self.sample_from_model(state, action)


class ModelTrainer:
    """Trainer for neural environment models.

    Args:
        model (NeuralModel): The neural environment model to train.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
    """

    def __init__(self, model: NeuralModel, lr: float = 1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_history = []

    def train_step(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        next_states: Union[np.ndarray, torch.Tensor],
        rewards: Union[np.ndarray, torch.Tensor],
    ) -> float:
        """Perform a single training step for the model.

        Args:
            states (Union[np.ndarray, torch.Tensor]): Batch of current states.
            actions (Union[np.ndarray, torch.Tensor]): Batch of actions taken.
            next_states (Union[np.ndarray, torch.Tensor]): Batch of next states.
            rewards (Union[np.ndarray, torch.Tensor]): Batch of rewards received.

        Returns:
            float: The total loss for this training step.
        """
        self.optimizer.zero_grad()

        # Convert to tensors and move to device
        states = torch.tensor(states, dtype=torch.float32).to(device)
        
        # Determine action dtype based on model's action_dim and if it's discrete
        if self.model.action_dim > 1 and actions.ndim == 1 and np.issubdtype(np.asarray(actions).dtype, np.integer):
            actions = torch.tensor(actions, dtype=torch.long).to(device) # Discrete actions
        else:
            actions = torch.tensor(actions, dtype=torch.float32).to(device) # Continuous or one-hot actions

        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        total_loss = 0

        # Train each model in ensemble
        for i in range(self.model.ensemble_size):
            pred_next_states, pred_rewards = self.model.forward(
                states, actions, model_idx=i
            )

            # Compute loss
            state_loss = F.mse_loss(pred_next_states, next_states)
            reward_loss = F.mse_loss(pred_rewards.squeeze(-1), rewards) # Ensure shapes match: [batch_size,] vs [batch_size, 1]

            loss = state_loss + reward_loss
            total_loss += loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train_batch(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        epochs: int = 10,
        batch_size: int = 32,
    ):
        """Train on a batch of data for multiple epochs.

        Args:
            data (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): A tuple containing
                                                                             (states, actions, next_states, rewards) as numpy arrays.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            batch_size (int, optional): Size of mini-batches. Defaults to 32.
        """
        states, actions, next_states, rewards = data
        n_samples = len(states)

        print(f"Starting training for {epochs} epochs with batch size {batch_size}...")

        for epoch in range(epochs):
            epoch_loss = 0.0
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

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            self.loss_history.append(avg_loss)

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        print("Training complete.")




class ReinforcementLearningAgent:
    """Base class for RL agents"""

    def __init__(self, num_states, num_actions, gamma=0.99, epsilon=0.1, alpha=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        self.403| # ... existing code ...
404| 
405| # =============================================================================
406| # HELPER FUNCTIONS AND UTILITIES
407| # =============================================================================
408| 
409| 
410| def set_seed(seed: int = 42):
411|     """Set random seeds for reproducibility"""
412|     torch.manual_seed(seed)
413|     np.random.seed(seed)
414|     random.seed(seed)
415|     if torch.cuda.is_available():
416|         torch.cuda.manual_seed(seed)
417|         torch.cuda.manual_seed_all(seed)
418| 
419|