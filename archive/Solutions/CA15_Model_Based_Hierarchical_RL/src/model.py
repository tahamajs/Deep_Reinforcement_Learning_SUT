import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Union
from src.utils import to_tensor, get_device
import random


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) for various components."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsModel(nn.Module):
    """Neural network model for environment dynamics."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state_predictor = MLP(state_dim + action_dim, state_dim, hidden_dim)
        self.reward_predictor = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward."""
        x = torch.cat([state, action], dim=-1)
        next_state = self.state_predictor(x)
        reward = self.reward_predictor(x)
        return next_state, reward.squeeze(-1) # Squeeze reward to be 1D


class ModelEnsemble:
    """Ensemble of dynamics models for uncertainty quantification."""

    def __init__(
        self, state_dim: int, action_dim: int, num_models: int = 5, hidden_dim: int = 256
    ):
        self.num_models = num_models
        self.models = nn.ModuleList(
            [DynamicsModel(state_dim, action_dim, hidden_dim) for _ in range(num_models)]
        ).to(get_device())
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=1e-4) for model in self.models
        ]
        self.criterion_state = nn.MSELoss()
        self.criterion_reward = nn.MSELoss()

    def train_step(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor
    ):
        """Train a randomly selected model from the ensemble."""
        model_idx = random.randint(0, self.num_models - 1)
        model = self.models[model_idx]
        optimizer = self.optimizers[model_idx]

        model.train()
        optimizer.zero_grad()

        predicted_next_states, predicted_rewards = model(states, actions)
        loss_state = self.criterion_state(predicted_next_states, next_states)
        loss_reward = self.criterion_reward(predicted_rewards, rewards)
        loss = loss_state + loss_reward

        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward using the ensemble (mean prediction)."""
        self.eval()
        next_states_preds = []
        rewards_preds = []
        with torch.no_grad():
            for model in self.models:
                next_state, reward = model(state, action)
                next_states_preds.append(next_state)
                rewards_preds.append(reward)
        return torch.stack(next_states_preds).mean(dim=0), torch.stack(rewards_preds).mean(dim=0)

    def eval(self):
        for model in self.models:
            model.eval()


class QNetwork(nn.Module):
    """Q-Network for DQN baseline."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    """Actor network for policy-based methods."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # For discrete actions, this would output logits. For continuous, actions directly.
        return self.net(state)


class Critic(nn.Module):
    """Critic network for value approximation."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(state_dim, 1, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class GoalConditionedActor(nn.Module):
    """Actor network for goal-conditioned policies."""

    def __init__(self, state_dim: int, action_dim: int, goal_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(state_dim + goal_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)


class GoalConditionedCritic(nn.Module):
    """Critic network for goal-conditioned value approximation."""

    def __init__(self, state_dim: int, goal_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(state_dim + goal_dim, 1, hidden_dim)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)


class FeudalManager(nn.Module):
    """Manager network for Feudal RL."""

    def __init__(self, state_dim: int, subgoal_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(state_dim, subgoal_dim, hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Outputs a subgoal vector
        return self.net(state)


class FeudalWorker(nn.Module):
    """Worker network for Feudal RL."""

    def __init__(self, state_dim: int, action_dim: int, subgoal_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = MLP(state_dim + subgoal_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, subgoal], dim=-1)
        return self.net(x)


