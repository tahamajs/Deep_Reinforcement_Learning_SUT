"""
Model-Based Reinforcement Learning Components

This module contains implementations of model-based RL algorithms including:
- Dynamics models for environment learning
- Model ensembles for uncertainty quantification
- Model predictive control
- Dyna-Q algorithm combining model-free and model-based learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import random
import copy
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicsModel(nn.Module):
    """Neural network model for environment dynamics."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1),  # next_state + reward
        )

        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + 1),  # Uncertainty for state + reward
            nn.Softplus(),  # Ensure positive uncertainty
        )

    def forward(self, state, action):
        """Predict next state and reward with uncertainty."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)

        if action.dtype == torch.long:
            action_one_hot = torch.zeros(action.size(0), self.action_dim).to(
                action.device
            )
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            action = action_one_hot

        input_tensor = torch.cat([state, action], dim=-1)

        prediction = self.transition_net(input_tensor)
        uncertainty = self.uncertainty_net(input_tensor)

        next_state_mean = prediction[:, : self.state_dim]
        reward_mean = prediction[:, self.state_dim :]

        next_state_std = uncertainty[:, : self.state_dim]
        reward_std = uncertainty[:, self.state_dim :]

        return {
            "next_state_mean": next_state_mean,
            "reward_mean": reward_mean,
            "next_state_std": next_state_std,
            "reward_std": reward_std,
        }

    def sample_prediction(self, state, action):
        """Sample from the predictive distribution."""
        output = self.forward(state, action)

        next_state = torch.normal(output["next_state_mean"], output["next_state_std"])
        reward = torch.normal(output["reward_mean"], output["reward_std"])

        return next_state.squeeze(), reward.squeeze()


class ModelEnsemble:
    """Ensemble of dynamics models for uncertainty quantification."""

    def __init__(self, state_dim, action_dim, ensemble_size=5):
        self.ensemble_size = ensemble_size
        self.models = []
        self.optimizers = []

        for _ in range(ensemble_size):
            model = DynamicsModel(state_dim, action_dim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            self.models.append(model)
            self.optimizers.append(optimizer)

    def train_step(self, states, actions, next_states, rewards):
        """Train all models in the ensemble."""
        total_loss = 0

        for model, optimizer in zip(self.models, self.optimizers):
            optimizer.zero_grad()

            output = model(states, actions)

            state_loss = F.mse_loss(output["next_state_mean"], next_states)
            reward_loss = F.mse_loss(output["reward_mean"], rewards.unsqueeze(-1))

            state_nll = 0.5 * torch.sum(
                ((output["next_state_mean"] - next_states) ** 2)
                / (output["next_state_std"] ** 2)
                + torch.log(output["next_state_std"] ** 2)
            )
            reward_nll = 0.5 * torch.sum(
                ((output["reward_mean"] - rewards.unsqueeze(-1)) ** 2)
                / (output["reward_std"] ** 2)
                + torch.log(output["reward_std"] ** 2)
            )

            loss = state_loss + reward_loss + 0.1 * (state_nll + reward_nll)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / self.ensemble_size

    def predict_ensemble(self, state, action):
        """Get predictions from all models in ensemble."""
        predictions = []

        for model in self.models:
            with torch.no_grad():
                pred = model.sample_prediction(state, action)
                predictions.append(pred)

        return predictions

    def predict_mean(self, state, action):
        """Get ensemble mean prediction."""
        predictions = self.predict_ensemble(state, action)

        next_states = torch.stack([pred[0] for pred in predictions])
        rewards = torch.stack([pred[1] for pred in predictions])

        return next_states.mean(dim=0), rewards.mean(dim=0)


class ModelPredictiveController:
    """Model Predictive Control using learned dynamics."""

    def __init__(self, model_ensemble, action_dim, horizon=10, num_samples=1000):
        self.model_ensemble = model_ensemble
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples

    def plan_action(self, state, goal_state=None):
        """Plan optimal action using MPC."""
        state = torch.FloatTensor(state).to(device)
        best_action = None
        best_value = float("-inf")

        for _ in range(self.num_samples):
            if isinstance(self.action_dim, int):  # Discrete actions
                actions = torch.randint(0, self.action_dim, (self.horizon,)).to(device)
            else:  # Continuous actions
                actions = torch.randn(self.horizon, self.action_dim).to(device)

            total_reward = 0
            current_state = state

            for t in range(self.horizon):
                next_state, reward = self.model_ensemble.predict_mean(
                    current_state, actions[t]
                )

                if goal_state is not None:
                    goal_state_tensor = torch.FloatTensor(goal_state).to(device)
                    goal_reward = -torch.norm(next_state - goal_state_tensor)
                    total_reward += goal_reward * (0.99**t)
                else:
                    total_reward += reward * (0.99**t)

                current_state = next_state

            if total_reward > best_value:
                best_value = total_reward
                best_action = actions[0]

        return (
            best_action.cpu().numpy()
            if best_action is not None
            else np.random.randint(self.action_dim)
        )


class DynaQAgent:
    """Dyna-Q algorithm combining model-free and model-based learning."""

    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        ).to(device)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.model_ensemble = ModelEnsemble(state_dim, action_dim)

        self.buffer = deque(maxlen=100000)

        self.training_stats = {
            "q_losses": [],
            "model_losses": [],
            "planning_rewards": [],
        }

    def get_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def update_q_function(self, batch_size=32):
        """Update Q-function using real experience."""
        if len(self.buffer) < batch_size:
            return 0

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.BoolTensor(dones).to(device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + 0.99 * next_q_values * (~dones)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        self.training_stats["q_losses"].append(loss.item())
        return loss.item()

    def update_model(self, batch_size=32):
        """Update dynamics model."""
        if len(self.buffer) < batch_size:
            return 0

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, _ = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)

        loss = self.model_ensemble.train_step(states, actions, next_states, rewards)
        self.training_stats["model_losses"].append(loss)
        return loss

    def planning_step(self, num_planning_steps=50):
        """Perform planning using the learned model."""
        if len(self.buffer) < 10:
            return 0

        total_planning_reward = 0

        for _ in range(num_planning_steps):
            state, _, _, _, _ = random.choice(self.buffer)
            state_tensor = torch.FloatTensor(state).to(device)

            action = np.random.randint(self.action_dim)
            action_tensor = torch.LongTensor([action]).to(device)

            next_state, reward = self.model_ensemble.predict_mean(
                state_tensor, action_tensor
            )

            with torch.no_grad():
                current_q = self.q_network(state_tensor.unsqueeze(0))[0, action]
                next_q = self.q_network(next_state.unsqueeze(0)).max()
                target_q = reward + 0.99 * next_q

            td_error = target_q - current_q
            q_values = self.q_network(state_tensor.unsqueeze(0))
            q_values[0, action] = current_q + 0.1 * td_error

            total_planning_reward += reward.item()

        avg_planning_reward = total_planning_reward / num_planning_steps
        self.training_stats["planning_rewards"].append(avg_planning_reward)
        return avg_planning_reward
