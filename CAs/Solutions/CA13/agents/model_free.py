"""Model-Free Reinforcement Learning Agents."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod


class ModelFreeAgent(ABC):
    """Base class for model-free RL agents."""

    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

    @abstractmethod
    def get_best_action(self, state):
        """Get best action according to current policy."""
        pass

    @abstractmethod
    def update(self, batch):
        """Update agent from batch of experiences."""
        pass


class DQNAgent(ModelFreeAgent):
    """Deep Q-Network agent (model-free)."""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        super().__init__(state_dim, action_dim, lr)
        self.gamma = gamma

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.target_network = self.q_network.__class__(*self.q_network.parameters())
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_count = 0
        self.losses = []

    def get_best_action(self, state):
        """Get action with highest Q-value."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update(self, batch):
        """Update Q-network using DQN loss."""
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (~dones))

        loss = F.mse_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.update_count += 1

        if self.update_count % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
