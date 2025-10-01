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

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=500,
    ):
        super().__init__(state_dim, action_dim, learning_rate)
        self.gamma = gamma
        self.hidden_dim = hidden_dim

        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.steps_done = 0

        # Q-Network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.q_network = self.network  # Alias for compatibility

        # Target Network
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.update_count = 0
        self.losses = []

        # Initialize replay buffer (will be set externally if needed)
        from ..buffers.replay_buffer import ReplayBuffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def act(self, state, epsilon=None):
        """Select action using epsilon-greedy policy."""
        if epsilon is None:
            # Use decaying epsilon
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                          np.exp(-1.0 * self.steps_done / self.epsilon_decay)
            self.steps_done += 1
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        return self.get_best_action(state)

    def get_best_action(self, state):
        """Get action with highest Q-value."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.network(state_tensor)
            return q_values.argmax().item()

    def update(self, batch_size=32):
        """Update Q-network using DQN loss."""
        if len(self.replay_buffer) < batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size)
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        current_q = self.network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * (~dones))

        loss = F.mse_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.update_count += 1

        if self.update_count % 100 == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss.item()
