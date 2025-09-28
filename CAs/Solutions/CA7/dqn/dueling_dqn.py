"""
Dueling DQN Implementation
=========================

This module implements Dueling DQN, which decomposes the Q-function into
separate value and advantage streams for better learning efficiency.

Key features:
- Separate value and advantage streams
- Better state value estimation
- Improved action selection through advantage learning
- Compatible with Double DQN

Author: CA7 Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import DQN, device
from .double_dqn import DoubleDQNAgent
import numpy as np


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network with separate value and advantage streams

    The dueling architecture decomposes Q(s,a) = V(s) + A(s,a), where:
    - V(s): State value function - how good is state s?
    - A(s,a): Advantage function - how much better is action a in state s?

    This decomposition allows better generalization and more efficient learning.
    """

    def __init__(
        self, state_dim, action_dim, hidden_dims=[256, 256], dueling_type="mean"
    ):
        """
        Initialize Dueling DQN network

        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions for feature extraction
            dueling_type: How to combine value and advantage ('mean', 'max', or 'naive')
        """
        super(DuelingDQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling_type = dueling_type

        # Shared feature extraction layers
        self.feature_layers = nn.Sequential()
        prev_dim = state_dim

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            self.feature_layers.add_module(f"fc{i}", nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.add_module(f"relu{i}", nn.ReLU())
            prev_dim = hidden_dim

        # Split into value and advantage streams
        feature_dim = hidden_dims[-1] if hidden_dims else prev_dim

        # Value stream - outputs single value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 1)
        )

        # Advantage stream - outputs advantage for each action A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, state):
        """
        Forward pass through dueling architecture

        Args:
            state: Batch of states [batch_size, state_dim]

        Returns:
            Tuple of (Q-values, values, advantages)
        """
        # Extract features
        features = self.feature_layers(state)

        # Compute value and advantage
        value = self.value_stream(features)  # [batch_size, 1]
        advantage = self.advantage_stream(features)  # [batch_size, action_dim]

        # Combine to get Q-values
        if self.dueling_type == "mean":
            # Subtract mean advantage to ensure identifiability
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        elif self.dueling_type == "max":
            # Subtract max advantage (alternative approach)
            q_values = value + advantage - advantage.max(dim=1, keepdim=True)[0]
        else:
            # Naive combination (not recommended - leads to identifiability issues)
            q_values = value + advantage

        return q_values, value, advantage

    def get_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            epsilon: Exploration probability

        Returns:
            Selected action (int)
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values, _, _ = self.forward(state_tensor)
                return q_values.argmax().item()


class DuelingDQNAgent(DoubleDQNAgent):
    """
    Dueling DQN Agent combining Double DQN with Dueling Architecture

    This agent combines the benefits of:
    - Double DQN: Reduced overestimation bias
    - Dueling Architecture: Better value decomposition and learning efficiency
    """

    def __init__(self, state_dim, action_dim, dueling_type="mean", **kwargs):
        """
        Initialize Dueling DQN agent

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            dueling_type: Dueling combination type ('mean', 'max', or 'naive')
            **kwargs: Additional arguments passed to parent DoubleDQNAgent
        """
        # Store dueling type before calling parent
        self.dueling_type = dueling_type

        # Initialize parent class first
        super().__init__(state_dim, action_dim, **kwargs)

        # Replace networks with dueling architecture
        self.q_network = DuelingDQN(
            state_dim, action_dim, dueling_type=dueling_type
        ).to(device)
        self.target_network = DuelingDQN(
            state_dim, action_dim, dueling_type=dueling_type
        ).to(device)

        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Update optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=kwargs.get("lr", 1e-4)
        )

        # Additional tracking for value/advantage analysis
        self.value_history = []
        self.advantage_history = []

    def train_step(self):
        """
        Dueling Double DQN training step

        Combines dueling architecture with double DQN target computation.

        Returns:
            Training loss (None if insufficient buffer size)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Current Q-values and decomposition
        current_q_values, current_values, current_advantages = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target computation with dueling
        with torch.no_grad():
            # Use main network to select actions
            next_q_main, _, _ = self.q_network(next_states)
            next_actions = next_q_main.argmax(1)

            # Use target network to evaluate selected actions
            next_q_target, _, _ = self.target_network(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(
                1
            )

            # Compute targets
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Store metrics
        self.losses.append(loss.item())
        self.epsilon_history.append(self.epsilon)

        # Track Q-values, values, and advantages
        with torch.no_grad():
            avg_q_value = current_q_values.mean().item()
            avg_value = current_values.mean().item()
            avg_advantage = current_advantages.mean().item()

            self.q_values_history.append(avg_q_value)
            self.value_history.append(avg_value)
            self.advantage_history.append(avg_advantage)

        return loss.item()

    def get_value_advantage_decomposition(self, state):
        """
        Get value and advantage decomposition for a state

        Args:
            state: Input state

        Returns:
            Dictionary with Q-values, value, and advantages
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values, value, advantage = self.q_network(state_tensor)

            return {
                "q_values": q_values.cpu().numpy().flatten(),
                "value": value.item(),
                "advantage": advantage.cpu().numpy().flatten(),
            }
