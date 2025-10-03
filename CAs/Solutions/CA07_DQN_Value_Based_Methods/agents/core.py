"""
Deep Q-Networks (DQN) Core Implementation
=======================================

This module contains the core DQN implementation including:
- DQN neural network architecture
- Experience replay buffer
- Basic DQN agent
- Training utilities

Author: CA7 Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import collections
from collections import deque, namedtuple
import warnings
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class DQN(nn.Module):
    """
    Deep Q-Network implementation

    A neural network that approximates the Q-function Q(s,a) for all actions
    given a state input.
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        """
        Initialize DQN network

        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        """
        Forward pass to compute Q-values

        Args:
            state: Batch of states [batch_size, state_dim]

        Returns:
            Q-values for all actions [batch_size, action_dim]
        """
        return self.network(state)

    def get_action(self, state, epsilon=0.0):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            epsilon: Exploration probability

        Returns:
            Selected action (int)
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.forward(state_tensor)
                return q_values.argmax().item()


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling experiences

    Implements the core idea of experience replay: store experiences and sample
    random batches to break temporal correlations and improve sample efficiency.
    """

    def __init__(self, capacity=100000):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        Store an experience in the buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample a batch of experiences

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)

        states = torch.FloatTensor([e.state for e in batch]).to(device)
        actions = torch.LongTensor([e.action for e in batch]).to(device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(device)
        dones = torch.BoolTensor([e.done for e in batch]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent

    Implements the complete DQN algorithm including training, evaluation,
    and experience management.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
    ):
        """
        Initialize DQN agent

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Experience replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        self.q_network = DQN(state_dim, action_dim).to(device)
        self.target_network = DQN(state_dim, action_dim).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        self.q_values_history = []
        self.epsilon_history = []

    def select_action(self, state):
        """
        Select action using current epsilon-greedy policy

        Args:
            state: Current state

        Returns:
            Selected action
        """
        return self.q_network.get_action(state, self.epsilon)

    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_target_network(self):
        """Update target network with main network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train_step(self):
        """
        Perform one training step

        Returns:
            Training loss (None if insufficient buffer size)
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        self.epsilon_history.append(self.epsilon)

        with torch.no_grad():
            avg_q_value = current_q_values.mean().item()
            self.q_values_history.append(avg_q_value)

        return loss.item()

    def train_episode(self, env, max_steps=1000):
        """
        Train for one episode

        Args:
            env: Gym environment
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (episode_reward, step_count)
        """
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0

        for step in range(max_steps):

            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.store_experience(state, action, reward, next_state, done)

            loss = self.train_step()

            episode_reward += reward
            step_count += 1
            state = next_state

            if done:
                break

        self.episode_rewards.append(episode_reward)
        return episode_reward, step_count

    def evaluate(self, env, num_episodes=10, render=False):
        """
        Evaluate the agent

        Args:
            env: Gym environment
            num_episodes: Number of evaluation episodes
            render: Whether to render environment

        Returns:
            Dictionary with evaluation metrics
        """
        eval_rewards = []

        original_epsilon = self.epsilon
        self.epsilon = 0.0

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            while True:
                action = self.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            eval_rewards.append(episode_reward)

        self.epsilon = original_epsilon

        return {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "min_reward": np.min(eval_rewards),
            "max_reward": np.max(eval_rewards),
        }

    def get_q_values(self, state):
        """
        Get Q-values for a given state

        Args:
            state: Input state

        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.q_network(state_tensor).cpu().numpy().flatten()

    def save(self, path):
        """
        Save agent state

        Args:
            path: Path to save agent
        """
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "episode_rewards": self.episode_rewards,
                "losses": self.losses,
                "q_values_history": self.q_values_history,
                "epsilon_history": self.epsilon_history,
            },
            path,
        )

    def load(self, path):
        """
        Load agent state

        Args:
            path: Path to load agent from
        """
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.losses = checkpoint["losses"]
        self.q_values_history = checkpoint["q_values_history"]
        self.epsilon_history = checkpoint["epsilon_history"]
