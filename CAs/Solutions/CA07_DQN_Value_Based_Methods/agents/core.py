"""
Core DQN Implementation for CA07
=================================
This module contains the core DQN implementation including basic DQN,
replay buffer, and utility functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

warnings.filterwarnings("ignore")


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-network for DQN"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Vanilla Deep Q-Network Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Training tracking
        self.losses = []
        self.epsilon_history = []
        self.update_count = 0
        self.episode_rewards = []
        self.episode_losses = []

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def update_epsilon(self):
        """Update epsilon for exploration decay"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def train_step(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q values
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Update network
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        loss_value = loss.item()
        self.losses.append(loss_value)

        return loss_value

    def train_episode(self, env: gym.Env, max_steps: int = 1000) -> Tuple[float, Dict]:
        """Train for one episode"""
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []

        while steps < max_steps:
            # Select and perform action
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            self.replay_buffer.push(state, action, reward, next_state, done)

            # Train
            loss = self.train_step()
            if loss is not None:
                episode_losses.append(loss)

            # Update exploration
            self.update_epsilon()

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        self.episode_rewards.append(total_reward)
        self.episode_losses.append(np.mean(episode_losses) if episode_losses else 0)

        return total_reward, {
            "steps": steps,
            "avg_loss": np.mean(episode_losses) if episode_losses else 0,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
        }

    def evaluate(
        self, env: gym.Env, num_episodes: int = 10, max_steps: int = 1000
    ) -> Dict[str, float]:
        """Evaluate agent performance"""
        rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0

            while steps < max_steps:
                action = self.select_action(state, epsilon=0.0)  # Greedy policy
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            rewards.append(episode_reward)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "median_reward": np.median(rewards),
        }

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
                "episode_rewards": self.episode_rewards,
                "losses": self.losses,
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.losses = checkpoint["losses"]

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().numpy().flatten()

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "total_episodes": len(self.episode_rewards),
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": np.max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": np.min(self.episode_rewards) if self.episode_rewards else 0,
            "mean_loss": np.mean(self.losses) if self.losses else 0,
            "current_epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "update_count": self.update_count,
        }
