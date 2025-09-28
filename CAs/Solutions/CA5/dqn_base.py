"""
Deep Q-Networks (DQN) - Base Implementation
==========================================

This module contains the fundamental DQN implementation including:
- Basic DQN network architecture
- Experience replay buffer
- Standard DQN agent with target networks
- Training and evaluation utilities

Author: CA5 Implementation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    """Deep Q-Network for discrete action spaces"""

    def __init__(self, state_size, action_size, hidden_sizes=[512, 256], dropout=0.1):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            )
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        """Initialize network weights"""
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, state):
        """Forward pass through network"""
        return self.network(state)


class ConvDQN(nn.Module):
    """Convolutional DQN for image-based observations"""

    def __init__(self, action_size, input_channels=4):
        super(ConvDQN, self).__init__()
        self.action_size = action_size

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_out_size = 64 * 7 * 7

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, action_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass through convolutional network"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ReplayBuffer:
    """Replay buffer for storing and sampling experiences"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done"]
        )

    def push(self, state, action, reward, next_state, done):
        """Store an experience tuple"""
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Complete DQN agent with experience replay and target networks"""

    def __init__(
        self,
        state_size,
        action_size,
        lr=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=100000,
        batch_size=32,
        target_update_freq=1000,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.memory = ReplayBuffer(buffer_size)

        self.step_count = 0
        self.episode_count = 0

        self.losses = []
        self.q_values = []
        self.episode_rewards = []

    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train_step(self):
        """Perform one training step if enough experiences are available"""
        if len(self.memory) < self.batch_size:
            return None

        experiences = self.memory.sample(self.batch_size)
        batch = self.experience_to_batch(experiences)

        states, actions, rewards, next_states, dones = batch

        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)

        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.losses.append(loss.item())
        avg_q_value = current_q_values.mean().item()
        self.q_values.append(avg_q_value)

        return loss.item()

    def experience_to_batch(self, experiences):
        """Convert batch of experiences to tensors"""
        states = torch.FloatTensor([e.state for e in experiences]).to(device)
        actions = (
            torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(device)
        )
        rewards = (
            torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(device)
        )
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(device)
        dones = torch.FloatTensor([e.done for e in experiences]).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones

    def train(self, env, num_episodes=1000, print_every=100):
        """Train the DQN agent"""
        scores = []
        losses_per_episode = []

        for episode in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Handle new gym API

            total_reward = 0
            episode_losses = []

            while True:
                action = self.get_action(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)

                self.store_experience(
                    state, action, reward, next_state, done or truncated
                )

                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward

                if done or truncated:
                    break

            scores.append(total_reward)
            losses_per_episode.append(np.mean(episode_losses) if episode_losses else 0)
            self.episode_rewards.append(total_reward)

            if (episode + 1) % print_every == 0:
                avg_score = np.mean(scores[-print_every:])
                avg_loss = np.mean(losses_per_episode[-print_every:])
                print(
                    f"Episode {episode + 1:4d} | "
                    f"Avg Score: {avg_score:7.2f} | "
                    f"Avg Loss: {avg_loss:8.4f} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Buffer Size: {len(self.memory)}"
                )

        return scores, losses_per_episode


def create_test_environment():
    """Create a test environment for DQN"""
    try:
        import gym

        env = gym.make("CartPole-v1")
        return env, env.observation_space.shape[0], env.action_space.n
    except:
        print("CartPole environment not available")
        return None, 4, 2


if __name__ == "__main__":
    print("DQN Base Implementation")
    print("=" * 50)

    dqn = DQN(4, 2)
    print(f"DQN Network: {dqn}")

    conv_dqn = ConvDQN(4)
    print(f"ConvDQN Network: {conv_dqn}")

    buffer = ReplayBuffer(1000)
    print(f"Replay Buffer created with capacity: {buffer.capacity}")

    print("✓ All components initialized successfully")
