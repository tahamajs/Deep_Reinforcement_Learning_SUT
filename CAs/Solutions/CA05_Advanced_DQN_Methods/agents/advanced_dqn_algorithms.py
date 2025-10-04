"""
Advanced DQN Algorithms - Complex Implementations
Includes: Noisy DQN, Distributional DQN, Multi-Step DQN, and more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any, Optional
import math


class NoisyLinear(nn.Module):
    """Noisy Linear layer for Noisy DQN"""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class CategoricalDQN(nn.Module):
    """Categorical DQN for Distributional RL"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.z = torch.linspace(v_min, v_max, num_atoms)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * num_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        logits = self.network(x)
        logits = logits.view(batch_size, -1, self.num_atoms)
        return F.softmax(logits, dim=2)


class MultiStepBuffer:
    """Multi-step experience replay buffer"""

    def __init__(self, capacity: int, n_step: int = 3, gamma: float = 0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        """Push single step"""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step:
            # Calculate n-step return
            n_step_return = 0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_return += (self.gamma**i) * r
                if d:
                    break

            # Get first state and action
            first_state, first_action, _, _, _ = self.n_step_buffer[0]
            last_state, _, _, _, last_done = self.n_step_buffer[-1]

            self.buffer.append(
                (first_state, first_action, n_step_return, last_state, last_done)
            )

    def sample(self, batch_size: int):
        """Sample batch from buffer"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class NoisyDQNAgent:
    """Noisy DQN Agent with parameter noise"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: str = "cpu",
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Networks
        self.q_network = self._build_network(state_dim, action_dim).to(device)
        self.target_network = self._build_network(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.step_count = 0

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _build_network(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build noisy network"""
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, output_dim),
        )

    def select_action(self, state: np.ndarray) -> int:
        """Select action using noisy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self) -> float:
        """Update network"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset noise
        for module in self.q_network.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


class DistributionalDQNAgent:
    """Distributional DQN Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        device: str = "cpu",
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.device = device

        # Networks
        self.q_network = CategoricalDQN(
            state_dim, action_dim, num_atoms=num_atoms, v_min=v_min, v_max=v_max
        ).to(device)
        self.target_network = CategoricalDQN(
            state_dim, action_dim, num_atoms=num_atoms, v_min=v_min, v_max=v_max
        ).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.step_count = 0

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: np.ndarray) -> int:
        """Select action using distributional network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.q_network(state_tensor)
            q_values = torch.sum(dist * self.q_network.z, dim=2)
        return q_values.argmax().item()

    def update(self) -> float:
        """Update network using distributional loss"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current distributions
        current_dist = self.q_network(states)
        current_dist = current_dist[range(self.batch_size), actions]

        # Target distributions
        with torch.no_grad():
            next_dist = self.target_network(next_states)
            next_actions = next_dist.sum(dim=2).argmax(dim=1)
            next_dist = next_dist[range(self.batch_size), next_actions]

            # Project to target atoms
            target_atoms = rewards.unsqueeze(
                1
            ) + self.gamma * self.q_network.z.unsqueeze(0) * (~dones).unsqueeze(1)
            target_atoms = torch.clamp(target_atoms, self.v_min, self.v_max)

            # Compute projection
            atom_indices = (target_atoms - self.v_min) / self.q_network.delta_z
            atom_indices = torch.clamp(atom_indices, 0, self.num_atoms - 1)

            # Distribute probability mass
            lower_indices = atom_indices.floor().long()
            upper_indices = atom_indices.ceil().long()
            lower_probs = next_dist * (upper_indices.float() - atom_indices)
            upper_probs = next_dist * (atom_indices - lower_indices.float())

            target_dist = torch.zeros_like(current_dist)
            target_dist.scatter_add_(1, lower_indices, lower_probs)
            target_dist.scatter_add_(1, upper_indices, upper_probs)

        # Compute loss
        loss = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=1).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


class MultiStepDQNAgent:
    """Multi-step DQN Agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        n_step: int = 3,
        device: str = "cpu",
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.device = device

        # Networks
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).to(device)

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).to(device)

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # Multi-step buffer
        self.replay_buffer = MultiStepBuffer(buffer_size, n_step, gamma)
        self.step_count = 0

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action with epsilon-greedy"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self) -> float:
        """Update network"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = (
                rewards + (self.gamma**self.n_step) * next_q_values * ~dones
            )

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


class HierarchicalDQNAgent:
    """Hierarchical DQN Agent with meta-controller and controller"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: str = "cpu",
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Meta-controller (goal selection)
        self.meta_controller = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, goal_dim),
        ).to(device)

        self.meta_target = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, goal_dim),
        ).to(device)

        # Controller (action selection given goal)
        self.controller = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).to(device)

        self.controller_target = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).to(device)

        # Optimizers
        self.meta_optimizer = torch.optim.Adam(self.meta_controller.parameters(), lr=lr)
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=lr
        )

        # Buffers
        self.meta_buffer = deque(maxlen=buffer_size)
        self.controller_buffer = deque(maxlen=buffer_size)
        self.step_count = 0

        # Copy weights to target networks
        self.meta_target.load_state_dict(self.meta_controller.state_dict())
        self.controller_target.load_state_dict(self.controller.state_dict())

    def select_goal(self, state: np.ndarray) -> int:
        """Select goal using meta-controller"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            goal_values = self.meta_controller(state_tensor)
        return goal_values.argmax().item()

    def select_action(self, state: np.ndarray, goal: int) -> int:
        """Select action given state and goal"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.zeros(1, self.goal_dim).to(self.device)
        goal_tensor[0, goal] = 1.0

        combined_input = torch.cat([state_tensor, goal_tensor], dim=1)
        with torch.no_grad():
            action_values = self.controller(combined_input)
        return action_values.argmax().item()

    def update(self) -> Tuple[float, float]:
        """Update both meta-controller and controller"""
        meta_loss = 0.0
        controller_loss = 0.0

        # Update meta-controller
        if len(self.meta_buffer) >= self.batch_size:
            batch = random.sample(self.meta_buffer, self.batch_size)
            states, goals, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(self.device)
            goals = torch.LongTensor(goals).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)

            current_q = self.meta_controller(states).gather(1, goals.unsqueeze(1))
            with torch.no_grad():
                next_q = self.meta_target(next_states).max(1)[0]
                target_q = rewards + self.gamma * next_q * ~dones

            meta_loss = F.mse_loss(current_q.squeeze(), target_q)

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

        # Update controller
        if len(self.controller_buffer) >= self.batch_size:
            batch = random.sample(self.controller_buffer, self.batch_size)
            states, goals, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states).to(self.device)
            goals = torch.LongTensor(goals).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.BoolTensor(dones).to(self.device)

            # One-hot encode goals
            goal_onehot = torch.zeros(self.batch_size, self.goal_dim).to(self.device)
            goal_onehot.scatter_(1, goals.unsqueeze(1), 1)

            combined_input = torch.cat([states, goal_onehot], dim=1)
            current_q = self.controller(combined_input).gather(1, actions.unsqueeze(1))

            with torch.no_grad():
                next_combined = torch.cat([next_states, goal_onehot], dim=1)
                next_q = self.controller_target(next_combined).max(1)[0]
                target_q = rewards + self.gamma * next_q * ~dones

            controller_loss = F.mse_loss(current_q.squeeze(), target_q)

            self.controller_optimizer.zero_grad()
            controller_loss.backward()
            self.controller_optimizer.step()

        # Update target networks
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.meta_target.load_state_dict(self.meta_controller.state_dict())
            self.controller_target.load_state_dict(self.controller.state_dict())

        return meta_loss, controller_loss


if __name__ == "__main__":
    print("Advanced DQN algorithms loaded successfully!")
    print("Available agents:")
    print("- NoisyDQNAgent: Parameter noise exploration")
    print("- DistributionalDQNAgent: Distributional RL")
    print("- MultiStepDQNAgent: Multi-step returns")
    print("- HierarchicalDQNAgent: Hierarchical RL")


