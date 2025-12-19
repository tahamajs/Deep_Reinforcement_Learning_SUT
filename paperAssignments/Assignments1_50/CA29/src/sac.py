"""Soft Actor-Critic (SAC) implementation for continuous control."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

from .config import SACConfig


class ReplayBuffer:
    """Experience replay buffer for SAC."""

    def __init__(self, size: int, state_dim: int, action_dim: int):
        self.size = size
        self.ptr = 0
        self.full = False
        # Use float32 for efficient storage and torch compatibility
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.next_states = np.zeros((size, state_dim), dtype=np.float32)
        self.dones = np.zeros((size, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the replay buffer.

        The inputs are stored as float32 NumPy arrays for efficient storage and
        compatibility with torch tensors. Shapes are expected to match the
        buffer initialization (state_dim, action_dim). Rewards and dones are
        stored as column vectors with shape (1,) for consistent batching.

        Args:
            state: Observation/state vector (1D NumPy array of length state_dim).
            action: Action vector (1D NumPy array of length action_dim).
            reward: Scalar reward.
            next_state: Next observation/state vector.
            done: Boolean indicating episode termination.
        """
        # Ensure we store values with the correct dtype
        self.states[self.ptr] = np.asarray(state, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        self.rewards[self.ptr] = np.asarray([reward], dtype=np.float32)
        self.next_states[self.ptr] = np.asarray(next_state, dtype=np.float32)
        self.dones[self.ptr] = np.asarray([float(done)], dtype=np.float32)

        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions from the buffer.

        Returns a tuple of torch.FloatTensors in the following order:
        (states, actions, rewards, next_states, dones)

        - states: Tensor of shape (batch_size, state_dim)
        - actions: Tensor of shape (batch_size, action_dim)
        - rewards: Tensor of shape (batch_size, 1)
        - next_states: Tensor of shape (batch_size, state_dim)
        - dones: Tensor of shape (batch_size, 1)

        All tensors are of dtype torch.float32 and ready to be moved to the
        appropriate device for training.
        """
        max_idx = self.size if self.full else self.ptr
        idxs = np.random.randint(0, max_idx, batch_size)

        return (
            torch.from_numpy(self.states[idxs]).float(),
            torch.from_numpy(self.actions[idxs]).float(),
            torch.from_numpy(self.rewards[idxs]).float(),
            torch.from_numpy(self.next_states[idxs]).float(),
            torch.from_numpy(self.dones[idxs]).float()
        )

    def __len__(self) -> int:
        return self.size if self.full else self.ptr


class Actor(nn.Module):
    """Stochastic policy network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """Q-function network for SAC."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SAC:
    """Soft Actor-Critic algorithm implementation."""

    def __init__(self, state_dim: int, action_dim: int, config: SACConfig, device: torch.device):
        self.config = config
        self.device = device

        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.lr_critic)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.lr_critic)

        self.buffer = ReplayBuffer(config.buffer_size, state_dim, action_dim)

        self.alpha = config.alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_actor)

        self.gamma = config.gamma

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action for a given state.

        Args:
            state: 1D NumPy array representing the observation.
            deterministic: If True, use the mean action (no sampling).

        Returns:
            A 1D NumPy array containing the selected action.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if deterministic:
            mean, _ = self.actor.forward(state)
            action = torch.tanh(mean)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def update(self) -> None:
        """Perform a single parameter update for actor, critics, and (optional) alpha.

        The method samples a minibatch from the replay buffer, computes the
        soft Q-targets using the target networks, updates critic networks via
        MSE loss, updates the policy via the reparameterization objective, and
        (if automatic entropy tuning is enabled) updates the temperature
        parameter alpha to match a target entropy.
        """
        if len(self.buffer) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update actor
        actions_new, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, actions_new)
        q2_new = self.critic2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (if using automatic entropy tuning behaviour is unchanged)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update targets (polyak / tau)
        tau = getattr(self.config, 'tau', 0.005)
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str) -> None:
        """Save the model and optimizer states."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            # Save log_alpha as a plain float for portability
            'log_alpha': float(self.log_alpha.detach().cpu().item()),
            'alpha': float(self.alpha)
        }, path)

    def load(self, path: str) -> None:
        """Load the model and optimizer states."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        # Restore log_alpha (stored as float) as a tensor with grad enabled on the correct device
        log_alpha_val = float(checkpoint.get('log_alpha', np.log(self.alpha)))
        self.log_alpha = torch.tensor(log_alpha_val, requires_grad=True, device=self.device)
        # Also restore alpha if present
        self.alpha = float(checkpoint.get('alpha', self.log_alpha.exp().item()))