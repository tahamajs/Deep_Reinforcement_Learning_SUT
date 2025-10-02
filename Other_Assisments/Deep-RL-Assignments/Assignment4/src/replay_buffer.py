"""
Replay Buffer for DDPG and TD3 algorithms.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import numpy as np
import torch
class ReplayBuffer(object):
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, buffer_size, burn_in, state_dim, action_dim, device):
        """Initialize the replay buffer.

        Args:
            buffer_size: (int) maximum size of the buffer
            burn_in: (int) number of transitions to collect before training starts
            state_dim: (int) dimension of the state space
            action_dim: (int) dimension of the action space
            device: (torch.device) device to store tensors on
        """
        self.buffer_size = buffer_size
        self.burn_in = burn_in
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.states = torch.zeros((buffer_size, state_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.next_states = torch.zeros((buffer_size, state_dim), device=device)
        self.dones = torch.zeros(buffer_size, device=device)

        self.current_size = 0
        self.ptr = 0

    @property
    def burned_in(self):
        """Check if the buffer has collected enough samples for training."""
        return self.current_size >= self.burn_in

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        Args:
            state: (torch.Tensor) current state
            action: (torch.Tensor) action taken
            reward: (torch.Tensor) reward received
            next_state: (torch.Tensor) next state
            done: (torch.Tensor) done flag
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def get_batch(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: (int) size of the batch to sample

        Returns:
            states: (torch.Tensor) batch of states
            actions: (torch.Tensor) batch of actions
            rewards: (torch.Tensor) batch of rewards
            next_states: (torch.Tensor) batch of next states
            dones: (torch.Tensor) batch of done flags
        """
        indices = np.random.choice(self.current_size, batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )