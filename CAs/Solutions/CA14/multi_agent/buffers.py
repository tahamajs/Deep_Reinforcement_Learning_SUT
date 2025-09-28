"""
Replay Buffers for Multi-Agent Reinforcement Learning

This module provides replay buffer implementations for multi-agent learning.
"""

import numpy as np
from typing import List, Tuple, Any


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent learning."""

    def __init__(self, capacity, num_agents, obs_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Store transition in buffer."""
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.next_states.append(None)
            self.dones.append(None)

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample batch from buffer."""
        if self.size < batch_size:
            return None

        indices = np.random.choice(self.size, batch_size, replace=False)

        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_rewards = [self.rewards[i] for i in indices]
        batch_next_states = [self.next_states[i] for i in indices]
        batch_dones = [self.dones[i] for i in indices]

        return (batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)