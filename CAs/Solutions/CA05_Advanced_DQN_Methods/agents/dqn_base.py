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
import random
from collections import namedtuple
from typing import Optional

from CAs.Solutions.CA05_Advanced_DQN_Methods.utils.replay_buffers import ReplayBuffer
from CAs.Solutions.CA05_Advanced_DQN_Methods.utils.network_architectures import QNetwork

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class DQNAgent:
    """
    Base Deep Q-Network (DQN) Agent.

    This class provides the fundamental structure and common functionalities for DQN
    agents, including network initialization, epsilon-greedy action selection,
    and handling of experience replay. Subclasses will implement the specific
    Q-value update rules.

    Attributes:
        state_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        gamma (float): Discount factor for future rewards.
        batch_size (int): Size of the batch sampled from the replay buffer.
        target_update_freq (int): Frequency (in steps) to update the target network.
        device (str): Device to run computations on ('cpu' or 'cuda').
        q_network (nn.Module): The online Q-network.
        target_network (nn.Module): The target Q-network.
        optimizer (torch.optim.Optimizer): Optimizer for the Q-network.
        replay_buffer (ReplayBuffer): Experience replay buffer.
        epsilon (float): Current exploration rate.
        epsilon_end (float): Minimum exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        steps (int): Total number of training steps.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        device: str = "cpu",
    ):
        """
        Initializes the DQNAgent.

        Args:
            state_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate for epsilon per step.
            buffer_size (int): Maximum capacity of the replay buffer.
            batch_size (int): Size of the batch sampled from the replay buffer.
            target_update_freq (int): Frequency (in steps) to update the target network.
            device (str): Device to run computations on ('cpu' or 'cuda').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Training stats
        self.steps = 0

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state from the environment.
            epsilon (Optional[float]): The exploration rate to use for this step.
                                       If None, uses the agent's current epsilon.

        Returns:
            int: The selected action.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def update(self) -> float:
        """
        Abstract method to update the Q-network.
        This method must be implemented by subclasses.

        Returns:
            float: The loss value from the update step.
        """
        raise NotImplementedError("Update method must be implemented by subclasses")

    def _common_update_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None
    ) -> float:
        """
        Performs a common Q-network update step.
        This handles loss calculation, backpropagation, and target network update.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.
            rewards (torch.Tensor): Batch of rewards.
            next_states (torch.Tensor): Batch of next states.
            dones (torch.Tensor): Batch of done flags.
            importance_weights (Optional[torch.Tensor]): Importance sampling weights for PER.

        Returns:
            float: The loss value.
        """
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values (Vanilla DQN style, can be overridden for Double DQN)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss and update
        loss = F.mse_loss(current_q, target_q, reduction="none")
        if importance_weights is not None:
            loss = (loss * importance_weights).mean()
        else:
            loss = loss.mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """
        Saves the agent's state to a file.

        Args:
            path (str): The file path to save the state.
        """
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str):
        """
        Loads the agent's state from a file.

        Args:
            path (str): The file path to load the state from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.eval()
