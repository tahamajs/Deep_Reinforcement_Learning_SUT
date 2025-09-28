"""
Actor Network for DDPG and TD3 algorithms.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class ActorNetwork(nn.Module):
    """Actor network for DDPG/TD3 algorithms."""

    def __init__(
        self, state_dim, action_dim, batch_size, tau, lr, device, custom_init=False
    ):
        """Initialize the Actor network.

        Args:
            state_dim: (int) dimension of the state space
            action_dim: (int) dimension of the action space
            batch_size: (int) batch size for training
            tau: (float) soft update parameter for target network
            lr: (float) learning rate for the optimizer
            device: (torch.device) device to run the network on
            custom_init: (bool) whether to use custom weight initialization
        """
        super(ActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.device = device

        # Create the policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

        # Create the target policy network
        self.policy_target = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh(),
        )

        # Copy weights to target network
        self.policy_target.load_state_dict(self.policy.state_dict())

        # Set target network to evaluation mode
        self.policy_target.eval()

        # Optimizer
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

        # Custom weight initialization
        if custom_init:
            for layer in self.policy:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, state):
        """Forward pass through the policy network."""
        return self.policy(state)

    def train(self, Q_value):
        """Train the policy network.

        Args:
            Q_value: (torch.Tensor) Q-values from the critic

        Returns:
            policy_loss: (float) loss value
        """
        policy_loss = -Q_value.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()

    def update_target(self):
        """Soft update the target network."""
        for target_param, param in zip(
            self.policy_target.parameters(), self.policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
