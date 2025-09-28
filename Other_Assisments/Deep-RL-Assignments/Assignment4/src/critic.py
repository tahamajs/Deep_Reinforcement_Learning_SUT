"""
Critic Networks for DDPG and TD3 algorithms.

Author: Saeed Reza Zouashkiani
Student ID: 400206262
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class CriticNetwork(nn.Module):
    """Critic network for DDPG algorithm."""

    def __init__(self, state_dim, action_dim, batch_size, tau, lr, gamma, device, custom_init=False):
        """Initialize the Critic network.

        Args:
            state_dim: (int) dimension of the state space
            action_dim: (int) dimension of the action space
            batch_size: (int) batch size for training
            tau: (float) soft update parameter for target network
            lr: (float) learning rate for the optimizer
            gamma: (float) discount factor
            device: (torch.device) device to run the network on
            custom_init: (bool) whether to use custom weight initialization
        """
        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.gamma = gamma
        self.device = device

        # Create the critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        # Create the target critic network
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        # Copy weights to target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Set target network to evaluation mode
        self.critic_target.eval()

        # Optimizer
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        # Loss function
        self.critic_loss = nn.MSELoss()

        # Custom weight initialization
        if custom_init:
            for layer in self.critic:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, state, action):
        """Forward pass through the critic network."""
        x = torch.cat([state, action], dim=1)
        return self.critic(x)

    def train(self, states, actions, rewards, next_states, dones, next_actions):
        """Train the critic network.

        Args:
            states: (torch.Tensor) batch of states
            actions: (torch.Tensor) batch of actions
            rewards: (torch.Tensor) batch of rewards
            next_states: (torch.Tensor) batch of next states
            dones: (torch.Tensor) batch of done flags
            next_actions: (torch.Tensor) batch of next actions

        Returns:
            critic_loss: (float) loss value
        """
        # Compute target Q-values
        with torch.no_grad():
            next_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * next_Q

        # Compute current Q-values
        current_Q = self.critic(states, actions)

        # Compute loss
        critic_loss = self.critic_loss(current_Q, target_Q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_target(self):
        """Soft update the target network."""
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class CriticNetworkTD3(nn.Module):
    """Twin Critic network for TD3 algorithm."""

    def __init__(self, state_dim, action_dim, batch_size, tau, lr, gamma, device, custom_init=False):
        """Initialize the Twin Critic network.

        Args:
            state_dim: (int) dimension of the state space
            action_dim: (int) dimension of the action space
            batch_size: (int) batch size for training
            tau: (float) soft update parameter for target network
            lr: (float) learning rate for the optimizer
            gamma: (float) discount factor
            device: (torch.device) device to run the network on
            custom_init: (bool) whether to use custom weight initialization
        """
        super(CriticNetworkTD3, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.tau = tau
        self.lr = lr
        self.gamma = gamma
        self.device = device

        # Create the first critic network
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        # Create the second critic network
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        # Create the target critic networks
        self.critic1_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        self.critic2_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

        # Copy weights to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Set target networks to evaluation mode
        self.critic1_target.eval()
        self.critic2_target.eval()

        # Optimizers
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr)

        # Loss function
        self.critic_loss = nn.MSELoss()

        # Custom weight initialization
        if custom_init:
            for critic in [self.critic1, self.critic2]:
                for layer in critic:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0.0)

    def forward(self, state, action):
        """Forward pass through both critic networks."""
        x = torch.cat([state, action], dim=1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)
        return q1, q2

    def get_Q(self, state, action):
        """Get the minimum Q-value from both critics."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def train(self, states, actions, rewards, next_states, dones, next_actions):
        """Train both critic networks.

        Args:
            states: (torch.Tensor) batch of states
            actions: (torch.Tensor) batch of actions
            rewards: (torch.Tensor) batch of rewards
            next_states: (torch.Tensor) batch of next states
            dones: (torch.Tensor) batch of done flags
            next_actions: (torch.Tensor) batch of next actions

        Returns:
            critic_loss: (float) average loss value
        """
        # Compute target Q-values (take minimum of both critics)
        with torch.no_grad():
            next_q1, next_q2 = self.critic1_target(next_states, next_actions), self.critic2_target(next_states, next_actions)
            next_Q = torch.min(next_q1, next_q2)
            target_Q = rewards + (1 - dones) * self.gamma * next_Q

        # Compute current Q-values
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))

        # Compute losses
        critic1_loss = self.critic_loss(current_q1, target_Q)
        critic2_loss = self.critic_loss(current_q2, target_Q)
        critic_loss = critic1_loss + critic2_loss

        # Update critics
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        return critic_loss.item()

    def update_target(self):
        """Soft update the target networks."""
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)