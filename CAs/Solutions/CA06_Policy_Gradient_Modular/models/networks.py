"""
Neural network models for policy gradient methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from utils.setup import device


class PolicyNetwork(nn.Module):
    """Neural network policy for discrete action spaces"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class ValueNetwork(nn.Module):
    """Value function network for baseline and actor-critic"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous action spaces using Gaussian distribution"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Mean and log_std for Gaussian policy
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Numerical stability

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared feature extractor"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ActorCriticNetwork, self).__init__()

        # Shared feature extractor
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head
        self.actor_fc = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shared features
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        # Actor output (policy)
        policy = F.softmax(self.actor_fc(x), dim=-1)

        # Critic output (value)
        value = self.critic_fc(x)

        return policy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value function estimate"""
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        return self.critic_fc(x)

    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        """Get policy distribution"""
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        return F.softmax(self.actor_fc(x), dim=-1)


class PPOActorCriticNetwork(nn.Module):
    """PPO Actor-Critic network with separate actor and critic"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPOActorCriticNetwork, self).__init__()

        # Actor network
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_fc3 = nn.Linear(hidden_dim, action_dim)

        # Critic network
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Actor forward pass
        actor_x = F.relu(self.actor_fc1(x))
        actor_x = F.relu(self.actor_fc2(actor_x))
        policy = F.softmax(self.actor_fc3(actor_x), dim=-1)

        # Critic forward pass
        critic_x = F.relu(self.critic_fc1(x))
        critic_x = F.relu(self.critic_fc2(critic_x))
        value = self.critic_fc3(critic_x)

        return policy, value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value function estimate"""
        x = F.relu(self.critic_fc1(x))
        x = F.relu(self.critic_fc2(x))
        return self.critic_fc3(x)

    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        """Get policy distribution"""
        x = F.relu(self.actor_fc1(x))
        x = F.relu(self.actor_fc2(x))
        return F.softmax(self.actor_fc3(x), dim=-1)


class ContinuousActorCriticNetwork(nn.Module):
    """Continuous Actor-Critic network for continuous action spaces"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(ContinuousActorCriticNetwork, self).__init__()

        # Shared feature extractor
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor head for continuous actions
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic_fc = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # Shared features
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        # Actor output (mean and log_std)
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        # Critic output
        value = self.critic_fc(x)

        return (mean, log_std), value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value function estimate"""
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        return self.critic_fc(x)

    def get_policy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get policy parameters"""
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))

        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std


