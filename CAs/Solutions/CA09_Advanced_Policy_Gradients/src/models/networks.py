import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class PolicyNetwork(nn.Module):
    """Base policy network for discrete action spaces"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, torch.Tensor]:
        """Sample action from policy"""
        logits = self.forward(state)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        log_prob = (
            F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
        )

        return action.item(), log_prob


class ValueNetwork(nn.Module):
    """Value function network"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous action spaces (Gaussian)"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        action_bound: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # Mean network
        self.mean_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bound mean between -1 and 1
        )

        # Standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - return mean and std"""
        mean = self.mean_network(state) * self.action_bound
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample action from Gaussian policy"""
        mean, std = self.forward(state)

        if deterministic:
            action = mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()

        # Compute log probability
        log_prob = self.compute_log_prob(action, mean, std)

        return action.detach().numpy(), log_prob

    def compute_log_prob(
        self, action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of action under policy"""
        var = std.pow(2)
        log_std = torch.log(std)

        # Log probability for multivariate Gaussian
        log_prob = -0.5 * (
            (action - mean).pow(2) / var + 2 * log_std + np.log(2 * np.pi)
        )
        return log_prob.sum(dim=-1)


