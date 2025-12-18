from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Small MLP policy producing action logits."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits (unnormalized)."""
        return self.head(self.net(x))

    def get_action(
        self, x: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        Returns (action, log_prob).
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            logp = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)
            return action, logp
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp


class ValueNetwork(nn.Module):
    """Small MLP value function."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)












