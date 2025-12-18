from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian actor that outputs mean and log_std."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes=(256, 256),
        log_std_min=-20,
        log_std_max=2,
    ):
        super().__init__()
        self.net = mlp(
            [state_dim] + list(hidden_sizes),
            activation=nn.ReLU,
            output_activation=nn.ReLU,
        )
        last_size = hidden_sizes[-1] if len(hidden_sizes) > 0 else state_dim
        self.mean_layer = nn.Linear(last_size, action_dim)
        self.log_std_layer = nn.Linear(last_size, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(state)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # log prob with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y_t, log_prob


class VectorizedCritic(nn.Module):
    """Ensemble of Q-networks implemented as a single module."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes=(256, 256),
        ensemble_size: int = 4,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        # each member is a separate MLP
        self.nets = nn.ModuleList(
            [
                mlp(
                    [state_dim + action_dim] + list(hidden_sizes) + [1],
                    activation=nn.ReLU,
                    output_activation=nn.Identity,
                )
                for _ in range(ensemble_size)
            ]
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (ensemble_size, batch, 1) or (ensemble_size, N, 1)."""
        sa = torch.cat([state, action], dim=-1)
        qs = [net(sa) for net in self.nets]  # list of (B,1)
        qs = torch.stack(qs, dim=0)  # (ensemble, B, 1)
        return qs









