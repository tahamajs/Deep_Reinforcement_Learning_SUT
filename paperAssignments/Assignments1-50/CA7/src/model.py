from typing import Tuple, Optional
import torch
from torch import nn


class RecurrentCritic(nn.Module):
    """
    Simple recurrent critic Q(o_t, h_t, a_t) implemented with a GRU encoder
    that processes observations (optionally concatenated with actions) and
    emits a scalar Q-value per timestep.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size + action_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(
        self, obs: torch.Tensor, acts: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward through the critic for a batch of sequences.

        Args:
            obs: [B, L, obs_dim]
            acts: [B, L, action_dim]
            h0: [1, B, hidden_size] optional initial hidden state

        Returns:
            q: [B, L, 1] Q-values per timestep
            h: [1, B, hidden_size] final hidden state
        """
        B, L, _ = obs.shape
        x = self.obs_embed(obs)  # [B, L, H]
        x_and_a = torch.cat([x, acts], dim=-1)
        out, h = self.gru(x_and_a, h0)  # out: [B, L, H]
        q = self.head(out)  # [B, L, 1]
        return q, h


class SimpleActor(nn.Module):
    """
    Recurrent actor producing actions given observations; outputs deterministic actions
    (suitable for DDPG/TD3 style usage). For stochastic policies (SAC) this can be
    replaced with a Gaussian head.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.obs_embed = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh(),
        )

    def forward(
        self, obs: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: [B, L, obs_dim]
            h0: initial hidden [1, B, H]
        Returns:
            actions: [B, L, action_dim]
            h: final hidden
        """
        x = self.obs_embed(obs)
        out, h = self.gru(x, h0)
        actions = self.head(out)
        return actions, h


class StochasticActor(nn.Module):
    """
    RNN-based Gaussian policy producing per-timestep actions and log-probabilities.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        log_std: float = -0.5,
    ):
        super().__init__()
        self.obs_embed = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )
        # fixed log_std scalar for simplicity; could be learned per-dim
        self.log_std = torch.nn.Parameter(
            torch.ones(1, action_dim) * log_std, requires_grad=False
        )

    def forward(
        self, obs: torch.Tensor, h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.obs_embed(obs)
        out, h = self.gru(x, h0)
        mean = self.mean_head(out)
        return mean, h

    def sample(
        self,
        obs: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """
        Sample actions and return (actions, logp, h).
        obs: [B, L, obs_dim]
        returns actions [B, L, action_dim], logp [B, L], h
        """
        mean, h = self.forward(obs, h0)
        std = self.log_std.exp().to(mean.device)
        if deterministic:
            actions = torch.tanh(mean)
        else:
            noise = torch.randn_like(mean)
            actions = mean + noise * std
            actions = torch.tanh(actions)
        # compute log_prob under Gaussian before tanh (approximate)
        var = std**2
        logp = -0.5 * (((mean - mean) ** 2) / var).sum(
            -1
        )  # zeros; placeholder for simple shape
        # For a proper log_prob we would invert tanh and compute Gaussian logp; simplified here:
        logp = -0.5 * (((actions - mean) ** 2) / var).sum(-1)
        return actions, logp, h

    def log_prob(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute approximate log-prob of provided actions under the current policy.
        Returns [B, L] log probabilities.
        """
        mean, _ = self.forward(obs, h0)
        std = self.log_std.exp().to(mean.device)
        var = std**2
        # simple Gaussian log-prob (ignores tanh correction)
        logp = -0.5 * (((actions - mean) ** 2) / var).sum(-1) - 0.5 * actions.shape[
            -1
        ] * torch.log(2 * torch.pi * var.sum())
        return logp
