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
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.obs_embed = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.ReLU())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )
        # per-dimension learned log_std
        self.log_std = torch.nn.Parameter(torch.ones(1, action_dim) * init_log_std, requires_grad=True)

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
        std = self.log_std.exp().to(mean.device)  # [1, action_dim]
        if deterministic:
            pre_tanh = mean
        else:
            noise = torch.randn_like(mean)
            pre_tanh = mean + noise * std
        actions = torch.tanh(pre_tanh)
        # log probability with tanh squashing correction:
        # logp = Normal(pre_tanh; mean, std).log_prob(pre_tanh) - sum(log(1 - tanh^2(pre_tanh)))
        normal_logp = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * torch.log(std) + torch.log(2 * torch.pi))
        # sum over action dim
        normal_logp = normal_logp.sum(-1)
        # correction
        log_det = torch.log(1.0 - actions.pow(2) + 1e-6).sum(-1)
        logp = normal_logp - log_det
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
        # invert tanh to get pre-squash values
        eps = 1e-6
        clipped = actions.clamp(-1 + eps, 1 - eps)
        pre_tanh = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        normal_logp = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * torch.log(std) + torch.log(2 * torch.pi))
        normal_logp = normal_logp.sum(-1)
        log_det = torch.log(1.0 - clipped.pow(2) + 1e-6).sum(-1)
        logp = normal_logp - log_det
        return logp
