from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """A simple MLP policy for discrete action spaces.

    This module produces action logits and action probabilities from
    observations. It also provides a convenience method to sample actions
    and compute log-probabilities for policy-gradient style algorithms.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int] = (64, 64)
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = tuple(hidden_sizes)

        layers = []
        last_dim = obs_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute action logits for a batch of observations.

        Args:
            obs: Float tensor with shape (batch, obs_dim) or (obs_dim,)

        Returns:
            logits: Float tensor with shape (batch, action_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        assert (
            obs.size(-1) == self.obs_dim
        ), f"Expected obs dim {self.obs_dim}, got {obs.size(-1)}"
        logits = self.net(obs)
        return logits

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or choose deterministic) action and return log-probability.

        Args:
            obs: observation tensor (obs_dim,) or (batch, obs_dim)
            deterministic: if True, choose action with highest prob

        Returns:
            actions: LongTensor of shape (batch,) or scalar
            log_probs: FloatTensor of shape (batch,)
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            actions = torch.argmax(probs, dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        # squeeze batch dim if input was single observation
        if obs.dim() == 1 and actions.dim() == 1 and actions.size(0) == 1:
            return actions.squeeze(0), log_probs.squeeze(0)
        return actions, log_probs

    def act_numpy(self, obs_np, deterministic: bool = False):
        """Utility to accept numpy arrays and return numpy action and logp."""
        import numpy as _np

        obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        action_t, logp_t = self.get_action(obs_t, deterministic=deterministic)
        return _np.asarray(action_t.cpu().numpy()), _np.asarray(logp_t.cpu().numpy())


