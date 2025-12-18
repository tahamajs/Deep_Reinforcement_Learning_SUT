import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoder(nn.Module):
    """Shared centralized encoder for joint observations."""

    def __init__(self, obs_dim: int, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, obs_dim)
        return self.net(x)


class SimpleDynamics(nn.Module):
    """Deterministic latent dynamics head: h_{t+1}, reward"""

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(latent_dim, 1)

    def forward(
        self, h: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # a is assumed one-hot or numeric vector (B, action_dim)
        x = torch.cat([h, a], dim=-1)
        h_next = self.fc(x)
        reward = self.reward_head(h_next).squeeze(-1)
        return h_next, reward


class PredictionHead(nn.Module):
    """Outputs policy logits (factored) and scalar value from latent."""

    def __init__(
        self,
        latent_dim: int,
        joint_action_dim: int,
        per_agent_action_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
        )
        self.joint_policy = nn.Linear(latent_dim, joint_action_dim)
        # optional per-agent heads (factored)
        self.per_agent_heads = None
        if per_agent_action_dims is not None:
            self.per_agent_heads = nn.ModuleList(
                [nn.Linear(latent_dim, a) for a in per_agent_action_dims]
            )

    def forward(self, h: torch.Tensor):
        v = self.value_head(h).squeeze(-1)
        logits_joint = self.joint_policy(h)
        logits_agents = None
        if self.per_agent_heads is not None:
            logits_agents = [head(h) for head in self.per_agent_heads]
        return logits_joint, logits_agents, v


class PrefixHead(nn.Module):
    """Predicts cumulative reward prefix (scalar)."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class MAEZV2Network(nn.Module):
    """Combines encoder, dynamics, prediction and prefix heads for MA-EZV2."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        joint_action_dim: int,
        per_agent_action_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.encoder = SimpleEncoder(obs_dim, latent_dim)
        self.dynamics = SimpleDynamics(
            latent_dim,
            (
                joint_action_dim
                if per_agent_action_dims is None
                else sum(per_agent_action_dims)
            ),
        )
        self.predict = PredictionHead(
            latent_dim, joint_action_dim, per_agent_action_dims
        )
        self.prefix = PrefixHead(latent_dim)

    def initial_latent(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_from_latent(self, h: torch.Tensor):
        logits_joint, logits_agents, value = self.predict(h)
        prefix = self.prefix(h)
        return logits_joint, logits_agents, value, prefix

    def unroll_dynamics(self, h: torch.Tensor, actions: torch.Tensor, steps: int):
        """Unroll dynamics deterministically for given actions.

        Args:
            h: (B, latent_dim)
            actions: (B, steps, action_dim) numeric representation
        Returns:
            list of tuples (h_k, reward_k)
        """
        outputs = []
        for k in range(steps):
            a_k = actions[:, k]
            h, r = self.dynamics(h, a_k)
            outputs.append((h, r))
        return outputs


if __name__ == "__main__":
    # quick smoke check (import safe because guarded)
    net = MAEZV2Network(obs_dim=10, latent_dim=64, joint_action_dim=8)
    x = torch.randn(4, 10)
    h0 = net.initial_latent(x)
    lj, la, v, z = net.predict_from_latent(h0)
    print("shapes:", lj.shape, v.shape, z.shape)














