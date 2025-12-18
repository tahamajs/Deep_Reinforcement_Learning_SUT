from typing import Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.ezv2_ma_net import MAEZV2Network
from ..mcts.gumbel_topk import topk_joint, topk_factored


class EfficientZeroV2Policy(nn.Module):
    """Light-weight implementation of MA-EZV2 policy combining network and search helpers.

    This class is intentionally self-contained and import-safe.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        joint_action_dim: int,
        per_agent_action_dims: Optional[Sequence[int]] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.net = MAEZV2Network(
            obs_dim,
            latent_dim,
            joint_action_dim,
            tuple(per_agent_action_dims) if per_agent_action_dims else None,
        ).to(self.device)
        self.joint_action_dim = joint_action_dim
        self.per_agent_action_dims = per_agent_action_dims

    def initial_infer(self, obs: torch.Tensor):
        """Encode observations and produce initial policy/value/prefix."""
        obs = obs.to(self.device)
        h0 = self.net.initial_latent(obs)
        logits_joint, logits_agents, value, prefix = self.net.predict_from_latent(h0)
        return logits_joint, logits_agents, value, prefix

    def compute_losses(
        self,
        h0: torch.Tensor,
        actions: torch.Tensor,
        pi_targets: torch.Tensor,
        value_targets: torch.Tensor,
        reward_targets: torch.Tensor,
        prefix_targets: torch.Tensor,
        alpha: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute sum of losses per unroll given targets.

        Simple, fully implemented losses: cross-entropy for policy, MSE for value/reward/prefix.
        """
        losses = {}
        loss_total = torch.tensor(0.0, device=h0.device)
        # predict at root
        logits_joint, logits_agents, v0, z0 = self.net.predict_from_latent(h0)
        # policy loss (CE with soft targets)
        logp = F.log_softmax(logits_joint, dim=-1)
        loss_pi = -(pi_targets * logp).sum(dim=-1).mean()
        losses["loss_pi"] = loss_pi.item()
        loss_total = loss_total + alpha.get("pi", 1.0) * loss_pi
        # value loss
        loss_v = F.mse_loss(v0, value_targets)
        losses["loss_v"] = loss_v.item()
        loss_total = loss_total + alpha.get("v", 1.0) * loss_v
        # reward loss: predict reward after applying dynamics on actions (one-step)
        h1, r1 = self.net.dynamics(h0, actions)
        loss_r = F.mse_loss(r1, reward_targets)
        losses["loss_r"] = loss_r.item()
        loss_total = loss_total + alpha.get("r", 1.0) * loss_r
        # prefix loss
        z_pred = self.net.prefix(h0)
        loss_z = F.mse_loss(z_pred, prefix_targets)
        losses["loss_prefix"] = loss_z.item()
        loss_total = loss_total + alpha.get("z", 1.0) * loss_z
        return loss_total, losses

    def gumbel_topk_actions(self, logits_joint: torch.Tensor, k: int = 8):
        """Return top-k joint indices and scores using Gumbel top-k."""
        return topk_joint(logits_joint, k)

    def factored_topk_actions(
        self, logits_agents: Sequence[torch.Tensor], k_each: int = 4, max_beam: int = 64
    ):
        return topk_factored(list(logits_agents), k_each, max_beam)

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None):
        sd = torch.load(path, map_location=map_location)
        self.net.load_state_dict(sd)


if __name__ == "__main__":
    # demo usage
    policy = EfficientZeroV2Policy(
        obs_dim=12,
        latent_dim=64,
        joint_action_dim=10,
        per_agent_action_dims=[5, 5],
        device="cpu",
    )
    obs = torch.randn(2, 12)
    lj, la, v, z = policy.initial_infer(obs)
    print("logits_joint", lj.shape, "value", v.shape, "prefix", z.shape)








