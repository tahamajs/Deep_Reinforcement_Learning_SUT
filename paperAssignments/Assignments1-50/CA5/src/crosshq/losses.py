from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn


class CrossHQLoss(nn.Module):
    """Generic CrossQ-style loss for joint batch normalization critics.

    Expects:
    - critic: instance with methods `q1_forward(x)` and `q2_forward(x)` taking a single
      tensor of shape (batch, feature_dim) where features are concatenated (state|action).
    - actor: policy with `rsample_and_logprob(obs)` returning (action, logp).
    """

    def __init__(
        self,
        critic,
        actor,
        gamma: float = 0.99,
        alpha: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.critic = critic
        self.actor = actor
        self.gamma = gamma
        self.alpha = alpha
        self.device = device or torch.device("cpu")

    def cross_forward(self, curr: torch.Tensor, nxt: torch.Tensor):
        """Concatenate along batch dim, forward through critic, and split outputs."""
        x = torch.cat([curr, nxt], dim=0)
        q1, q2 = self.critic.forward(x)
        # q1 and q2 are tensors of shape (2*B, 1)
        q1_curr, q1_next = torch.chunk(q1, 2, dim=0)
        q2_curr, q2_next = torch.chunk(q2, 2, dim=0)
        return q1_curr, q1_next, q2_curr, q2_next

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Compute CrossQ critic loss. obs/next_obs should be already concatenated (state|goal) or (state) depending on level.
        action/next_action should be primitive action (worker) or goal (manager) vectors accordingly.
        Shapes: all tensors (B, D) except reward (B, 1) mask (B, 1).
        """
        # Sample next action using current actor (no target)
        with torch.no_grad():
            next_action, logp_next = self.actor.rsample_and_logprob(next_obs)

        # Build concatenated feature vectors (state|action)
        curr = torch.cat([obs, action], dim=-1)
        nxt = torch.cat([next_obs, next_action], dim=-1)

        # Ensure critic updates BN stats
        self.critic.train()

        q1_curr, q1_next, q2_curr, q2_next = self.cross_forward(curr, nxt)

        min_q_next = torch.min(q1_next, q2_next)
        target_v = min_q_next - self.alpha * logp_next
        target_q = reward + mask * (self.gamma * target_v.detach())

        loss_q1 = F.mse_loss(q1_curr, target_q)
        loss_q2 = F.mse_loss(q2_curr, target_q)
        loss = loss_q1 + loss_q2
        return loss















