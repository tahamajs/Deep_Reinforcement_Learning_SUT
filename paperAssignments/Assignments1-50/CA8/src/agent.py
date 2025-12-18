"""
MaxSinkAgent: a minimal, import-safe implementation of the MaxSink idea.
This implementation targets discrete action spaces and scalar returns.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from config import cfg
from losses import SinkhornWrapper
from utils import set_seed, soft_update


class DistributionalCritic(nn.Module):
    """
    Simple MLP critic that outputs `particles` scalar samples per action.
    Input: state vector (flattened). Output shape: [B, A, N, 1]
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        particles: int = cfg.particles,
        hidden=(256, 256),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.particles = particles
        # shared encoder
        layers = []
        in_ch = state_dim
        for h in hidden:
            layers.append(nn.Linear(in_ch, h))
            layers.append(nn.ReLU())
            in_ch = h
        self.encoder = nn.Sequential(*layers)
        # final head outputs n_actions * particles scalars
        self.head = nn.Linear(in_ch, n_actions * particles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, state_dim]
        returns: [B, n_actions, particles, 1]
        """
        h = self.encoder(x)
        out = self.head(h)
        B = out.shape[0]
        out = out.view(B, self.n_actions, self.particles, 1)
        return out


class MaxSinkAgent:
    """
    Minimal agent glue: critic, target critic, optimizer, sinkhorn loss, and update step.
    Discrete-action, deterministic policy via argmax of expectation.
    """

    def __init__(self, state_dim: int, n_actions: int, device: Optional[str] = None):
        self.device = device or cfg.device
        set_seed(cfg.seed)
        self.critic = DistributionalCritic(state_dim, n_actions, cfg.particles).to(
            self.device
        )
        self.target_critic = DistributionalCritic(
            state_dim, n_actions, cfg.particles
        ).to(self.device)
        # copy weights
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.opt = optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.sinkhorn = SinkhornWrapper(
            blur=cfg.sinkhorn_blur, scaling=cfg.sinkhorn_scaling, p=cfg.sinkhorn_p
        )
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.beta = cfg.beta

    def act(self, state: torch.Tensor) -> int:
        """
        Deterministic action selection by expected value (mean of particles).
        state: [state_dim] or [1, state_dim]
        returns: scalar action index
        """
        self.critic.eval()
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_dist = self.critic(state.to(self.device))  # [1, A, N, 1]
            mean_q = q_dist.mean(dim=2).squeeze(-1)  # [1, A]
            a = int(mean_q.argmax(dim=1).item())
        self.critic.train()
        return a

    def update(self, batch: dict) -> Tuple[float, float]:
        """
        batch expected keys: 'state', 'action', 'reward', 'next_state', 'done'
        Shapes:
          state: [B, state_dim]
          action: [B] (int64)
          reward: [B] (float) -- already transformed reward r'
          next_state: [B, state_dim]
          done: [B] (0/1)
        Returns: (loss.item(), sinkhorn_mean)
        """
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device).unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device).unsqueeze(-1).unsqueeze(-1)

        B = state.shape[0]
        # predictions X: [B, N, 1] for the taken action
        all_pred = self.critic(state)  # [B, A, N, 1]
        # gather per-action predictions
        idx = action.view(B, 1, 1, 1).expand(-1, 1, cfg.particles, 1)
        x = torch.gather(all_pred, 1, idx).squeeze(1)  # [B, N, 1]

        with torch.no_grad():
            # compute next action via target critic expectation
            next_pred_all = self.target_critic(next_state)  # [B, A, N, 1]
            next_mean = next_pred_all.mean(dim=2).squeeze(-1)  # [B, A]
            a_star = next_mean.argmax(dim=1)  # [B]
            idx_next = a_star.view(B, 1, 1, 1).expand(-1, 1, cfg.particles, 1)
            y_particles = torch.gather(next_pred_all, 1, idx_next).squeeze(
                1
            )  # [B, N, 1]
            # target y = reward + gamma * (1 - done) * y_particles
            y = reward + self.gamma * (1.0 - done) * y_particles

        # compute sinkhorn loss per batch element
        loss_per_batch = self.sinkhorn(x, y)  # [B]
        loss = loss_per_batch.mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.opt.step()

        # soft update target
        soft_update(self.target_critic, self.critic, self.tau)

        # return particles for optional visualization (numpy)
        try:
            x_np = x.detach().cpu().numpy()  # [B, N, 1]
            y_np = y.detach().cpu().numpy()  # [B, N, 1]
        except Exception:
            x_np = None
            y_np = None

        return float(loss.item()), float(loss_per_batch.mean().item()), x_np, y_np










