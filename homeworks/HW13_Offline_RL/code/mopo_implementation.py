"""
MOPO implementation (ensemble dynamics + penalized rollouts).
Lightweight, intended for integration in notebooks.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class DynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        h = self.net(x)
        delta = self.delta_head(h)
        reward = self.reward_head(h)
        next_state = state + delta
        return next_state, reward.squeeze(-1)


class MOPO:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        lambda_u: float = 1.0,
        lr: float = 1e-3,
    ):
        self.ensemble: List[DynamicsModel] = [
            DynamicsModel(state_dim, action_dim) for _ in range(ensemble_size)
        ]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.ensemble]
        self.lambda_u = lambda_u

    def train_dynamics(
        self,
        dataset: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        epochs: int = 10,
    ):
        # dataset: list of (s, a, r, s', done) as tensors
        for epoch in range(epochs):
            for model, opt in zip(self.ensemble, self.optimizers):
                for s, a, r, s_next, _ in dataset:
                    pred_next, pred_r = model(s.unsqueeze(0), a.unsqueeze(0))
                    loss = ((pred_next.squeeze(0) - s_next) ** 2).mean() + (
                        (pred_r - r) ** 2
                    ).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def model_rollout(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # predict with ensemble, compute mean and std
        preds_next = []
        preds_r = []
        for model in self.ensemble:
            ns, r = model(state.unsqueeze(0), action.unsqueeze(0))
            preds_next.append(ns.squeeze(0))
            preds_r.append(r.squeeze(0))
        preds_next = torch.stack(preds_next, dim=0)
        preds_r = torch.stack(preds_r, dim=0)

        next_state_mean = preds_next.mean(dim=0)
        reward_mean = preds_r.mean(dim=0)
        uncertainty = preds_next.std(dim=0).mean()

        penalty = self.lambda_u * uncertainty
        adjusted_reward = reward_mean - penalty
        return next_state_mean, adjusted_reward


if __name__ == "__main__":
    print("mopo_implementation: dynamics ensemble + model_rollout utilities.")



