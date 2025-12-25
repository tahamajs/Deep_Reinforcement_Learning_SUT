"""
IQL implementation (V network, Q network, and policy) for offline training.
Import-safe; example usage guarded by __main__.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class VNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * (diff**2)).mean()


class IQL:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        expectile: float = 0.7,
        temperature: float = 0.05,
        lr: float = 3e-4,
    ):
        # Q and policy architectures re-used from cql style networks (simple)
        from cql_implementation import (
            QNetwork,
            PolicyNetwork,
        )  # local import to avoid circulars at top-level

        self.Q = QNetwork(state_dim, action_dim)
        self.V = VNetwork(state_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)

        self.q_optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        self.v_optimizer = optim.Adam(self.V.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.expectile = expectile
        self.temperature = temperature
        self.gamma = 0.99

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        # 1. V update (expectile regression)
        with torch.no_grad():
            q_values = self.Q(states, actions)

        v_pred = self.V(states).squeeze(-1)
        v_loss = expectile_loss(q_values.squeeze(-1) - v_pred, self.expectile)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # 2. Q update using V
        with torch.no_grad():
            v_next = self.V(next_states).squeeze(-1)
            q_target = (
                rewards.squeeze(-1) + (1 - dones.squeeze(-1)) * self.gamma * v_next
            )

        q_pred = self.Q(states, actions).squeeze(-1)
        q_loss = F.mse_loss(q_pred, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 3. Policy update with advantage-weighted regression
        with torch.no_grad():
            q_val = self.Q(states, actions).squeeze(-1)
            v_val = self.V(states).squeeze(-1)
            advantage = q_val - v_val
            weights = (
                torch.exp(advantage / self.temperature).clamp(max=100).unsqueeze(-1)
            )

        # policy log prob for given actions
        dist = self.policy.forward(states)
        log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        policy_loss = -(weights.detach() * log_probs).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
        }


if __name__ == "__main__":
    print("iql_implementation module: define IQL, import in notebooks to train.")






