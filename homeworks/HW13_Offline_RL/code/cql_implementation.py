"""
CQL implementation (lightweight, import-safe).
Provides: QNetwork, PolicyNetwork, CQL class and train_step utility.
No heavy execution on import.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Normal:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(state)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
        return torch.tanh(raw_action), log_prob


class CQL:
    """Conservative Q-Learning agent (minimal, ready for offline training loops).

    This implementation is intentionally small but fully functional for integration
    into training notebooks or scripts. It avoids any execution at import time.
    """

    def __init__(
        self, state_dim: int, action_dim: int, alpha: float = 1.0, lr: float = 3e-4
    ):
        self.Q1 = QNetwork(state_dim, action_dim)
        self.Q2 = QNetwork(state_dim, action_dim)
        self.Q1_target = QNetwork(state_dim, action_dim)
        self.Q2_target = QNetwork(state_dim, action_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)

        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.q_optimizer = optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=lr
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.alpha = alpha
        self.gamma = 0.99
        self.tau = 0.005

    def cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        q1 = self.Q1(states, actions)
        q2 = self.Q2(states, actions)

        with torch.no_grad():
            next_actions, _ = self.policy.sample(next_states)
            q1_next = self.Q1_target(next_states, next_actions)
            q2_next = self.Q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        bellman_error_1 = F.mse_loss(q1, q_target)
        bellman_error_2 = F.mse_loss(q2, q_target)

        # CQL regularization: sample random actions + policy actions
        batch_size = states.shape[0]
        action_dim = actions.shape[1]
        random_actions = (
            torch.rand(batch_size, action_dim, device=states.device) * 2.0
        ) - 1.0
        curr_actions, _ = self.policy.sample(states)

        q1_rand = self.Q1(states, random_actions)
        q1_curr = self.Q1(states, curr_actions)
        q1_data = self.Q1(states, actions)

        q2_rand = self.Q2(states, random_actions)
        q2_curr = self.Q2(states, curr_actions)
        q2_data = self.Q2(states, actions)

        cat_q1 = torch.cat([q1_rand, q1_curr], dim=0)
        cat_q2 = torch.cat([q2_rand, q2_curr], dim=0)

        cql_q1_loss = torch.logsumexp(cat_q1, dim=0).mean() - q1_data.mean()
        cql_q2_loss = torch.logsumexp(cat_q2, dim=0).mean() - q2_data.mean()

        q_loss = (
            bellman_error_1 + bellman_error_2 + self.alpha * (cql_q1_loss + cql_q2_loss)
        )
        return q_loss

    def policy_loss(self, states: torch.Tensor) -> torch.Tensor:
        actions, log_probs = self.policy.sample(states)
        q1 = self.Q1(states, actions)
        q2 = self.Q2(states, actions)
        q = torch.min(q1, q2)
        return -(q - 0.01 * log_probs).mean()

    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> dict:
        states, actions, rewards, next_states, dones = batch

        self.q_optimizer.zero_grad()
        q_loss = self.cql_loss(states, actions, rewards, next_states, dones)
        q_loss.backward()
        self.q_optimizer.step()

        self.policy_optimizer.zero_grad()
        p_loss = self.policy_loss(states)
        p_loss.backward()
        self.policy_optimizer.step()

        # soft updates
        for param, target_param in zip(
            self.Q1.parameters(), self.Q1_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.Q2.parameters(), self.Q2_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {"q_loss": q_loss.item(), "policy_loss": p_loss.item()}


if __name__ == "__main__":
    print("cql_implementation module: define CQL, import in notebooks to train.")





