import torch
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple

from src.models.networks import PolicyNetwork, ValueNetwork
from src.config import Config

class REINFORCEAgent:
    """REINFORCE algorithm with optional baseline"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        use_baseline: bool = False,
        gamma: float = Config.REINFORCE_GAMMA,
        lr: float = Config.REINFORCE_LR,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_baseline = use_baseline
        self.gamma = gamma

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        if use_baseline:
            self.value_net = ValueNetwork(state_dim, hidden_dim)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.episode_buffer = []

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Select action using current policy"""
        return self.policy.get_action(state)

    def store_transition(
        self, state: torch.Tensor, action: int, reward: float, log_prob: torch.Tensor
    ):
        """Store transition in episode buffer"""
        self.episode_buffer.append(
            {"state": state, "action": action, "reward": reward, "log_prob": log_prob}
        )

    def update(self) -> Dict[str, float]:
        """Update policy using REINFORCE"""
        if not self.episode_buffer:
            return {}

        # Compute returns
        rewards = [t["reward"] for t in self.episode_buffer]
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=Config.DEVICE)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute advantages if using baseline
        if self.use_baseline:
            states = torch.stack([t["state"] for t in self.episode_buffer]).to(Config.DEVICE)
            values = self.value_net(states).squeeze()
            advantages = returns - values.detach()

            # Update value network
            value_loss = torch.nn.functional.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        else:
            advantages = returns

        # Compute policy loss
        log_probs = torch.stack([t["log_prob"] for t in self.episode_buffer]).to(Config.DEVICE)
        policy_loss = -(log_probs * advantages).mean()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear buffer
        self.episode_buffer = []

        return {
            "policy_loss": policy_loss.item(),
            "returns_mean": returns.mean().item(),
            "returns_std": returns.std().item(),
        }
