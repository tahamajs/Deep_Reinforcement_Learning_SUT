import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Tuple

from src.models.networks import ContinuousPolicyNetwork, ValueNetwork
from src.config import Config

class ContinuousPPOAgent:
    """PPO agent for continuous control"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        action_bound: float = Config.CONTINUOUS_PPO_ACTION_BOUND,
        gamma: float = Config.PPO_GAMMA,
        gae_lambda: float = Config.PPO_GAE_LAMBDA,
        clip_ratio: float = Config.PPO_CLIP_RATIO,
        lr: float = Config.PPO_LR,
        value_coeff: float = Config.PPO_VALUE_COEFF,
        entropy_coeff: float = Config.PPO_ENTROPY_COEFF,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        self.actor = ContinuousPolicyNetwork(
            state_dim, action_dim, hidden_dim, action_bound
        ).to(Config.DEVICE)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(Config.DEVICE)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.buffer = deque(maxlen=Config.CONTINUOUS_PPO_UPDATE_FREQ)

    def select_action(
        self, state: torch.Tensor
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Select action for continuous control"""
        state = state.to(Config.DEVICE)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state)
            value = self.critic(state).item()
        return action, log_prob, value

    def store_transition(
        self,
        state: torch.Tensor,
        action: np.ndarray,
        reward: float,
        log_prob: torch.Tensor,
        value: float,
        done: bool,
    ):
        """Store transition"""
        self.buffer.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "log_prob": log_prob,
                "value": value,
                "done": done,
            }
        )

    def compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> torch.Tensor:
        """Compute GAE for continuous actions"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32, device=Config.DEVICE)

    def update(self) -> Dict[str, float]:
        """Update continuous PPO agent"""
        if len(self.buffer) < Config.PPO_BATCH_SIZE:  # Use PPO_BATCH_SIZE for consistency
            return {}

        # Convert buffer to batch
        batch = list(self.buffer)
        states = torch.stack([t["state"] for t in batch]).to(Config.DEVICE)
        actions = torch.tensor(
            np.array([t["action"] for t in batch]), dtype=torch.float32, device=Config.DEVICE
        )
        old_log_probs = torch.stack([t["log_prob"] for t in batch]).to(Config.DEVICE)
        rewards = [t["reward"] for t in batch]
        values = [t["value"] for t in batch]
        dones = [t["done"] for t in batch]

        # Compute advantages and returns
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=Config.DEVICE)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        mean, std = self.actor(states)
        new_log_probs = self.actor.compute_log_prob(actions, mean, std)
        values_pred = self.critic(states).squeeze()

        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_loss = -(
            torch.min(ratio * advantages, clipped_ratio * advantages)
        ).mean()

        # Value loss
        value_loss = F.mse_loss(values_pred, returns)

        # Entropy bonus
        entropy = -0.5 * (2 * torch.log(std) + 1 + np.log(2 * np.pi)).sum(dim=-1).mean() # Corrected entropy calculation

        # Total loss
        total_loss = (
            policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        )

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }


