import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple

from src.models.networks import PolicyNetwork, ValueNetwork
from src.config import Config

class PPOAgent:
    """Proximal Policy Optimization (PPO) agent"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        gamma: float = Config.PPO_GAMMA,
        gae_lambda: float = Config.PPO_GAE_LAMBDA,
        clip_ratio: float = Config.PPO_CLIP_RATIO,
        lr: float = Config.PPO_LR,
        value_coeff: float = Config.PPO_VALUE_COEFF,
        entropy_coeff: float = Config.PPO_ENTROPY_COEFF,
        ppo_epochs: int = Config.PPO_EPOCHS,
        batch_size: int = Config.PPO_BATCH_SIZE,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(Config.DEVICE)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(Config.DEVICE)
        self.actor_old = PolicyNetwork(state_dim, action_dim, hidden_dim).to(Config.DEVICE)
        self.critic_old = ValueNetwork(state_dim, hidden_dim).to(Config.DEVICE)

        # Copy parameters
        self._update_old_networks()

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        self.buffer = deque(maxlen=Config.PPO_UPDATE_FREQ)

    def _update_old_networks(self):
        """Update old networks with current parameters"""
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

    def select_action(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action using current policy"""
        state = state.to(Config.DEVICE)
        with torch.no_grad():
            action, log_prob = self.actor_old.get_action(state)
            value = self.critic_old(state).item()
        return action, log_prob, value

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: float,
        done: bool,
    ):
        """Store transition in buffer"""
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
        """Compute Generalized Advantage Estimation"""
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
        """Update PPO agent"""
        if len(self.buffer) < self.batch_size:
            return {}

        # Convert buffer to tensors
        batch = list(self.buffer)
        states = torch.stack([t["state"] for t in batch]).to(Config.DEVICE)
        actions = torch.tensor([t["action"] for t in batch], dtype=torch.long, device=Config.DEVICE)
        old_log_probs = torch.stack([t["log_prob"] for t in batch]).to(Config.DEVICE)
        rewards = [t["reward"] for t in batch]
        values = [t["value"] for t in batch]
        dones = [t["done"] for t in batch]

        # Compute returns and advantages
        advantages = self.compute_gae(rewards, values, dones)
        returns = advantages + torch.tensor(values, dtype=torch.float32, device=Config.DEVICE)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.ppo_epochs):
            # Sample mini-batch
            indices = torch.randperm(len(batch), device=Config.DEVICE)[: self.batch_size]

            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_old_log_probs = old_log_probs[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]

            # Get current policy outputs
            logits = self.actor(batch_states)
            values_pred = self.critic(batch_states).squeeze()

            new_log_probs = (
                F.log_softmax(logits, dim=-1)
                .gather(1, batch_actions.unsqueeze(-1))
                .squeeze(-1)
            )
            entropy = (
                -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1))
                .sum(dim=-1)
                .mean()
            )

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -(
                torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages)
            ).mean()

            # Value loss
            value_loss = F.mse_loss(values_pred, batch_returns)

            # Total loss
            loss = (
                policy_loss
                + self.value_coeff * value_loss
                - self.entropy_coeff * entropy
            )

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        # Update old networks
        self._update_old_networks()

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy": total_entropy / self.ppo_epochs,
        }

