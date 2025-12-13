import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple

from src.models.networks import PolicyNetwork, ValueNetwork
from src.config import Config

class ActorCriticAgent:
    """Actor-Critic agent with advantage estimation"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        gamma: float = Config.AC_GAMMA,
        gae_lambda: float = Config.AC_GAE_LAMBDA,
        lr: float = Config.AC_LR,
        value_coeff: float = Config.AC_VALUE_COEFF,
        entropy_coeff: float = Config.AC_ENTROPY_COEFF,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim).to(Config.DEVICE)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(Config.DEVICE)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def select_action(
        self, state: torch.Tensor
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Select action and return value"""
        state = state.to(Config.DEVICE)
        action, log_prob = self.actor.get_action(state)
        value = self.critic(state).item()
        return action, log_prob, value

    def compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32, device=Config.DEVICE)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """Update actor and critic"""

        # Get current policy outputs
        states = states.to(Config.DEVICE)
        actions = actions.to(Config.DEVICE)
        log_probs = log_probs.to(Config.DEVICE)
        returns = returns.to(Config.DEVICE)
        advantages = advantages.to(Config.DEVICE)

        new_logits = self.actor(states)
        new_log_probs = (
            F.log_softmax(new_logits, dim=-1)
            .gather(1, actions.unsqueeze(-1))
            .squeeze(-1)
        )
        entropy = (
            -(F.softmax(new_logits, dim=-1) * F.log_softmax(new_logits, dim=-1))
            .sum(dim=-1)
            .mean()
        )

        # Policy loss (note: PPO clipping is applied here for demonstration, but a pure AC would not have it)
        # For a pure A2C, policy_loss = -(new_log_probs * advantages).mean()
        ratio = torch.exp(new_log_probs - log_probs)
        # This is essentially PPO's clipped objective, adapted for A2C context
        # A pure A2C would simply be: policy_loss = -(new_log_probs * advantages).mean()
        policy_loss = -(torch.min(ratio * advantages, torch.clamp(ratio, 1 - Config.PPO_CLIP_RATIO, 1 + Config.PPO_CLIP_RATIO) * advantages)).mean()


        # Value loss
        values = self.critic(states).squeeze()
        value_loss = F.mse_loss(values, returns)

        # Total loss
        total_loss = (
            policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
        )

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }

