"""A2C Agent implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .model import ActorCritic
from .utils import set_seed


# Small helper type alias for rollouts
Rollouts = Dict[str, List[torch.Tensor]]


class A2CAgent:
    """Advantage Actor-Critic (A2C) agent."""

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        config: dict,
        device: str = "cpu"
    ):
        """Initialize the A2C agent.

        Args:
            num_inputs: State dimension.
            num_actions: Number of actions.
            config: Configuration dictionary.
            device: Device to run on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.model = ActorCritic(num_inputs, num_actions).to(self.device)
        lr = float(config.get("learning_rate", 1e-3))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = float(config.get("gamma", 0.99))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 0.5))

    def compute_returns(self, rewards: List[float], dones: List[bool], next_value: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute discounted returns.

        Args:
            rewards: List of rewards collected along the rollout.
            dones: List of done flags for each step in the rollout.
            next_value: Value estimate for the state following the rollout (or None).

        Returns:
            Tensor of discounted returns on the agent device with dtype float32.
        """
        returns: List[float] = []
        R = float(next_value.item()) if next_value is not None else 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0.0
            R = float(reward) + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given a single state, return (action, log_prob, value).

        The `state` may be a single state (shape [features]) or a batch of states
        (shape [N, features]). The returned tensors are on the agent device.
        """
        state = state.to(self.device)
        action_logits, value = self.model(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def update(self, rollouts: dict) -> dict:
        """Update the model using A2C loss.

        Args:
            rollouts: Dictionary containing states, actions, log_probs, values, rewards, dones.

        Returns:
            Dictionary with loss components.
        """
        states = torch.stack(rollouts["states"]).to(self.device)
        actions = torch.stack(rollouts["actions"]).to(self.device)
        old_log_probs = torch.stack(rollouts["log_probs"]).to(self.device)
        old_values = torch.stack(rollouts["values"]).to(self.device)
        rewards = rollouts["rewards"]
        dones = rollouts["dones"]

        # Compute returns
        with torch.no_grad():
            _, next_value = self.model(states[-1:])
        returns = self.compute_returns(rewards, dones, next_value)

        # Compute advantages
        advantages = returns - old_values.squeeze()

        # Get current policy and value
        action_logits, values = self.model(states)
        dist = Categorical(logits=action_logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Policy loss
        policy_loss = -(new_log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)

        # Entropy loss
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }