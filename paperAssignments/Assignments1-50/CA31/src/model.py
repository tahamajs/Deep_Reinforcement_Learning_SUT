"""Neural network models for Actor-Critic."""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple


class ActorCritic(nn.Module):
    """Actor-Critic neural network.

    Combines policy (actor) and value (critic) functions in a single network.
    """

    def __init__(self, num_inputs: int, num_outputs: int, hidden_size: int = 128):
        """Initialize the Actor-Critic network.

        Args:
            num_inputs: Number of input features (state dimension).
            num_outputs: Number of possible actions.
            hidden_size: Size of the hidden layers.
        """
        super().__init__()
        self.num_outputs = num_outputs

        self.shared = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_size, num_outputs)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input tensor (batch of states).

        Returns:
            Tuple of (action logits, value estimates).
        """
        shared_out = self.shared(x)
        action_logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_logits, value

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the policy for a batch of states.

        Args:
            state: Batch of current states.

        Returns:
            Tuple of (actions, log_probs, values).
        """
        action_logits, values = self.forward(state)
        dist = Categorical(logits=action_logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values.squeeze(-1)