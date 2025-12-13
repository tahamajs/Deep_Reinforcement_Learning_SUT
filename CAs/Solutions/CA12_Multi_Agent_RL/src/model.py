import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

class MLP(nn.Module):
    """A simple Multi-Layer Perceptron network."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: nn.Module = nn.ReLU(),
        output_activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation:
            layers.append(output_activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

class ActorNetwork(nn.Module):
    """Actor network for policy-based methods, outputting action logits for discrete actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.network = MLP(obs_dim, action_dim, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network, returns action logits."""
        return self.network(obs)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """Sample action from policy given observation."""
        logits = self.forward(obs)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
        return action.item(), log_prob

class CriticNetwork(nn.Module):
    """Centralized critic network for value estimation in multi-agent settings.
    It takes flattened observations and actions from all agents as input."""

    def __init__(
        self,
        total_obs_dim: int,
        total_action_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        input_dim = total_obs_dim + total_action_dim
        self.network = MLP(input_dim, 1, hidden_dims, activation)

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through critic network.
        Args:
            obs (torch.Tensor): Concatenated observations from all agents [batch_size, total_obs_dim].
            actions (torch.Tensor): Concatenated actions from all agents [batch_size, total_action_dim].
        Returns:
            torch.Tensor: Predicted Q-value [batch_size, 1].
        """
        x = torch.cat([obs.flatten(start_dim=1), actions.flatten(start_dim=1)], dim=-1)
        return self.network(x).squeeze(-1)

class CommunicationEncoder(nn.Module):
    """Encodes an agent's observation into a message vector."""
    def __init__(
        self,
        obs_dim: int,
        message_dim: int,
        hidden_dims: List[int] = [64],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.network = MLP(obs_dim, message_dim, hidden_dims, activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns the message."""
        return self.network(obs)

class CommunicationDecoder(nn.Module):
    """Decodes messages from other agents into a context vector for the current agent."""
    def __init__(
        self,
        num_other_agents: int,
        message_dim: int,
        output_context_dim: int,
        hidden_dims: List[int] = [64],
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        input_dim = num_other_agents * message_dim
        self.network = MLP(input_dim, output_context_dim, hidden_dims, activation)

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns the decoded message context."""
        return self.network(messages.flatten(start_dim=1))

