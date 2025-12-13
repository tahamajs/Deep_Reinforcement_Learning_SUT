"""
Neural network models for multi-agent RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class ActorNetwork(nn.Module):
    """Actor network for policy-based methods."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: Optional[str] = None,
    ):
        super(ActorNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        return self.network(obs)


class CriticNetwork(nn.Module):
    """Critic network for value estimation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 0,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super(CriticNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        input_dim = obs_dim + action_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            prev_dim = hidden_dim

        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self, obs: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through critic network."""
        if actions is not None:
            x = torch.cat([obs, actions], dim=-1)
        else:
            x = obs
        return self.network(x).squeeze(-1)


class QNetwork(nn.Module):
    """Q-network for value-based methods."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dueling: bool = False,
    ):
        super(QNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dueling = dueling

        # Build shared layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims[:-1]:  # Exclude last layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        if dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1),
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim),
            )
        else:
            # Standard Q-network
            self.q_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], action_dim),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q-network."""
        x = self.shared_layers(obs)

        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_values
        else:
            return self.q_head(x)


class PolicyNetwork(nn.Module):
    """Stochastic policy network with continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super(PolicyNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build network layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through policy network."""
        x = self.shared_layers(obs)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()

        # Tanh squashing
        action = torch.tanh(x_t)

        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, mean


class CentralizedCriticNetwork(nn.Module):
    """Centralized critic for multi-agent settings."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super(CentralizedCriticNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        # Input includes all agents' observations and actions
        total_obs_dim = obs_dim * n_agents
        total_action_dim = action_dim * n_agents
        input_dim = total_obs_dim + total_action_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            prev_dim = hidden_dim

        # Output layer (single value for team)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass through centralized critic."""
        # Flatten observations and actions from all agents
        obs_flat = obs.view(obs.size(0), -1)  # [batch, n_agents * obs_dim]
        actions_flat = actions.view(
            actions.size(0), -1
        )  # [batch, n_agents * action_dim]

        x = torch.cat([obs_flat, actions_flat], dim=-1)
        return self.network(x).squeeze(-1)


class IndividualCriticNetwork(nn.Module):
    """Individual critic for each agent in multi-agent setting."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super(IndividualCriticNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents

        # Each agent's critic sees its own obs/action + global context
        agent_input_dim = obs_dim + action_dim
        global_context_dim = obs_dim * (n_agents - 1) + action_dim * (n_agents - 1)
        input_dim = agent_input_dim + global_context_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            prev_dim = hidden_dim

        # Output layer (single value for this agent)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        agent_obs: torch.Tensor,
        agent_action: torch.Tensor,
        other_obs: torch.Tensor,
        other_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through individual critic."""
        # Concatenate agent's own obs/action with others' obs/actions
        agent_input = torch.cat([agent_obs, agent_action], dim=-1)
        others_input = torch.cat(
            [
                other_obs.view(other_obs.size(0), -1),
                other_actions.view(other_actions.size(0), -1),
            ],
            dim=-1,
        )

        x = torch.cat([agent_input, others_input], dim=-1)
        return self.network(x).squeeze(-1)
