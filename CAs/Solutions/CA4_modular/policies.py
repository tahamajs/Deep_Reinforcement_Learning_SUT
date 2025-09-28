"""
Policy networks for Policy Gradient Methods
CA4: Policy Gradient Methods and Neural Networks in RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """Neural network policy for discrete action spaces"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """Initialize policy network

        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            hidden_size: Size of hidden layers
        """
        super(PolicyNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            state: State tensor

        Returns:
            Action probabilities (logits)
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        action_logits = self.fc3(x)
        return action_logits

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities from state

        Args:
            state: State tensor

        Returns:
            Action probability distribution
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)

    def sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Sample action from policy

        Args:
            state: State tensor

        Returns:
            Tuple of (action, log_probability)
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def get_log_prob(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """Get log probability of action under current policy

        Args:
            state: State tensor
            action: Action taken

        Returns:
            Log probability of action
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        return dist.log_prob(torch.tensor(action))


class ValueNetwork(nn.Module):
    """Value network for state value estimation"""

    def __init__(self, state_size: int, hidden_size: int = 128):
        """Initialize value network

        Args:
            state_size: Dimension of state space
            hidden_size: Size of hidden layers
        """
        super(ValueNetwork, self).__init__()
        self.state_size = state_size

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            state: State tensor

        Returns:
            State value estimate
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)
        return value


class SharedFeatureNetwork(nn.Module):
    """Shared feature extraction for Actor-Critic"""

    def __init__(self, state_size: int, hidden_size: int = 128, feature_size: int = 64):
        """Initialize shared feature network

        Args:
            state_size: Dimension of state space
            hidden_size: Size of hidden layers
            feature_size: Size of output features
        """
        super(SharedFeatureNetwork, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feature_size),
            nn.ReLU()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Extract features from state

        Args:
            state: State tensor

        Returns:
            Feature representation
        """
        return self.shared_layers(state)


class AdvancedActorCritic(nn.Module):
    """Advanced Actor-Critic with shared features"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, feature_size: int = 64):
        """Initialize advanced actor-critic network

        Args:
            state_size: Dimension of state space
            action_size: Number of discrete actions
            hidden_size: Size of hidden layers
            feature_size: Size of shared features
        """
        super(AdvancedActorCritic, self).__init__()

        self.shared_features = SharedFeatureNetwork(state_size, hidden_size, feature_size)

        self.actor_head = nn.Sequential(
            nn.Linear(feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor-critic

        Args:
            state: State tensor

        Returns:
            Tuple of (action_probs, value)
        """
        features = self.shared_features(state)
        action_probs = self.actor_head(features)
        value = self.critic_head(features)
        return action_probs, value.squeeze()

    def get_action_and_value(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action and value from state

        Args:
            state: State tensor

        Returns:
            Tuple of (action, log_prob, value)
        """
        action_probs, value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value


class ContinuousPolicyNetwork(nn.Module):
    """Policy network for continuous action spaces"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """Initialize continuous policy network

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super(ContinuousPolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network

        Args:
            state: State tensor

        Returns:
            Tuple of (mean, log_std)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))  # Bounded actions [-1, 1]
        log_std = torch.clamp(self.log_std_head(x), -20, 2)  # Prevent extreme values

        return mu, log_std

    def sample_action(self, state: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample action from continuous policy

        Args:
            state: State tensor

        Returns:
            Tuple of (action, log_probability)
        """
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach().numpy(), log_prob

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action under current policy

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Tuple of (log_prob, entropy, value) - Note: value is None for policy-only network
        """
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, None


class ContinuousActorCriticAgent(nn.Module):
    """Actor-Critic for continuous action spaces"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """Initialize continuous actor-critic

        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space
            hidden_size: Size of hidden layers
        """
        super(ContinuousActorCriticAgent, self).__init__()

        self.policy_net = ContinuousPolicyNetwork(state_size, action_size, hidden_size)
        self.value_net = ValueNetwork(state_size, hidden_size)

    def get_action(self, state: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample action from policy

        Args:
            state: State tensor

        Returns:
            Tuple of (action, log_probability)
        """
        return self.policy_net.sample_action(state)

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate action under current policy and get value

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            Tuple of (log_prob, entropy, value)
        """
        log_prob, entropy, _ = self.policy_net.evaluate_action(state, action)
        value = self.value_net(state).squeeze()
        return log_prob, entropy, value


def create_policy_network(state_size: int, action_size: int, continuous: bool = False,
                         shared_features: bool = False, **kwargs) -> nn.Module:
    """Factory function to create policy network

    Args:
        state_size: Dimension of state space
        action_size: Dimension of action space
        continuous: Whether action space is continuous
        shared_features: Whether to use shared features
        **kwargs: Additional arguments for network

    Returns:
        Policy network instance
    """
    if continuous:
        if shared_features:
            return ContinuousActorCriticAgent(state_size, action_size, **kwargs)
        else:
            return ContinuousPolicyNetwork(state_size, action_size, **kwargs)
    else:
        if shared_features:
            return AdvancedActorCritic(state_size, action_size, **kwargs)
        else:
            return PolicyNetwork(state_size, action_size, **kwargs)


def test_policy_network(network: nn.Module, state_size: int, continuous: bool = False):
    """Test policy network with dummy input

    Args:
        network: Policy network to test
        state_size: Dimension of state space
        continuous: Whether action space is continuous

    Returns:
        Test results dictionary
    """
    # Create dummy state
    state = torch.randn(1, state_size)

    try:
        if continuous:
            if hasattr(network, 'get_action'):
                action, log_prob = network.get_action(state)
                return {
                    'success': True,
                    'action_shape': action.shape,
                    'log_prob_shape': log_prob.shape,
                    'action_sample': action
                }
            else:
                mu, log_std = network(state)
                return {
                    'success': True,
                    'mu_shape': mu.shape,
                    'log_std_shape': log_std.shape,
                    'mu_sample': mu.detach().numpy(),
                    'std_sample': torch.exp(log_std).detach().numpy()
                }
        else:
            if hasattr(network, 'get_action_and_value'):
                action, log_prob, value = network.get_action_and_value(state)
                return {
                    'success': True,
                    'action': action,
                    'log_prob_shape': log_prob.shape,
                    'value_shape': value.shape
                }
            else:
                probs = network.get_action_probs(state)
                action, log_prob = network.sample_action(state)
                return {
                    'success': True,
                    'action_probs_shape': probs.shape,
                    'sampled_action': action,
                    'log_prob_shape': log_prob.shape
                }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }