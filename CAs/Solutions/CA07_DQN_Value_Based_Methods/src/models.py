import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QNetwork(nn.Module):
    """
    Standard Q-network architecture for Deep Q-Networks.

    This network takes a state as input and outputs Q-values for each action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initializes the QNetwork.

        Args:
            state_dim: The dimensionality of the input state space.
            action_dim: The dimensionality of the action space.
            hidden_dim: The number of neurons in the hidden layers.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x: The input state tensor.

        Returns:
            The output Q-values for each action.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network architecture.

    This network separates the value and advantage streams to estimate Q-values,
    potentially improving learning in environments with many similar-valued actions.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initializes the DuelingQNetwork.

        Args:
            state_dim: The dimensionality of the input state space.
            action_dim: The dimensionality of the action space.
            hidden_dim: The number of neurons in the hidden layers.
        """
        super(DuelingQNetwork, self).__init__()

        # Feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            x: The input state tensor.

        Returns:
            The output Q-values for each action.
        """
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage: Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for exploration in DQN.

    Replaces standard linear layers to enable parameter-space noise for exploration,
    as described in "Noisy Networks for Exploration" (Fortunato et al., 2017).
    """

    def __init__(self, in_features: int, out_features: int, sigm_init: float = 0.5):
        """
        Initializes the NoisyLinear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            sigm_init: Initial standard deviation of the noise.
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigm_init = sigm_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer(
            "weight_epsilon", torch.FloatTensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer("bias_epsilon", torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Resets the parameters (weights and biases) of the layer.
        """
        fan_in = self.in_features
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        nn.init.constant_(self.weight_sigma, self.sigm_init / (fan_in**0.5))
        nn.init.constant_(self.bias_sigma, self.sigm_init / (fan_in**0.5))

    def scale_noise(self, size: int) -> torch.Tensor:
        """
        Generates and scales noise for parameters.
        """
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """
        Resets the noise by sampling new epsilon values.
        """
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the noisy layer.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    """
    Q-network with NoisyLinear layers for exploration.

    Instead of epsilon-greedy, this network uses parameter-space noise to drive exploration.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 128, sigm_init: float = 0.5
    ):
        """
        Initializes the NoisyQNetwork.

        Args:
            state_dim: The dimensionality of the input state space.
            action_dim: The dimensionality of the action space.
            hidden_dim: The number of neurons in the hidden layers.
            sigm_init: Initial standard deviation of the noise for NoisyLinear layers.
        """
        super(NoisyQNetwork, self).__init__()
        self.fc1 = NoisyLinear(state_dim, hidden_dim, sigm_init)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim, sigm_init)
        self.fc3 = NoisyLinear(hidden_dim, action_dim, sigm_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def reset_noise(self):
        """
        Resets the noise in all NoisyLinear layers.
        """
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class CategoricalQNetwork(nn.Module):
    """
    Categorical Q-network for Distributional Reinforcement Learning (C51).

    This network outputs a probability distribution over a set of atoms for each action,
    instead of a single Q-value. The Q-value is the expectation of this distribution.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, n_atoms: int
    ):
        """
        Initializes the CategoricalQNetwork.

        Args:
            state_dim: The dimensionality of the input state space.
            action_dim: The dimensionality of the action space.
            hidden_dim: The number of neurons in the hidden layers.
            n_atoms: The number of atoms in the value distribution.
        """
        super(CategoricalQNetwork, self).__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * n_atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network, outputting logits for each atom.

        Args:
            x: The input state tensor.

        Returns:
            A tensor of logits with shape (batch_size, action_dim, n_atoms).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.action_dim, self.n_atoms)

    def get_q_values(self, x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Calculates expected Q-values from the predicted categorical distribution.

        Args:
            x: The input state tensor.
            support: A tensor defining the support (atom values) of the distribution.

        Returns:
            A tensor of expected Q-values with shape (batch_size, action_dim).
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        # Q-values are the expectation of the distribution
        return torch.sum(probs * support, dim=-1)
