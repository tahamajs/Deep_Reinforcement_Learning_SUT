"""
Advanced Neural Network Architectures for Policy Gradient Methods
CA4: Policy Gradient Methods and Neural Networks in RL - Advanced Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import math


class CNNPolicyNetwork(nn.Module):
    """Convolutional Neural Network Policy for Image-based Environments"""

    def __init__(
        self, input_channels: int = 3, action_size: int = 4, hidden_size: int = 512
    ):
        """Initialize CNN policy network

        Args:
            input_channels: Number of input channels
            action_size: Number of actions
            hidden_size: Hidden layer size
        """
        super(CNNPolicyNetwork, self).__init__()

        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Third convolutional block
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Calculate the size after convolutions
        self.conv_output_size = self._get_conv_output_size(input_channels)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, action_size),
        )

    def _get_conv_output_size(self, input_channels: int) -> int:
        """Calculate the output size of convolutional layers"""
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        with torch.no_grad():
            conv_output = self.conv_layers(dummy_input)
        return int(np.prod(conv_output.size()[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Action logits
        """
        # Ensure input is in correct format
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension

        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc_layers(x)

        return x

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
        probs = F.softmax(logits, dim=-1)

        # Sample action
        action = torch.multinomial(probs, 1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action)

        return action.item(), log_prob.squeeze()


class LSTMPolicyNetwork(nn.Module):
    """LSTM-based Policy Network for Sequential Decision Making"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize LSTM policy network

        Args:
            state_size: State space dimension
            action_size: Number of actions
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMPolicyNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_projection = nn.Linear(state_size, hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_size),
        )

        # Initialize hidden state
        self.hidden_state = None

    def init_hidden(self, batch_size: int = 1):
        """Initialize hidden state

        Args:
            batch_size: Batch size
        """
        self.hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            x: Input tensor (batch_size, sequence_length, state_size) or (batch_size, state_size)

        Returns:
            Action logits
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size = x.size(0)

        # Initialize hidden state if not set
        if self.hidden_state is None:
            self.init_hidden(batch_size)

        # Input projection
        x = self.input_projection(x)

        # LSTM forward pass
        lstm_out, self.hidden_state = self.lstm(x, self.hidden_state)

        # Use the last timestep output
        last_output = lstm_out[:, -1, :]

        # Output layers
        logits = self.output_layers(last_output)

        return logits

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
        probs = F.softmax(logits, dim=-1)

        # Sample action
        action = torch.multinomial(probs, 1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action)

        return action.item(), log_prob.squeeze()


class TransformerPolicyNetwork(nn.Module):
    """Transformer-based Policy Network for Complex Sequential Decision Making"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        """Initialize Transformer policy network

        Args:
            state_size: State space dimension
            action_size: Number of actions
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super(TransformerPolicyNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(state_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            x: Input tensor (batch_size, sequence_length, state_size) or (batch_size, state_size)

        Returns:
            Action logits
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer forward pass
        transformer_out = self.transformer(x)

        # Use the last timestep output
        last_output = transformer_out[:, -1, :]

        # Output layers
        logits = self.output_layers(last_output)

        return logits

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
        probs = F.softmax(logits, dim=-1)

        # Sample action
        action = torch.multinomial(probs, 1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action)

        return action.item(), log_prob.squeeze()


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding

        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input

        Args:
            x: Input tensor

        Returns:
            Input with positional encoding
        """
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """Residual block for deep networks"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """Initialize residual block

        Args:
            hidden_size: Hidden layer size
            dropout: Dropout rate
        """
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return x + self.layers(x)


class DeepResidualPolicyNetwork(nn.Module):
    """Deep Residual Policy Network"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        num_blocks: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize deep residual policy network

        Args:
            state_size: State space dimension
            action_size: Number of actions
            hidden_size: Hidden layer size
            num_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super(DeepResidualPolicyNetwork, self).__init__()

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            x: Input tensor

        Returns:
            Action logits
        """
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.output_layer(x)
        return x

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
        probs = F.softmax(logits, dim=-1)

        # Sample action
        action = torch.multinomial(probs, 1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action)

        return action.item(), log_prob.squeeze()


class AttentionPolicyNetwork(nn.Module):
    """Attention-based Policy Network"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize attention policy network

        Args:
            state_size: State space dimension
            action_size: Number of actions
            hidden_size: Hidden layer size
            num_heads: Number of attention heads
            num_layers: Number of attention layers
            dropout: Dropout rate
        """
        super(AttentionPolicyNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # Input projection
        self.input_projection = nn.Linear(state_size, hidden_size)

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_size, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )

        # Feed-forward networks
        self.feed_forward = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size),
                )
                for _ in range(num_layers)
            ]
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            x: Input tensor (batch_size, sequence_length, state_size) or (batch_size, state_size)

        Returns:
            Action logits
        """
        # Handle single timestep input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Input projection
        x = self.input_projection(x)

        # Apply attention layers
        for attention, layer_norm, ff in zip(
            self.attention_layers, self.layer_norms, self.feed_forward
        ):
            # Self-attention
            attn_output, _ = attention(x, x, x)
            x = layer_norm(x + attn_output)

            # Feed-forward
            ff_output = ff(x)
            x = layer_norm(x + ff_output)

        # Use the last timestep output
        last_output = x[:, -1, :]

        # Output layer
        logits = self.output_layer(last_output)

        return logits

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
        probs = F.softmax(logits, dim=-1)

        # Sample action
        action = torch.multinomial(probs, 1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action)

        return action.item(), log_prob.squeeze()


class EnsemblePolicyNetwork(nn.Module):
    """Ensemble Policy Network with multiple policy heads"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
        num_policies: int = 5,
        dropout: float = 0.1,
    ):
        """Initialize ensemble policy network

        Args:
            state_size: State space dimension
            action_size: Number of actions
            hidden_size: Hidden layer size
            num_policies: Number of policies in ensemble
            dropout: Dropout rate
        """
        super(EnsemblePolicyNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_policies = num_policies

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Individual policy heads
        self.policy_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, action_size),
                )
                for _ in range(num_policies)
            ]
        )

        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(num_policies) / num_policies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network

        Args:
            x: Input tensor

        Returns:
            Ensemble action logits
        """
        # Extract features
        features = self.feature_extractor(x)

        # Get predictions from each policy head
        policy_outputs = []
        for head in self.policy_heads:
            policy_outputs.append(head(features))

        # Stack outputs
        policy_outputs = torch.stack(
            policy_outputs, dim=1
        )  # (batch_size, num_policies, action_size)

        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_output = torch.sum(policy_outputs * weights.view(1, -1, 1), dim=1)

        return ensemble_output

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
        probs = F.softmax(logits, dim=-1)

        # Sample action
        action = torch.multinomial(probs, 1)
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action)

        return action.item(), log_prob.squeeze()

    def get_individual_predictions(self, state: torch.Tensor) -> torch.Tensor:
        """Get predictions from individual policy heads

        Args:
            state: State tensor

        Returns:
            Individual policy predictions
        """
        features = self.feature_extractor(state)

        policy_outputs = []
        for head in self.policy_heads:
            policy_outputs.append(head(features))

        return torch.stack(policy_outputs, dim=1)


def create_advanced_policy_network(
    network_type: str, state_size: int, action_size: int, **kwargs
) -> nn.Module:
    """Factory function to create advanced policy networks

    Args:
        network_type: Type of network ('cnn', 'lstm', 'transformer', 'residual', 'attention', 'ensemble')
        state_size: State space dimension
        action_size: Action space dimension
        **kwargs: Additional arguments

    Returns:
        Policy network instance
    """
    if network_type.lower() == "cnn":
        return CNNPolicyNetwork(action_size=action_size, **kwargs)
    elif network_type.lower() == "lstm":
        return LSTMPolicyNetwork(state_size, action_size, **kwargs)
    elif network_type.lower() == "transformer":
        return TransformerPolicyNetwork(state_size, action_size, **kwargs)
    elif network_type.lower() == "residual":
        return DeepResidualPolicyNetwork(state_size, action_size, **kwargs)
    elif network_type.lower() == "attention":
        return AttentionPolicyNetwork(state_size, action_size, **kwargs)
    elif network_type.lower() == "ensemble":
        return EnsemblePolicyNetwork(state_size, action_size, **kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def test_network_performance(
    network: nn.Module, input_shape: tuple, num_tests: int = 1000
) -> Dict[str, float]:
    """Test network performance and efficiency

    Args:
        network: Network to test
        input_shape: Input tensor shape
        num_tests: Number of test iterations

    Returns:
        Performance metrics
    """
    import time

    network.eval()

    # Test inference speed
    dummy_input = torch.randn(1, *input_shape)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_tests):
            _ = network(dummy_input)
    end_time = time.time()

    inference_time = (end_time - start_time) / num_tests

    # Test memory usage
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)

    return {
        "inference_time_ms": inference_time * 1000,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "memory_efficiency": (
            total_params / (inference_time * 1000) if inference_time > 0 else 0
        ),
    }

