"""
Energy-Efficient Reinforcement Learning

This module implements energy-efficient learning approaches for RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time


class EarlyExitNetwork(nn.Module):
    """Network with early exit capabilities for energy efficiency."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [128, 128],
        exit_thresholds: List[float] = [0.8, 0.9],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.exit_thresholds = exit_thresholds

        # Build network layers
        self.layers = nn.ModuleList()
        self.exit_heads = nn.ModuleList()

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # Main layer
            self.layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
                )
            )

            # Exit head
            self.exit_heads.append(nn.Linear(hidden_dim, output_dim))

            prev_dim = hidden_dim

        # Final layer
        self.final_layer = nn.Linear(prev_dim, output_dim)

        # Energy tracking
        self.energy_consumption = 0.0
        self.exit_statistics = {
            "early_exits": [0] * len(exit_thresholds),
            "total_forward_passes": 0,
            "energy_savings": 0.0,
        }

    def forward(
        self, x: torch.Tensor, force_full: bool = False
    ) -> Tuple[torch.Tensor, int, float]:
        """Forward pass with early exit capability."""
        current_input = x
        energy_used = 0.0

        for i, (layer, exit_head) in enumerate(zip(self.layers, self.exit_heads)):
            # Forward through layer
            current_input = layer(current_input)
            energy_used += self._compute_layer_energy(i, current_input.shape)

            # Check for early exit
            if not force_full and i < len(self.exit_thresholds):
                exit_output = exit_head(current_input)
                exit_probs = F.softmax(exit_output, dim=-1)
                max_confidence = torch.max(exit_probs, dim=-1)[0]

                if max_confidence.item() > self.exit_thresholds[i]:
                    # Early exit
                    self.exit_statistics["early_exits"][i] += 1
                    self.exit_statistics["total_forward_passes"] += 1
                    self.energy_consumption += energy_used

                    return exit_output, i, energy_used

        # Full forward pass
        final_output = self.final_layer(current_input)
        energy_used += self._compute_layer_energy(len(self.layers), final_output.shape)

        self.exit_statistics["total_forward_passes"] += 1
        self.energy_consumption += energy_used

        return final_output, len(self.layers), energy_used

    def _compute_layer_energy(self, layer_idx: int, output_shape: Tuple) -> float:
        """Compute energy consumption for a layer."""
        # Simplified energy model: energy = operations * energy_per_operation
        operations = np.prod(output_shape)
        energy_per_operation = 1e-12  # 1 pJ per operation

        return operations * energy_per_operation

    def get_energy_statistics(self) -> Dict[str, Any]:
        """Get energy consumption statistics."""
        total_early_exits = sum(self.exit_statistics["early_exits"])
        early_exit_rate = total_early_exits / max(
            1, self.exit_statistics["total_forward_passes"]
        )

        return {
            "total_energy_consumption": self.energy_consumption,
            "early_exit_rate": early_exit_rate,
            "exit_statistics": self.exit_statistics.copy(),
            "energy_savings": self.exit_statistics["energy_savings"],
        }


class AdaptiveComputation(nn.Module):
    """Adaptive computation for energy-efficient RL."""

    def __init__(self, input_dim: int, output_dim: int, max_layers: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_layers = max_layers

        # Adaptive layers
        self.layers = nn.ModuleList()
        for i in range(max_layers):
            layer = nn.Sequential(
                nn.Linear(input_dim if i == 0 else 128, 128), nn.ReLU(), nn.Dropout(0.1)
            )
            self.layers.append(layer)

        # Output layer
        self.output_layer = nn.Linear(128, output_dim)

        # Computation controller
        self.computation_controller = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        # Energy tracking
        self.energy_consumption = 0.0
        self.computation_statistics = {
            "layer_usage": [0] * max_layers,
            "total_forward_passes": 0,
            "adaptive_decisions": [],
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, float]:
        """Forward pass with adaptive computation."""
        # Decide how many layers to use
        computation_decision = self.computation_controller(x)
        num_layers = int(computation_decision.item() * self.max_layers) + 1
        num_layers = min(num_layers, self.max_layers)

        # Forward through selected layers
        current_input = x
        energy_used = 0.0

        for i in range(num_layers):
            current_input = self.layers[i](current_input)
            energy_used += self._compute_layer_energy(i, current_input.shape)

            # Update statistics
            self.computation_statistics["layer_usage"][i] += 1

        # Output layer
        output = self.output_layer(current_input)
        energy_used += self._compute_layer_energy(num_layers, output.shape)

        # Update statistics
        self.computation_statistics["total_forward_passes"] += 1
        self.computation_statistics["adaptive_decisions"].append(num_layers)
        self.energy_consumption += energy_used

        return output, num_layers, energy_used

    def _compute_layer_energy(self, layer_idx: int, output_shape: Tuple) -> float:
        """Compute energy consumption for a layer."""
        operations = np.prod(output_shape)
        energy_per_operation = 1e-12  # 1 pJ per operation
        return operations * energy_per_operation

    def get_computation_statistics(self) -> Dict[str, Any]:
        """Get adaptive computation statistics."""
        avg_layers_used = np.mean(self.computation_statistics["adaptive_decisions"])
        layer_usage_rates = [
            usage / max(1, self.computation_statistics["total_forward_passes"])
            for usage in self.computation_statistics["layer_usage"]
        ]

        return {
            "total_energy_consumption": self.energy_consumption,
            "average_layers_used": avg_layers_used,
            "layer_usage_rates": layer_usage_rates,
            "computation_statistics": self.computation_statistics.copy(),
        }


class EnergyEfficientRL(nn.Module):
    """Energy-efficient reinforcement learning agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        method: str = "early_exit",
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.method = method
        self.device = device

        # Initialize network based on method
        if method == "early_exit":
            self.network = EarlyExitNetwork(state_dim, action_dim)
        elif method == "adaptive":
            self.network = AdaptiveComputation(state_dim, action_dim)
        else:
            # Standard network
            self.network = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Energy tracking
        self.energy_consumption = 0.0
        self.performance_history = {
            "rewards": [],
            "energy_consumption": [],
            "efficiency_ratios": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.method in ["early_exit", "adaptive"]:
            output, num_layers, energy = self.network(x)
            self.energy_consumption += energy
            return output
        else:
            return self.network(x)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action with energy efficiency."""
        with torch.no_grad():
            if self.method == "early_exit":
                action_logits, num_layers, energy = self.network(state.unsqueeze(0))
            elif self.method == "adaptive":
                action_logits, num_layers, energy = self.network(state.unsqueeze(0))
            else:
                action_logits = self.network(state.unsqueeze(0))

            action_probs = F.softmax(action_logits, dim=-1)

            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action = torch.multinomial(action_probs, 1)

            return action.item()

    def update(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> float:
        """Update policy with energy efficiency tracking."""
        # Forward pass
        if self.method in ["early_exit", "adaptive"]:
            action_logits, num_layers, energy = self.network(states)
        else:
            action_logits = self.network(states)

        # Compute loss
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        loss = -(log_probs * rewards.unsqueeze(1)).mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update performance history
        self.performance_history["rewards"].append(rewards.mean().item())
        self.performance_history["energy_consumption"].append(self.energy_consumption)

        # Compute efficiency ratio
        if self.energy_consumption > 0:
            efficiency_ratio = rewards.mean().item() / self.energy_consumption
            self.performance_history["efficiency_ratios"].append(efficiency_ratio)

        return loss.item()

    def get_energy_statistics(self) -> Dict[str, Any]:
        """Get energy consumption statistics."""
        stats = {
            "total_energy_consumption": self.energy_consumption,
            "performance_history": self.performance_history.copy(),
        }

        if self.method == "early_exit":
            stats["early_exit_stats"] = self.network.get_energy_statistics()
        elif self.method == "adaptive":
            stats["adaptive_stats"] = self.network.get_computation_statistics()

        return stats

    def compute_energy_efficiency(self) -> float:
        """Compute energy efficiency metric."""
        if not self.performance_history["rewards"] or self.energy_consumption == 0:
            return 0.0

        avg_reward = np.mean(self.performance_history["rewards"])
        return avg_reward / self.energy_consumption

    def reset_energy_tracking(self):
        """Reset energy consumption tracking."""
        self.energy_consumption = 0.0
        if hasattr(self.network, "energy_consumption"):
            self.network.energy_consumption = 0.0
        if hasattr(self.network, "exit_statistics"):
            self.network.exit_statistics = {
                "early_exits": [0] * len(self.network.exit_thresholds),
                "total_forward_passes": 0,
                "energy_savings": 0.0,
            }
        if hasattr(self.network, "computation_statistics"):
            self.network.computation_statistics = {
                "layer_usage": [0] * self.network.max_layers,
                "total_forward_passes": 0,
                "adaptive_decisions": [],
            }


class EnergyAwareTraining:
    """Energy-aware training for RL agents."""

    def __init__(self, agent: EnergyEfficientRL, energy_budget: float = 1.0):
        self.agent = agent
        self.energy_budget = energy_budget
        self.current_energy_usage = 0.0

        # Training parameters
        self.energy_penalty = 0.1
        self.performance_threshold = 0.8

        # Training history
        self.training_history = {
            "energy_usage": [],
            "performance": [],
            "efficiency": [],
        }

    def train_step(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> float:
        """Training step with energy awareness."""
        # Check energy budget
        if self.current_energy_usage >= self.energy_budget:
            return 0.0  # No training if energy budget exceeded

        # Forward pass
        if self.agent.method in ["early_exit", "adaptive"]:
            action_logits, num_layers, energy = self.agent.network(states)
        else:
            action_logits = self.agent.network(states)
            energy = 0.0

        # Compute loss with energy penalty
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        policy_loss = -(log_probs * rewards.unsqueeze(1)).mean()

        # Energy penalty
        energy_penalty = self.energy_penalty * energy
        total_loss = policy_loss + energy_penalty

        # Backward pass
        self.agent.optimizer.zero_grad()
        total_loss.backward()
        self.agent.optimizer.step()

        # Update energy usage
        self.current_energy_usage += energy

        # Update training history
        self.training_history["energy_usage"].append(self.current_energy_usage)
        self.training_history["performance"].append(rewards.mean().item())
        self.training_history["efficiency"].append(
            rewards.mean().item() / max(energy, 1e-8)
        )

        return total_loss.item()

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "energy_budget": self.energy_budget,
            "current_energy_usage": self.current_energy_usage,
            "energy_utilization": self.current_energy_usage / self.energy_budget,
            "training_history": self.training_history.copy(),
        }

    def reset_energy_budget(self, new_budget: float = None):
        """Reset energy budget."""
        if new_budget is not None:
            self.energy_budget = new_budget
        self.current_energy_usage = 0.0
