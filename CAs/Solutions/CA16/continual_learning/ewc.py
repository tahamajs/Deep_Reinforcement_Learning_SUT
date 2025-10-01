"""
Elastic Weight Consolidation (EWC) for Continual Learning

This module implements EWC to prevent catastrophic forgetting in neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import copy


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""

    def __init__(
        self, model: nn.Module, lambda_ewc: float = 1000.0, device: str = "cpu"
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.device = device

        # Store optimal parameters and Fisher information for each task
        self.optimal_params = {}
        self.fisher_information = {}
        self.task_count = 0

        # Training history
        self.ewc_losses = []
        self.task_performances = {}

    def compute_fisher_information(
        self, dataloader, task_id: int, num_samples: int = 1000
    ):
        """Compute Fisher information matrix for current task."""
        self.model.eval()

        # Initialize Fisher information
        fisher_info = {}
        for name, param in self.model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)

        # Compute Fisher information
        sample_count = 0
        for batch_idx, (states, actions, rewards) in enumerate(dataloader):
            if sample_count >= num_samples:
                break

            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)

            # Forward pass
            outputs = self.model(states)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take first output if tuple

            # Compute loss
            if actions.dtype == torch.long:
                loss = F.cross_entropy(outputs, actions)
            else:
                loss = F.mse_loss(outputs, actions.float())

            # Compute gradients
            self.model.zero_grad()
            loss.backward()

            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data**2

            sample_count += len(states)

        # Normalize Fisher information
        for name in fisher_info:
            fisher_info[name] /= sample_count

        # Store Fisher information
        self.fisher_information[task_id] = fisher_info

        return fisher_info

    def save_optimal_params(self, task_id: int):
        """Save optimal parameters for current task."""
        optimal_params = {}
        for name, param in self.model.named_parameters():
            optimal_params[name] = param.data.clone()

        self.optimal_params[task_id] = optimal_params
        self.task_count += 1

    def compute_ewc_loss(self, task_id: int) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if task_id not in self.optimal_params or task_id not in self.fisher_information:
            return torch.tensor(0.0, device=self.device)

        ewc_loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if (
                name in self.optimal_params[task_id]
                and name in self.fisher_information[task_id]
            ):
                optimal_param = self.optimal_params[task_id][name]
                fisher_info = self.fisher_information[task_id][name]

                # EWC loss: λ/2 * Σ F_i * (θ_i - θ_i*)^2
                ewc_loss += torch.sum(fisher_info * (param - optimal_param) ** 2)

        return self.lambda_ewc * ewc_loss / 2.0

    def update_task_performance(self, task_id: int, performance: float):
        """Update performance tracking for a task."""
        if task_id not in self.task_performances:
            self.task_performances[task_id] = []
        self.task_performances[task_id].append(performance)

    def get_forgetting_measure(self, task_id: int) -> float:
        """Compute forgetting measure for a task."""
        if (
            task_id not in self.task_performances
            or len(self.task_performances[task_id]) < 2
        ):
            return 0.0

        performances = self.task_performances[task_id]
        initial_performance = performances[0]
        final_performance = performances[-1]

        return max(0.0, initial_performance - final_performance)

    def get_transfer_measure(self, task_id: int) -> float:
        """Compute transfer measure for a task."""
        if (
            task_id not in self.task_performances
            or len(self.task_performances[task_id]) < 2
        ):
            return 0.0

        performances = self.task_performances[task_id]
        initial_performance = performances[0]
        final_performance = performances[-1]

        return max(0.0, final_performance - initial_performance)

    def get_statistics(self) -> Dict[str, Any]:
        """Get EWC statistics."""
        stats = {
            "num_tasks": self.task_count,
            "ewc_losses": self.ewc_losses.copy(),
            "task_performances": {
                k: v.copy() for k, v in self.task_performances.items()
            },
            "forgetting_measures": {},
            "transfer_measures": {},
        }

        for task_id in self.task_performances:
            stats["forgetting_measures"][task_id] = self.get_forgetting_measure(task_id)
            stats["transfer_measures"][task_id] = self.get_transfer_measure(task_id)

        return stats

    def set_lambda_ewc(self, lambda_ewc: float):
        """Set EWC regularization strength."""
        self.lambda_ewc = lambda_ewc

    def reset(self):
        """Reset EWC state."""
        self.optimal_params.clear()
        self.fisher_information.clear()
        self.task_count = 0
        self.ewc_losses.clear()
        self.task_performances.clear()


class EWCNetwork(nn.Module):
    """Neural network with EWC support."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # EWC component
        self.ewc = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)

    def setup_ewc(self, lambda_ewc: float = 1000.0, device: str = "cpu"):
        """Setup EWC for this network."""
        self.ewc = ElasticWeightConsolidation(self, lambda_ewc, device)

    def train_task(
        self, dataloader, task_id: int, num_epochs: int = 10, lr: float = 1e-3
    ) -> List[float]:
        """Train network on a specific task with EWC."""
        if self.ewc is None:
            raise ValueError("EWC not setup. Call setup_ewc() first.")

        optimizer = optim.Adam(self.parameters(), lr=lr)
        losses = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for states, actions, rewards in dataloader:
                states = states.to(self.ewc.device)
                actions = actions.to(self.ewc.device)
                rewards = rewards.to(self.ewc.device)

                # Forward pass
                outputs = self(states)

                # Compute task loss
                if actions.dtype == torch.long:
                    task_loss = F.cross_entropy(outputs, actions)
                else:
                    task_loss = F.mse_loss(outputs, actions.float())

                # Compute EWC loss
                ewc_loss = self.ewc.compute_ewc_loss(task_id)

                # Total loss
                total_loss = task_loss + ewc_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            self.ewc.ewc_losses.append(avg_loss)

        return losses

    def compute_fisher_information(
        self, dataloader, task_id: int, num_samples: int = 1000
    ):
        """Compute Fisher information for current task."""
        if self.ewc is None:
            raise ValueError("EWC not setup. Call setup_ewc() first.")

        return self.ewc.compute_fisher_information(dataloader, task_id, num_samples)

    def save_optimal_params(self, task_id: int):
        """Save optimal parameters for current task."""
        if self.ewc is None:
            raise ValueError("EWC not setup. Call setup_ewc() first.")

        self.ewc.save_optimal_params(task_id)

    def get_ewc_statistics(self) -> Dict[str, Any]:
        """Get EWC statistics."""
        if self.ewc is None:
            return {}

        return self.ewc.get_statistics()
