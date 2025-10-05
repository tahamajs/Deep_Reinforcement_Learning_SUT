"""
Progressive Neural Networks for Continual Learning

This module implements progressive neural networks that grow with new tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy


class ProgressiveNetwork(nn.Module):
    """Progressive Neural Network that grows with new tasks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        num_columns: int = 3,
        lateral_connections: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_columns = num_columns
        self.lateral_connections = lateral_connections

        # Create columns
        self.columns = nn.ModuleList()
        self.lateral_layers = nn.ModuleList()

        # Initialize first column
        self._add_column()

    def _add_column(self):
        """Add a new column to the network."""
        column_id = len(self.columns)

        # Create main column
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        column = nn.Sequential(*layers)
        self.columns.append(column)

        # Create lateral connections if enabled
        if self.lateral_connections and column_id > 0:
            lateral_layer = nn.ModuleList()
            for prev_column_id in range(column_id):
                # Lateral connection from previous column
                lateral_dim = (
                    self.hidden_dims[-1] if self.hidden_dims else self.output_dim
                )
                lateral_conn = nn.Linear(lateral_dim, lateral_dim)
                lateral_layer.append(lateral_conn)
            self.lateral_layers.append(lateral_layer)

    def forward(self, x: torch.Tensor, column_id: int = 0) -> torch.Tensor:
        """Forward pass through specified column."""
        if column_id >= len(self.columns):
            raise ValueError(f"Column {column_id} does not exist")

        # Forward through main column
        output = self.columns[column_id](x)

        # Add lateral connections if available
        if self.lateral_connections and column_id < len(self.lateral_layers):
            lateral_outputs = []
            for prev_column_id, lateral_conn in enumerate(
                self.lateral_layers[column_id]
            ):
                # Get output from previous column
                prev_output = self.columns[prev_column_id](x)
                # Apply lateral connection
                lateral_output = lateral_conn(prev_output)
                lateral_outputs.append(lateral_output)

            # Combine lateral outputs
            if lateral_outputs:
                combined_lateral = torch.stack(lateral_outputs, dim=0).sum(dim=0)
                output = output + combined_lateral

        return output

    def add_column(self):
        """Add a new column for a new task."""
        self._add_column()
        self.num_columns += 1

    def get_column_output(self, x: torch.Tensor, column_id: int) -> torch.Tensor:
        """Get output from a specific column."""
        return self.forward(x, column_id)

    def get_all_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get outputs from all columns."""
        outputs = []
        for column_id in range(len(self.columns)):
            output = self.forward(x, column_id)
            outputs.append(output)
        return outputs

    def get_column_statistics(self, column_id: int) -> Dict[str, Any]:
        """Get statistics for a specific column."""
        if column_id >= len(self.columns):
            return {"error": f"Column {column_id} does not exist"}

        column = self.columns[column_id]
        total_params = sum(p.numel() for p in column.parameters())

        stats = {
            "column_id": column_id,
            "total_parameters": total_params,
            "layers": len(column),
        }

        # Add lateral connection statistics
        if self.lateral_connections and column_id < len(self.lateral_layers):
            lateral_params = sum(
                p.numel()
                for lateral_layer in self.lateral_layers[column_id]
                for p in lateral_layer.parameters()
            )
            stats["lateral_parameters"] = lateral_params
            stats["lateral_connections"] = len(self.lateral_layers[column_id])

        return stats

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get overall network statistics."""
        stats = {
            "total_columns": len(self.columns),
            "lateral_connections": self.lateral_connections,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
        }

        # Column-wise statistics
        column_stats = []
        total_params = 0

        for column_id in range(len(self.columns)):
            col_stats = self.get_column_statistics(column_id)
            column_stats.append(col_stats)
            total_params += col_stats["total_parameters"]

            if "lateral_parameters" in col_stats:
                total_params += col_stats["lateral_parameters"]

        stats["column_statistics"] = column_stats
        stats["total_parameters"] = total_params

        return stats


class DynamicProgressiveNetwork(nn.Module):
    """Dynamic Progressive Network with adaptive column selection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        max_columns: int = 10,
        lateral_connections: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.max_columns = max_columns
        self.lateral_connections = lateral_connections

        # Create columns
        self.columns = nn.ModuleList()
        self.lateral_layers = nn.ModuleList()
        self.column_weights = nn.Parameter(torch.ones(1))  # Will be expanded

        # Task assignment
        self.task_to_column = {}
        self.column_to_task = {}

        # Initialize first column
        self._add_column()

    def _add_column(self):
        """Add a new column to the network."""
        column_id = len(self.columns)

        # Create main column
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.output_dim))
        column = nn.Sequential(*layers)
        self.columns.append(column)

        # Create lateral connections if enabled
        if self.lateral_connections and column_id > 0:
            lateral_layer = nn.ModuleList()
            for prev_column_id in range(column_id):
                lateral_dim = (
                    self.hidden_dims[-1] if self.hidden_dims else self.output_dim
                )
                lateral_conn = nn.Linear(lateral_dim, lateral_dim)
                lateral_layer.append(lateral_conn)
            self.lateral_layers.append(lateral_layer)

        # Update column weights
        new_weights = torch.ones(len(self.columns))
        if hasattr(self, "column_weights"):
            new_weights[: len(self.columns) - 1] = self.column_weights.data
        self.column_weights = nn.Parameter(new_weights)

    def add_task_column(self, task_id: int) -> int:
        """Add a new column for a specific task."""
        if len(self.columns) >= self.max_columns:
            # Reuse existing column
            column_id = len(self.columns) - 1
        else:
            self._add_column()
            column_id = len(self.columns) - 1

        # Assign task to column
        self.task_to_column[task_id] = column_id
        self.column_to_task[column_id] = task_id

        return column_id

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[int] = None,
        column_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass through the network."""
        if column_id is None:
            if task_id is None:
                # Use all columns with weights
                return self._forward_weighted(x)
            else:
                column_id = self.task_to_column.get(task_id, 0)

        if column_id >= len(self.columns):
            raise ValueError(f"Column {column_id} does not exist")

        # Forward through specified column
        output = self.columns[column_id](x)

        # Add lateral connections if available
        if self.lateral_connections and column_id < len(self.lateral_layers):
            lateral_outputs = []
            for prev_column_id, lateral_conn in enumerate(
                self.lateral_layers[column_id]
            ):
                prev_output = self.columns[prev_column_id](x)
                lateral_output = lateral_conn(prev_output)
                lateral_outputs.append(lateral_output)

            if lateral_outputs:
                combined_lateral = torch.stack(lateral_outputs, dim=0).sum(dim=0)
                output = output + combined_lateral

        return output

    def _forward_weighted(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using weighted combination of all columns."""
        outputs = []
        weights = F.softmax(self.column_weights, dim=0)

        for column_id in range(len(self.columns)):
            output = self.forward(x, column_id=column_id)
            outputs.append(output)

        # Weighted combination
        weighted_output = torch.stack(outputs, dim=0)
        weighted_output = (weighted_output * weights.view(-1, 1, 1)).sum(dim=0)

        return weighted_output

    def get_task_column(self, task_id: int) -> int:
        """Get column assigned to a task."""
        return self.task_to_column.get(task_id, 0)

    def reassign_task(self, task_id: int, new_column_id: int):
        """Reassign a task to a different column."""
        if new_column_id >= len(self.columns):
            raise ValueError(f"Column {new_column_id} does not exist")

        old_column_id = self.task_to_column.get(task_id)
        if old_column_id is not None:
            del self.column_to_task[old_column_id]

        self.task_to_column[task_id] = new_column_id
        self.column_to_task[new_column_id] = task_id

    def get_column_usage(self) -> Dict[int, List[int]]:
        """Get which tasks are using which columns."""
        usage = {}
        for column_id in range(len(self.columns)):
            usage[column_id] = []
            for task_id, assigned_column in self.task_to_column.items():
                if assigned_column == column_id:
                    usage[column_id].append(task_id)
        return usage

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get overall network statistics."""
        stats = {
            "total_columns": len(self.columns),
            "max_columns": self.max_columns,
            "assigned_tasks": len(self.task_to_column),
            "lateral_connections": self.lateral_connections,
        }

        # Column usage statistics
        column_usage = self.get_column_usage()
        stats["column_usage"] = column_usage

        # Weight statistics
        if hasattr(self, "column_weights"):
            weights = F.softmax(self.column_weights, dim=0)
            stats["column_weights"] = weights.detach().cpu().numpy().tolist()
            stats["dominant_column"] = torch.argmax(weights).item()

        return stats


class ProgressiveNetworkTrainer:
    """Trainer for progressive neural networks."""

    def __init__(
        self,
        network: ProgressiveNetwork,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.network = network
        self.device = device
        self.network.to(device)

        # Optimizers for each column
        self.optimizers = {}
        self._create_optimizers(lr)

        # Training history
        self.training_history = {
            "column_losses": {},
            "column_performances": {},
        }

    def _create_optimizers(self, lr: float):
        """Create optimizers for all columns."""
        for column_id in range(len(self.network.columns)):
            optimizer = torch.optim.Adam(
                self.network.columns[column_id].parameters(), lr=lr
            )
            self.optimizers[column_id] = optimizer

    def train_column(
        self,
        column_id: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """Train a specific column."""
        if column_id >= len(self.network.columns):
            raise ValueError(f"Column {column_id} does not exist")

        # Forward pass
        action_logits = self.network.forward(states, column_id=column_id)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))

        # Compute loss
        loss = -(log_probs.squeeze() * rewards).mean()

        # Backward pass
        optimizer = self.optimizers[column_id]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.network.columns[column_id].parameters(), max_norm=1.0
        )
        optimizer.step()

        # Record training history
        if column_id not in self.training_history["column_losses"]:
            self.training_history["column_losses"][column_id] = []
            self.training_history["column_performances"][column_id] = []

        self.training_history["column_losses"][column_id].append(loss.item())
        self.training_history["column_performances"][column_id].append(
            rewards.mean().item()
        )

        return {
            "loss": loss.item(),
            "avg_reward": rewards.mean().item(),
            "column_id": column_id,
        }

    def add_new_column(self, lr: float = 1e-3):
        """Add a new column and its optimizer."""
        old_num_columns = len(self.network.columns)
        self.network.add_column()

        # Create optimizer for new column
        new_column_id = len(self.network.columns) - 1
        optimizer = torch.optim.Adam(
            self.network.columns[new_column_id].parameters(), lr=lr
        )
        self.optimizers[new_column_id] = optimizer

        return new_column_id

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            "total_columns": len(self.network.columns),
            "column_losses": self.training_history["column_losses"],
            "column_performances": self.training_history["column_performances"],
        }

        # Compute average losses and performances
        avg_losses = {}
        avg_performances = {}

        for column_id in self.training_history["column_losses"]:
            losses = self.training_history["column_losses"][column_id]
            performances = self.training_history["column_performances"][column_id]

            avg_losses[column_id] = np.mean(losses) if losses else 0.0
            avg_performances[column_id] = np.mean(performances) if performances else 0.0

        stats["avg_losses"] = avg_losses
        stats["avg_performances"] = avg_performances

        return stats
