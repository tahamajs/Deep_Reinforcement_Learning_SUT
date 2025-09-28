"""
Progressive Networks for Continual Learning

This module implements Progressive Networks, which add new neural network columns
for each new task while keeping previous columns frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from collections import OrderedDict


class ProgressiveColumn(nn.Module):
    """
    A single column in a progressive network.

    Each column specializes in one task and can leverage knowledge
    from previous columns through lateral connections.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        prev_columns: List["ProgressiveColumn"] = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.prev_columns = prev_columns or []

        # Main network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                ]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Lateral connections from previous columns
        self.lateral_weights = nn.ModuleList()
        if self.prev_columns:
            # Learnable weights for combining outputs from previous columns
            for prev_col in self.prev_columns:
                self.lateral_weights.append(nn.Linear(prev_col.output_size, prev_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the column."""
        # Main network forward
        output = self.network(x)

        # Add lateral connections from previous columns
        if self.prev_columns and self.lateral_weights:
            lateral_output = 0
            for i, prev_col in enumerate(self.prev_columns):
                # Get output from previous column (assuming same input)
                prev_output = prev_col(x)
                # Apply lateral weight
                lateral_contrib = self.lateral_weights[i](prev_output)
                lateral_output += lateral_contrib

            # Combine main output with lateral contributions
            output = output + lateral_output

        return output


class ProgressiveNetwork(nn.Module):
    """
    Progressive Network that grows by adding new columns for each task.

    The network maintains separate columns for each task while allowing
    knowledge transfer through lateral connections.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_tasks: int = 10,
        shared_representation: bool = True,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_tasks = num_tasks
        self.shared_representation = shared_representation

        # Shared representation network (optional)
        if shared_representation:
            shared_layers = []
            prev_size = input_size
            for hidden_size in hidden_sizes[:-1]:  # All but last hidden layer
                shared_layers.extend(
                    [
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_size),
                    ]
                )
                prev_size = hidden_size

            self.shared_network = nn.Sequential(*shared_layers)
            column_input_size = prev_size
        else:
            self.shared_network = None
            column_input_size = input_size

        # Task-specific columns
        self.columns: nn.ModuleList = nn.ModuleList()

        # Task output sizes (to be set when adding tasks)
        self.task_output_sizes: Dict[int, int] = {}

        # Current number of tasks
        self.current_tasks = 0

    def add_task(self, task_id: int, output_size: int):
        """
        Add a new task column to the network.

        Args:
            task_id: Unique identifier for the task
            output_size: Output dimension for this task
        """
        if task_id in self.task_output_sizes:
            raise ValueError(f"Task {task_id} already exists")

        # Determine input size for new column
        if self.shared_representation:
            column_input_size = (
                self.hidden_sizes[-2] if len(self.hidden_sizes) > 1 else self.input_size
            )
        else:
            column_input_size = self.input_size

        # Create previous columns list (excluding current task)
        prev_columns = list(self.columns) if self.current_tasks > 0 else []

        # Create new column
        new_column = ProgressiveColumn(
            input_size=column_input_size,
            hidden_sizes=(
                [self.hidden_sizes[-1]]
                if self.shared_representation
                else self.hidden_sizes
            ),
            output_size=output_size,
            prev_columns=prev_columns,
        )

        self.columns.append(new_column)
        self.task_output_sizes[task_id] = output_size
        self.current_tasks += 1

        return new_column

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass for a specific task.

        Args:
            x: Input tensor
            task_id: Task identifier

        Returns:
            Output for the specified task
        """
        if task_id not in self.task_output_sizes:
            raise ValueError(
                f"Task {task_id} not found. Available tasks: {list(self.task_output_sizes.keys())}"
            )

        # Get column index for this task
        task_idx = list(self.task_output_sizes.keys()).index(task_id)

        # Shared representation
        if self.shared_representation:
            shared_repr = self.shared_network(x)
        else:
            shared_repr = x

        # Forward through appropriate column
        return self.columns[task_idx](shared_repr)

    def get_task_columns(self) -> Dict[int, ProgressiveColumn]:
        """Get mapping of task IDs to their columns."""
        return dict(zip(self.task_output_sizes.keys(), self.columns))

    def freeze_previous_tasks(self, current_task_id: int):
        """
        Freeze all columns except the current task's column.

        Args:
            current_task_id: The task that should remain trainable
        """
        for task_id, column in zip(self.task_output_sizes.keys(), self.columns):
            if task_id != current_task_id:
                for param in column.parameters():
                    param.requires_grad = False
            else:
                for param in column.parameters():
                    param.requires_grad = True

    def get_trainable_parameters(self, task_id: int) -> List[torch.Tensor]:
        """
        Get trainable parameters for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            List of trainable parameters
        """
        task_idx = list(self.task_output_sizes.keys()).index(task_id)
        column = self.columns[task_idx]

        trainable_params = []
        for param in column.parameters():
            if param.requires_grad:
                trainable_params.append(param)

        return trainable_params

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown."""
        total_params = sum(p.numel() for p in self.parameters())

        shared_params = 0
        if self.shared_representation:
            shared_params = sum(p.numel() for p in self.shared_network.parameters())

        column_params = 0
        for column in self.columns:
            column_params += sum(p.numel() for p in column.parameters())

        return {
            "total": total_params,
            "shared": shared_params,
            "columns": column_params,
            "num_columns": len(self.columns),
        }


class ProgressiveNetworkTrainer:
    """
    Trainer for Progressive Networks.

    Manages the training process for progressive networks across multiple tasks.
    """

    def __init__(self, network: ProgressiveNetwork, device: str = "cpu"):
        self.network = network
        self.device = device
        self.task_optimizers: Dict[int, torch.optim.Optimizer] = {}
        self.task_schedulers: Dict[int, Any] = {}

        # Training history
        self.training_history: Dict[int, List[Dict[str, float]]] = {}

    def add_task_optimizer(
        self,
        task_id: int,
        optimizer_class: type = torch.optim.Adam,
        lr: float = 1e-3,
        **optimizer_kwargs,
    ):
        """
        Add optimizer for a specific task.

        Args:
            task_id: Task identifier
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            lr: Learning rate
            **optimizer_kwargs: Additional optimizer arguments
        """
        if task_id not in self.network.task_output_sizes:
            raise ValueError(f"Task {task_id} not found in network")

        # Get parameters for this task
        task_params = self.network.get_trainable_parameters(task_id)

        # Create optimizer
        optimizer = optimizer_class(task_params, lr=lr, **optimizer_kwargs)
        self.task_optimizers[task_id] = optimizer

        # Optional scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        self.task_schedulers[task_id] = scheduler

    def train_task(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        criterion: callable = F.cross_entropy,
    ) -> Dict[str, Any]:
        """
        Train the network on a specific task.

        Args:
            task_id: Task identifier
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            criterion: Loss function

        Returns:
            Training metrics
        """
        if task_id not in self.task_optimizers:
            raise ValueError(f"No optimizer configured for task {task_id}")

        optimizer = self.task_optimizers[task_id]
        scheduler = self.task_schedulers.get(task_id)

        # Freeze other tasks
        self.network.freeze_previous_tasks(task_id)

        self.network.train()
        history = []

        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.network(inputs, task_id)
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # Validation
            val_acc = 0.0
            if val_loader is not None:
                val_acc = self._evaluate_task(task_id, val_loader)

            # Step scheduler
            if scheduler:
                scheduler.step()

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }

            history.append(epoch_metrics)

            print(
                f"Task {task_id}, Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

        self.training_history[task_id] = history

        return {
            "task_id": task_id,
            "final_train_loss": history[-1]["train_loss"],
            "final_train_accuracy": history[-1]["train_accuracy"],
            "final_val_accuracy": history[-1]["val_accuracy"],
            "epochs": epochs,
        }

    def _evaluate_task(
        self, task_id: int, dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Evaluate accuracy for a specific task."""
        self.network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.network(inputs, task_id)
                _, predicted = torch.max(outputs.data, 1)

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return correct / total

    def evaluate_all_tasks(
        self, task_loaders: Dict[int, torch.utils.data.DataLoader]
    ) -> Dict[int, float]:
        """
        Evaluate performance on all tasks.

        Args:
            task_loaders: Dictionary mapping task IDs to their data loaders

        Returns:
            Dictionary mapping task IDs to accuracy scores
        """
        results = {}

        for task_id, loader in task_loaders.items():
            if task_id in self.network.task_output_sizes:
                accuracy = self._evaluate_task(task_id, loader)
                results[task_id] = accuracy

        return results

    def get_network_growth_stats(self) -> Dict[str, Any]:
        """Get statistics about network growth."""
        param_counts = self.network.get_parameter_count()

        return {
            "total_parameters": param_counts["total"],
            "shared_parameters": param_counts["shared"],
            "column_parameters": param_counts["columns"],
            "num_tasks": param_counts["num_columns"],
            "parameters_per_task": param_counts["columns"]
            / max(1, param_counts["num_columns"]),
            "shared_ratio": param_counts["shared"] / max(1, param_counts["total"]),
        }
