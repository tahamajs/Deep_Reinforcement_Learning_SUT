"""
Elastic Weight Consolidation (EWC) for Continual Learning

This module implements Elastic Weight Consolidation, a regularization-based approach
for continual learning that prevents catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from collections import defaultdict


class ElasticWeightConsolidation(nn.Module):
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.

    EWC constrains parameter updates to stay close to their values at the end
    of previous tasks, weighted by the importance of each parameter.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        super().__init__()
        self.model = model
        self.device = device

        # Fisher Information Matrix for each task
        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}

        # Parameter values at the end of each task
        self.task_parameters: Dict[int, Dict[str, torch.Tensor]] = {}

        # Current task ID
        self.current_task = 0

    def start_task(self, task_id: int):
        """Start a new task."""
        self.current_task = task_id

    def end_task(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: Callable = F.cross_entropy,
    ):
        """
        End current task and compute Fisher Information Matrix.

        Args:
            dataloader: DataLoader for the current task's data
            criterion: Loss function for computing Fisher information
        """
        # Store current parameters
        self.task_parameters[self.current_task] = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Compute Fisher Information Matrix
        fisher_info = self._compute_fisher_information(dataloader, criterion)
        self.fisher_matrices[self.current_task] = fisher_info

    def _compute_fisher_information(
        self, dataloader: torch.utils.data.DataLoader, criterion: Callable
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix for current task.

        The Fisher Information Matrix measures the importance of each parameter
        for the current task.
        """
        fisher_info = {}
        self.model.eval()

        # Initialize Fisher information for each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)

        # Accumulate Fisher information over samples
        num_samples = 0

        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Compute gradients
            loss.backward()

            # Accumulate squared gradients (Fisher information)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data**2

            num_samples += len(inputs)

        # Average over samples
        for name in fisher_info:
            fisher_info[name] /= num_samples

        return fisher_info

    def ewc_loss(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """
        Compute EWC regularization loss.

        Args:
            lambda_ewc: Regularization strength

        Returns:
            EWC regularization loss
        """
        loss = 0.0

        for task_id in range(self.current_task):
            if task_id in self.fisher_matrices and task_id in self.task_parameters:
                fisher = self.fisher_matrices[task_id]
                params_old = self.task_parameters[task_id]

                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in fisher and name in params_old:
                        # EWC penalty: (1/2) * λ * F_i * (θ - θ*_i)^2
                        param_diff = param - params_old[name]
                        loss += torch.sum(fisher[name] * param_diff**2)

        return (lambda_ewc / 2.0) * loss

    def get_parameter_importance(self) -> Dict[str, torch.Tensor]:
        """
        Get accumulated parameter importance across all tasks.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        importance = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = torch.zeros_like(param)

        # Sum importance across all tasks
        for task_id in self.fisher_matrices:
            fisher = self.fisher_matrices[task_id]
            for name in fisher:
                if name in importance:
                    importance[name] += fisher[name]

        return importance

    def consolidate(
        self,
        task_id: int,
        dataloader: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        epochs: int = 1,
        lambda_ewc: float = 1000.0,
        criterion: Callable = F.cross_entropy,
    ):
        """
        Consolidate knowledge for current task using EWC.

        Args:
            task_id: Current task ID
            dataloader: DataLoader for current task
            optimizer: Optimizer for consolidation
            epochs: Number of consolidation epochs
            lambda_ewc: EWC regularization strength
            criterion: Loss function
        """
        self.start_task(task_id)
        self.model.train()

        for epoch in range(epochs):
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Add EWC regularization
                if self.current_task > 0:
                    ewc_penalty = self.ewc_loss(lambda_ewc)
                    loss += ewc_penalty

                # Backward pass
                loss.backward()
                optimizer.step()

        # End task and compute Fisher information
        self.end_task(dataloader, criterion)


class EWCTrainer:
    """
    Trainer for Elastic Weight Consolidation.

    Provides a complete training pipeline for continual learning with EWC.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str = "cpu",
        lambda_ewc: float = 1000.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_ewc = lambda_ewc

        self.ewc = ElasticWeightConsolidation(model, device)
        self.task_performance = defaultdict(list)

    def train_task(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        criterion: Callable = F.cross_entropy,
    ) -> Dict[str, float]:
        """
        Train on a new task with EWC regularization.

        Args:
            task_id: Task identifier
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            criterion: Loss function

        Returns:
            Training metrics
        """
        self.ewc.start_task(task_id)
        self.model.train()

        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Training loop
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Add EWC regularization for subsequent tasks
                if task_id > 0:
                    ewc_penalty = self.ewc.ewc_loss(self.lambda_ewc)
                    loss += ewc_penalty

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)

            # Validation
            if val_loader is not None:
                val_acc = self._evaluate(val_loader)
                val_accuracies.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            print(
                f"Task {task_id}, Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
                if val_loader
                else ""
            )

        # Consolidate task knowledge
        self.ewc.end_task(train_loader, criterion)

        # Store performance
        final_metrics = {
            "task_id": task_id,
            "final_train_loss": train_losses[-1],
            "best_val_accuracy": best_val_acc if val_loader else 0.0,
            "epochs_trained": epochs,
        }

        self.task_performance[task_id].append(final_metrics)

        return final_metrics

    def _evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate model accuracy."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
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
            accuracy = self._evaluate(loader)
            results[task_id] = accuracy

        return results

    def get_forgetting_measure(
        self, task_loaders: Dict[int, torch.utils.data.DataLoader]
    ) -> Dict[str, float]:
        """
        Compute forgetting measures across tasks.

        Returns:
            Dictionary with forgetting metrics
        """
        # Evaluate current performance on all tasks
        current_performance = self.evaluate_all_tasks(task_loaders)

        # Get best performance for each task (from training history)
        best_performance = {}
        for task_id in current_performance.keys():
            if task_id in self.task_performance:
                # Best validation accuracy for this task
                best_acc = max(
                    [
                        metrics["best_val_accuracy"]
                        for metrics in self.task_performance[task_id]
                    ]
                )
                best_performance[task_id] = best_acc

        # Compute forgetting
        forgetting = {}
        for task_id in current_performance.keys():
            if task_id in best_performance:
                forgetting[task_id] = (
                    best_performance[task_id] - current_performance[task_id]
                )

        # Average forgetting
        avg_forgetting = np.mean(list(forgetting.values())) if forgetting else 0.0

        return {
            "average_forgetting": avg_forgetting,
            "task_forgetting": forgetting,
            "current_performance": current_performance,
            "best_performance": best_performance,
        }
