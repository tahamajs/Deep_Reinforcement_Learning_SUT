"""
Meta Learning for Continual Learning

This module implements meta-learning approaches for continual learning,
including Model-Agnostic Meta-Learning (MAML) adapted for sequential tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import numpy as np
from collections import defaultdict
import copy


class MAMLAgent(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) agent for continual learning.

    MAML learns a good initialization that can be quickly adapted to new tasks
    with few gradient steps.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        adaptation_steps: int = 5,
    ):
        super().__init__()

        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps

        # Meta optimizer
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)

        # Task-specific adapted models
        self.adapted_models: Dict[int, nn.Module] = {}

        # Meta-learning statistics
        self.task_adaptation_history: Dict[int, List[Dict[str, float]]] = {}

    def adapt_to_task(
        self,
        task_id: int,
        support_data: torch.utils.data.DataLoader,
        adaptation_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt the model to a new task using inner-loop adaptation.

        Args:
            task_id: Task identifier
            support_data: Support set for adaptation
            adaptation_steps: Number of adaptation steps (uses default if None)

        Returns:
            Adapted model for the task
        """
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps

        # Start with current meta-model parameters
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()

        # Create inner optimizer
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        adaptation_history = []

        for step in range(adaptation_steps):
            # Sample batch from support data
            try:
                batch = next(iter(support_data))
                inputs, targets = batch
            except StopIteration:
                # Reset iterator if exhausted
                support_iter = iter(support_data)
                batch = next(support_iter)
                inputs, targets = batch

            # Inner adaptation step
            inner_optimizer.zero_grad()

            outputs = adapted_model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            inner_optimizer.step()

            adaptation_history.append({"step": step + 1, "loss": loss.item()})

        # Store adapted model
        self.adapted_models[task_id] = adapted_model
        self.task_adaptation_history[task_id] = adaptation_history

        return adapted_model

    def meta_update(self, task_losses: Dict[int, torch.Tensor]):
        """
        Perform meta-update using losses from adapted models.

        Args:
            task_losses: Dictionary mapping task IDs to their losses
        """
        self.meta_optimizer.zero_grad()

        # Compute meta-loss as sum of task losses
        meta_loss = sum(task_losses.values())

        # Backward pass through meta-model
        meta_loss.backward()
        self.meta_optimizer.step()

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass using adapted model if available, otherwise base model.

        Args:
            x: Input tensor
            task_id: Task identifier (uses base model if None)

        Returns:
            Model output
        """
        if task_id is not None and task_id in self.adapted_models:
            return self.adapted_models[task_id](x)
        else:
            return self.model(x)

    def get_adaptation_stats(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get adaptation statistics for a task."""
        if task_id not in self.task_adaptation_history:
            return None

        history = self.task_adaptation_history[task_id]

        return {
            "task_id": task_id,
            "adaptation_steps": len(history),
            "initial_loss": history[0]["loss"] if history else None,
            "final_loss": history[-1]["loss"] if history else None,
            "loss_improvement": (
                (history[0]["loss"] - history[-1]["loss"]) if history else 0.0
            ),
            "loss_trajectory": [h["loss"] for h in history],
        }


class MetaLearner:
    """
    Meta-learning trainer for continual learning scenarios.

    Manages the meta-learning process across multiple tasks with proper
    train/validation splits for meta-training.
    """

    def __init__(
        self,
        base_model: nn.Module,
        device: str = "cpu",
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        adaptation_steps: int = 5,
    ):
        self.device = device
        self.maml_agent = MAMLAgent(base_model, inner_lr, meta_lr, adaptation_steps)
        self.maml_agent.to(device)

        # Task data management
        self.task_data: Dict[int, Dict[str, torch.utils.data.DataLoader]] = {}

        # Meta-training history
        self.meta_training_history: List[Dict[str, Any]] = []

    def add_task(
        self,
        task_id: int,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        """
        Add a new task to the meta-learner.

        Args:
            task_id: Unique task identifier
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            test_loader: Test data loader (optional)
        """
        self.task_data[task_id] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

    def meta_train_epoch(
        self, tasks_per_batch: int = 4, adaptation_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform one epoch of meta-training.

        Args:
            tasks_per_batch: Number of tasks to sample per meta-batch
            adaptation_steps: Adaptation steps per task

        Returns:
            Meta-training metrics
        """
        # Sample tasks for this meta-batch
        available_tasks = list(self.task_data.keys())
        if len(available_tasks) < tasks_per_batch:
            tasks_per_batch = len(available_tasks)

        sampled_tasks = np.random.choice(
            available_tasks, tasks_per_batch, replace=False
        )

        task_losses = {}
        total_meta_loss = 0.0

        for task_id in sampled_tasks:
            # Adapt to task
            adapted_model = self.maml_agent.adapt_to_task(
                task_id, self.task_data[task_id]["train"], adaptation_steps
            )

            # Evaluate on validation set
            if self.task_data[task_id]["val"] is not None:
                val_loss = self._compute_loss(
                    adapted_model, self.task_data[task_id]["val"]
                )
                task_losses[task_id] = val_loss
                total_meta_loss += val_loss.item()

        # Meta-update
        self.maml_agent.meta_update(task_losses)

        epoch_metrics = {
            "meta_loss": total_meta_loss / len(sampled_tasks),
            "tasks_sampled": len(sampled_tasks),
            "task_losses": {k: v.item() for k, v in task_losses.items()},
        }

        self.meta_training_history.append(epoch_metrics)

        return epoch_metrics

    def _compute_loss(
        self, model: nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> torch.Tensor:
        """Compute loss for a model on a dataset."""
        model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size

        return torch.tensor(total_loss / num_samples)

    def continual_learn(
        self, new_task_id: int, epochs: int = 10, adaptation_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Continual learning: Adapt to new task while preserving knowledge.

        Args:
            new_task_id: ID of the new task
            epochs: Number of meta-training epochs
            adaptation_steps: Adaptation steps for new task

        Returns:
            Continual learning results
        """
        print(f"Starting continual learning for task {new_task_id}")

        # Adapt to new task
        adapted_model = self.maml_agent.adapt_to_task(
            new_task_id, self.task_data[new_task_id]["train"], adaptation_steps
        )

        # Fine-tune meta-parameters on new task
        for epoch in range(epochs):
            metrics = self.meta_train_epoch(tasks_per_batch=len(self.task_data))
            print(
                f"Meta-training epoch {epoch+1}/{epochs}, Loss: {metrics['meta_loss']:.4f}"
            )

        # Evaluate on all tasks
        all_task_performance = self.evaluate_all_tasks()

        # Get adaptation statistics
        adaptation_stats = self.maml_agent.get_adaptation_stats(new_task_id)

        return {
            "new_task_id": new_task_id,
            "adapted_model": adapted_model,
            "final_performance": all_task_performance,
            "adaptation_stats": adaptation_stats,
            "meta_training_epochs": epochs,
        }

    def evaluate_all_tasks(self) -> Dict[int, Dict[str, float]]:
        """
        Evaluate performance on all tasks.

        Returns:
            Dictionary mapping task IDs to performance metrics
        """
        results = {}

        for task_id, loaders in self.task_data.items():
            task_results = {}

            # Use adapted model if available, otherwise base model
            model = self.maml_agent.adapted_models.get(task_id) or self.maml_agent.model

            # Evaluate on all available splits
            for split_name, loader in loaders.items():
                if loader is not None:
                    loss = self._compute_loss(model, loader)
                    accuracy = self._compute_accuracy(model, loader)

                    task_results[f"{split_name}_loss"] = loss.item()
                    task_results[f"{split_name}_accuracy"] = accuracy

            results[task_id] = task_results

        return results

    def _compute_accuracy(
        self, model: nn.Module, dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Compute accuracy for a model on a dataset."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        return correct / total

    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        if not self.meta_training_history:
            return {}

        losses = [h["meta_loss"] for h in self.meta_training_history]

        return {
            "total_meta_epochs": len(self.meta_training_history),
            "average_meta_loss": np.mean(losses),
            "min_meta_loss": np.min(losses),
            "max_meta_loss": np.max(losses),
            "final_meta_loss": losses[-1],
            "meta_loss_trajectory": losses,
            "num_tasks": len(self.task_data),
        }

    def save_checkpoint(self, path: str):
        """Save meta-learner checkpoint."""
        checkpoint = {
            "maml_agent": self.maml_agent.state_dict(),
            "meta_optimizer": self.maml_agent.meta_optimizer.state_dict(),
            "task_data_keys": list(self.task_data.keys()),
            "adapted_models": {
                k: v.state_dict() for k, v in self.maml_agent.adapted_models.items()
            },
            "meta_training_history": self.meta_training_history,
            "task_adaptation_history": self.maml_agent.task_adaptation_history,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load meta-learner checkpoint."""
        checkpoint = torch.load(path)

        self.maml_agent.load_state_dict(checkpoint["maml_agent"])
        self.maml_agent.meta_optimizer.load_state_dict(checkpoint["meta_optimizer"])

        # Restore adapted models
        for task_id, state_dict in checkpoint.get("adapted_models", {}).items():
            adapted_model = copy.deepcopy(self.maml_agent.model)
            adapted_model.load_state_dict(state_dict)
            self.maml_agent.adapted_models[task_id] = adapted_model

        self.meta_training_history = checkpoint.get("meta_training_history", [])
        self.maml_agent.task_adaptation_history = checkpoint.get(
            "task_adaptation_history", {}
        )
