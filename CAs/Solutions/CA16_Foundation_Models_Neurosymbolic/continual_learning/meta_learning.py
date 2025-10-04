"""
Meta-Learning for Continual Learning

This module implements meta-learning algorithms like MAML and Reptile for continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy


class MAML:
    """Model-Agnostic Meta-Learning (MAML) implementation."""

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        adaptation_steps: int = 5,
        device: str = "cpu",
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.device = device

        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

        # Training history
        self.meta_losses = []
        self.adaptation_losses = []

    def meta_update(
        self,
        support_sets: List[Dict[str, torch.Tensor]],
        query_sets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """Perform meta-update using support and query sets."""
        self.model.train()
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        meta_losses = []
        adaptation_losses = []

        # Process each task
        for support_set, query_set in zip(support_sets, query_sets):
            # Inner loop: adapt to support set
            adapted_params = self._adapt_to_task(support_set)
            
            # Compute adaptation loss
            adaptation_loss = self._compute_loss(support_set, adapted_params)
            adaptation_losses.append(adaptation_loss.item())

            # Outer loop: compute meta-loss on query set
            meta_loss = self._compute_loss(query_set, adapted_params)
            meta_losses.append(meta_loss)

        # Average meta-loss
        avg_meta_loss = torch.stack(meta_losses).mean()

        # Meta-update
        self.meta_optimizer.zero_grad()
        avg_meta_loss.backward()
        self.meta_optimizer.step()

        # Record losses
        self.meta_losses.append(avg_meta_loss.item())
        self.adaptation_losses.extend(adaptation_losses)

        return {
            "meta_loss": avg_meta_loss.item(),
            "avg_adaptation_loss": np.mean(adaptation_losses),
        }

    def _adapt_to_task(self, support_set: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt model to a specific task using support set."""
        # Create temporary optimizer for inner loop
        temp_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Adaptation steps
        for _ in range(self.adaptation_steps):
            # Forward pass
            loss = self._compute_loss(support_set, original_params)
            
            # Compute gradients
            temp_optimizer.zero_grad()
            loss.backward()
            temp_optimizer.step()

        # Return adapted parameters
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data.copy_(original_params[name])
        
        return adapted_params

    def _compute_loss(self, data: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for given data and parameters."""
        # Temporarily set parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for name, param in self.model.named_parameters():
            param.data.copy_(params[name])

        # Forward pass
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        rewards = data["rewards"].to(self.device)

        # Simple policy gradient loss
        action_logits = self.model(states)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        loss = -(log_probs.squeeze() * rewards).mean()

        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data.copy_(original_params[name])

        return loss

    def evaluate(
        self,
        test_tasks: List[Dict[str, torch.Tensor]],
        adaptation_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate MAML on test tasks."""
        self.model.eval()
        
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps

        test_losses = []
        test_rewards = []

        with torch.no_grad():
            for task in test_tasks:
                # Adapt to task
                adapted_params = self._adapt_to_task(task)
                
                # Evaluate on adapted model
                loss = self._compute_loss(task, adapted_params)
                test_losses.append(loss.item())
                
                # Compute reward (simplified)
                reward = -loss.item()
                test_rewards.append(reward)

        return {
            "avg_test_loss": np.mean(test_losses),
            "avg_test_reward": np.mean(test_rewards),
            "std_test_reward": np.std(test_rewards),
        }


class Reptile:
    """Reptile meta-learning algorithm."""

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        adaptation_steps: int = 5,
        device: str = "cpu",
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.device = device

        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

        # Training history
        self.meta_losses = []
        self.adaptation_losses = []

    def meta_update(
        self,
        support_sets: List[Dict[str, torch.Tensor]],
        query_sets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """Perform meta-update using Reptile algorithm."""
        self.model.train()
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        meta_losses = []
        adaptation_losses = []
        parameter_updates = []

        # Process each task
        for support_set, query_set in zip(support_sets, query_sets):
            # Inner loop: adapt to support set
            adapted_params, adaptation_loss = self._adapt_to_task(support_set)
            adaptation_losses.append(adaptation_loss.item())

            # Compute parameter update
            param_update = {}
            for name, param in self.model.named_parameters():
                param_update[name] = adapted_params[name] - original_params[name]
            parameter_updates.append(param_update)

            # Compute meta-loss on query set
            meta_loss = self._compute_loss(query_set, adapted_params)
            meta_losses.append(meta_loss)

        # Average parameter updates
        avg_param_update = {}
        for name in original_params.keys():
            updates = [param_update[name] for param_update in parameter_updates]
            avg_param_update[name] = torch.stack(updates).mean(dim=0)

        # Apply averaged parameter update
        for name, param in self.model.named_parameters():
            param.data.add_(avg_param_update[name], alpha=self.meta_lr)

        # Record losses
        avg_meta_loss = torch.stack(meta_losses).mean()
        self.meta_losses.append(avg_meta_loss.item())
        self.adaptation_losses.extend(adaptation_losses)

        return {
            "meta_loss": avg_meta_loss.item(),
            "avg_adaptation_loss": np.mean(adaptation_losses),
        }

    def _adapt_to_task(self, support_set: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Adapt model to a specific task using support set."""
        # Create temporary optimizer for inner loop
        temp_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Adaptation steps
        total_loss = 0.0
        for _ in range(self.adaptation_steps):
            # Forward pass
            loss = self._compute_loss(support_set, original_params)
            total_loss += loss.item()
            
            # Compute gradients
            temp_optimizer.zero_grad()
            loss.backward()
            temp_optimizer.step()

        # Return adapted parameters
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data.copy_(original_params[name])
        
        return adapted_params, torch.tensor(total_loss / self.adaptation_steps)

    def _compute_loss(self, data: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for given data and parameters."""
        # Temporarily set parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for name, param in self.model.named_parameters():
            param.data.copy_(params[name])

        # Forward pass
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        rewards = data["rewards"].to(self.device)

        # Simple policy gradient loss
        action_logits = self.model(states)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        loss = -(log_probs.squeeze() * rewards).mean()

        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data.copy_(original_params[name])

        return loss

    def evaluate(
        self,
        test_tasks: List[Dict[str, torch.Tensor]],
        adaptation_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate Reptile on test tasks."""
        self.model.eval()
        
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps

        test_losses = []
        test_rewards = []

        with torch.no_grad():
            for task in test_tasks:
                # Adapt to task
                adapted_params, _ = self._adapt_to_task(task)
                
                # Evaluate on adapted model
                loss = self._compute_loss(task, adapted_params)
                test_losses.append(loss.item())
                
                # Compute reward (simplified)
                reward = -loss.item()
                test_rewards.append(reward)

        return {
            "avg_test_loss": np.mean(test_losses),
            "avg_test_reward": np.mean(test_rewards),
            "std_test_reward": np.std(test_rewards),
        }


class MetaLearningTrainer:
    """Trainer for meta-learning algorithms."""

    def __init__(
        self,
        meta_algorithm,
        num_meta_tasks: int = 5,
        num_adaptation_tasks: int = 3,
        batch_size: int = 32,
    ):
        self.meta_algorithm = meta_algorithm
        self.num_meta_tasks = num_meta_tasks
        self.num_adaptation_tasks = num_adaptation_tasks
        self.batch_size = batch_size

        # Training history
        self.training_history = {
            "meta_losses": [],
            "adaptation_losses": [],
            "test_performances": [],
        }

    def train_epoch(
        self,
        task_generator,
        num_epochs: int = 1,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_meta_losses = []
        epoch_adaptation_losses = []

        for _ in range(num_epochs):
            # Generate meta-tasks
            meta_tasks = []
            for _ in range(self.num_meta_tasks):
                task = task_generator.generate_task()
                meta_tasks.append(task)

            # Split into support and query sets
            support_sets = []
            query_sets = []
            
            for task in meta_tasks:
                # Split task data
                split_idx = len(task["states"]) // 2
                support_set = {
                    "states": task["states"][:split_idx],
                    "actions": task["actions"][:split_idx],
                    "rewards": task["rewards"][:split_idx],
                }
                query_set = {
                    "states": task["states"][split_idx:],
                    "actions": task["actions"][split_idx:],
                    "rewards": task["rewards"][split_idx:],
                }
                
                support_sets.append(support_set)
                query_sets.append(query_set)

            # Meta-update
            meta_loss_info = self.meta_algorithm.meta_update(support_sets, query_sets)
            
            epoch_meta_losses.append(meta_loss_info["meta_loss"])
            epoch_adaptation_losses.append(meta_loss_info["avg_adaptation_loss"])

        # Record training history
        avg_meta_loss = np.mean(epoch_meta_losses)
        avg_adaptation_loss = np.mean(epoch_adaptation_losses)
        
        self.training_history["meta_losses"].append(avg_meta_loss)
        self.training_history["adaptation_losses"].append(avg_adaptation_loss)

        return {
            "avg_meta_loss": avg_meta_loss,
            "avg_adaptation_loss": avg_adaptation_loss,
        }

    def evaluate(
        self,
        task_generator,
        num_test_tasks: int = 10,
    ) -> Dict[str, float]:
        """Evaluate meta-learning performance."""
        # Generate test tasks
        test_tasks = []
        for _ in range(num_test_tasks):
            task = task_generator.generate_task()
            test_tasks.append(task)

        # Evaluate
        eval_results = self.meta_algorithm.evaluate(test_tasks)
        
        # Record test performance
        self.training_history["test_performances"].append(eval_results["avg_test_reward"])

        return eval_results

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_epochs": len(self.training_history["meta_losses"]),
            "avg_meta_loss": np.mean(self.training_history["meta_losses"]) if self.training_history["meta_losses"] else 0.0,
            "avg_adaptation_loss": np.mean(self.training_history["adaptation_losses"]) if self.training_history["adaptation_losses"] else 0.0,
            "avg_test_performance": np.mean(self.training_history["test_performances"]) if self.training_history["test_performances"] else 0.0,
            "latest_meta_loss": self.training_history["meta_losses"][-1] if self.training_history["meta_losses"] else 0.0,
            "latest_test_performance": self.training_history["test_performances"][-1] if self.training_history["test_performances"] else 0.0,
        }