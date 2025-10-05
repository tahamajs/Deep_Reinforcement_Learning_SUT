"""
Elastic Weight Consolidation (EWC) for Continual Learning

This module implements EWC and related algorithms for preventing catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        device: str = "cpu",
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.device = device

        # Store Fisher information and optimal parameters for each task
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_importance = {}

        # EWC statistics
        self.ewc_penalties = []
        self.task_count = 0

    def compute_fisher_information(
        self,
        dataloader,
        task_id: int,
        num_samples: int = 1000,
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher information matrix for a task."""
        self.model.eval()

        # Initialize Fisher information
        fisher_info = {}
        for name, param in self.model.named_parameters():
            fisher_info[name] = torch.zeros_like(param.data)

        # Compute Fisher information
        sample_count = 0
        with torch.no_grad():
            for batch in dataloader:
                if sample_count >= num_samples:
                    break

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                states = batch["states"]
                actions = batch["actions"]
                rewards = batch["rewards"]

                # Compute log-likelihood
                action_logits = self.model(states)
                action_probs = F.softmax(action_logits, dim=-1)
                log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))

                # Compute gradients
                self.model.zero_grad()
                log_probs.sum().backward()

                # Accumulate Fisher information
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_info[name] += param.grad.data**2

                sample_count += len(states)

        # Average Fisher information
        for name in fisher_info:
            fisher_info[name] /= sample_count

        # Store Fisher information and optimal parameters
        self.fisher_information[task_id] = fisher_info
        self.optimal_params[task_id] = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

        self.task_count += 1
        self.model.train()

        return fisher_info

    def compute_ewc_penalty(self, task_id: int) -> torch.Tensor:
        """Compute EWC penalty for a specific task."""
        if task_id not in self.fisher_information:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        fisher_info = self.fisher_information[task_id]
        optimal_params = self.optimal_params[task_id]

        for name, param in self.model.named_parameters():
            if name in fisher_info and name in optimal_params:
                fisher = fisher_info[name]
                optimal = optimal_params[name]
                penalty += (fisher * (param - optimal) ** 2).sum()

        return penalty

    def compute_total_ewc_penalty(self) -> torch.Tensor:
        """Compute total EWC penalty for all previous tasks."""
        total_penalty = torch.tensor(0.0, device=self.device)

        for task_id in self.fisher_information:
            penalty = self.compute_ewc_penalty(task_id)
            total_penalty += penalty

        return total_penalty

    def update_ewc_params(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        task_id: int,
        num_samples: int = 100,
    ):
        """Update EWC parameters for a task."""
        self.model.eval()

        # Initialize Fisher information
        fisher_info = {}
        for name, param in self.model.named_parameters():
            fisher_info[name] = torch.zeros_like(param.data)

        # Compute Fisher information on provided data
        with torch.no_grad():
            for i in range(min(num_samples, len(states))):
                state = states[i : i + 1].to(self.device)
                action = actions[i : i + 1].to(self.device)

                # Compute log-likelihood
                action_logits = self.model(state)
                action_probs = F.softmax(action_logits, dim=-1)
                log_probs = torch.log(action_probs.gather(1, action.unsqueeze(1)))

                # Compute gradients
                self.model.zero_grad()
                log_probs.sum().backward()

                # Accumulate Fisher information
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_info[name] += param.grad.data**2

        # Average Fisher information
        for name in fisher_info:
            fisher_info[name] /= min(num_samples, len(states))

        # Store Fisher information and optimal parameters
        self.fisher_information[task_id] = fisher_info
        self.optimal_params[task_id] = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }

        self.model.train()

    def get_parameter_importance(self, task_id: int) -> Dict[str, torch.Tensor]:
        """Get parameter importance for a specific task."""
        if task_id not in self.fisher_information:
            return {}

        return self.fisher_information[task_id].copy()

    def get_ewc_statistics(self) -> Dict[str, Any]:
        """Get EWC statistics."""
        stats = {
            "lambda_ewc": self.lambda_ewc,
            "task_count": self.task_count,
            "stored_tasks": list(self.fisher_information.keys()),
            "ewc_penalties": self.ewc_penalties,
        }

        # Compute average penalty
        if self.ewc_penalties:
            stats["avg_ewc_penalty"] = np.mean(self.ewc_penalties)
            stats["std_ewc_penalty"] = np.std(self.ewc_penalties)

        return stats

    def clear_ewc_data(self):
        """Clear all EWC data."""
        self.fisher_information.clear()
        self.optimal_params.clear()
        self.task_importance.clear()
        self.ewc_penalties.clear()
        self.task_count = 0


class OnlineEWC:
    """Online Elastic Weight Consolidation for streaming data."""

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        gamma: float = 0.9,
        device: str = "cpu",
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.gamma = gamma  # Decay factor for old tasks
        self.device = device

        # Online Fisher information and optimal parameters
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_weights = {}

    def update_fisher_information(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        task_id: int,
        num_samples: int = 100,
    ):
        """Update Fisher information online."""
        self.model.eval()

        # Initialize Fisher information if not exists
        if task_id not in self.fisher_information:
            self.fisher_information[task_id] = {}
            self.optimal_params[task_id] = {}
            self.task_weights[task_id] = 1.0

            for name, param in self.model.named_parameters():
                self.fisher_information[task_id][name] = torch.zeros_like(param.data)
                self.optimal_params[task_id][name] = param.data.clone()

        # Compute Fisher information on provided data
        with torch.no_grad():
            for i in range(min(num_samples, len(states))):
                state = states[i : i + 1].to(self.device)
                action = actions[i : i + 1].to(self.device)

                # Compute log-likelihood
                action_logits = self.model(state)
                action_probs = F.softmax(action_logits, dim=-1)
                log_probs = torch.log(action_probs.gather(1, action.unsqueeze(1)))

                # Compute gradients
                self.model.zero_grad()
                log_probs.sum().backward()

                # Update Fisher information
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.fisher_information[task_id][name] += param.grad.data**2

        # Average Fisher information
        for name in self.fisher_information[task_id]:
            self.fisher_information[task_id][name] /= min(num_samples, len(states))

        # Update optimal parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[task_id][name] = param.data.clone()

        self.model.train()

    def compute_online_ewc_penalty(self) -> torch.Tensor:
        """Compute online EWC penalty."""
        total_penalty = torch.tensor(0.0, device=self.device)

        for task_id in self.fisher_information:
            weight = self.task_weights[task_id]
            fisher_info = self.fisher_information[task_id]
            optimal_params = self.optimal_params[task_id]

            for name, param in self.model.named_parameters():
                if name in fisher_info and name in optimal_params:
                    fisher = fisher_info[name]
                    optimal = optimal_params[name]
                    penalty = weight * (fisher * (param - optimal) ** 2).sum()
                    total_penalty += penalty

        return total_penalty

    def decay_task_weights(self):
        """Decay weights of old tasks."""
        for task_id in self.task_weights:
            self.task_weights[task_id] *= self.gamma

    def get_task_weights(self) -> Dict[int, float]:
        """Get current task weights."""
        return self.task_weights.copy()


class EWCWrapper:
    """Wrapper for applying EWC to any PyTorch model."""

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        device: str = "cpu",
    ):
        self.model = model
        self.ewc = ElasticWeightConsolidation(model, ewc_lambda, device)
        self.device = device

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        task_id: int,
        optimizer: torch.optim.Optimizer,
        use_ewc: bool = True,
    ) -> Dict[str, float]:
        """Single training step with EWC."""
        # Compute main loss
        action_logits = self.model(states)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        main_loss = -(log_probs.squeeze() * rewards).mean()

        # Compute EWC penalty
        ewc_penalty = torch.tensor(0.0, device=self.device)
        if use_ewc and task_id > 0:
            ewc_penalty = self.ewc.compute_total_ewc_penalty()

        # Total loss
        total_loss = main_loss + self.ewc.lambda_ewc * ewc_penalty

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        return {
            "main_loss": main_loss.item(),
            "ewc_penalty": ewc_penalty.item(),
            "total_loss": total_loss.item(),
        }

    def update_ewc_params(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        task_id: int,
    ):
        """Update EWC parameters for a task."""
        self.ewc.update_ewc_params(states, actions, task_id)

    def get_ewc_statistics(self) -> Dict[str, Any]:
        """Get EWC statistics."""
        return self.ewc.get_ewc_statistics()

    def clear_ewc_data(self):
        """Clear all EWC data."""
        self.ewc.clear_ewc_data()
