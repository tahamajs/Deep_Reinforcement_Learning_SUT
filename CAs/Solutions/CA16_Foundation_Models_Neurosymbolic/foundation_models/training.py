"""
Training utilities for Foundation Models

This module contains training classes and utilities for foundation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .algorithms import (
    DecisionTransformer,
    MultiTaskDecisionTransformer,
    InContextLearner,
)


class FoundationModelTrainer:
    """Trainer for foundation models."""

    def __init__(
        self,
        model: DecisionTransformer,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.model.to(device)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

        self.training_history = {"losses": [], "rewards": [], "episodes": []}

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> float:
        """Single training step."""
        self.model.train()

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns_to_go = returns_to_go.to(self.device)
        timesteps = timesteps.to(self.device)

        # Forward pass
        predicted_actions = self.model(states, actions, returns_to_go, timesteps)

        # Compute loss (MSE for continuous actions, CrossEntropy for discrete)
        if actions.dtype == torch.long:
            loss = nn.CrossEntropyLoss()(
                predicted_actions.view(-1, predicted_actions.size(-1)), actions.view(-1)
            )
        else:
            loss = nn.MSELoss()(predicted_actions, actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_episode(self, episode_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train on a single episode."""
        states = episode_data["states"]
        actions = episode_data["actions"]
        returns_to_go = episode_data["returns_to_go"]
        timesteps = episode_data["timesteps"]

        loss = self.train_step(states, actions, returns_to_go, timesteps)

        return {"loss": loss}

    def evaluate(self, test_data: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.eval()
        total_loss = 0.0
        num_episodes = len(test_data)

        with torch.no_grad():
            for episode_data in test_data:
                states = episode_data["states"].to(self.device)
                actions = episode_data["actions"].to(self.device)
                returns_to_go = episode_data["returns_to_go"].to(self.device)
                timesteps = episode_data["timesteps"].to(self.device)

                predicted_actions = self.model(
                    states, actions, returns_to_go, timesteps
                )

                if actions.dtype == torch.long:
                    loss = nn.CrossEntropyLoss()(
                        predicted_actions.view(-1, predicted_actions.size(-1)),
                        actions.view(-1),
                    )
                else:
                    loss = nn.MSELoss()(predicted_actions, actions)

                total_loss += loss.item()

        return {"avg_loss": total_loss / num_episodes}


class MultiTaskTrainer:
    """Trainer for multi-task foundation models."""

    def __init__(
        self,
        model: MultiTaskDecisionTransformer,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.model.to(device)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

        self.task_losses = {}
        self.task_performances = {}

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> float:
        """Single training step for multi-task learning."""
        self.model.train()

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns_to_go = returns_to_go.to(self.device)
        timesteps = timesteps.to(self.device)
        task_ids = task_ids.to(self.device)

        # Forward pass
        predicted_actions = self.model(
            states, actions, returns_to_go, timesteps, task_ids
        )

        # Compute loss
        if actions.dtype == torch.long:
            loss = nn.CrossEntropyLoss()(
                predicted_actions.view(-1, predicted_actions.size(-1)), actions.view(-1)
            )
        else:
            loss = nn.MSELoss()(predicted_actions, actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train_episode(self, episode_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train on a single episode with task information."""
        states = episode_data["states"]
        actions = episode_data["actions"]
        returns_to_go = episode_data["returns_to_go"]
        timesteps = episode_data["timesteps"]
        task_ids = episode_data["task_ids"]

        loss = self.train_step(states, actions, returns_to_go, timesteps, task_ids)

        # Track task-specific losses
        task_id = task_ids[0].item()
        if task_id not in self.task_losses:
            self.task_losses[task_id] = []
        self.task_losses[task_id].append(loss)

        return {"loss": loss, "task_id": task_id}

    def evaluate_task(
        self, test_data: List[Dict[str, torch.Tensor]], task_id: int
    ) -> Dict[str, float]:
        """Evaluate model on specific task."""
        self.model.eval()
        total_loss = 0.0
        num_episodes = 0

        with torch.no_grad():
            for episode_data in test_data:
                if episode_data["task_ids"][0].item() != task_id:
                    continue

                states = episode_data["states"].to(self.device)
                actions = episode_data["actions"].to(self.device)
                returns_to_go = episode_data["returns_to_go"].to(self.device)
                timesteps = episode_data["timesteps"].to(self.device)
                task_ids = episode_data["task_ids"].to(self.device)

                predicted_actions = self.model(
                    states, actions, returns_to_go, timesteps, task_ids
                )

                if actions.dtype == torch.long:
                    loss = nn.CrossEntropyLoss()(
                        predicted_actions.view(-1, predicted_actions.size(-1)),
                        actions.view(-1),
                    )
                else:
                    loss = nn.MSELoss()(predicted_actions, actions)

                total_loss += loss.item()
                num_episodes += 1

        if num_episodes == 0:
            return {"avg_loss": float("inf")}

        return {"avg_loss": total_loss / num_episodes}


class InContextTrainer:
    """Trainer for in-context learning."""

    def __init__(
        self,
        model: InContextLearner,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.model.to(device)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

        self.few_shot_performance = {}

    def train_step(
        self,
        context_states: torch.Tensor,
        context_actions: torch.Tensor,
        context_returns: torch.Tensor,
        query_states: torch.Tensor,
        query_actions: torch.Tensor,
    ) -> float:
        """Single training step for in-context learning."""
        self.model.train()

        # Move to device
        context_states = context_states.to(self.device)
        context_actions = context_actions.to(self.device)
        context_returns = context_returns.to(self.device)
        query_states = query_states.to(self.device)
        query_actions = query_actions.to(self.device)

        # Forward pass
        predicted_actions = self.model(
            context_states, context_actions, context_returns, query_states
        )

        # Compute loss
        if query_actions.dtype == torch.long:
            loss = nn.CrossEntropyLoss()(
                predicted_actions.view(-1, predicted_actions.size(-1)),
                query_actions.view(-1),
            )
        else:
            loss = nn.MSELoss()(predicted_actions, query_actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def evaluate_few_shot(
        self,
        context_data: Dict[str, torch.Tensor],
        query_data: Dict[str, torch.Tensor],
        num_shots: int,
    ) -> Dict[str, float]:
        """Evaluate few-shot performance."""
        self.model.eval()

        with torch.no_grad():
            # Use first num_shots examples as context
            context_states = context_data["states"][:num_shots].to(self.device)
            context_actions = context_data["actions"][:num_shots].to(self.device)
            context_returns = context_data["returns"][:num_shots].to(self.device)

            # Use remaining examples as queries
            query_states = query_data["states"].to(self.device)
            query_actions = query_data["actions"].to(self.device)

            predicted_actions = self.model(
                context_states, context_actions, context_returns, query_states
            )

            # Compute accuracy
            if query_actions.dtype == torch.long:
                predicted_classes = torch.argmax(predicted_actions, dim=-1)
                accuracy = (predicted_classes == query_actions).float().mean().item()
            else:
                mse = nn.MSELoss()(predicted_actions, query_actions).item()
                accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like metric

            return {"accuracy": accuracy, "num_shots": num_shots}
