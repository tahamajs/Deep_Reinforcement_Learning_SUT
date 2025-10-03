"""
Dynamic Architectures for Continual Learning

This module implements dynamic neural architectures that can grow and adapt
to new tasks while preserving knowledge from previous tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class DynamicNetwork(nn.Module):
    """
    Dynamic neural network that can add/remove neurons and layers.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()

        # Build initial layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        if hidden_dims:  # Only add output layer if there are hidden layers
            self.layers.append(nn.Linear(prev_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dynamic network."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation except for last layer
                x = F.relu(x)
        return x


class TaskSpecificHead(nn.Module):
    """
    Task-specific output head for multi-task learning.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through task head."""
        return self.head(x)


class KnowledgeDistillation(nn.Module):
    """
    Knowledge distillation for transferring knowledge between networks.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both teacher and student."""
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)

        student_logits = self.student_model(x)

        return teacher_logits, student_logits

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ground_truth: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""

        # Soft targets from teacher
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)

        # Soft predictions from student
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)

        # Distillation loss
        distillation_loss = F.kl_div(
            soft_predictions, soft_targets, reduction="batchmean"
        )

        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, ground_truth)

        # Combined loss
        total_loss = (
            alpha * distillation_loss * (self.temperature**2) + (1 - alpha) * hard_loss
        )

        return total_loss
