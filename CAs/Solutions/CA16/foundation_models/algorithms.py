"""
Foundation Model Algorithms

This module implements core foundation model algorithms for reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[: x.size(0), :]


class DecisionTransformer(nn.Module):
    """Decision Transformer for reinforcement learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        context_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.context_length = context_length

        # Embeddings
        self.state_embedding = nn.Linear(state_dim, model_dim)
        self.action_embedding = nn.Linear(action_dim, model_dim)
        self.return_embedding = nn.Linear(1, model_dim)
        self.timestep_embedding = nn.Embedding(1000, model_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(model_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.action_head = nn.Linear(model_dim, action_dim)
        self.value_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of Decision Transformer."""
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embeddings
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        return_emb = self.return_embedding(returns_to_go.unsqueeze(-1))
        timestep_emb = self.timestep_embedding(timesteps)

        # Stack embeddings
        stacked_inputs = torch.stack((state_emb, action_emb, return_emb), dim=1)
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(
            batch_size, 3 * seq_len, self.model_dim
        )

        # Add timestep embeddings
        stacked_inputs = stacked_inputs + timestep_emb.repeat_interleave(3, dim=1)

        # Add positional encoding
        stacked_inputs = self.pos_encoding(stacked_inputs.transpose(0, 1)).transpose(
            0, 1
        )

        # Transformer
        transformer_outputs = self.transformer(stacked_inputs)

        # Extract action predictions
        action_outputs = transformer_outputs[
            :, 1::3
        ]  # Actions are at positions 1, 4, 7, ...
        actions_pred = self.action_head(action_outputs)

        return actions_pred


class MultiTaskDecisionTransformer(nn.Module):
    """Multi-task Decision Transformer for learning across multiple tasks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_tasks: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        context_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tasks = num_tasks

        # Shared transformer backbone
        self.shared_transformer = DecisionTransformer(
            state_dim,
            action_dim,
            model_dim,
            num_heads,
            num_layers,
            context_length,
            dropout,
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList(
            [nn.Linear(model_dim, action_dim) for _ in range(num_tasks)]
        )

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, model_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with task-specific heads."""
        # Get shared representations
        shared_outputs = self.shared_transformer(
            states, actions, returns_to_go, timesteps
        )

        # Apply task-specific heads
        batch_size = states.shape[0]
        task_outputs = []

        for i in range(batch_size):
            task_id = task_ids[i].item()
            task_output = self.task_heads[task_id](shared_outputs[i])
            task_outputs.append(task_output)

        return torch.stack(task_outputs, dim=0)


class InContextLearner(nn.Module):
    """In-context learning for few-shot adaptation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim

        # Embeddings
        self.state_embedding = nn.Linear(state_dim, model_dim)
        self.action_embedding = nn.Linear(action_dim, model_dim)
        self.return_embedding = nn.Linear(1, model_dim)

        # Context encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Query encoder
        self.query_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention for context-query interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Output head
        self.action_head = nn.Linear(model_dim, action_dim)

    def forward(
        self,
        context_states: torch.Tensor,
        context_actions: torch.Tensor,
        context_returns: torch.Tensor,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for in-context learning."""
        # Encode context
        context_state_emb = self.state_embedding(context_states)
        context_action_emb = self.action_embedding(context_actions)
        context_return_emb = self.return_embedding(context_returns.unsqueeze(-1))

        context_inputs = torch.cat(
            [context_state_emb, context_action_emb, context_return_emb], dim=1
        )
        context_features = self.context_encoder(context_inputs)

        # Encode query
        query_state_emb = self.state_embedding(query_states)
        query_features = self.query_encoder(query_state_emb)

        # Cross-attention
        attended_features, _ = self.cross_attention(
            query_features, context_features, context_features
        )

        # Predict actions
        actions_pred = self.action_head(attended_features)

        return actions_pred


class ScalingAnalyzer:
    """Analyzer for scaling laws in foundation models."""

    def __init__(self):
        self.results = {}

    def analyze_scaling(
        self,
        model_sizes: List[int],
        performances: List[float],
        dataset_sizes: List[int] = None,
        compute_budgets: List[float] = None,
    ) -> Dict[str, float]:
        """Analyze scaling relationships."""
        # Model size scaling
        if len(model_sizes) > 1:
            log_sizes = np.log(model_sizes)
            log_perfs = np.log(performances)
            beta = np.polyfit(log_sizes, log_perfs, 1)[0]
            self.results["model_scaling_exponent"] = beta

        # Dataset size scaling
        if dataset_sizes is not None and len(dataset_sizes) > 1:
            log_data = np.log(dataset_sizes)
            log_perfs = np.log(performances)
            gamma = np.polyfit(log_data, log_perfs, 1)[0]
            self.results["data_scaling_exponent"] = gamma

        # Compute scaling
        if compute_budgets is not None and len(compute_budgets) > 1:
            log_compute = np.log(compute_budgets)
            log_perfs = np.log(performances)
            delta = np.polyfit(log_compute, log_perfs, 1)[0]
            self.results["compute_scaling_exponent"] = delta

        return self.results

    def predict_performance(
        self, model_size: int, dataset_size: int = None, compute_budget: float = None
    ) -> float:
        """Predict performance based on scaling laws."""
        if not self.results:
            raise ValueError("Must run analyze_scaling first")

        performance = 1.0

        if "model_scaling_exponent" in self.results:
            performance *= model_size ** self.results["model_scaling_exponent"]

        if dataset_size is not None and "data_scaling_exponent" in self.results:
            performance *= dataset_size ** self.results["data_scaling_exponent"]

        if compute_budget is not None and "compute_scaling_exponent" in self.results:
            performance *= compute_budget ** self.results["compute_scaling_exponent"]

        return performance
