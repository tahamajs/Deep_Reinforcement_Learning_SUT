"""
Foundation Models Algorithms

This module contains implementations of foundation model approaches for RL:
- Decision Transformers for sequence-based RL
- Multi-task foundation models
- In-context learning
- Foundation model training framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based RL models."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class DecisionTransformer(nn.Module):
    """Decision Transformer for sequence-based RL."""

    def __init__(
        self,
        state_dim,
        action_dim,
        model_dim=512,
        num_heads=8,
        num_layers=6,
        max_length=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_dim = model_dim
        self.max_length = max_length

        # Embedding layers for states, actions, and returns-to-go
        self.state_embedding = nn.Linear(state_dim, model_dim)
        self.action_embedding = nn.Linear(action_dim, model_dim)
        self.return_embedding = nn.Linear(1, model_dim)
        self.timestep_embedding = nn.Embedding(max_length, model_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            model_dim, max_length * 3
        )  # 3x for s,a,r tokens

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(model_dim)

        # Output heads
        self.action_head = nn.Linear(model_dim, action_dim)
        self.value_head = nn.Linear(model_dim, 1)
        self.return_head = nn.Linear(model_dim, 1)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize transformer weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        """
        Forward pass through Decision Transformer.

        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len, action_dim)
            returns_to_go: (batch_size, seq_len, 1)
            timesteps: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len * 3)
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embed inputs
        state_embeddings = self.state_embedding(states)
        action_embeddings = self.action_embedding(actions)
        return_embeddings = self.return_embedding(returns_to_go)
        time_embeddings = self.timestep_embedding(timesteps)

        # Add time embeddings
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        return_embeddings += time_embeddings

        # Stack embeddings: [return, state, action] * seq_len
        # Shape: (batch_size, seq_len * 3, model_dim)
        stacked_inputs = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings], dim=2
        ).reshape(batch_size, 3 * seq_len, self.model_dim)

        # Apply positional encoding
        stacked_inputs = self.pos_encoding(stacked_inputs.transpose(0, 1)).transpose(
            0, 1
        )
        stacked_inputs = self.layer_norm(stacked_inputs)
        stacked_inputs = self.dropout(stacked_inputs)

        # Apply transformer
        transformer_output = self.transformer(
            stacked_inputs, src_key_padding_mask=attention_mask
        )

        # Reshape back to (batch_size, seq_len, 3, model_dim)
        transformer_output = transformer_output.reshape(
            batch_size, seq_len, 3, self.model_dim
        )

        # Extract outputs for each token type
        return_preds = self.return_head(transformer_output[:, :, 0])  # Return tokens
        state_preds = transformer_output[:, :, 1]  # State tokens (for representation)
        action_preds = self.action_head(transformer_output[:, :, 2])  # Action tokens
        value_preds = self.value_head(
            transformer_output[:, :, 1]
        )  # Value from state tokens

        return {
            "action_preds": action_preds,
            "value_preds": value_preds,
            "return_preds": return_preds,
            "state_representations": state_preds,
        }

    def get_action(self, states, actions, returns_to_go, timesteps, temperature=1.0):
        """Get action for inference."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(states, actions, returns_to_go, timesteps)
            action_logits = outputs["action_preds"][:, -1] / temperature

            # For discrete actions
            if self.action_dim > 1:
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1)
            else:
                action = torch.tanh(action_logits)  # For continuous actions

            return action


class MultiTaskRLFoundationModel(nn.Module):
    """Multi-task foundation model for RL."""

    def __init__(
        self, state_dim, action_dim, task_dim, model_dim=512, num_heads=8, num_layers=6
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.task_dim = task_dim
        self.model_dim = model_dim

        # Task-specific embeddings
        self.task_embedding = nn.Embedding(task_dim, model_dim)

        # Core Decision Transformer
        self.decision_transformer = DecisionTransformer(
            state_dim, action_dim, model_dim, num_heads, num_layers
        )

        # Task-specific output heads
        self.task_heads = nn.ModuleDict(
            {f"task_{i}": nn.Linear(model_dim, action_dim) for i in range(task_dim)}
        )

        # Meta-learning components
        self.context_encoder = nn.LSTM(model_dim, model_dim, batch_first=True)
        self.adaptation_network = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim)
        )

    def forward(
        self, states, actions, returns_to_go, timesteps, task_ids, context_length=10
    ):
        """Forward pass with task conditioning."""
        batch_size = states.shape[0]

        # Get task embeddings
        task_embeds = self.task_embedding(task_ids)  # (batch_size, model_dim)
        task_embeds = task_embeds.unsqueeze(1).expand(-1, states.shape[1], -1)

        # Modify states with task conditioning
        conditioned_states = states + task_embeds[:, :, : self.state_dim]

        # Forward through decision transformer
        outputs = self.decision_transformer(
            conditioned_states, actions, returns_to_go, timesteps
        )

        # Task-specific action prediction
        state_representations = outputs["state_representations"]
        task_specific_actions = []

        for i, task_id in enumerate(task_ids):
            task_head = self.task_heads[f"task_{task_id.item()}"]
            task_action = task_head(state_representations[i])
            task_specific_actions.append(task_action)

        outputs["task_specific_actions"] = torch.stack(task_specific_actions)

        return outputs

    def adapt_to_new_task(self, context_trajectories, num_adaptation_steps=5):
        """Few-shot adaptation to new task using in-context learning."""
        # Encode context trajectories
        context_features = []

        for trajectory in context_trajectories:
            # Extract features from demonstration trajectory
            states, actions, returns = (
                trajectory["states"],
                trajectory["actions"],
                trajectory["returns"],
            )
            timesteps = torch.arange(len(states))

            with torch.no_grad():
                outputs = self.decision_transformer(states, actions, returns, timesteps)
                context_features.append(outputs["state_representations"].mean(dim=1))

        # Aggregate context
        context_features = torch.stack(context_features)
        context_encoding, _ = self.context_encoder(context_features.unsqueeze(0))

        # Compute adaptation parameters
        adaptation_params = self.adaptation_network(
            context_encoding.squeeze(0).mean(dim=0)
        )

        return adaptation_params


class InContextLearningRL:
    """In-context learning for RL foundation models."""

    def __init__(self, foundation_model, context_length=50):
        self.foundation_model = foundation_model
        self.context_length = context_length
        self.context_buffer = deque(maxlen=context_length)

    def add_context(self, state, action, reward, next_state, done):
        """Add experience to context buffer."""
        self.context_buffer.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
        )

    def get_action(self, current_state, desired_return, temperature=1.0):
        """Get action using in-context learning."""
        if len(self.context_buffer) == 0:
            # Random action if no context
            return np.random.randint(self.foundation_model.action_dim)

        # Prepare context sequence
        context_states = []
        context_actions = []
        context_returns = []
        context_timesteps = []

        # Build context from buffer
        cumulative_return = 0
        for i, exp in enumerate(reversed(list(self.context_buffer))):
            context_states.append(exp["state"])
            context_actions.append(exp["action"])
            cumulative_return += exp["reward"]
            context_returns.append([cumulative_return])
            context_timesteps.append(len(self.context_buffer) - i - 1)

        # Reverse to get chronological order
        context_states.reverse()
        context_actions.reverse()
        context_returns.reverse()
        context_timesteps.reverse()

        # Add current state
        context_states.append(current_state)
        context_actions.append(
            np.zeros(self.foundation_model.action_dim)
        )  # Placeholder
        context_returns.append([desired_return])
        context_timesteps.append(len(self.context_buffer))

        # Convert to tensors
        states = torch.FloatTensor(context_states).unsqueeze(0).to(device)
        actions = torch.FloatTensor(context_actions).unsqueeze(0).to(device)
        returns_to_go = torch.FloatTensor(context_returns).unsqueeze(0).to(device)
        timesteps = torch.LongTensor(context_timesteps).unsqueeze(0).to(device)

        # Get action from foundation model
        with torch.no_grad():
            action = self.foundation_model.get_action(
                states, actions, returns_to_go, timesteps, temperature
            )

        return action.cpu().numpy().flatten()


class FoundationModelTrainer:
    """Training framework for RL foundation models."""

    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-2):
        self.model = model
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

        self.training_stats = {
            "losses": [],
            "action_losses": [],
            "value_losses": [],
            "return_losses": [],
        }

    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Unpack batch
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        returns_to_go = batch["returns_to_go"].to(device)
        timesteps = batch["timesteps"].to(device)
        target_actions = batch["target_actions"].to(device)
        target_returns = batch["target_returns"].to(device)

        # Forward pass
        outputs = self.model(states, actions, returns_to_go, timesteps)

        # Compute losses
        action_loss = F.mse_loss(outputs["action_preds"], target_actions)
        value_loss = F.mse_loss(outputs["value_preds"], target_returns)
        return_loss = F.mse_loss(outputs["return_preds"], target_returns)

        # Combined loss
        total_loss = action_loss + 0.5 * value_loss + 0.1 * return_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update statistics
        self.training_stats["losses"].append(total_loss.item())
        self.training_stats["action_losses"].append(action_loss.item())
        self.training_stats["value_losses"].append(value_loss.item())
        self.training_stats["return_losses"].append(return_loss.item())

        return total_loss.item()
