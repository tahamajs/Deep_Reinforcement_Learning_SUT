"""
Foundation Models in Reinforcement Learning

This module implements foundation model approaches for RL including:
- Decision Transformers for sequence modeling
- Trajectory Transformers for trajectory generation
- Multi-task RL foundation models
- In-context learning for RL
- Foundation model training utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for offline RL
    
    Models the conditional probability P(a_t | s_{1:t}, a_{1:t-1}, R_{t:T})
    where R_{t:T} represents the desired return-to-go from time t to episode end T.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        context_length: int = 1024,
        dropout: float = 0.1,
        max_timestep: int = 1000,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.context_length = context_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Embeddings
        self.state_embedding = nn.Linear(state_dim, model_dim)
        self.action_embedding = nn.Linear(action_dim, model_dim)
        self.return_embedding = nn.Linear(1, model_dim)
        self.timestep_embedding = nn.Embedding(max_timestep, model_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(model_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.action_head = nn.Linear(model_dim, action_dim)
        self.value_head = nn.Linear(model_dim, 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(model_dim)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Decision Transformer
        
        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len, action_dim)
            returns_to_go: (batch_size, seq_len)
            timesteps: (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing action predictions and value estimates
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embeddings
        state_emb = self.state_embedding(states)  # (batch_size, seq_len, model_dim)
        action_emb = self.action_embedding(actions)  # (batch_size, seq_len, model_dim)
        return_emb = self.return_embedding(returns_to_go.unsqueeze(-1))  # (batch_size, seq_len, model_dim)
        timestep_emb = self.timestep_embedding(timesteps)  # (batch_size, seq_len, model_dim)
        
        # Stack embeddings: [state, action, return] for each timestep
        stacked_inputs = torch.stack((state_emb, action_emb, return_emb), dim=2)
        stacked_inputs = stacked_inputs.permute(0, 1, 2, 3).reshape(
            batch_size, 3 * seq_len, self.model_dim
        )
        
        # Add timestep embeddings
        timestep_emb_expanded = timestep_emb.repeat_interleave(3, dim=1)
        stacked_inputs = stacked_inputs + timestep_emb_expanded
        
        # Add positional encoding
        stacked_inputs = self.pos_encoding(stacked_inputs.transpose(0, 1)).transpose(0, 1)
        
        # Layer normalization
        stacked_inputs = self.layer_norm(stacked_inputs)
        
        # Transformer
        transformer_outputs = self.transformer(stacked_inputs, src_key_padding_mask=attention_mask)
        
        # Extract action predictions (actions are at positions 1, 4, 7, ...)
        action_outputs = transformer_outputs[:, 1::3]
        actions_pred = self.action_head(action_outputs)
        
        # Extract value estimates (returns are at positions 2, 5, 8, ...)
        value_outputs = transformer_outputs[:, 2::3]
        values_pred = self.value_head(value_outputs)
        
        return {
            'actions': actions_pred,
            'values': values_pred,
            'hidden_states': transformer_outputs
        }
    
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Get action prediction for given sequence"""
        with torch.no_grad():
            outputs = self.forward(states, actions, returns_to_go, timesteps)
            action_logits = outputs['actions'][:, -1] / temperature
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1)
            return action


class TrajectoryTransformer(nn.Module):
    """
    Trajectory Transformer for modeling entire trajectories
    
    Models P(τ | g) = ∏_{t=0}^T P(s_{t+1}, r_t, a_t | s_{1:t}, a_{1:t-1}, g)
    where g represents the goal or task specification.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
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
        self.reward_embedding = nn.Linear(1, model_dim)
        self.goal_embedding = nn.Linear(goal_dim, model_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(model_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.state_head = nn.Linear(model_dim, state_dim)
        self.action_head = nn.Linear(model_dim, action_dim)
        self.reward_head = nn.Linear(model_dim, 1)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        goals: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Trajectory Transformer
        
        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len, action_dim)
            rewards: (batch_size, seq_len)
            goals: (batch_size, goal_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing predictions for next states, actions, and rewards
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Embeddings
        state_emb = self.state_embedding(states)
        action_emb = self.action_embedding(actions)
        reward_emb = self.reward_embedding(rewards.unsqueeze(-1))
        goal_emb = self.goal_embedding(goals).unsqueeze(1).repeat(1, seq_len, 1)
        
        # Stack embeddings: [state, action, reward] for each timestep
        stacked_inputs = torch.stack((state_emb, action_emb, reward_emb), dim=2)
        stacked_inputs = stacked_inputs.permute(0, 1, 2, 3).reshape(
            batch_size, 3 * seq_len, self.model_dim
        )
        
        # Add goal embeddings
        goal_emb_expanded = goal_emb.repeat_interleave(3, dim=1)
        stacked_inputs = stacked_inputs + goal_emb_expanded
        
        # Add positional encoding
        stacked_inputs = self.pos_encoding(stacked_inputs.transpose(0, 1)).transpose(0, 1)
        
        # Transformer
        transformer_outputs = self.transformer(stacked_inputs, src_key_padding_mask=attention_mask)
        
        # Extract predictions
        state_outputs = transformer_outputs[:, 0::3]
        action_outputs = transformer_outputs[:, 1::3]
        reward_outputs = transformer_outputs[:, 2::3]
        
        next_states_pred = self.state_head(state_outputs)
        actions_pred = self.action_head(action_outputs)
        rewards_pred = self.reward_head(reward_outputs)
        
        return {
            'next_states': next_states_pred,
            'actions': actions_pred,
            'rewards': rewards_pred,
            'hidden_states': transformer_outputs
        }


class MultiTaskRLFoundationModel(nn.Module):
    """
    Multi-task RL Foundation Model
    
    A foundation model trained on multiple RL tasks that can be adapted
    to new tasks through fine-tuning or in-context learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        num_tasks: int = 10,
        context_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_tasks = num_tasks
        
        # Shared backbone
        self.backbone = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            context_length=context_length,
            dropout=dropout,
        )
        
        # Task-specific adapters
        self.task_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim // 4),
                nn.ReLU(),
                nn.Linear(model_dim // 4, model_dim),
            ) for _ in range(num_tasks)
        ])
        
        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, model_dim)
        
        # Output heads
        self.action_head = nn.Linear(model_dim, action_dim)
        self.value_head = nn.Linear(model_dim, 1)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task-specific adaptation
        
        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len, action_dim)
            returns_to_go: (batch_size, seq_len)
            timesteps: (batch_size, seq_len)
            task_ids: (batch_size,) - task identifiers
            
        Returns:
            Dictionary containing task-adapted predictions
        """
        # Get backbone outputs
        backbone_outputs = self.backbone(states, actions, returns_to_go, timesteps)
        hidden_states = backbone_outputs['hidden_states']
        
        # Apply task-specific adapters
        adapted_states = []
        for i, task_id in enumerate(task_ids):
            adapter = self.task_adapters[task_id.item()]
            task_emb = self.task_embedding(task_id)
            adapted_state = hidden_states[i] + adapter(hidden_states[i]) + task_emb.unsqueeze(0)
            adapted_states.append(adapted_state)
        
        adapted_states = torch.stack(adapted_states)
        
        # Extract action predictions
        action_outputs = adapted_states[:, 1::3]
        actions_pred = self.action_head(action_outputs)
        
        # Extract value estimates
        value_outputs = adapted_states[:, 2::3]
        values_pred = self.value_head(value_outputs)
        
        return {
            'actions': actions_pred,
            'values': values_pred,
            'hidden_states': adapted_states,
            'task_embeddings': self.task_embedding(task_ids)
        }


class InContextLearningRL(nn.Module):
    """
    In-Context Learning for RL
    
    Learns to adapt to new tasks through in-context examples without
    requiring gradient updates.
    """
    
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
        
        # Base transformer
        self.transformer = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            context_length=context_length,
            dropout=dropout,
        )
        
        # In-context learning components
        self.example_encoder = nn.Linear(state_dim + action_dim + 1, model_dim)  # state + action + reward
        self.task_encoder = nn.Linear(model_dim, model_dim)
        
    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        examples: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with in-context learning
        
        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len, action_dim)
            returns_to_go: (batch_size, seq_len)
            timesteps: (batch_size, seq_len)
            examples: Optional in-context examples
            
        Returns:
            Dictionary containing predictions
        """
        if examples is not None:
            # Encode in-context examples
            example_states = examples['states']  # (batch_size, num_examples, state_dim)
            example_actions = examples['actions']  # (batch_size, num_examples, action_dim)
            example_rewards = examples['rewards']  # (batch_size, num_examples)
            
            # Combine example information
            example_inputs = torch.cat([
                example_states,
                example_actions,
                example_rewards.unsqueeze(-1)
            ], dim=-1)
            
            # Encode examples
            example_embeddings = self.example_encoder(example_inputs)
            
            # Create task representation from examples
            task_representation = torch.mean(example_embeddings, dim=1)  # (batch_size, model_dim)
            task_representation = self.task_encoder(task_representation)
            
            # Add task representation to states
            task_representation = task_representation.unsqueeze(1).repeat(1, states.shape[1], 1)
            states = states + task_representation
        
        # Forward through base transformer
        return self.transformer(states, actions, returns_to_go, timesteps)


class FoundationModelTrainer:
    """
    Trainer for foundation models in RL
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps)
        )
        
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Batch of training data
            loss_weights: Optional weights for different loss components
            
        Returns:
            Dictionary containing loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            states=batch['states'],
            actions=batch['actions'],
            returns_to_go=batch['returns_to_go'],
            timesteps=batch['timesteps'],
        )
        
        # Compute losses
        losses = self._compute_losses(batch, outputs, loss_weights)
        
        # Backward pass
        total_loss = sum(losses.values())
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1
        
        return {k: v.item() for k, v in losses.items()}
    
    def _compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses"""
        if loss_weights is None:
            loss_weights = {'action': 1.0, 'value': 0.1}
        
        losses = {}
        
        # Action prediction loss
        action_loss = F.mse_loss(outputs['actions'], batch['actions'])
        losses['action'] = loss_weights['action'] * action_loss
        
        # Value prediction loss
        if 'values' in outputs and 'values' in batch:
            value_loss = F.mse_loss(outputs['values'], batch['values'])
            losses['value'] = loss_weights['value'] * value_loss
        
        return losses
    
    def evaluate(
        self,
        dataloader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on validation set
        
        Args:
            dataloader: Validation dataloader
            num_batches: Number of batches to evaluate (None for all)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_losses = {}
        num_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    states=batch['states'],
                    actions=batch['actions'],
                    returns_to_go=batch['returns_to_go'],
                    timesteps=batch['timesteps'],
                )
                
                # Compute losses
                losses = self._compute_losses(batch, outputs)
                
                # Accumulate losses
                batch_size = batch['states'].shape[0]
                for key, loss in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += loss.item() * batch_size
                
                num_samples += batch_size
        
        # Average losses
        avg_losses = {k: v / num_samples for k, v in total_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step_count': self.step_count,
            'metadata': metadata or {},
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step_count = checkpoint['step_count']
        return checkpoint.get('metadata', {})
