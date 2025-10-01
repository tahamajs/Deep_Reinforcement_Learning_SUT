"""
Progressive Networks for Continual Learning

This module implements progressive networks that grow with each new task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict


class ProgressiveColumn(nn.Module):
    """A single column in a progressive network."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Lateral connections (will be added by ProgressiveNetwork)
        self.lateral_connections = nn.ModuleList()
    
    def forward(self, x: torch.Tensor, lateral_inputs: List[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional lateral inputs."""
        if lateral_inputs is None:
            lateral_inputs = []
        
        # Combine input with lateral connections
        if lateral_inputs:
            # Concatenate lateral inputs
            lateral_concat = torch.cat(lateral_inputs, dim=-1)
            x = torch.cat([x, lateral_concat], dim=-1)
        
        return self.network(x)
    
    def add_lateral_connection(self, input_dim: int):
        """Add a lateral connection from another column."""
        lateral_layer = nn.Linear(input_dim, self.input_dim)
        self.lateral_connections.append(lateral_layer)
        return lateral_layer


class LateralConnection(nn.Module):
    """Lateral connection between progressive network columns."""
    
    def __init__(self, input_dim: int, output_dim: int, connection_type: str = 'linear'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.connection_type = connection_type
        
        if connection_type == 'linear':
            self.connection = nn.Linear(input_dim, output_dim)
        elif connection_type == 'attention':
            self.connection = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=4, dropout=0.1, batch_first=True
            )
        else:
            raise ValueError(f"Unknown connection type: {connection_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through lateral connection."""
        if self.connection_type == 'linear':
            return self.connection(x)
        elif self.connection_type == 'attention':
            # For attention, we need to add sequence dimension
            x = x.unsqueeze(1)  # Add sequence dimension
            output, _ = self.connection(x, x, x)
            return output.squeeze(1)  # Remove sequence dimension


class ProgressiveNetwork(nn.Module):
    """Progressive network that grows with each new task."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Columns for each task
        self.columns = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        
        # Task-specific information
        self.task_columns = {}  # Maps task_id to column index
        self.column_output_dims = {}  # Maps column index to output dimension
        
        # Training history
        self.training_history = {
            'task_losses': defaultdict(list),
            'column_performances': defaultdict(list),
            'lateral_connection_weights': defaultdict(list)
        }
    
    def add_task_column(self, task_id: int, output_dim: int = None) -> int:
        """Add a new column for a task."""
        if output_dim is None:
            output_dim = self.output_dim
        
        # Create new column
        column = ProgressiveColumn(self.input_dim, output_dim, self.hidden_dims)
        self.columns.append(column)
        
        # Add lateral connections from previous columns
        if len(self.columns) > 1:
            lateral_connections = []
            for i in range(len(self.columns) - 1):
                prev_output_dim = self.column_output_dims.get(i, self.output_dim)
                lateral_conn = LateralConnection(prev_output_dim, self.input_dim)
                lateral_connections.append(lateral_conn)
            
            self.lateral_connections.append(nn.ModuleList(lateral_connections))
        
        # Update tracking
        column_idx = len(self.columns) - 1
        self.task_columns[task_id] = column_idx
        self.column_output_dims[column_idx] = output_dim
        
        return column_idx

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass for a specific task."""
        if task_id not in self.task_columns:
            raise ValueError(f"Task {task_id} not found. Add task column first.")
        
        column_idx = self.task_columns[task_id]
        column = self.columns[column_idx]
        
        # Get lateral inputs from previous columns
        lateral_inputs = []
        if column_idx > 0 and column_idx - 1 < len(self.lateral_connections):
            lateral_conns = self.lateral_connections[column_idx - 1]
            for i, lateral_conn in enumerate(lateral_conns):
                prev_output = self.columns[i](x, lateral_inputs=[])
                lateral_output = lateral_conn(prev_output)
                lateral_inputs.append(lateral_output)
        
        # Forward pass through target column
        output = column(x, lateral_inputs)
        
        return output
    
    def train_task(self, dataloader, task_id: int, num_epochs: int = 10, 
                   lr: float = 1e-3, freeze_previous: bool = True) -> List[float]:
        """Train network on a specific task."""
        if task_id not in self.task_columns:
            raise ValueError(f"Task {task_id} not found. Add task column first.")
        
        # Setup optimizer
        if freeze_previous:
            # Only train the current task's column and its lateral connections
            current_column_idx = self.task_columns[task_id]
            trainable_params = list(self.columns[current_column_idx].parameters())
            
            if current_column_idx > 0 and current_column_idx - 1 < len(self.lateral_connections):
                trainable_params.extend(list(self.lateral_connections[current_column_idx - 1].parameters()))
            else:
            # Train all parameters
            trainable_params = list(self.parameters())
        
        optimizer = optim.Adam(trainable_params, lr=lr)
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for states, actions, rewards in dataloader:
                states = states.to(next(self.parameters()).device)
                actions = actions.to(next(self.parameters()).device)
                rewards = rewards.to(next(self.parameters()).device)
                
                # Forward pass
                outputs = self(states, task_id)
                
                # Compute loss
                if actions.dtype == torch.long:
                    loss = F.cross_entropy(outputs, actions)
                else:
                    loss = F.mse_loss(outputs, actions.float())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            self.training_history['task_losses'][task_id].append(avg_loss)
        
        return losses
    
    def evaluate_task(self, dataloader, task_id: int) -> Dict[str, float]:
        """Evaluate performance on a specific task."""
        if task_id not in self.task_columns:
            raise ValueError(f"Task {task_id} not found.")
        
        self.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for states, actions, rewards in dataloader:
                states = states.to(next(self.parameters()).device)
                actions = actions.to(next(self.parameters()).device)
                
                outputs = self(states, task_id)
                
                # Compute loss
                if actions.dtype == torch.long:
                    loss = F.cross_entropy(outputs, actions)
                    predictions = torch.argmax(outputs, dim=-1)
                    correct_predictions += (predictions == actions).sum().item()
                    total_predictions += actions.size(0)
                else:
                    loss = F.mse_loss(outputs, actions.float())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Update performance history
        self.training_history['column_performances'][task_id].append(accuracy)

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def get_column_statistics(self, task_id: int) -> Dict[str, Any]:
        """Get statistics for a specific column."""
        if task_id not in self.task_columns:
            return {}
        
        column_idx = self.task_columns[task_id]
        
        stats = {
            'column_index': column_idx,
            'output_dim': self.column_output_dims[column_idx],
            'num_lateral_connections': len(self.lateral_connections[column_idx - 1]) if column_idx > 0 else 0,
            'training_losses': self.training_history['task_losses'][task_id].copy(),
            'performances': self.training_history['column_performances'][task_id].copy()
        }
        
        return stats
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get overall network statistics."""
        stats = {
            'num_columns': len(self.columns),
            'num_tasks': len(self.task_columns),
            'task_columns': self.task_columns.copy(),
            'column_output_dims': self.column_output_dims.copy(),
            'training_history': {
                'task_losses': dict(self.training_history['task_losses']),
                'column_performances': dict(self.training_history['column_performances'])
            }
        }
        
        return stats
    
    def freeze_column(self, task_id: int):
        """Freeze parameters of a specific column."""
        if task_id not in self.task_columns:
            raise ValueError(f"Task {task_id} not found.")
        
        column_idx = self.task_columns[task_id]
        for param in self.columns[column_idx].parameters():
            param.requires_grad = False
    
    def unfreeze_column(self, task_id: int):
        """Unfreeze parameters of a specific column."""
        if task_id not in self.task_columns:
            raise ValueError(f"Task {task_id} not found.")
        
        column_idx = self.task_columns[task_id]
        for param in self.columns[column_idx].parameters():
            param.requires_grad = True