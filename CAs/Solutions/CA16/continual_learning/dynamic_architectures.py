"""
Dynamic Architectures for Continual Learning

This module implements dynamic architectures that adapt to new tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy


class DynamicNetwork(nn.Module):
    """Dynamic network that can grow and adapt to new tasks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_hidden_dim: int = 128,
        max_hidden_dim: int = 512,
        growth_factor: float = 1.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initial_hidden_dim = initial_hidden_dim
        self.max_hidden_dim = max_hidden_dim
        self.growth_factor = growth_factor

        # Initial architecture
        self.hidden_dim = initial_hidden_dim
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, output_dim),
            ]
        )

        # Task-specific components
        self.task_heads = nn.ModuleList()
        self.task_embeddings = nn.Embedding(10, self.hidden_dim)  # Max 10 tasks

        # Growth history
        self.growth_history = []
        self.task_assignments = {}

    def add_task_head(self, task_id: int) -> int:
        """Add a task-specific head."""
        if task_id >= len(self.task_heads):
            # Add new task head
            head = nn.Linear(self.hidden_dim, self.output_dim)
            self.task_heads.append(head)

            # Update task embeddings if needed
            if task_id >= self.task_embeddings.num_embeddings:
                new_embeddings = nn.Embedding(task_id + 1, self.hidden_dim)
                new_embeddings.weight.data[: self.task_embeddings.num_embeddings] = (
                    self.task_embeddings.weight.data
                )
                self.task_embeddings = new_embeddings

        return len(self.task_heads) - 1

    def grow_network(self):
        """Grow the network by increasing hidden dimension."""
        old_hidden_dim = self.hidden_dim
        new_hidden_dim = min(
            int(old_hidden_dim * self.growth_factor), self.max_hidden_dim
        )

        if new_hidden_dim == old_hidden_dim:
            return  # No growth possible

        # Create new layers with larger hidden dimension
        new_layers = nn.ModuleList(
            [
                nn.Linear(self.input_dim, new_hidden_dim),
                nn.ReLU(),
                nn.Linear(new_hidden_dim, new_hidden_dim),
                nn.ReLU(),
                nn.Linear(new_hidden_dim, self.output_dim),
            ]
        )

        # Copy weights from old layers
        with torch.no_grad():
            # First layer
            new_layers[0].weight.data[:old_hidden_dim, :] = self.layers[0].weight.data
            new_layers[0].bias.data[:old_hidden_dim] = self.layers[0].bias.data

            # Second layer
            new_layers[2].weight.data[:old_hidden_dim, :old_hidden_dim] = self.layers[
                2
            ].weight.data
            new_layers[2].bias.data[:old_hidden_dim] = self.layers[2].bias.data

            # Output layer
            new_layers[4].weight.data[:, :old_hidden_dim] = self.layers[4].weight.data
            new_layers[4].bias.data = self.layers[4].bias.data

        # Update task embeddings
        new_task_embeddings = nn.Embedding(
            self.task_embeddings.num_embeddings, new_hidden_dim
        )
        new_task_embeddings.weight.data[:, :old_hidden_dim] = (
            self.task_embeddings.weight.data
        )

        # Replace layers and embeddings
        self.layers = new_layers
        self.task_embeddings = new_task_embeddings
        self.hidden_dim = new_hidden_dim

        # Update task heads
        for i, head in enumerate(self.task_heads):
            new_head = nn.Linear(new_hidden_dim, self.output_dim)
            new_head.weight.data[:, :old_hidden_dim] = head.weight.data
            new_head.bias.data = head.bias.data
            self.task_heads[i] = new_head

        # Record growth
        self.growth_history.append(
            {
                "old_hidden_dim": old_hidden_dim,
                "new_hidden_dim": new_hidden_dim,
                "growth_factor": new_hidden_dim / old_hidden_dim,
            }
        )

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the network."""
        # Shared backbone
        hidden = self.layers[0](x)  # Linear
        hidden = self.layers[1](hidden)  # ReLU
        hidden = self.layers[2](hidden)  # Linear
        hidden = self.layers[3](hidden)  # ReLU

        if task_id is not None and task_id < len(self.task_heads):
            # Task-specific head
            task_embedding = self.task_embeddings(
                torch.tensor(task_id, device=x.device)
            )
            hidden = hidden + task_embedding
            output = self.task_heads[task_id](hidden)
        else:
            # Shared output
            output = self.layers[4](hidden)

        return output

    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        total_params = sum(p.numel() for p in self.parameters())

        stats = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "current_hidden_dim": self.hidden_dim,
            "max_hidden_dim": self.max_hidden_dim,
            "total_parameters": total_params,
            "num_task_heads": len(self.task_heads),
            "growth_history": self.growth_history,
        }

        return stats


class AdaptiveNetwork(nn.Module):
    """Adaptive network that can modify its architecture based on task complexity."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_hidden_dim: int = 128,
        complexity_threshold: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_hidden_dim = base_hidden_dim
        self.complexity_threshold = complexity_threshold

        # Core network
        self.core_network = nn.Sequential(
            nn.Linear(input_dim, base_hidden_dim),
            nn.ReLU(),
            nn.Linear(base_hidden_dim, base_hidden_dim),
            nn.ReLU(),
        )

        # Adaptive components
        self.adaptive_layers = nn.ModuleList()
        self.task_complexities = {}
        self.activation_history = {}

    def add_adaptive_layer(self, task_id: int, complexity: float):
        """Add an adaptive layer based on task complexity."""
        if complexity > self.complexity_threshold:
            # High complexity - add more layers
            adaptive_layer = nn.Sequential(
                nn.Linear(self.base_hidden_dim, self.base_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.base_hidden_dim, self.base_hidden_dim),
                nn.ReLU(),
            )
        else:
            # Low complexity - add simple layer
            adaptive_layer = nn.Sequential(
                nn.Linear(self.base_hidden_dim, self.base_hidden_dim),
                nn.ReLU(),
            )

        self.adaptive_layers.append(adaptive_layer)
        self.task_complexities[task_id] = complexity

        return len(self.adaptive_layers) - 1

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass through the adaptive network."""
        # Core network
        hidden = self.core_network(x)

        # Adaptive layers
        if task_id < len(self.adaptive_layers):
            hidden = self.adaptive_layers[task_id](hidden)

        # Output layer
        output = nn.Linear(self.base_hidden_dim, self.output_dim)(hidden)

        # Record activation for complexity analysis
        self._record_activation(task_id, hidden)

        return output

    def _record_activation(self, task_id: int, activations: torch.Tensor):
        """Record activations for complexity analysis."""
        if task_id not in self.activation_history:
            self.activation_history[task_id] = []

        # Store activation statistics
        activation_stats = {
            "mean": activations.mean().item(),
            "std": activations.std().item(),
            "max": activations.max().item(),
            "min": activations.min().item(),
        }

        self.activation_history[task_id].append(activation_stats)

    def analyze_complexity(self, task_id: int) -> Dict[str, Any]:
        """Analyze task complexity based on activation patterns."""
        if task_id not in self.activation_history:
            return {"complexity": 0.0, "analysis": "No data available"}

        activations = self.activation_history[task_id]

        # Compute complexity metrics
        mean_activation = np.mean([a["mean"] for a in activations])
        activation_variance = np.var([a["std"] for a in activations])
        activation_range = np.mean([a["max"] - a["min"] for a in activations])

        # Combine metrics to estimate complexity
        complexity = (activation_variance + activation_range) / (1 + mean_activation)

        return {
            "complexity": complexity,
            "mean_activation": mean_activation,
            "activation_variance": activation_variance,
            "activation_range": activation_range,
            "num_samples": len(activations),
        }

    def get_adaptive_statistics(self) -> Dict[str, Any]:
        """Get adaptive network statistics."""
        stats = {
            "num_adaptive_layers": len(self.adaptive_layers),
            "task_complexities": self.task_complexities,
            "activation_history_size": len(self.activation_history),
        }

        # Complexity analysis for all tasks
        complexity_analysis = {}
        for task_id in self.task_complexities:
            complexity_analysis[task_id] = self.analyze_complexity(task_id)

        stats["complexity_analysis"] = complexity_analysis

        return stats


class ModularNetwork(nn.Module):
    """Modular network with reusable components."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        module_dim: int = 128,
        max_modules: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.module_dim = module_dim
        self.max_modules = max_modules

        # Input projection
        self.input_projection = nn.Linear(input_dim, module_dim)

        # Modular components
        self.modules = nn.ModuleList()
        self.module_usage = {}
        self.task_modules = {}

        # Output projection
        self.output_projection = nn.Linear(module_dim, output_dim)

    def add_module(self, module_type: str = "standard") -> int:
        """Add a new module to the network."""
        if len(self.modules) >= self.max_modules:
            # Reuse existing module
            return len(self.modules) - 1

        if module_type == "standard":
            module = nn.Sequential(
                nn.Linear(self.module_dim, self.module_dim),
                nn.ReLU(),
                nn.Linear(self.module_dim, self.module_dim),
                nn.ReLU(),
            )
        elif module_type == "residual":
            module = nn.Sequential(
                nn.Linear(self.module_dim, self.module_dim),
                nn.ReLU(),
                nn.Linear(self.module_dim, self.module_dim),
            )
        else:
            # Default module
            module = nn.Sequential(
                nn.Linear(self.module_dim, self.module_dim),
                nn.ReLU(),
            )

        self.modules.append(module)
        self.module_usage[len(self.modules) - 1] = 0

        return len(self.modules) - 1

    def assign_modules_to_task(self, task_id: int, module_ids: List[int]):
        """Assign specific modules to a task."""
        self.task_modules[task_id] = module_ids

        # Update usage statistics
        for module_id in module_ids:
            if module_id < len(self.modules):
                self.module_usage[module_id] += 1

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass through assigned modules."""
        # Input projection
        hidden = self.input_projection(x)

        # Apply assigned modules
        if task_id in self.task_modules:
            module_ids = self.task_modules[task_id]
            for module_id in module_ids:
                if module_id < len(self.modules):
                    module = self.modules[module_id]
                    if len(module) > 2 and "residual" in str(module):
                        # Residual connection
                        hidden = hidden + module(hidden)
                    else:
                        hidden = module(hidden)

        # Output projection
        output = self.output_projection(hidden)

        return output

    def get_module_statistics(self) -> Dict[str, Any]:
        """Get module usage statistics."""
        stats = {
            "total_modules": len(self.modules),
            "max_modules": self.max_modules,
            "module_usage": self.module_usage,
            "task_assignments": self.task_modules,
        }

        # Compute module efficiency
        total_usage = sum(self.module_usage.values())
        if total_usage > 0:
            usage_distribution = {
                k: v / total_usage for k, v in self.module_usage.items()
            }
            stats["usage_distribution"] = usage_distribution

        return stats

    def optimize_module_assignment(self, task_id: int, performance_metric: float):
        """Optimize module assignment based on performance."""
        if task_id not in self.task_modules:
            return

        current_modules = self.task_modules[task_id]

        # Simple optimization: try different module combinations
        if performance_metric < 0.5:  # Low performance
            # Try adding more modules
            if len(current_modules) < len(self.modules):
                new_modules = current_modules + [len(current_modules)]
                self.assign_modules_to_task(task_id, new_modules)
        elif performance_metric > 0.8:  # High performance
            # Try reducing modules
            if len(current_modules) > 1:
                new_modules = current_modules[:-1]
                self.assign_modules_to_task(task_id, new_modules)
