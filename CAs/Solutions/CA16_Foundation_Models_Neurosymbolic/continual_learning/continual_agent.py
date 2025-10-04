"""
Continual Learning Agent

This module implements agents that can learn continuously without catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy


class ContinualLearningAgent:
    """RL agent that learns continuously across multiple tasks."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_tasks: int = 5,
        lr: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.current_task = 0

        # Shared backbone network
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Task-specific heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) for _ in range(num_tasks)
        ])

        # Task-specific value heads
        self.value_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_tasks)
        ])

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Continual learning components
        self.ewc_lambda = 1000.0
        self.fisher_information = {}
        self.optimal_params = {}

        # Learning history
        self.task_performances = {}
        self.learning_history = {"losses": [], "rewards": [], "tasks": []}

    def select_action(self, state: torch.Tensor, task_id: Optional[int] = None) -> int:
        """Select action for given task."""
        if task_id is None:
            task_id = self.current_task

        with torch.no_grad():
            features = self.backbone(state)
            action_logits = self.task_heads[task_id](features)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1)

            return action.item()

    def get_action_probs(self, state: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Get action probabilities for given task."""
        if task_id is None:
            task_id = self.current_task

        features = self.backbone(state)
        action_logits = self.task_heads[task_id](features)
        return F.softmax(action_logits, dim=-1)

    def get_value(self, state: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Get state value for given task."""
        if task_id is None:
            task_id = self.current_task

        features = self.backbone(state)
        value = self.value_heads[task_id](features)
        return value.squeeze(-1)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        task_id: int,
        use_ewc: bool = True,
    ) -> Dict[str, float]:
        """Update agent on current task."""
        self.current_task = task_id

        # Forward pass
        features = self.backbone(states)
        action_logits = self.task_heads[task_id](features)
        values = self.value_heads[task_id](features).squeeze(-1)

        # Compute losses
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        
        # Policy loss (simplified REINFORCE)
        policy_loss = -(log_probs.squeeze() * rewards).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, rewards)

        # EWC penalty
        ewc_penalty = torch.tensor(0.0)
        if use_ewc and task_id > 0:
            ewc_penalty = self._compute_ewc_penalty()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + self.ewc_lambda * ewc_penalty

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Record learning history
        self.learning_history["losses"].append(total_loss.item())
        self.learning_history["rewards"].append(rewards.mean().item())
        self.learning_history["tasks"].append(task_id)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "ewc_penalty": ewc_penalty.item(),
            "total_loss": total_loss.item(),
        }

    def _compute_ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty for previous tasks."""
        penalty = torch.tensor(0.0)
        
        for task_id in range(self.current_task):
            if task_id in self.fisher_information:
                for name, param in self.named_parameters():
                    if name in self.fisher_information[task_id]:
                        fisher = self.fisher_information[task_id][name]
                        optimal = self.optimal_params[task_id][name]
                        penalty += (fisher * (param - optimal) ** 2).sum()
        
        return penalty

    def compute_fisher_information(self, states: torch.Tensor, actions: torch.Tensor, task_id: int):
        """Compute Fisher information matrix for current task."""
        self.eval()
        
        # Forward pass
        features = self.backbone(states)
        action_logits = self.task_heads[task_id](features)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))

        # Compute gradients
        self.zero_grad()
        log_probs.sum().backward()

        # Store Fisher information
        fisher_info = {}
        optimal_params = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                fisher_info[name] = param.grad.data.clone() ** 2
                optimal_params[name] = param.data.clone()

        self.fisher_information[task_id] = fisher_info
        self.optimal_params[task_id] = optimal_params

        self.train()

    def switch_task(self, task_id: int):
        """Switch to a different task."""
        if 0 <= task_id < self.num_tasks:
            self.current_task = task_id

    def get_task_performance(self, task_id: int) -> Dict[str, float]:
        """Get performance metrics for a specific task."""
        if task_id not in self.task_performances:
            return {"avg_reward": 0.0, "episodes": 0}

        return self.task_performances[task_id]

    def update_task_performance(self, task_id: int, reward: float):
        """Update performance metrics for a task."""
        if task_id not in self.task_performances:
            self.task_performances[task_id] = {
                "rewards": [],
                "avg_reward": 0.0,
                "episodes": 0,
            }

        self.task_performances[task_id]["rewards"].append(reward)
        self.task_performances[task_id]["episodes"] += 1
        self.task_performances[task_id]["avg_reward"] = np.mean(
            self.task_performances[task_id]["rewards"]
        )

    def get_continual_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about continual learning performance."""
        stats = {
            "current_task": self.current_task,
            "total_tasks": self.num_tasks,
            "task_performances": self.task_performances,
            "ewc_penalty": self.ewc_lambda,
            "fisher_matrices": len(self.fisher_information),
        }

        # Compute forgetting metrics
        if len(self.task_performances) > 1:
            forgetting_metrics = self._compute_forgetting_metrics()
            stats.update(forgetting_metrics)

        return stats

    def _compute_forgetting_metrics(self) -> Dict[str, float]:
        """Compute forgetting metrics."""
        if len(self.task_performances) < 2:
            return {"forgetting": 0.0, "retention": 1.0}

        # Compute average forgetting
        forgetting_scores = []
        retention_scores = []

        for task_id in range(len(self.task_performances) - 1):
            if task_id in self.task_performances:
                task_perf = self.task_performances[task_id]
                if len(task_perf["rewards"]) > 10:
                    # Use first 10 episodes as baseline
                    baseline_perf = np.mean(task_perf["rewards"][:10])
                    # Use last 10 episodes as current performance
                    current_perf = np.mean(task_perf["rewards"][-10:])
                    
                    forgetting = max(0, baseline_perf - current_perf)
                    retention = current_perf / baseline_perf if baseline_perf > 0 else 0
                    
                    forgetting_scores.append(forgetting)
                    retention_scores.append(retention)

        return {
            "forgetting": np.mean(forgetting_scores) if forgetting_scores else 0.0,
            "retention": np.mean(retention_scores) if retention_scores else 1.0,
        }

    def save_task_checkpoint(self, task_id: int, path: str):
        """Save checkpoint for a specific task."""
        checkpoint = {
            "task_id": task_id,
            "model_state_dict": self.state_dict(),
            "fisher_information": self.fisher_information.get(task_id, {}),
            "optimal_params": self.optimal_params.get(task_id, {}),
            "task_performance": self.task_performances.get(task_id, {}),
        }
        torch.save(checkpoint, path)

    def load_task_checkpoint(self, path: str):
        """Load checkpoint for a specific task."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        
        task_id = checkpoint["task_id"]
        if "fisher_information" in checkpoint:
            self.fisher_information[task_id] = checkpoint["fisher_information"]
        if "optimal_params" in checkpoint:
            self.optimal_params[task_id] = checkpoint["optimal_params"]
        if "task_performance" in checkpoint:
            self.task_performances[task_id] = checkpoint["task_performance"]

    def parameters(self):
        """Get all model parameters."""
        return list(self.backbone.parameters()) + list(self.task_heads.parameters()) + list(self.value_heads.parameters())

    def named_parameters(self):
        """Get named parameters."""
        params = []
        for name, param in self.backbone.named_parameters():
            params.append((f"backbone.{name}", param))
        for i, head in enumerate(self.task_heads):
            for name, param in head.named_parameters():
                params.append((f"task_heads.{i}.{name}", param))
        for i, head in enumerate(self.value_heads):
            for name, param in head.named_parameters():
                params.append((f"value_heads.{i}.{name}", param))
        return params

    def eval(self):
        """Set model to evaluation mode."""
        self.backbone.eval()
        for head in self.task_heads:
            head.eval()
        for head in self.value_heads:
            head.eval()

    def train(self):
        """Set model to training mode."""
        self.backbone.train()
        for head in self.task_heads:
            head.train()
        for head in self.value_heads:
            head.train()

    def state_dict(self):
        """Get state dictionary."""
        return {
            "backbone": self.backbone.state_dict(),
            "task_heads": [head.state_dict() for head in self.task_heads],
            "value_heads": [head.state_dict() for head in self.value_heads],
        }

    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.backbone.load_state_dict(state_dict["backbone"])
        for i, head_state in enumerate(state_dict["task_heads"]):
            self.task_heads[i].load_state_dict(head_state)
        for i, head_state in enumerate(state_dict["value_heads"]):
            self.value_heads[i].load_state_dict(head_state)