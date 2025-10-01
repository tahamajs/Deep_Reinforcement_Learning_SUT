"""
Neurosymbolic Policy Components

This module implements neural-symbolic policy learning and integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from .knowledge_base import SymbolicKnowledgeBase, LogicalPredicate, LogicalRule


class NeuralPerceptionModule(nn.Module):
    """Neural module for perceiving and processing raw observations."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Attention mechanism for focusing on relevant features
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Process raw observations into neural features."""
        batch_size = observations.shape[0]

        # Extract features
        features = self.feature_extractor(observations)

        # Apply attention (treating each feature as a sequence)
        features = features.unsqueeze(1)  # Add sequence dimension
        attended_features, attention_weights = self.attention(
            features, features, features
        )
        attended_features = attended_features.squeeze(1)  # Remove sequence dimension

        # Layer normalization
        output = self.layer_norm(attended_features)

        return output, attention_weights


class SymbolicReasoningModule(nn.Module):
    """Symbolic module for logical reasoning and rule application."""

    def __init__(
        self,
        knowledge_base: SymbolicKnowledgeBase,
        neural_dim: int = 64,
        symbolic_dim: int = 32,
    ):
        super().__init__()
        self.knowledge_base = knowledge_base
        self.neural_dim = neural_dim
        self.symbolic_dim = symbolic_dim

        # Neural to symbolic mapping
        self.neural_to_symbolic = nn.Sequential(
            nn.Linear(neural_dim, symbolic_dim),
            nn.ReLU(),
            nn.Linear(symbolic_dim, symbolic_dim),
        )

        # Symbolic reasoning layers
        self.reasoning_layers = nn.Sequential(
            nn.Linear(symbolic_dim, symbolic_dim),
            nn.ReLU(),
            nn.Linear(symbolic_dim, symbolic_dim),
        )

        # Rule application weights
        self.rule_weights = nn.Parameter(torch.ones(len(knowledge_base.rules)))

    def forward(self, neural_features: torch.Tensor) -> torch.Tensor:
        """Perform symbolic reasoning on neural features."""
        batch_size = neural_features.shape[0]

        # Map neural features to symbolic space
        symbolic_features = self.neural_to_symbolic(neural_features)

        # Apply symbolic reasoning
        reasoned_features = self.reasoning_layers(symbolic_features)

        # Apply rule-based reasoning
        rule_applications = self._apply_rules(symbolic_features)

        # Combine neural and symbolic reasoning
        combined_features = reasoned_features + rule_applications

        return combined_features

    def _apply_rules(self, symbolic_features: torch.Tensor) -> torch.Tensor:
        """Apply symbolic rules to features."""
        batch_size = symbolic_features.shape[0]
        rule_outputs = torch.zeros_like(symbolic_features)

        # Apply each rule with its weight
        for i, rule in enumerate(self.knowledge_base.rules):
            if i < len(self.rule_weights):
                weight = torch.sigmoid(self.rule_weights[i])
                # Simplified rule application
                rule_output = weight * symbolic_features
                rule_outputs += rule_output

        return rule_outputs


class NeurosymbolicPolicy(nn.Module):
    """Neurosymbolic policy combining neural perception and symbolic reasoning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        knowledge_base: SymbolicKnowledgeBase,
        hidden_dim: int = 128,
        symbolic_dim: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.knowledge_base = knowledge_base

        # Neural perception module
        self.neural_perception = NeuralPerceptionModule(
            input_dim=state_dim, hidden_dim=hidden_dim, output_dim=hidden_dim
        )

        # Symbolic reasoning module
        self.symbolic_reasoning = SymbolicReasoningModule(
            knowledge_base=knowledge_base,
            neural_dim=hidden_dim,
            symbolic_dim=symbolic_dim,
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + symbolic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + symbolic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Attention weights for combining neural and symbolic features
        self.attention_weights = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass of neurosymbolic policy."""
        batch_size = states.shape[0]

        # Neural perception
        neural_features, attention_weights = self.neural_perception(states)

        # Symbolic reasoning
        symbolic_features = self.symbolic_reasoning(neural_features)

        # Combine neural and symbolic features
        attention_probs = F.softmax(self.attention_weights, dim=0)
        combined_features = (
            attention_probs[0] * neural_features
            + attention_probs[1] * symbolic_features
        )

        # Policy and value outputs
        action_logits = self.policy_head(combined_features)
        values = self.value_head(combined_features)

        # Additional outputs for interpretability
        interpretability_info = {
            "neural_features": neural_features,
            "symbolic_features": symbolic_features,
            "attention_weights": attention_weights,
            "attention_probs": attention_probs,
            "rule_weights": self.symbolic_reasoning.rule_weights,
        }

        return action_logits, values, interpretability_info

    def get_action_probs(self, states: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        action_logits, _, _ = self.forward(states)
        return F.softmax(action_logits, dim=-1)

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        """Get state values."""
        _, values, _ = self.forward(states)
        return values.squeeze(-1)


class NeurosymbolicAgent:
    """Neurosymbolic RL agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        knowledge_base: SymbolicKnowledgeBase,
        lr: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.knowledge_base = knowledge_base

        # Policy network
        self.policy = NeurosymbolicPolicy(state_dim, action_dim, knowledge_base)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Training history
        self.training_history = {"losses": [], "rewards": [], "rule_activations": []}

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Select action using neurosymbolic policy."""
        with torch.no_grad():
            action_probs = self.policy.get_action_probs(state.unsqueeze(0))

            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action = torch.multinomial(action_probs, 1)

            return action.item()

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy using policy gradient."""
        # Forward pass
        action_logits, values, interpretability_info = self.policy(states)
        action_probs = F.softmax(action_logits, dim=-1)

        # Compute policy loss
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        policy_loss = -(log_probs * advantages.unsqueeze(1)).mean()

        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(-1), rewards)

        # Compute total loss
        total_loss = policy_loss + 0.5 * value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Record training history
        self.training_history["losses"].append(total_loss.item())
        self.training_history["rewards"].append(rewards.mean().item())

        # Record rule activations
        rule_stats = self.knowledge_base.get_rule_statistics()
        self.training_history["rule_activations"].append(
            rule_stats["total_activations"]
        )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def get_interpretability_info(self, states: torch.Tensor) -> Dict[str, Any]:
        """Get interpretability information for states."""
        with torch.no_grad():
            _, _, interpretability_info = self.policy(states)
            return interpretability_info

    def add_rule(self, rule: LogicalRule):
        """Add a new rule to the knowledge base."""
        self.knowledge_base.add_rule(rule)
        # Update rule weights parameter
        new_weights = torch.cat(
            [self.policy.symbolic_reasoning.rule_weights, torch.tensor([1.0])]
        )
        self.policy.symbolic_reasoning.rule_weights = nn.Parameter(new_weights)

    def get_rule_importance(self) -> Dict[str, float]:
        """Get importance scores for rules."""
        rule_weights = torch.sigmoid(self.policy.symbolic_reasoning.rule_weights)
        rule_importance = {}

        for i, rule in enumerate(self.knowledge_base.rules):
            if i < len(rule_weights):
                rule_importance[str(rule)] = rule_weights[i].item()

        return rule_importance
