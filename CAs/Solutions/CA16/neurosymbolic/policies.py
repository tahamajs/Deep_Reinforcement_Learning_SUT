"""
Neurosymbolic Policies and Agents

This module contains policy networks and agents that combine neural and symbolic approaches:
- Neurosymbolic policy networks
- Hybrid agents
- Symbolic constraint satisfaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .knowledge_base import SymbolicKnowledgeBase, SymbolicAtom
from .neural_components import (
    NeuralPerceptionModule,
    SymbolicReasoningModule,
    NeuralSymbolicInterface,
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeurosymbolicPolicy(nn.Module):
    """Policy network that combines neural and symbolic reasoning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        perception_module: NeuralPerceptionModule,
        reasoning_module: SymbolicReasoningModule,
        knowledge_base: SymbolicKnowledgeBase,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Neural-symbolic components
        self.perception = perception_module
        self.reasoning = reasoning_module
        self.interface = NeuralSymbolicInterface(
            perception_module, reasoning_module, knowledge_base
        )

        # Policy networks
        self.neural_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.symbolic_policy = nn.Sequential(
            nn.Linear(reasoning_module.symbol_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(action_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value network
        self.value_network = nn.Sequential(
            nn.Linear(state_dim + reasoning_module.symbol_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Symbolic constraints
        self.symbolic_constraints = self._init_symbolic_constraints()

    def _init_symbolic_constraints(self) -> Dict[str, List[SymbolicAtom]]:
        """Initialize symbolic constraints for safe actions."""
        constraints = {
            "safe_actions": [],  # Actions that are always safe
            "forbidden_actions": [],  # Actions that are never allowed
            "conditional_actions": [],  # Actions allowed under certain conditions
        }
        return constraints

    def forward(
        self, state: torch.Tensor, symbolic_queries: Optional[List[SymbolicAtom]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through neurosymbolic policy."""
        # Neural perception and reasoning
        if symbolic_queries:
            reasoning_results = self.interface.perceive_and_reason(
                state, symbolic_queries
            )
            symbolic_features = reasoning_results["reasoned_features"]
        else:
            symbolic_features = self.perception(state)

        # Neural policy
        neural_action_logits = self.neural_policy(state)

        # Symbolic policy
        symbolic_action_logits = self.symbolic_policy(symbolic_features)

        # Fuse policies
        combined_logits = torch.cat(
            [neural_action_logits, symbolic_action_logits], dim=-1
        )
        fused_action_logits = self.fusion_network(combined_logits)

        # Apply symbolic constraints
        constrained_logits = self._apply_symbolic_constraints(
            fused_action_logits, symbolic_queries
        )

        # Value estimation
        value_input = torch.cat([state, symbolic_features], dim=-1)
        value = self.value_network(value_input)

        return constrained_logits, value

    def _apply_symbolic_constraints(
        self,
        action_logits: torch.Tensor,
        symbolic_queries: Optional[List[SymbolicAtom]] = None,
    ) -> torch.Tensor:
        """Apply symbolic constraints to action logits."""
        constrained_logits = action_logits.clone()

        if symbolic_queries is None:
            return constrained_logits

        # Check for forbidden actions
        for i, query in enumerate(symbolic_queries):
            if str(query).startswith("forbidden("):
                # Extract action from query
                action_name = str(query).split("(")[1].split(")")[0]
                if action_name.isdigit():
                    action_idx = int(action_name)
                    if action_idx < self.action_dim:
                        constrained_logits[0, action_idx] = -float("inf")

        # Could add more sophisticated constraint checking here
        return constrained_logits

    def get_action(
        self,
        state: torch.Tensor,
        symbolic_queries: Optional[List[SymbolicAtom]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get action from policy."""
        self.eval()
        with torch.no_grad():
            action_logits, value = self.forward(state, symbolic_queries)

            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action_dist = torch.distributions.Categorical(logits=action_logits)
                action = action_dist.sample()

            log_prob = torch.log_softmax(action_logits, dim=-1)[0, action]

            info = {
                "log_prob": log_prob,
                "value": value,
                "action_logits": action_logits,
            }

        return action, value, info

    def get_action_probabilities(
        self, state: torch.Tensor, symbolic_queries: Optional[List[SymbolicAtom]] = None
    ) -> torch.Tensor:
        """Get action probabilities."""
        self.eval()
        with torch.no_grad():
            action_logits, _ = self.forward(state, symbolic_queries)
            action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs


class NeurosymbolicAgent:
    """Agent that uses neurosymbolic reasoning for decision making."""

    def __init__(
        self,
        policy: NeurosymbolicPolicy,
        knowledge_base: SymbolicKnowledgeBase,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.policy = policy
        self.kb = knowledge_base

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.symbolic_queries = []

    def act(
        self, state: np.ndarray, symbolic_queries: Optional[List[SymbolicAtom]] = None
    ) -> Tuple[int, Dict]:
        """Select action based on current state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        action, value, info = self.policy.get_action(state_tensor, symbolic_queries)

        action_scalar = action.item()
        log_prob = info["log_prob"].item()
        value_scalar = value.item()

        # Store experience
        self.states.append(state)
        self.actions.append(action_scalar)
        self.values.append(value_scalar)
        self.log_probs.append(log_prob)
        self.symbolic_queries.append(symbolic_queries or [])

        return action_scalar, info

    def store_reward(self, reward: float, done: bool):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)

    def update_policy(self, next_value: float = 0.0):
        """Update policy using PPO-style algorithm."""
        if len(self.states) == 0:
            return

        # Convert to tensors
        states = torch.FloatTensor(self.states).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        dones = torch.FloatTensor(self.dones).to(device)
        values = torch.FloatTensor(self.values).to(device)

        # Compute advantages and returns
        advantages, returns = self._compute_advantages(
            rewards, values, dones, next_value
        )

        # Update policy
        for _ in range(4):  # PPO epochs
            # Get current policy outputs
            action_logits, current_values = self.policy(states)

            # Compute ratios
            current_log_probs = torch.log_softmax(action_logits, dim=-1)
            current_log_probs = current_log_probs.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)
            ratios = torch.exp(current_log_probs - old_log_probs)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(current_values.squeeze(), returns)

            # Total loss
            total_loss = policy_loss + 0.5 * value_loss

            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Clear experience buffer
        self._clear_buffer()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns using GAE."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Bootstrap next value
        next_value_tensor = torch.tensor(next_value).to(device)

        # Compute advantages and returns backwards
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value_tensor
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def _clear_buffer(self):
        """Clear experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.symbolic_queries = []

    def save_model(self, path: str):
        """Save agent model."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "kb": self.kb,  # This might need special handling for pickle
            },
            path,
        )

    def load_model(self, path: str):
        """Load agent model."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # KB loading would need to be handled separately


class SafeNeurosymbolicAgent(NeurosymbolicAgent):
    """Neurosymbolic agent with safety constraints."""

    def __init__(
        self,
        policy: NeurosymbolicPolicy,
        knowledge_base: SymbolicKnowledgeBase,
        safety_threshold: float = 0.8,
    ):
        super().__init__(policy, knowledge_base)
        self.safety_threshold = safety_threshold

    def act(
        self, state: np.ndarray, symbolic_queries: Optional[List[SymbolicAtom]] = None
    ) -> Tuple[int, Dict]:
        """Select safe action."""
        # Get action probabilities
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.policy.get_action_probabilities(
            state_tensor, symbolic_queries
        )

        # Check safety of each action
        safe_actions = []
        for action_idx in range(self.policy.action_dim):
            # Create safety query
            safety_query = SymbolicAtom(
                self.kb.predicates.get(
                    "safe", self.kb.predicates["at"]
                ),  # Fallback predicate
                (f"action_{action_idx}", "current_state"),
            )

            # Check if action is safe
            if self.kb.query(safety_query):
                safe_actions.append(action_idx)

        if safe_actions:
            # Filter probabilities to safe actions only
            safe_probs = action_probs[0, safe_actions]
            safe_probs = safe_probs / safe_probs.sum()  # Renormalize

            # Sample from safe actions
            safe_action_idx = torch.multinomial(safe_probs, 1).item()
            action = safe_actions[safe_action_idx]
        else:
            # No safe actions, fall back to regular policy
            action, info = super().act(state, symbolic_queries)
            return action, info

        # Get log prob for the selected action
        log_prob = torch.log(action_probs[0, action])

        info = {
            "log_prob": log_prob.item(),
            "safe_actions": safe_actions,
            "action_probs": action_probs,
        }

        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob.item())
        self.symbolic_queries.append(symbolic_queries or [])

        return action, info
