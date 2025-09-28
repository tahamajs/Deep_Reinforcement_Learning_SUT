import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from collections import deque
import copy
import random


# Safety Constraints
class SafetyConstraints:
    """Safety constraints for reinforcement learning"""

    def __init__(
        self,
        state_bounds: Optional[np.ndarray] = None,
        action_bounds: Optional[np.ndarray] = None,
        safety_threshold: float = 0.1,
    ):
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.safety_threshold = safety_threshold

    def check_state_safety(self, state: np.ndarray) -> bool:
        """Check if state is within safe bounds"""
        if self.state_bounds is None:
            return True
        return np.all(np.abs(state) <= self.state_bounds)

    def check_action_safety(self, action: np.ndarray) -> bool:
        """Check if action is within safe bounds"""
        if self.action_bounds is None:
            return True
        return np.all(np.abs(action) <= self.action_bounds)

    def project_to_safe_action(self, action: np.ndarray) -> np.ndarray:
        """Project action to safe action space"""
        if self.action_bounds is not None:
            return np.clip(action, -self.action_bounds, self.action_bounds)
        return action


# Robust Policy
class RobustPolicy:
    """Robust policy that handles adversarial perturbations"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        robustness_level: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.robustness_level = robustness_level

        # Main policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Robustness network (adversarial training)
        self.robust_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def get_action(
        self, state: torch.Tensor, adversarial: bool = False
    ) -> torch.Tensor:
        """Get robust action"""
        main_action = self.policy_net(state)

        if adversarial:
            # Add adversarial perturbation
            perturbation = self.robust_net(state) * self.robustness_level
            robust_action = main_action + perturbation
        else:
            robust_action = main_action

        return torch.clamp(robust_action, -1, 1)

    def train_robustness(
        self, states: torch.Tensor, actions: torch.Tensor, perturbations: torch.Tensor
    ):
        """Train robustness against perturbations"""
        # Adversarial training objective
        perturbed_states = states + perturbations
        robust_actions = self.robust_net(perturbed_states)

        # Minimize difference between main and robust policies
        main_actions = self.policy_net(states)
        robustness_loss = F.mse_loss(robust_actions, main_actions.detach())

        return robustness_loss


# Constrained Policy Optimization
class ConstrainedPolicyOptimization:
    """Constrained policy optimization with safety constraints"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        cost_limit: float = 0.1,
        lambda_lr: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Cost value network (for safety constraints)
        self.cost_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Lagrange multiplier for constraint
        self.lambda_param = torch.tensor(1.0, requires_grad=True)
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.cost_optimizer = torch.optim.Adam(self.cost_net.parameters(), lr=1e-3)
        self.lambda_optimizer = torch.optim.Adam([self.lambda_param], lr=lambda_lr)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get action from constrained policy"""
        with torch.no_grad():
            return self.policy(state)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Update constrained policy"""

        # Compute advantages
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            advantages = rewards + 0.99 * next_values * (1 - dones) - values

            cost_values = self.cost_net(states).squeeze()
            next_cost_values = self.cost_net(next_states).squeeze()
            cost_advantages = (
                costs + 0.99 * next_cost_values * (1 - dones) - cost_values
            )

        # Policy loss with constraints
        log_probs = self._compute_log_probs(states, actions)
        policy_loss = -(log_probs * advantages).mean()

        # Cost constraint
        cost_penalty = torch.max(
            torch.tensor(0.0),
            self.lambda_param * (cost_advantages.mean() - self.cost_limit),
        )

        total_policy_loss = policy_loss + cost_penalty

        # Update policy
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        # Update value function
        value_loss = F.mse_loss(values, rewards + 0.99 * next_values * (1 - dones))
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update cost value function
        cost_loss = F.mse_loss(
            cost_values, costs + 0.99 * next_cost_values * (1 - dones)
        )
        self.cost_optimizer.zero_grad()
        cost_loss.backward()
        self.cost_optimizer.step()

        # Update Lagrange multiplier
        lambda_loss = -self.lambda_param * (cost_advantages.mean() - self.cost_limit)
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        # Clip lambda to be positive
        with torch.no_grad():
            self.lambda_param.data.clamp_(min=0)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "cost_loss": cost_loss.item(),
            "lambda": self.lambda_param.item(),
            "cost_penalty": cost_penalty.item(),
        }

    def _compute_log_probs(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities of actions"""
        policy_output = self.policy(states)
        dist = torch.distributions.Normal(policy_output, 0.1)
        return dist.log_prob(actions).sum(dim=-1)


# Risk-Sensitive RL
class RiskSensitiveRL:
    """Risk-sensitive reinforcement learning"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        risk_sensitivity: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.risk_sensitivity = risk_sensitivity

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Value network (risk-adjusted)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()), lr=1e-3
        )

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get risk-sensitive action"""
        with torch.no_grad():
            return self.policy(state)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Update with risk-sensitive objective"""

        # Compute risk-adjusted returns
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()

            # Exponential utility for risk sensitivity
            if self.risk_sensitivity > 0:
                # Risk-seeking (higher moments)
                adjusted_rewards = torch.exp(self.risk_sensitivity * rewards)
            else:
                # Risk-averse (lower moments)
                adjusted_rewards = -torch.exp(-self.risk_sensitivity * rewards)

            targets = adjusted_rewards + 0.99 * next_values * (1 - dones)

        # Value loss
        value_loss = F.mse_loss(values, targets)

        # Policy loss (risk-sensitive)
        log_probs = self._compute_log_probs(states, actions)
        policy_loss = -(log_probs * (targets - values.detach())).mean()

        total_loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "risk_sensitivity": self.risk_sensitivity,
        }

    def _compute_log_probs(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities"""
        policy_output = self.policy(states)
        dist = torch.distributions.Normal(policy_output, 0.1)
        return dist.log_prob(actions).sum(dim=-1)


# Adversarial Training
class AdversarialTraining:
    """Adversarial training for robustness"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        adversarial_budget: float = 0.1,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.adversarial_budget = adversarial_budget

        # Main policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Adversarial perturbation generator
        self.adversary = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh(),
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.adversary_optimizer = torch.optim.Adam(
            self.adversary.parameters(), lr=1e-3
        )

    def get_action(self, state: torch.Tensor, robust: bool = True) -> torch.Tensor:
        """Get action with adversarial robustness"""
        if robust:
            # Generate adversarial perturbation
            perturbation = self.adversary(state) * self.adversarial_budget
            perturbed_state = state + perturbation

            # Get action for perturbed state
            action = self.policy(perturbed_state)
        else:
            action = self.policy(state)

        return action

    def adversarial_update(
        self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ):
        """Adversarial training update"""

        # Generate adversarial perturbations
        perturbations = self.adversary(states) * self.adversarial_budget
        perturbed_states = states + perturbations

        # Policy should perform well on perturbed states
        perturbed_actions = self.policy(perturbed_states)
        policy_loss = F.mse_loss(perturbed_actions, actions)

        # Adversary should make policy perform worse
        original_actions = self.policy(states)
        perturbed_actions = self.policy(perturbed_states)
        adversary_loss = -F.mse_loss(perturbed_actions, original_actions)

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Update adversary
        self.adversary_optimizer.zero_grad()
        adversary_loss.backward()
        self.adversary_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "adversary_loss": adversary_loss.item(),
        }


# Safety Monitor
class SafetyMonitor:
    """Real-time safety monitoring and intervention"""

    def __init__(
        self, safety_threshold: float = 0.1, intervention_probability: float = 0.1
    ):
        self.safety_threshold = safety_threshold
        self.intervention_probability = intervention_probability
        self.safety_violations = []
        self.interventions = []

    def monitor_safety(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> Dict:
        """Monitor safety of state-action transitions"""

        # Simple safety metrics
        state_norm = np.linalg.norm(state)
        action_norm = np.linalg.norm(action)
        state_change = np.linalg.norm(next_state - state)

        # Safety violation detection
        violation = (
            state_norm > 5  # State too extreme
            or action_norm > 1  # Action too extreme
            or state_change > 2  # Too much state change
        )

        self.safety_violations.append(violation)

        # Intervention decision
        intervention = violation and np.random.random() < self.intervention_probability

        if intervention:
            self.interventions.append(True)
            # Safe intervention: go to origin
            safe_action = -0.5 * state[: len(action)]
        else:
            self.interventions.append(False)
            safe_action = action

        return {
            "violation": violation,
            "intervention": intervention,
            "safe_action": safe_action,
            "safety_metrics": {
                "state_norm": state_norm,
                "action_norm": action_norm,
                "state_change": state_change,
            },
        }

    def get_safety_report(self) -> Dict:
        """Get safety monitoring report"""
        total_steps = len(self.safety_violations)
        violation_rate = np.mean(self.safety_violations) if total_steps > 0 else 0
        intervention_rate = np.mean(self.interventions) if total_steps > 0 else 0

        return {
            "total_steps": total_steps,
            "violation_rate": violation_rate,
            "intervention_rate": intervention_rate,
            "safety_score": 1 - violation_rate,
        }


print("✅ Advanced Safety implementations complete!")
print("Components implemented:")
print("- SafetyConstraints: Safety bounds and constraints")
print("- RobustPolicy: Adversarial robustness")
print("- ConstrainedPolicyOptimization: Safety-constrained optimization")
print("- RiskSensitiveRL: Risk-aware decision making")
print("- AdversarialTraining: Robustness through adversarial training")
print("- SafetyMonitor: Real-time safety monitoring and intervention")
