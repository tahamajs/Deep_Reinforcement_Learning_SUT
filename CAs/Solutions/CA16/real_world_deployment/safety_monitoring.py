"""
Safety Monitoring and Robustness Testing

This module contains classes for monitoring safety, testing robustness,
and implementing fail-safe mechanisms in RL systems.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import random


@dataclass
class SafetyViolation:
    """Record of a safety violation."""

    timestamp: float
    violation_type: str
    severity: str
    description: str
    state: np.ndarray
    action: int
    context: Dict[str, Any]


@dataclass
class RobustnessTest:
    """Robustness test configuration."""

    test_name: str
    test_type: str
    parameters: Dict[str, Any]
    expected_behavior: str
    threshold: float


class SafetyMonitor:
    """Monitor safety violations and system behavior."""

    def __init__(self, safety_thresholds: Dict[str, float] = None):
        self.safety_thresholds = safety_thresholds or {
            "max_action_magnitude": 1.0,
            "max_state_deviation": 2.0,
            "min_confidence": 0.5,
            "max_uncertainty": 0.3,
        }

        self.violations = deque(maxlen=1000)
        self.safety_metrics = {
            "total_violations": 0,
            "critical_violations": 0,
            "warning_violations": 0,
            "safety_score": 1.0,
        }

        # Safety callbacks
        self.safety_callbacks = []

    def check_safety(
        self, state: np.ndarray, action: int, confidence: float = 1.0
    ) -> bool:
        """Check if the action is safe given the state."""
        is_safe = True
        violations = []

        # Check action magnitude
        if abs(action) > self.safety_thresholds["max_action_magnitude"]:
            is_safe = False
            violations.append(
                {
                    "type": "action_magnitude",
                    "severity": "critical",
                    "description": f"Action magnitude {abs(action)} exceeds threshold {self.safety_thresholds['max_action_magnitude']}",
                }
            )

        # Check state deviation
        state_norm = np.linalg.norm(state)
        if state_norm > self.safety_thresholds["max_state_deviation"]:
            is_safe = False
            violations.append(
                {
                    "type": "state_deviation",
                    "severity": "warning",
                    "description": f"State deviation {state_norm} exceeds threshold {self.safety_thresholds['max_state_deviation']}",
                }
            )

        # Check confidence
        if confidence < self.safety_thresholds["min_confidence"]:
            is_safe = False
            violations.append(
                {
                    "type": "low_confidence",
                    "severity": "warning",
                    "description": f"Confidence {confidence} below threshold {self.safety_thresholds['min_confidence']}",
                }
            )

        # Record violations
        for violation in violations:
            self._record_violation(state, action, violation)

        return is_safe

    def _record_violation(
        self, state: np.ndarray, action: int, violation: Dict[str, Any]
    ):
        """Record a safety violation."""
        safety_violation = SafetyViolation(
            timestamp=time.time(),
            violation_type=violation["type"],
            severity=violation["severity"],
            description=violation["description"],
            state=state.copy(),
            action=action,
            context={},
        )

        self.violations.append(safety_violation)

        # Update metrics
        self.safety_metrics["total_violations"] += 1
        if violation["severity"] == "critical":
            self.safety_metrics["critical_violations"] += 1
        else:
            self.safety_metrics["warning_violations"] += 1

        # Update safety score
        self._update_safety_score()

        # Trigger callbacks
        self._trigger_safety_callbacks(safety_violation)

    def _update_safety_score(self):
        """Update the overall safety score."""
        if self.safety_metrics["total_violations"] == 0:
            self.safety_metrics["safety_score"] = 1.0
        else:
            # Penalize based on violation count and severity
            critical_penalty = self.safety_metrics["critical_violations"] * 0.1
            warning_penalty = self.safety_metrics["warning_violations"] * 0.05
            self.safety_metrics["safety_score"] = max(
                0.0, 1.0 - critical_penalty - warning_penalty
            )

    def _trigger_safety_callbacks(self, violation: SafetyViolation):
        """Trigger registered safety callbacks."""
        for callback in self.safety_callbacks:
            try:
                callback(violation)
            except Exception as e:
                logging.error(f"Error in safety callback: {e}")

    def register_safety_callback(self, callback: Callable[[SafetyViolation], None]):
        """Register a safety callback function."""
        self.safety_callbacks.append(callback)

    def get_safety_report(self) -> Dict[str, Any]:
        """Get a comprehensive safety report."""
        recent_violations = [
            v for v in self.violations if v.timestamp > time.time() - 3600
        ]  # Last hour

        return {
            "safety_metrics": self.safety_metrics,
            "recent_violations": len(recent_violations),
            "violation_types": self._get_violation_type_counts(),
            "safety_trend": self._calculate_safety_trend(),
            "recommendations": self._generate_safety_recommendations(),
        }

    def _get_violation_type_counts(self) -> Dict[str, int]:
        """Get counts of different violation types."""
        type_counts = {}
        for violation in self.violations:
            violation_type = violation.violation_type
            type_counts[violation_type] = type_counts.get(violation_type, 0) + 1
        return type_counts

    def _calculate_safety_trend(self) -> str:
        """Calculate safety trend over time."""
        if len(self.violations) < 10:
            return "insufficient_data"

        # Compare recent vs older violations
        recent_violations = [
            v for v in self.violations if v.timestamp > time.time() - 1800
        ]  # Last 30 minutes
        older_violations = [
            v for v in self.violations if v.timestamp <= time.time() - 1800
        ]  # Older than 30 minutes

        if len(recent_violations) < len(older_violations):
            return "improving"
        elif len(recent_violations) > len(older_violations):
            return "deteriorating"
        else:
            return "stable"

    def _generate_safety_recommendations(self) -> List[str]:
        """Generate safety recommendations based on violations."""
        recommendations = []

        if self.safety_metrics["critical_violations"] > 5:
            recommendations.append("Consider reducing action magnitude limits")

        if self.safety_metrics["warning_violations"] > 20:
            recommendations.append("Review state normalization and preprocessing")

        if self.safety_metrics["safety_score"] < 0.7:
            recommendations.append("Implement additional safety constraints")

        return recommendations


class RobustnessTester:
    """Test system robustness against various perturbations."""

    def __init__(self, model: nn.Module, state_dim: int, action_dim: int):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.test_results = []

    def test_noise_robustness(
        self, test_states: torch.Tensor, noise_levels: List[float]
    ) -> Dict[str, Any]:
        """Test robustness to input noise."""
        results = {
            "noise_levels": noise_levels,
            "accuracy_drop": [],
            "action_consistency": [],
        }

        for noise_level in noise_levels:
            # Add noise to test states
            noise = torch.randn_like(test_states) * noise_level
            noisy_states = test_states + noise

            # Get predictions
            with torch.no_grad():
                original_outputs = self.model(test_states)
                noisy_outputs = self.model(noisy_states)

            # Calculate accuracy drop
            if original_outputs.dim() > 1:
                original_actions = torch.argmax(original_outputs, dim=-1)
                noisy_actions = torch.argmax(noisy_outputs, dim=-1)
                accuracy_drop = (
                    1.0 - (original_actions == noisy_actions).float().mean().item()
                )
            else:
                accuracy_drop = F.mse_loss(original_outputs, noisy_outputs).item()

            results["accuracy_drop"].append(accuracy_drop)

            # Calculate action consistency
            action_consistency = 1.0 - accuracy_drop
            results["action_consistency"].append(action_consistency)

        return results

    def test_adversarial_robustness(
        self, test_states: torch.Tensor, epsilon: float = 0.1
    ) -> Dict[str, Any]:
        """Test robustness to adversarial perturbations."""
        results = {
            "epsilon": epsilon,
            "successful_attacks": 0,
            "total_attacks": len(test_states),
            "attack_success_rate": 0.0,
        }

        for i, state in enumerate(test_states):
            # Generate adversarial perturbation
            state.requires_grad_(True)
            output = self.model(state.unsqueeze(0))

            if output.dim() > 1:
                target_action = torch.argmax(output, dim=-1)
                loss = F.cross_entropy(output, target_action)
            else:
                loss = output.sum()

            # Compute gradient
            grad = torch.autograd.grad(loss, state, create_graph=False)[0]

            # Generate adversarial example
            adversarial_state = state + epsilon * grad.sign()

            # Test if attack is successful
            with torch.no_grad():
                original_output = self.model(state.unsqueeze(0))
                adversarial_output = self.model(adversarial_state.unsqueeze(0))

            if original_output.dim() > 1:
                original_action = torch.argmax(original_output, dim=-1)
                adversarial_action = torch.argmax(adversarial_output, dim=-1)
                if original_action != adversarial_action:
                    results["successful_attacks"] += 1
            else:
                if torch.abs(original_output - adversarial_output).item() > epsilon:
                    results["successful_attacks"] += 1

        results["attack_success_rate"] = (
            results["successful_attacks"] / results["total_attacks"]
        )
        return results

    def test_distribution_shift(
        self, original_states: torch.Tensor, shifted_states: torch.Tensor
    ) -> Dict[str, Any]:
        """Test robustness to distribution shift."""
        with torch.no_grad():
            original_outputs = self.model(original_states)
            shifted_outputs = self.model(shifted_states)

        # Calculate performance drop
        if original_outputs.dim() > 1:
            original_actions = torch.argmax(original_outputs, dim=-1)
            shifted_actions = torch.argmax(shifted_outputs, dim=-1)
            performance_drop = (
                1.0 - (original_actions == shifted_actions).float().mean().item()
            )
        else:
            performance_drop = F.mse_loss(original_outputs, shifted_outputs).item()

        return {
            "performance_drop": performance_drop,
            "distribution_shift_severity": (
                "high"
                if performance_drop > 0.3
                else "medium" if performance_drop > 0.1 else "low"
            ),
        }


class AdversarialTester:
    """Test system against adversarial attacks."""

    def __init__(self, model: nn.Module, attack_methods: List[str] = None):
        self.model = model
        self.attack_methods = attack_methods or ["fgsm", "pgd", "carlini_wagner"]
        self.attack_results = []

    def fgsm_attack(
        self, state: torch.Tensor, target: torch.Tensor, epsilon: float = 0.1
    ) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        state.requires_grad_(True)
        output = self.model(state.unsqueeze(0))

        if output.dim() > 1:
            loss = F.cross_entropy(output, target.unsqueeze(0))
        else:
            loss = F.mse_loss(output, target.unsqueeze(0))

        grad = torch.autograd.grad(loss, state, create_graph=False)[0]
        adversarial_state = state + epsilon * grad.sign()

        return adversarial_state

    def pgd_attack(
        self,
        state: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 0.1,
        num_steps: int = 10,
        step_size: float = 0.01,
    ) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        adversarial_state = state.clone()

        for _ in range(num_steps):
            adversarial_state.requires_grad_(True)
            output = self.model(adversarial_state.unsqueeze(0))

            if output.dim() > 1:
                loss = F.cross_entropy(output, target.unsqueeze(0))
            else:
                loss = F.mse_loss(output, target.unsqueeze(0))

            grad = torch.autograd.grad(loss, adversarial_state, create_graph=False)[0]

            # Update adversarial state
            adversarial_state = adversarial_state + step_size * grad.sign()

            # Project back to epsilon ball
            adversarial_state = torch.clamp(
                adversarial_state, state - epsilon, state + epsilon
            )

        return adversarial_state

    def test_attack(
        self,
        test_states: torch.Tensor,
        test_targets: torch.Tensor,
        attack_method: str,
        epsilon: float = 0.1,
    ) -> Dict[str, Any]:
        """Test a specific attack method."""
        results = {
            "attack_method": attack_method,
            "epsilon": epsilon,
            "successful_attacks": 0,
            "total_attacks": len(test_states),
            "attack_success_rate": 0.0,
            "avg_perturbation": 0.0,
        }

        perturbations = []

        for i, (state, target) in enumerate(zip(test_states, test_targets)):
            try:
                if attack_method == "fgsm":
                    adversarial_state = self.fgsm_attack(state, target, epsilon)
                elif attack_method == "pgd":
                    adversarial_state = self.pgd_attack(state, target, epsilon)
                else:
                    continue

                # Calculate perturbation
                perturbation = torch.norm(adversarial_state - state).item()
                perturbations.append(perturbation)

                # Test if attack is successful
                with torch.no_grad():
                    original_output = self.model(state.unsqueeze(0))
                    adversarial_output = self.model(adversarial_state.unsqueeze(0))

                if original_output.dim() > 1:
                    original_action = torch.argmax(original_output, dim=-1)
                    adversarial_action = torch.argmax(adversarial_output, dim=-1)
                    if original_action != adversarial_action:
                        results["successful_attacks"] += 1
                else:
                    if torch.abs(original_output - adversarial_output).item() > epsilon:
                        results["successful_attacks"] += 1

            except Exception as e:
                logging.error(f"Attack failed for sample {i}: {e}")
                continue

        results["attack_success_rate"] = (
            results["successful_attacks"] / results["total_attacks"]
        )
        results["avg_perturbation"] = np.mean(perturbations) if perturbations else 0.0

        return results


class FailSafeMechanism:
    """Implement fail-safe mechanisms for RL systems."""

    def __init__(self, safety_monitor: SafetyMonitor):
        self.safety_monitor = safety_monitor
        self.fail_safe_actions = []
        self.emergency_stop = False
        self.fail_safe_history = []

    def check_fail_safe_condition(
        self, state: np.ndarray, action: int, confidence: float
    ) -> bool:
        """Check if fail-safe condition is triggered."""
        # Check safety
        if not self.safety_monitor.check_safety(state, action, confidence):
            return True

        # Check for emergency stop
        if self.emergency_stop:
            return True

        # Check for critical violations
        if self.safety_monitor.safety_metrics["critical_violations"] > 10:
            return True

        return False

    def trigger_fail_safe(self, state: np.ndarray, reason: str) -> int:
        """Trigger fail-safe mechanism."""
        fail_safe_action = self._get_fail_safe_action(state)

        # Record fail-safe event
        fail_safe_event = {
            "timestamp": time.time(),
            "reason": reason,
            "state": state.copy(),
            "action": fail_safe_action,
            "safety_score": self.safety_monitor.safety_metrics["safety_score"],
        }
        self.fail_safe_history.append(fail_safe_event)

        return fail_safe_action

    def _get_fail_safe_action(self, state: np.ndarray) -> int:
        """Get the appropriate fail-safe action."""
        # Simple fail-safe: return to safe state or stop
        # In practice, this would be more sophisticated
        return 0  # Stop action

    def reset_fail_safe(self):
        """Reset fail-safe mechanism."""
        self.emergency_stop = False
        self.fail_safe_history.clear()


class SafetyGuarantee:
    """Provide formal safety guarantees for RL systems."""

    def __init__(self, safety_constraints: List[Callable]):
        self.safety_constraints = safety_constraints
        self.guarantee_history = []

    def verify_safety_guarantee(self, state: np.ndarray, action: int) -> bool:
        """Verify that the action satisfies safety guarantees."""
        for constraint in self.safety_constraints:
            if not constraint(state, action):
                return False
        return True

    def get_safety_guarantee(self, state: np.ndarray, action: int) -> Dict[str, Any]:
        """Get formal safety guarantee for the action."""
        guarantee = {
            "timestamp": time.time(),
            "state": state.copy(),
            "action": action,
            "satisfies_constraints": self.verify_safety_guarantee(state, action),
            "constraint_violations": [],
        }

        # Check each constraint
        for i, constraint in enumerate(self.safety_constraints):
            if not constraint(state, action):
                guarantee["constraint_violations"].append(i)

        self.guarantee_history.append(guarantee)
        return guarantee


class FormalVerifier:
    """Formal verification of RL system properties."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.verification_results = []

    def verify_reachability(
        self,
        initial_states: torch.Tensor,
        target_states: torch.Tensor,
        max_steps: int = 100,
    ) -> Dict[str, Any]:
        """Verify reachability property."""
        results = {
            "reachable_states": 0,
            "unreachable_states": 0,
            "reachability_rate": 0.0,
            "avg_steps_to_reach": 0.0,
        }

        steps_to_reach = []

        for initial_state in initial_states:
            current_state = initial_state.clone()
            steps = 0

            while steps < max_steps:
                # Get action from model
                with torch.no_grad():
                    action_logits = self.model(current_state.unsqueeze(0))
                    action = torch.argmax(action_logits, dim=-1)

                # Simulate state transition (simplified)
                # In practice, this would use the actual environment dynamics
                current_state = current_state + 0.1 * torch.randn_like(current_state)

                # Check if target is reached
                if torch.norm(current_state - target_states[0]) < 0.1:
                    results["reachable_states"] += 1
                    steps_to_reach.append(steps)
                    break

                steps += 1
            else:
                results["unreachable_states"] += 1

        results["reachability_rate"] = results["reachable_states"] / len(initial_states)
        results["avg_steps_to_reach"] = (
            np.mean(steps_to_reach) if steps_to_reach else 0.0
        )

        return results

    def verify_safety_invariant(
        self, states: torch.Tensor, safety_function: Callable
    ) -> Dict[str, Any]:
        """Verify safety invariant property."""
        results = {
            "safe_states": 0,
            "unsafe_states": 0,
            "safety_rate": 0.0,
        }

        for state in states:
            if safety_function(state):
                results["safe_states"] += 1
            else:
                results["unsafe_states"] += 1

        results["safety_rate"] = results["safe_states"] / len(states)
        return results
