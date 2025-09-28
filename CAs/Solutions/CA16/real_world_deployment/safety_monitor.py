"""
Safety Monitor

This module provides safety monitoring and risk assessment for RL systems,
including constraint checking and safety validation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
from collections import deque
import threading
import time
import logging


class SafetyConstraints:
    """
    Safety constraints for RL systems.

    Defines acceptable ranges and forbidden states/actions.
    """

    def __init__(self):
        self.state_bounds = {}  # {'state_var': (min, max)}

        self.action_bounds = {}  # {'action_var': (min, max)}
        self.forbidden_actions = set()

        self.safety_rules = []  # List of callable safety rules

        self.risk_thresholds = {"high_risk": 0.8, "medium_risk": 0.5, "low_risk": 0.2}

    def add_state_constraint(self, variable: str, min_val: float, max_val: float):
        """Add constraint on state variable."""
        self.state_bounds[variable] = (min_val, max_val)

    def add_action_constraint(self, variable: str, min_val: float, max_val: float):
        """Add constraint on action variable."""
        self.action_bounds[variable] = (min_val, max_val)

    def add_forbidden_action(self, action):
        """Add forbidden action."""
        self.forbidden_actions.add(action)

    def add_safety_rule(self, rule: Callable[[Dict[str, Any]], bool], description: str):
        """
        Add safety rule.

        Args:
            rule: Function that takes state dict and returns True if safe
            description: Description of the rule
        """
        self.safety_rules.append((rule, description))

    def check_state_constraints(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if state satisfies constraints.

        Args:
            state: State dictionary

        Returns:
            (is_safe, violation_messages)
        """
        violations = []

        for var, (min_val, max_val) in self.state_bounds.items():
            if var in state:
                val = state[var]
                if not (min_val <= val <= max_val):
                    violations.append(
                        f"State {var}={val} out of bounds [{min_val}, {max_val}]"
                    )

        return len(violations) == 0, violations

    def check_action_constraints(self, action: Any) -> Tuple[bool, List[str]]:
        """
        Check if action satisfies constraints.

        Args:
            action: Action value

        Returns:
            (is_safe, violation_messages)
        """
        violations = []

        if action in self.forbidden_actions:
            violations.append(f"Action {action} is forbidden")

        if isinstance(action, dict):
            for var, (min_val, max_val) in self.action_bounds.items():
                if var in action:
                    val = action[var]
                    if not (min_val <= val <= max_val):
                        violations.append(
                            f"Action {var}={val} out of bounds [{min_val}, {max_val}]"
                        )

        return len(violations) == 0, violations

    def check_safety_rules(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check safety rules.

        Args:
            state: State dictionary

        Returns:
            (is_safe, violation_messages)
        """
        violations = []

        for rule, description in self.safety_rules:
            try:
                if not rule(state):
                    violations.append(f"Safety rule violation: {description}")
            except Exception as e:
                violations.append(f"Safety rule error: {description} - {e}")

        return len(violations) == 0, violations

    def assess_risk(self, state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        Assess risk level of state-action pair.

        Args:
            state: Current state
            action: Proposed action

        Returns:
            Risk assessment dictionary
        """
        risk_score = 0.0
        risk_factors = []

        state_safe, state_violations = self.check_state_constraints(state)
        if not state_safe:
            risk_score += 0.4
            risk_factors.extend(state_violations)

        action_safe, action_violations = self.check_action_constraints(action)
        if not action_safe:
            risk_score += 0.4
            risk_factors.extend(action_violations)

        rules_safe, rule_violations = self.check_safety_rules(state)
        if not rules_safe:
            risk_score += 0.2
            risk_factors.extend(rule_violations)

        if risk_score >= self.risk_thresholds["high_risk"]:
            risk_level = "high"
        elif risk_score >= self.risk_thresholds["medium_risk"]:
            risk_level = "medium"
        elif risk_score >= self.risk_thresholds["low_risk"]:
            risk_level = "low"
        else:
            risk_level = "safe"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "is_safe": risk_score < self.risk_thresholds["medium_risk"],
        }


class RiskAssessor:
    """
    Risk assessment system using machine learning models.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, device: str = "cpu"):
        self.device = device

        self.risk_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output risk probability
        ).to(device)

        self.optimizer = torch.optim.Adam(self.risk_model.parameters(), lr=1e-3)
        self.risk_history = deque(maxlen=1000)

    def assess_risk(self, state: torch.Tensor) -> float:
        """
        Assess risk of a state using ML model.

        Args:
            state: State tensor

        Returns:
            Risk probability (0-1)
        """
        self.risk_model.eval()
        with torch.no_grad():
            state = (
                state.to(self.device).unsqueeze(0)
                if state.dim() == 1
                else state.to(self.device)
            )
            risk_prob = self.risk_model(state).item()

        self.risk_history.append(risk_prob)
        return risk_prob

    def update_risk_model(self, states: torch.Tensor, risk_labels: torch.Tensor):
        """
        Update risk assessment model.

        Args:
            states: Batch of states
            risk_labels: Corresponding risk labels (0-1)
        """
        self.risk_model.train()

        states = states.to(self.device)
        risk_labels = risk_labels.to(self.device)

        self.optimizer.zero_grad()
        predicted_risks = self.risk_model(states).squeeze()
        loss = F.binary_cross_entropy(predicted_risks, risk_labels)

        loss.backward()
        self.optimizer.step()

    def get_risk_stats(self) -> Dict[str, float]:
        """Get risk assessment statistics."""
        if not self.risk_history:
            return {"mean_risk": 0.0, "max_risk": 0.0, "risk_variance": 0.0}

        risks = list(self.risk_history)
        return {
            "mean_risk": np.mean(risks),
            "max_risk": np.max(risks),
            "risk_variance": np.var(risks),
            "high_risk_count": sum(1 for r in risks if r > 0.8),
        }


class SafetyMonitor:
    """
    Safety monitoring system for RL agents.

    Monitors agent behavior, detects unsafe actions, and provides safety interventions.
    """

    def __init__(
        self,
        constraints: SafetyConstraints,
        risk_assessor: Optional[RiskAssessor] = None,
    ):
        self.constraints = constraints
        self.risk_assessor = risk_assessor

        self.monitoring_active = False
        self.violation_count = 0
        self.intervention_count = 0
        self.monitoring_history = deque(maxlen=10000)

        self.alert_threshold = 5  # Violations per minute
        self.alert_callbacks = []

        self.logger = logging.getLogger("SafetyMonitor")
        self.logger.setLevel(logging.WARNING)

        self.monitor_thread = None

    def start_monitoring(self):
        """Start safety monitoring."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop safety monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def check_safety(self, state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        Check safety of state-action pair.

        Args:
            state: Current state
            action: Proposed action

        Returns:
            Safety assessment
        """
        rule_assessment = self.constraints.assess_risk(state, action)

        ml_risk = 0.0
        if self.risk_assessor is not None and isinstance(state, torch.Tensor):
            ml_risk = self.risk_assessor.assess_risk(state)

        combined_risk = max(rule_assessment["risk_score"], ml_risk)
        is_safe = combined_risk < self.constraints.risk_thresholds["medium_risk"]

        assessment = {
            "is_safe": is_safe,
            "risk_score": combined_risk,
            "rule_based_risk": rule_assessment["risk_score"],
            "ml_risk": ml_risk,
            "risk_factors": rule_assessment["risk_factors"],
            "intervention_needed": not is_safe,
        }

        self.monitoring_history.append(
            {
                "timestamp": time.time(),
                "state": state.copy() if isinstance(state, dict) else state,
                "action": action,
                "assessment": assessment.copy(),
            }
        )

        if not is_safe:
            self.violation_count += 1
            self.logger.warning(
                f"Safety violation detected: {assessment['risk_factors']}"
            )

        return assessment

    def intervene(self, state: Dict[str, Any]) -> Any:
        """
        Provide safe intervention action.

        Args:
            state: Current state

        Returns:
            Safe action
        """
        self.intervention_count += 1

        if isinstance(state, dict):
            return self._get_domain_safe_action(state)
        else:
            return 0  # Default safe action

    def _get_domain_safe_action(self, state: Dict[str, Any]) -> Any:
        """Get domain-specific safe action."""
        return {}  # Default safe action dict

    def _monitoring_loop(self):
        """Monitoring loop for periodic checks."""
        last_check = time.time()
        last_violation_count = 0

        while self.monitoring_active:
            current_time = time.time()

            if current_time - last_check >= 60:
                violations_per_minute = self.violation_count - last_violation_count

                if violations_per_minute >= self.alert_threshold:
                    self._trigger_alerts(violations_per_minute)

                last_check = current_time
                last_violation_count = self.violation_count

            time.sleep(1)  # Check every second

    def _trigger_alerts(self, violation_rate: float):
        """Trigger safety alerts."""
        alert_message = (
            f"High safety violation rate: {violation_rate} violations/minute"
        )

        self.logger.critical(alert_message)

        for callback in self.alert_callbacks:
            try:
                callback(alert_message, violation_rate)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")

    def add_alert_callback(self, callback: Callable[[str, float], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        recent_violations = sum(
            1
            for entry in list(self.monitoring_history)[-100:]
            if not entry["assessment"]["is_safe"]
        )

        return {
            "total_violations": self.violation_count,
            "total_interventions": self.intervention_count,
            "monitoring_active": self.monitoring_active,
            "recent_violations": recent_violations,
            "violation_rate": self.violation_count
            / max(1, len(self.monitoring_history)),
            "history_size": len(self.monitoring_history),
        }

    def export_violations(self, filepath: str):
        """Export violation history to file."""
        violations = [
            entry
            for entry in self.monitoring_history
            if not entry["assessment"]["is_safe"]
        ]

        with open(filepath, "w") as f:
            for violation in violations:
                f.write(f"{violation}\n")
