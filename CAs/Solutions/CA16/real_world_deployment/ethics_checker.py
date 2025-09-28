"""
Ethics Checker

This module provides ethical evaluation and bias detection for RL systems,
ensuring fair and responsible AI behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
from collections import defaultdict, deque
import threading
import time
import logging


class BiasDetector:
    """
    Bias detection system for RL policies.

    Detects and measures bias in decision-making across different groups or contexts.
    """

    def __init__(self, protected_attributes: List[str], device: str = "cpu"):
        self.protected_attributes = protected_attributes
        self.device = device

        # Bias tracking
        self.decision_history = defaultdict(list)
        self.bias_metrics = {}

        # Statistical tests
        self.significance_threshold = 0.05

    def record_decision(
        self,
        protected_attr_values: Dict[str, Any],
        decision: Any,
        outcome: Optional[Any] = None,
    ):
        """
        Record a decision for bias analysis.

        Args:
            protected_attr_values: Values of protected attributes
            decision: Decision made
            outcome: Outcome of the decision (optional)
        """
        for attr in self.protected_attributes:
            if attr in protected_attr_values:
                key = (attr, protected_attr_values[attr])
                self.decision_history[key].append(
                    {"decision": decision, "outcome": outcome, "timestamp": time.time()}
                )

    def compute_bias_metrics(self) -> Dict[str, Any]:
        """
        Compute bias metrics across protected attributes.

        Returns:
            Dictionary of bias metrics
        """
        metrics = {}

        for attr in self.protected_attributes:
            attr_metrics = self._compute_attribute_bias(attr)
            metrics[attr] = attr_metrics

        self.bias_metrics = metrics
        return metrics

    def _compute_attribute_bias(self, attribute: str) -> Dict[str, Any]:
        """Compute bias metrics for a specific attribute."""
        attr_data = {}

        # Group decisions by attribute value
        for key, decisions in self.decision_history.items():
            attr_name, attr_value = key
            if attr_name == attribute:
                attr_data[attr_value] = decisions

        if len(attr_data) < 2:
            return {"insufficient_data": True}

        # Compute decision distribution differences
        decision_distributions = {}
        outcome_distributions = {}

        for attr_value, decisions in attr_data.items():
            # Decision distribution
            decision_counts = defaultdict(int)
            outcome_counts = defaultdict(int)

            for d in decisions:
                decision_counts[d["decision"]] += 1
                if d["outcome"] is not None:
                    outcome_counts[d["outcome"]] += 1

            total_decisions = len(decisions)
            decision_distributions[attr_value] = {
                k: v / total_decisions for k, v in decision_counts.items()
            }

            if outcome_counts:
                total_outcomes = sum(outcome_counts.values())
                outcome_distributions[attr_value] = {
                    k: v / total_outcomes for k, v in outcome_counts.items()
                }

        # Statistical tests for bias
        bias_indicators = self._statistical_bias_tests(
            decision_distributions, outcome_distributions
        )

        return {
            "group_sizes": {k: len(v) for k, v in attr_data.items()},
            "decision_distributions": decision_distributions,
            "outcome_distributions": outcome_distributions,
            "bias_indicators": bias_indicators,
        }

    def _statistical_bias_tests(
        self, decision_dist: Dict, outcome_dist: Dict
    ) -> Dict[str, Any]:
        """Perform statistical tests for bias detection."""
        # Simplified bias detection
        # In practice, would use proper statistical tests (chi-square, etc.)

        indicators = {
            "decision_disparity": self._compute_distribution_disparity(decision_dist),
            "outcome_disparity": (
                self._compute_distribution_disparity(outcome_dist)
                if outcome_dist
                else 0.0
            ),
            "representation_bias": self._compute_representation_bias(decision_dist),
        }

        return indicators

    def _compute_distribution_disparity(self, distributions: Dict) -> float:
        """Compute disparity between distributions."""
        if len(distributions) < 2:
            return 0.0

        # Simple disparity measure (max difference in decision rates)
        all_decisions = set()
        for dist in distributions.values():
            all_decisions.update(dist.keys())

        max_disparity = 0.0
        for decision in all_decisions:
            rates = [dist.get(decision, 0.0) for dist in distributions.values()]
            disparity = max(rates) - min(rates)
            max_disparity = max(max_disparity, disparity)

        return max_disparity

    def _compute_representation_bias(self, distributions: Dict) -> float:
        """Compute representation bias."""
        # Check if decisions are equally distributed across groups
        group_sizes = [sum(dist.values()) for dist in distributions.values()]

        if not group_sizes:
            return 0.0

        expected_size = np.mean(group_sizes)
        disparities = [
            abs(size - expected_size) / expected_size for size in group_sizes
        ]

        return np.mean(disparities)

    def get_bias_report(self) -> Dict[str, Any]:
        """Generate comprehensive bias report."""
        if not self.bias_metrics:
            self.compute_bias_metrics()

        report = {
            "timestamp": time.time(),
            "protected_attributes": self.protected_attributes,
            "bias_metrics": self.bias_metrics,
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate bias mitigation recommendations."""
        recommendations = []

        for attr, metrics in self.bias_metrics.items():
            if "bias_indicators" in metrics:
                indicators = metrics["bias_indicators"]

                if indicators.get("decision_disparity", 0) > 0.2:
                    recommendations.append(
                        f"High decision disparity detected for {attr}. "
                        "Consider fairness-aware training."
                    )

                if indicators.get("outcome_disparity", 0) > 0.15:
                    recommendations.append(
                        f"Outcome disparity detected for {attr}. "
                        "Review decision-making process."
                    )

                if indicators.get("representation_bias", 0) > 0.3:
                    recommendations.append(
                        f"Representation bias in {attr}. "
                        "Ensure balanced data collection."
                    )

        if not recommendations:
            recommendations.append("No significant bias detected.")

        return recommendations


class FairnessEvaluator:
    """
    Fairness evaluation system for RL systems.

    Evaluates fairness metrics and provides fairness constraints.
    """

    def __init__(self, fairness_metrics: List[str] = None):
        if fairness_metrics is None:
            fairness_metrics = [
                "demographic_parity",
                "equal_opportunity",
                "fairness_through_awareness",
            ]

        self.fairness_metrics = fairness_metrics
        self.evaluation_history = deque(maxlen=1000)

    def evaluate_fairness(
        self,
        predictions: torch.Tensor,
        sensitive_attributes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Evaluate fairness metrics.

        Args:
            predictions: Model predictions
            sensitive_attributes: Sensitive attribute values
            labels: True labels (for some metrics)

        Returns:
            Fairness metric scores
        """
        scores = {}

        for metric in self.fairness_metrics:
            if metric == "demographic_parity":
                scores[metric] = self._demographic_parity(
                    predictions, sensitive_attributes
                )
            elif metric == "equal_opportunity" and labels is not None:
                scores[metric] = self._equal_opportunity(
                    predictions, labels, sensitive_attributes
                )
            elif metric == "fairness_through_awareness":
                scores[metric] = self._fairness_through_awareness(
                    predictions, sensitive_attributes
                )

        # Record evaluation
        self.evaluation_history.append(
            {
                "timestamp": time.time(),
                "scores": scores.copy(),
                "num_samples": len(predictions),
            }
        )

        return scores

    def _demographic_parity(
        self, predictions: torch.Tensor, sensitive_attrs: torch.Tensor
    ) -> float:
        """Compute demographic parity difference."""
        # P(Y=1 | A=0) - P(Y=1 | A=1)
        groups = torch.unique(sensitive_attrs)

        if len(groups) < 2:
            return 0.0

        group_rates = {}
        for group in groups:
            mask = sensitive_attrs == group
            if mask.sum() > 0:
                group_rates[group.item()] = predictions[mask].mean().item()

        if len(group_rates) >= 2:
            rates = list(group_rates.values())
            return max(rates) - min(rates)

        return 0.0

    def _equal_opportunity(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        sensitive_attrs: torch.Tensor,
    ) -> float:
        """Compute equal opportunity difference."""
        # P(Y=1 | A=0, Y_true=1) - P(Y=1 | A=1, Y_true=1)
        groups = torch.unique(sensitive_attrs)

        if len(groups) < 2:
            return 0.0

        group_rates = {}
        for group in groups:
            mask = (sensitive_attrs == group) & (labels == 1)
            if mask.sum() > 0:
                group_rates[group.item()] = predictions[mask].mean().item()

        if len(group_rates) >= 2:
            rates = list(group_rates.values())
            return max(rates) - min(rates)

        return 0.0

    def _fairness_through_awareness(
        self, predictions: torch.Tensor, sensitive_attrs: torch.Tensor
    ) -> float:
        """Simplified fairness through awareness metric."""
        # This is a complex metric - simplified implementation
        # In practice, would require access to all relevant attributes

        # Simple proxy: correlation between predictions and sensitive attributes
        pred_mean = predictions.mean()
        attr_mean = sensitive_attrs.float().mean()

        correlation = torch.corrcoef(
            torch.stack([predictions, sensitive_attrs.float()])
        )[0, 1]

        return abs(correlation.item())

    def get_fairness_report(self) -> Dict[str, Any]:
        """Generate fairness evaluation report."""
        if not self.evaluation_history:
            return {"error": "No evaluations performed"}

        recent_evals = list(self.evaluation_history)[-10:]  # Last 10 evaluations

        avg_scores = defaultdict(float)
        for eval_data in recent_evals:
            for metric, score in eval_data["scores"].items():
                avg_scores[metric] += score

        for metric in avg_scores:
            avg_scores[metric] /= len(recent_evals)

        report = {
            "timestamp": time.time(),
            "metrics_evaluated": self.fairness_metrics,
            "average_scores": dict(avg_scores),
            "recent_trend": self._compute_fairness_trend(recent_evals),
            "recommendations": self._generate_fairness_recommendations(avg_scores),
        }

        return report

    def _compute_fairness_trend(self, evaluations: List[Dict]) -> Dict[str, str]:
        """Compute trend in fairness metrics."""
        trends = {}

        for metric in self.fairness_metrics:
            scores = [eval_data["scores"].get(metric, 0) for eval_data in evaluations]

            if len(scores) >= 2:
                # Simple trend analysis
                first_half = np.mean(scores[: len(scores) // 2])
                second_half = np.mean(scores[len(scores) // 2 :])

                if second_half > first_half * 1.1:
                    trends[metric] = "worsening"
                elif second_half < first_half * 0.9:
                    trends[metric] = "improving"
                else:
                    trends[metric] = "stable"
            else:
                trends[metric] = "insufficient_data"

        return trends

    def _generate_fairness_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate fairness improvement recommendations."""
        recommendations = []

        for metric, score in scores.items():
            if metric == "demographic_parity" and score > 0.1:
                recommendations.append(
                    "High demographic parity difference. Consider reweighting training data."
                )
            elif metric == "equal_opportunity" and score > 0.1:
                recommendations.append(
                    "Equal opportunity violation detected. Review positive outcome distribution."
                )
            elif metric == "fairness_through_awareness" and score > 0.5:
                recommendations.append(
                    "High correlation with sensitive attributes. Consider adversarial debiasing."
                )

        if not recommendations:
            recommendations.append("Fairness metrics within acceptable ranges.")

        return recommendations


class EthicsChecker:
    """
    Comprehensive ethics checking system for RL systems.

    Combines bias detection, fairness evaluation, and ethical guidelines.
    """

    def __init__(
        self,
        protected_attributes: List[str] = None,
        ethical_guidelines: List[str] = None,
    ):
        self.bias_detector = BiasDetector(
            protected_attributes or ["gender", "race", "age"]
        )
        self.fairness_evaluator = FairnessEvaluator()

        # Ethical guidelines
        if ethical_guidelines is None:
            ethical_guidelines = [
                "beneficence",
                "non_maleficence",
                "autonomy",
                "justice",
                "explicability",
                "fairness",
                "privacy",
                "safety",
            ]

        self.ethical_guidelines = ethical_guidelines
        self.guideline_scores = {guideline: 0.0 for guideline in ethical_guidelines}

        # Ethics monitoring
        self.ethics_violations = []
        self.monitoring_active = False

        # Logging
        self.logger = logging.getLogger("EthicsChecker")
        self.logger.setLevel(logging.INFO)

    def evaluate_ethics(
        self,
        state: Dict[str, Any],
        action: Any,
        predictions: Optional[torch.Tensor] = None,
        sensitive_attrs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive ethics evaluation.

        Args:
            state: Current state
            action: Proposed action
            predictions: Model predictions (optional)
            sensitive_attrs: Sensitive attributes (optional)
            labels: True labels (optional)

        Returns:
            Ethics evaluation results
        """
        results = {
            "timestamp": time.time(),
            "ethical_score": 0.0,
            "violations": [],
            "recommendations": [],
        }

        # Bias evaluation
        if sensitive_attrs is not None and predictions is not None:
            fairness_scores = self.fairness_evaluator.evaluate_fairness(
                predictions, sensitive_attrs, labels
            )
            results["fairness_scores"] = fairness_scores

            # Check for fairness violations
            for metric, score in fairness_scores.items():
                if score > 0.15:  # Threshold for concern
                    results["violations"].append(
                        f"Fairness violation: {metric} = {score:.3f}"
                    )
                    results["ethical_score"] -= 0.2

        # Guideline compliance
        guideline_compliance = self._check_guideline_compliance(state, action)
        results["guideline_compliance"] = guideline_compliance

        for guideline, compliant in guideline_compliance.items():
            if not compliant:
                results["violations"].append(f"Guideline violation: {guideline}")
                results["ethical_score"] -= 0.1

        # Overall ethical score (0-1 scale, higher is better)
        results["ethical_score"] = max(0.0, min(1.0, 0.5 - results["ethical_score"]))

        # Generate recommendations
        results["recommendations"] = self._generate_ethics_recommendations(results)

        # Record violations
        if results["violations"]:
            self.ethics_violations.append(results)

        return results

    def _check_guideline_compliance(
        self, state: Dict[str, Any], action: Any
    ) -> Dict[str, bool]:
        """Check compliance with ethical guidelines."""
        compliance = {}

        # Simplified guideline checks (would be more sophisticated in practice)

        # Beneficence: Does the action provide benefit?
        compliance["beneficence"] = self._check_beneficence(state, action)

        # Non-maleficence: Does the action avoid harm?
        compliance["non_maleficence"] = self._check_non_maleficence(state, action)

        # Autonomy: Does the action respect user autonomy?
        compliance["autonomy"] = self._check_autonomy(state, action)

        # Justice: Is the action fair?
        compliance["justice"] = self._check_justice(state, action)

        # Safety: Is the action safe?
        compliance["safety"] = self._check_safety(state, action)

        # Other guidelines (simplified)
        compliance["explicability"] = True  # Assume explainable by default
        compliance["fairness"] = True  # Checked separately
        compliance["privacy"] = True  # Assume private by default

        return compliance

    def _check_beneficence(self, state: Dict[str, Any], action: Any) -> bool:
        """Check if action provides benefit."""
        # Domain-specific logic would go here
        return True  # Simplified

    def _check_non_maleficence(self, state: Dict[str, Any], action: Any) -> bool:
        """Check if action avoids harm."""
        # Check for obviously harmful actions
        if isinstance(action, dict):
            # Example: check for dangerous speed in autonomous driving
            if "speed" in action and action["speed"] > 50:
                return False
        return True

    def _check_autonomy(self, state: Dict[str, Any], action: Any) -> bool:
        """Check if action respects autonomy."""
        # Check if overriding user preferences
        return True  # Simplified

    def _check_justice(self, state: Dict[str, Any], action: Any) -> bool:
        """Check if action is fair."""
        # Would integrate with fairness evaluator
        return True  # Simplified

    def _check_safety(self, state: Dict[str, Any], action: Any) -> bool:
        """Check if action is safe."""
        # Basic safety checks
        return True  # Simplified

    def _generate_ethics_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generate ethics improvement recommendations."""
        recommendations = []

        if evaluation["ethical_score"] < 0.7:
            recommendations.append("Overall ethical performance needs improvement.")

        if "violations" in evaluation and evaluation["violations"]:
            recommendations.append("Address identified ethical violations immediately.")

        if evaluation.get("fairness_scores"):
            fairness_report = self.fairness_evaluator.get_fairness_report()
            recommendations.extend(fairness_report.get("recommendations", []))

        bias_report = self.bias_detector.get_bias_report()
        recommendations.extend(bias_report.get("recommendations", []))

        return recommendations

    def get_ethics_report(self) -> Dict[str, Any]:
        """Generate comprehensive ethics report."""
        return {
            "timestamp": time.time(),
            "bias_analysis": self.bias_detector.get_bias_report(),
            "fairness_analysis": self.fairness_evaluator.get_fairness_report(),
            "ethical_guidelines": self.ethical_guidelines,
            "guideline_scores": self.guideline_scores,
            "violation_history": self.ethics_violations[-10:],  # Last 10 violations
            "overall_assessment": self._assess_overall_ethics(),
        }

    def _assess_overall_ethics(self) -> Dict[str, Any]:
        """Assess overall ethical performance."""
        total_violations = len(self.ethics_violations)

        if total_violations == 0:
            assessment = "excellent"
            score = 1.0
        elif total_violations < 10:
            assessment = "good"
            score = 0.8
        elif total_violations < 50:
            assessment = "concerning"
            score = 0.6
        else:
            assessment = "critical"
            score = 0.3

        return {
            "assessment": assessment,
            "score": score,
            "total_violations": total_violations,
            "recommendation": self._get_assessment_recommendation(assessment),
        }

    def _get_assessment_recommendation(self, assessment: str) -> str:
        """Get recommendation based on assessment."""
        recommendations = {
            "excellent": "Continue current ethical practices.",
            "good": "Monitor for potential issues and maintain vigilance.",
            "concerning": "Implement additional ethical safeguards and review processes.",
            "critical": "Immediate action required: pause deployment and conduct ethical review.",
        }

        return recommendations.get(
            assessment, "Conduct comprehensive ethical evaluation."
        )
