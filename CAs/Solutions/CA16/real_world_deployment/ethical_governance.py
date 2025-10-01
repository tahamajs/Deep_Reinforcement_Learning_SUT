"""
Ethical Governance and Responsible AI

This module contains classes for implementing ethical considerations,
fairness monitoring, bias detection, and responsible AI practices.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score


@dataclass
class BiasReport:
    """Report of bias detection results."""
    timestamp: float
    bias_type: str
    severity: str
    affected_groups: List[str]
    metrics: Dict[str, float]
    recommendations: List[str]


@dataclass
class FairnessMetrics:
    """Fairness metrics for different groups."""
    group_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    demographic_parity: float
    equalized_odds: float


@dataclass
class PrivacyViolation:
    """Record of a privacy violation."""
    timestamp: float
    violation_type: str
    severity: str
    description: str
    affected_data: str
    mitigation_applied: bool


class FairnessMonitor:
    """Monitor fairness across different demographic groups."""
    
    def __init__(self, protected_attributes: List[str]):
        self.protected_attributes = protected_attributes
        self.fairness_metrics = {}
        self.bias_reports = deque(maxlen=1000)
        self.fairness_thresholds = {
            "demographic_parity": 0.1,
            "equalized_odds": 0.1,
            "accuracy_difference": 0.05,
        }
        
    def compute_fairness_metrics(self, predictions: torch.Tensor, labels: torch.Tensor, 
                                groups: Dict[str, torch.Tensor]) -> Dict[str, FairnessMetrics]:
        """Compute fairness metrics for different groups."""
        metrics = {}
        
        for group_name, group_mask in groups.items():
            group_predictions = predictions[group_mask]
            group_labels = labels[group_mask]
            
            if len(group_predictions) == 0:
                continue
                
            # Convert to numpy for sklearn
            pred_np = group_predictions.cpu().numpy()
            label_np = group_labels.cpu().numpy()
            
            # Handle binary classification
            if pred_np.ndim > 1:
                pred_np = np.argmax(pred_np, axis=1)
                
            # Compute metrics
            accuracy = accuracy_score(label_np, pred_np)
            precision = precision_score(label_np, pred_np, average='weighted', zero_division=0)
            recall = recall_score(label_np, pred_np, average='weighted', zero_division=0)
            f1 = f1_score(label_np, pred_np, average='weighted', zero_division=0)
            
            # Compute confusion matrix
            cm = confusion_matrix(label_np, pred_np)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            else:
                fpr = 0.0
                fnr = 0.0
                
            # Compute demographic parity and equalized odds
            demographic_parity = self._compute_demographic_parity(predictions, groups, group_name)
            equalized_odds = self._compute_equalized_odds(predictions, labels, groups, group_name)
            
            metrics[group_name] = FairnessMetrics(
                group_name=group_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                false_positive_rate=fpr,
                false_negative_rate=fnr,
                demographic_parity=demographic_parity,
                equalized_odds=equalized_odds,
            )
            
        return metrics
        
    def _compute_demographic_parity(self, predictions: torch.Tensor, groups: Dict[str, torch.Tensor], 
                                   target_group: str) -> float:
        """Compute demographic parity for a target group."""
        if target_group not in groups:
            return 0.0
            
        target_group_predictions = predictions[groups[target_group]]
        other_predictions = torch.cat([predictions[groups[g]] for g in groups if g != target_group])
        
        if len(target_group_predictions) == 0 or len(other_predictions) == 0:
            return 0.0
            
        # Compute positive prediction rates
        target_positive_rate = (target_group_predictions > 0.5).float().mean().item()
        other_positive_rate = (other_predictions > 0.5).float().mean().item()
        
        return abs(target_positive_rate - other_positive_rate)
        
    def _compute_equalized_odds(self, predictions: torch.Tensor, labels: torch.Tensor, 
                               groups: Dict[str, torch.Tensor], target_group: str) -> float:
        """Compute equalized odds for a target group."""
        if target_group not in groups:
            return 0.0
            
        target_group_predictions = predictions[groups[target_group]]
        target_group_labels = labels[groups[target_group]]
        other_predictions = torch.cat([predictions[groups[g]] for g in groups if g != target_group])
        other_labels = torch.cat([labels[groups[g]] for g in groups if g != target_group])
        
        if len(target_group_predictions) == 0 or len(other_predictions) == 0:
            return 0.0
            
        # Compute TPR and FPR for both groups
        target_tpr = self._compute_tpr(target_group_predictions, target_group_labels)
        target_fpr = self._compute_fpr(target_group_predictions, target_group_labels)
        other_tpr = self._compute_tpr(other_predictions, other_labels)
        other_fpr = self._compute_fpr(other_predictions, other_labels)
        
        # Equalized odds is the maximum difference in TPR and FPR
        tpr_diff = abs(target_tpr - other_tpr)
        fpr_diff = abs(target_fpr - other_fpr)
        
        return max(tpr_diff, fpr_diff)
        
    def _compute_tpr(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute true positive rate."""
        if len(predictions) == 0:
            return 0.0
            
        pred_binary = (predictions > 0.5).float()
        true_positives = ((pred_binary == 1) & (labels == 1)).float().sum()
        total_positives = (labels == 1).float().sum()
        
        return (true_positives / total_positives).item() if total_positives > 0 else 0.0
        
    def _compute_fpr(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute false positive rate."""
        if len(predictions) == 0:
            return 0.0
            
        pred_binary = (predictions > 0.5).float()
        false_positives = ((pred_binary == 1) & (labels == 0)).float().sum()
        total_negatives = (labels == 0).float().sum()
        
        return (false_positives / total_negatives).item() if total_negatives > 0 else 0.0
        
    def detect_bias(self, fairness_metrics: Dict[str, FairnessMetrics]) -> List[BiasReport]:
        """Detect bias based on fairness metrics."""
        bias_reports = []
        
        # Check demographic parity
        for group_name, metrics in fairness_metrics.items():
            if metrics.demographic_parity > self.fairness_thresholds["demographic_parity"]:
                bias_reports.append(BiasReport(
                    timestamp=time.time(),
                    bias_type="demographic_parity",
                    severity="high" if metrics.demographic_parity > 0.2 else "medium",
                    affected_groups=[group_name],
                    metrics={"demographic_parity": metrics.demographic_parity},
                    recommendations=["Consider rebalancing training data", "Apply fairness constraints"],
                ))
                
            # Check equalized odds
            if metrics.equalized_odds > self.fairness_thresholds["equalized_odds"]:
                bias_reports.append(BiasReport(
                    timestamp=time.time(),
                    bias_type="equalized_odds",
                    severity="high" if metrics.equalized_odds > 0.2 else "medium",
                    affected_groups=[group_name],
                    metrics={"equalized_odds": metrics.equalized_odds},
                    recommendations=["Implement equalized odds constraints", "Review feature selection"],
                ))
                
        return bias_reports
        
    def get_fairness_report(self) -> Dict[str, Any]:
        """Get comprehensive fairness report."""
        return {
            "fairness_metrics": self.fairness_metrics,
            "bias_reports": list(self.bias_reports),
            "fairness_thresholds": self.fairness_thresholds,
            "recommendations": self._generate_fairness_recommendations(),
        }
        
    def _generate_fairness_recommendations(self) -> List[str]:
        """Generate fairness recommendations."""
        recommendations = []
        
        if len(self.bias_reports) > 5:
            recommendations.append("Consider implementing fairness constraints in training")
            
        if any(report.severity == "high" for report in self.bias_reports):
            recommendations.append("Review model architecture and training data")
            
        return recommendations


class BiasDetector:
    """Detect various types of bias in ML models."""
    
    def __init__(self, bias_types: List[str] = None):
        self.bias_types = bias_types or ["demographic", "historical", "measurement", "aggregation"]
        self.detection_results = []
        
    def detect_demographic_bias(self, predictions: torch.Tensor, labels: torch.Tensor, 
                               demographic_groups: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Detect demographic bias."""
        results = {
            "bias_detected": False,
            "bias_metrics": {},
            "affected_groups": [],
        }
        
        for group_name, group_mask in demographic_groups.items():
            group_predictions = predictions[group_mask]
            group_labels = labels[group_mask]
            
            if len(group_predictions) == 0:
                continue
                
            # Compute group-specific metrics
            group_accuracy = (group_predictions.argmax(dim=-1) == group_labels).float().mean().item()
            group_positive_rate = (group_predictions.argmax(dim=-1) == 1).float().mean().item()
            
            results["bias_metrics"][group_name] = {
                "accuracy": group_accuracy,
                "positive_rate": group_positive_rate,
            }
            
            # Check for significant differences
            if group_accuracy < 0.7:  # Threshold for bias detection
                results["bias_detected"] = True
                results["affected_groups"].append(group_name)
                
        return results
        
    def detect_historical_bias(self, training_data: torch.Tensor, historical_data: torch.Tensor) -> Dict[str, Any]:
        """Detect historical bias in training data."""
        results = {
            "bias_detected": False,
            "distribution_difference": 0.0,
            "recommendations": [],
        }
        
        # Compare distributions
        training_mean = training_data.mean(dim=0)
        historical_mean = historical_data.mean(dim=0)
        
        distribution_difference = torch.norm(training_mean - historical_mean).item()
        results["distribution_difference"] = distribution_difference
        
        if distribution_difference > 0.5:  # Threshold for bias detection
            results["bias_detected"] = True
            results["recommendations"].append("Consider data augmentation to reduce historical bias")
            results["recommendations"].append("Review data collection methods")
            
        return results
        
    def detect_measurement_bias(self, predictions: torch.Tensor, ground_truth: torch.Tensor, 
                               measurement_groups: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Detect measurement bias."""
        results = {
            "bias_detected": False,
            "measurement_errors": {},
            "recommendations": [],
        }
        
        for group_name, group_mask in measurement_groups.items():
            group_predictions = predictions[group_mask]
            group_ground_truth = ground_truth[group_mask]
            
            if len(group_predictions) == 0:
                continue
                
            # Compute measurement error
            measurement_error = F.mse_loss(group_predictions, group_ground_truth).item()
            results["measurement_errors"][group_name] = measurement_error
            
            # Check for significant measurement bias
            if measurement_error > 0.1:  # Threshold for bias detection
                results["bias_detected"] = True
                results["recommendations"].append(f"Review measurement methods for group {group_name}")
                
        return results
        
    def detect_aggregation_bias(self, individual_predictions: List[torch.Tensor], 
                               aggregated_predictions: torch.Tensor) -> Dict[str, Any]:
        """Detect aggregation bias."""
        results = {
            "bias_detected": False,
            "aggregation_error": 0.0,
            "recommendations": [],
        }
        
        # Compute expected aggregation
        expected_aggregation = torch.stack(individual_predictions).mean(dim=0)
        
        # Compare with actual aggregation
        aggregation_error = F.mse_loss(aggregated_predictions, expected_aggregation).item()
        results["aggregation_error"] = aggregation_error
        
        if aggregation_error > 0.05:  # Threshold for bias detection
            results["bias_detected"] = True
            results["recommendations"].append("Review aggregation methods")
            results["recommendations"].append("Consider weighted aggregation")
            
        return results


class PrivacyProtector:
    """Protect privacy in RL systems."""
    
    def __init__(self, privacy_budget: float = 1.0):
        self.privacy_budget = privacy_budget
        self.privacy_violations = deque(maxlen=1000)
        self.privacy_metrics = {
            "total_violations": 0,
            "privacy_budget_used": 0.0,
            "remaining_budget": privacy_budget,
        }
        
    def add_differential_privacy(self, data: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
        """Add differential privacy noise to data."""
        # Check privacy budget
        if self.privacy_metrics["remaining_budget"] < epsilon:
            raise ValueError("Insufficient privacy budget")
            
        # Add Laplace noise for differential privacy
        noise = torch.distributions.Laplace(0, 1/epsilon).sample(data.shape)
        private_data = data + noise
        
        # Update privacy budget
        self.privacy_metrics["privacy_budget_used"] += epsilon
        self.privacy_metrics["remaining_budget"] -= epsilon
        
        return private_data
        
    def detect_privacy_violation(self, original_data: torch.Tensor, reconstructed_data: torch.Tensor, 
                                threshold: float = 0.1) -> bool:
        """Detect potential privacy violations."""
        # Compute reconstruction error
        reconstruction_error = F.mse_loss(original_data, reconstructed_data).item()
        
        if reconstruction_error < threshold:
            # Potential privacy violation
            violation = PrivacyViolation(
                timestamp=time.time(),
                violation_type="reconstruction_attack",
                severity="high" if reconstruction_error < 0.05 else "medium",
                description=f"Data reconstruction error: {reconstruction_error}",
                affected_data="sensitive_features",
                mitigation_applied=False,
            )
            
            self.privacy_violations.append(violation)
            self.privacy_metrics["total_violations"] += 1
            
            return True
            
        return False
        
    def apply_k_anonymity(self, data: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Apply k-anonymity to data."""
        # Simple k-anonymity implementation
        # In practice, this would be more sophisticated
        
        # Group similar records
        grouped_data = []
        for i in range(0, len(data), k):
            group = data[i:i+k]
            if len(group) >= k:
                # Use group mean for anonymization
                group_mean = group.mean(dim=0, keepdim=True)
                grouped_data.append(group_mean.repeat(len(group), 1))
                
        if grouped_data:
            return torch.cat(grouped_data, dim=0)
        else:
            return data
            
    def get_privacy_report(self) -> Dict[str, Any]:
        """Get privacy protection report."""
        return {
            "privacy_metrics": self.privacy_metrics,
            "privacy_violations": list(self.privacy_violations),
            "recommendations": self._generate_privacy_recommendations(),
        }
        
    def _generate_privacy_recommendations(self) -> List[str]:
        """Generate privacy recommendations."""
        recommendations = []
        
        if self.privacy_metrics["total_violations"] > 5:
            recommendations.append("Consider increasing privacy budget")
            
        if self.privacy_metrics["remaining_budget"] < 0.1:
            recommendations.append("Privacy budget nearly exhausted")
            
        return recommendations


class RegulatoryCompliance:
    """Ensure compliance with AI regulations."""
    
    def __init__(self, regulations: List[str] = None):
        self.regulations = regulations or ["GDPR", "CCPA", "AI_ACT", "HIGHLY_AI"]
        self.compliance_status = {}
        self.compliance_history = deque(maxlen=1000)
        
    def check_gdpr_compliance(self, data_processing: Dict[str, Any]) -> Dict[str, Any]:
        """Check GDPR compliance."""
        compliance = {
            "compliant": True,
            "violations": [],
            "recommendations": [],
        }
        
        # Check data minimization
        if data_processing.get("data_amount", 0) > data_processing.get("purpose_required", 0):
            compliance["compliant"] = False
            compliance["violations"].append("Data minimization violation")
            compliance["recommendations"].append("Reduce data collection to minimum necessary")
            
        # Check purpose limitation
        if not data_processing.get("purpose_specified", False):
            compliance["compliant"] = False
            compliance["violations"].append("Purpose limitation violation")
            compliance["recommendations"].append("Specify clear purpose for data processing")
            
        # Check consent
        if not data_processing.get("consent_obtained", False):
            compliance["compliant"] = False
            compliance["violations"].append("Consent violation")
            compliance["recommendations"].append("Obtain explicit consent for data processing")
            
        return compliance
        
    def check_ccpa_compliance(self, data_processing: Dict[str, Any]) -> Dict[str, Any]:
        """Check CCPA compliance."""
        compliance = {
            "compliant": True,
            "violations": [],
            "recommendations": [],
        }
        
        # Check right to know
        if not data_processing.get("disclosure_provided", False):
            compliance["compliant"] = False
            compliance["violations"].append("Right to know violation")
            compliance["recommendations"].append("Provide clear disclosure of data collection")
            
        # Check right to delete
        if not data_processing.get("deletion_mechanism", False):
            compliance["compliant"] = False
            compliance["violations"].append("Right to delete violation")
            compliance["recommendations"].append("Implement data deletion mechanism")
            
        return compliance
        
    def check_ai_act_compliance(self, ai_system: Dict[str, Any]) -> Dict[str, Any]:
        """Check AI Act compliance."""
        compliance = {
            "compliant": True,
            "violations": [],
            "recommendations": [],
        }
        
        # Check risk assessment
        if not ai_system.get("risk_assessment", False):
            compliance["compliant"] = False
            compliance["violations"].append("Risk assessment missing")
            compliance["recommendations"].append("Conduct comprehensive risk assessment")
            
        # Check transparency
        if not ai_system.get("transparency_measures", False):
            compliance["compliant"] = False
            compliance["violations"].append("Transparency violation")
            compliance["recommendations"].append("Implement transparency measures")
            
        # Check human oversight
        if not ai_system.get("human_oversight", False):
            compliance["compliant"] = False
            compliance["violations"].append("Human oversight missing")
            compliance["recommendations"].append("Implement human oversight mechanisms")
            
        return compliance
        
    def get_compliance_report(self) -> Dict[str, Any]:
        """Get comprehensive compliance report."""
        return {
            "regulations": self.regulations,
            "compliance_status": self.compliance_status,
            "compliance_history": list(self.compliance_history),
            "overall_compliance": self._calculate_overall_compliance(),
        }
        
    def _calculate_overall_compliance(self) -> float:
        """Calculate overall compliance score."""
        if not self.compliance_status:
            return 1.0
            
        compliant_count = sum(1 for status in self.compliance_status.values() if status.get("compliant", False))
        total_count = len(self.compliance_status)
        
        return compliant_count / total_count if total_count > 0 else 1.0


class AIGovernance:
    """AI governance framework."""
    
    def __init__(self):
        self.governance_policies = {}
        self.decision_history = deque(maxlen=1000)
        self.audit_trail = deque(maxlen=1000)
        
    def add_governance_policy(self, policy_name: str, policy: Dict[str, Any]):
        """Add a governance policy."""
        self.governance_policies[policy_name] = {
            "policy": policy,
            "created_at": time.time(),
            "last_updated": time.time(),
        }
        
    def evaluate_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a decision against governance policies."""
        evaluation = {
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "risk_level": "low",
        }
        
        # Check against each policy
        for policy_name, policy_info in self.governance_policies.items():
            policy = policy_info["policy"]
            
            # Check decision against policy
            if not self._check_policy_compliance(decision, policy):
                evaluation["compliant"] = False
                evaluation["violations"].append(policy_name)
                evaluation["recommendations"].append(f"Review decision against {policy_name}")
                
        # Determine risk level
        if len(evaluation["violations"]) > 2:
            evaluation["risk_level"] = "high"
        elif len(evaluation["violations"]) > 0:
            evaluation["risk_level"] = "medium"
            
        # Record decision
        decision_record = {
            "timestamp": time.time(),
            "decision": decision,
            "evaluation": evaluation,
        }
        self.decision_history.append(decision_record)
        
        return evaluation
        
    def _check_policy_compliance(self, decision: Dict[str, Any], policy: Dict[str, Any]) -> bool:
        """Check if decision complies with policy."""
        # Simple compliance check
        # In practice, this would be more sophisticated
        
        required_fields = policy.get("required_fields", [])
        for field in required_fields:
            if field not in decision:
                return False
                
        return True
        
    def audit_system(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Audit the AI system."""
        audit_result = {
            "timestamp": time.time(),
            "system_state": system_state,
            "compliance_score": 0.0,
            "issues_found": [],
            "recommendations": [],
        }
        
        # Check system compliance
        compliance_score = 0.0
        total_checks = 0
        
        for policy_name, policy_info in self.governance_policies.items():
            total_checks += 1
            if self._check_system_compliance(system_state, policy_info["policy"]):
                compliance_score += 1
            else:
                audit_result["issues_found"].append(f"Non-compliance with {policy_name}")
                
        audit_result["compliance_score"] = compliance_score / total_checks if total_checks > 0 else 1.0
        
        # Record audit
        self.audit_trail.append(audit_result)
        
        return audit_result
        
    def _check_system_compliance(self, system_state: Dict[str, Any], policy: Dict[str, Any]) -> bool:
        """Check if system state complies with policy."""
        # Simple compliance check
        # In practice, this would be more sophisticated
        
        required_components = policy.get("required_components", [])
        for component in required_components:
            if component not in system_state:
                return False
                
        return True
        
    def get_governance_report(self) -> Dict[str, Any]:
        """Get governance report."""
        return {
            "governance_policies": self.governance_policies,
            "decision_history": list(self.decision_history),
            "audit_trail": list(self.audit_trail),
            "compliance_trend": self._calculate_compliance_trend(),
        }
        
    def _calculate_compliance_trend(self) -> str:
        """Calculate compliance trend over time."""
        if len(self.audit_trail) < 2:
            return "insufficient_data"
            
        recent_audits = list(self.audit_trail)[-5:]
        older_audits = list(self.audit_trail)[-10:-5]
        
        if not older_audits:
            return "insufficient_data"
            
        recent_compliance = np.mean([audit["compliance_score"] for audit in recent_audits])
        older_compliance = np.mean([audit["compliance_score"] for audit in older_audits])
        
        if recent_compliance > older_compliance:
            return "improving"
        elif recent_compliance < older_compliance:
            return "deteriorating"
        else:
            return "stable"


class ResponsibleAIFramework:
    """Comprehensive responsible AI framework."""
    
    def __init__(self):
        self.fairness_monitor = FairnessMonitor(protected_attributes=["gender", "race", "age"])
        self.bias_detector = BiasDetector()
        self.privacy_protector = PrivacyProtector()
        self.regulatory_compliance = RegulatoryCompliance()
        self.ai_governance = AIGovernance()
        
        # Framework metrics
        self.framework_metrics = {
            "total_assessments": 0,
            "fairness_violations": 0,
            "bias_incidents": 0,
            "privacy_violations": 0,
            "compliance_violations": 0,
        }
        
    def assess_system(self, predictions: torch.Tensor, labels: torch.Tensor, 
                     demographic_groups: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Comprehensive system assessment."""
        assessment = {
            "timestamp": time.time(),
            "fairness_assessment": {},
            "bias_assessment": {},
            "privacy_assessment": {},
            "compliance_assessment": {},
            "overall_score": 0.0,
            "recommendations": [],
        }
        
        # Fairness assessment
        fairness_metrics = self.fairness_monitor.compute_fairness_metrics(
            predictions, labels, demographic_groups
        )
        bias_reports = self.fairness_monitor.detect_bias(fairness_metrics)
        assessment["fairness_assessment"] = {
            "metrics": fairness_metrics,
            "bias_reports": bias_reports,
        }
        
        # Bias assessment
        demographic_bias = self.bias_detector.detect_demographic_bias(
            predictions, labels, demographic_groups
        )
        assessment["bias_assessment"] = demographic_bias
        
        # Privacy assessment
        privacy_report = self.privacy_protector.get_privacy_report()
        assessment["privacy_assessment"] = privacy_report
        
        # Compliance assessment
        compliance_report = self.regulatory_compliance.get_compliance_report()
        assessment["compliance_assessment"] = compliance_report
        
        # Calculate overall score
        assessment["overall_score"] = self._calculate_overall_score(assessment)
        
        # Generate recommendations
        assessment["recommendations"] = self._generate_recommendations(assessment)
        
        # Update framework metrics
        self.framework_metrics["total_assessments"] += 1
        if bias_reports:
            self.framework_metrics["fairness_violations"] += len(bias_reports)
        if demographic_bias["bias_detected"]:
            self.framework_metrics["bias_incidents"] += 1
        if privacy_report["privacy_metrics"]["total_violations"] > 0:
            self.framework_metrics["privacy_violations"] += 1
        if compliance_report["overall_compliance"] < 1.0:
            self.framework_metrics["compliance_violations"] += 1
            
        return assessment
        
    def _calculate_overall_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate overall responsible AI score."""
        scores = []
        
        # Fairness score
        fairness_score = 1.0
        if assessment["fairness_assessment"]["bias_reports"]:
            fairness_score = max(0.0, 1.0 - len(assessment["fairness_assessment"]["bias_reports"]) * 0.1)
        scores.append(fairness_score)
        
        # Bias score
        bias_score = 0.0 if assessment["bias_assessment"]["bias_detected"] else 1.0
        scores.append(bias_score)
        
        # Privacy score
        privacy_violations = assessment["privacy_assessment"]["privacy_metrics"]["total_violations"]
        privacy_score = max(0.0, 1.0 - privacy_violations * 0.1)
        scores.append(privacy_score)
        
        # Compliance score
        compliance_score = assessment["compliance_assessment"]["overall_compliance"]
        scores.append(compliance_score)
        
        return np.mean(scores)
        
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment."""
        recommendations = []
        
        # Fairness recommendations
        if assessment["fairness_assessment"]["bias_reports"]:
            recommendations.append("Implement fairness constraints in model training")
            
        # Bias recommendations
        if assessment["bias_assessment"]["bias_detected"]:
            recommendations.append("Review training data for bias")
            recommendations.append("Consider bias mitigation techniques")
            
        # Privacy recommendations
        if assessment["privacy_assessment"]["privacy_metrics"]["total_violations"] > 0:
            recommendations.append("Strengthen privacy protection measures")
            
        # Compliance recommendations
        if assessment["compliance_assessment"]["overall_compliance"] < 1.0:
            recommendations.append("Address compliance violations")
            
        return recommendations
        
    def get_framework_report(self) -> Dict[str, Any]:
        """Get comprehensive framework report."""
        return {
            "framework_metrics": self.framework_metrics,
            "fairness_report": self.fairness_monitor.get_fairness_report(),
            "privacy_report": self.privacy_protector.get_privacy_report(),
            "compliance_report": self.regulatory_compliance.get_compliance_report(),
            "governance_report": self.ai_governance.get_governance_report(),
        }
