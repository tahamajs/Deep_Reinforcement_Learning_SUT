"""
Evaluation Metrics for Causal Multi-Modal RL
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, f1_score
import networkx as nx


class CausalDiscoveryMetrics:
    """Metrics for evaluating causal discovery algorithms"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.structural_hamming_distance = 0
        self.computation_times = []

    def update(
        self,
        true_graph: nx.DiGraph,
        predicted_graph: nx.DiGraph,
        computation_time: float = 0.0,
    ):
        """Update metrics with new predictions"""
        self.computation_times.append(computation_time)

        # Get adjacency matrices
        nodes = list(true_graph.nodes())
        true_adj = nx.adjacency_matrix(true_graph, nodelist=nodes).toarray()
        pred_adj = nx.adjacency_matrix(predicted_graph, nodelist=nodes).toarray()

        # Calculate confusion matrix
        tp = np.sum((true_adj == 1) & (pred_adj == 1))
        fp = np.sum((true_adj == 0) & (pred_adj == 1))
        tn = np.sum((true_adj == 0) & (pred_adj == 0))
        fn = np.sum((true_adj == 1) & (pred_adj == 0))

        self.true_positives += tp
        self.false_positives += fp
        self.true_negatives += tn
        self.false_negatives += fn

        # Structural Hamming Distance
        shd = fp + fn  # Simplified SHD
        self.structural_hamming_distance += shd

    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics"""
        total = (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )

        if total == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "structural_hamming_distance": 0.0,
                "avg_computation_time": 0.0,
            }

        accuracy = (self.true_positives + self.true_negatives) / total
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0
            else 0.0
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "structural_hamming_distance": (
                self.structural_hamming_distance / len(self.computation_times)
                if self.computation_times
                else 0.0
            ),
            "avg_computation_time": (
                np.mean(self.computation_times) if self.computation_times else 0.0
            ),
        }


class MultiModalMetrics:
    """Metrics for evaluating multi-modal learning"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.modality_contributions = {"visual": [], "textual": [], "state": []}
        self.fusion_accuracy = []
        self.cross_modal_correlation = []
        self.missing_modality_robustness = []

    def update(
        self,
        modality_features: Dict[str, np.ndarray],
        predictions: np.ndarray,
        targets: np.ndarray,
        missing_modalities: List[str] = None,
    ):
        """Update metrics with new predictions"""
        # Modality contributions (simplified)
        for modality, features in modality_features.items():
            if modality in self.modality_contributions:
                contribution = np.linalg.norm(features)
                self.modality_contributions[modality].append(contribution)

        # Fusion accuracy
        accuracy = accuracy_score(targets, predictions)
        self.fusion_accuracy.append(accuracy)

        # Cross-modal correlation
        if len(modality_features) >= 2:
            modalities = list(modality_features.keys())
            corr = np.corrcoef(
                modality_features[modalities[0]].flatten(),
                modality_features[modalities[1]].flatten(),
            )[0, 1]
            self.cross_modal_correlation.append(abs(corr))

        # Missing modality robustness
        if missing_modalities:
            robustness = 1.0 - len(missing_modalities) / len(modality_features)
            self.missing_modality_robustness.append(robustness)

    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics"""
        metrics = {
            "avg_fusion_accuracy": (
                np.mean(self.fusion_accuracy) if self.fusion_accuracy else 0.0
            ),
            "avg_cross_modal_correlation": (
                np.mean(self.cross_modal_correlation)
                if self.cross_modal_correlation
                else 0.0
            ),
            "avg_missing_modality_robustness": (
                np.mean(self.missing_modality_robustness)
                if self.missing_modality_robustness
                else 0.0
            ),
        }

        # Modality contributions
        for modality, contributions in self.modality_contributions.items():
            metrics[f"{modality}_contribution"] = (
                np.mean(contributions) if contributions else 0.0
            )

        return metrics


class CausalRLMetrics:
    """Metrics for evaluating causal RL agents"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = []
        self.sample_efficiency = []
        self.intervention_accuracy = []
        self.counterfactual_quality = []
        self.causal_reasoning_accuracy = []

    def update(
        self,
        episode_reward: float,
        steps: int,
        intervention_results: Dict[str, Any] = None,
        counterfactual_results: Dict[str, Any] = None,
    ):
        """Update metrics with new episode data"""
        self.episode_rewards.append(episode_reward)

        # Sample efficiency (reward per step)
        efficiency = episode_reward / steps if steps > 0 else 0.0
        self.sample_efficiency.append(efficiency)

        # Intervention accuracy
        if intervention_results:
            accuracy = intervention_results.get("accuracy", 0.0)
            self.intervention_accuracy.append(accuracy)

        # Counterfactual quality
        if counterfactual_results:
            quality = counterfactual_results.get("quality", 0.0)
            self.counterfactual_quality.append(quality)

    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics"""
        return {
            "avg_episode_reward": (
                np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            ),
            "std_episode_reward": (
                np.std(self.episode_rewards) if self.episode_rewards else 0.0
            ),
            "avg_sample_efficiency": (
                np.mean(self.sample_efficiency) if self.sample_efficiency else 0.0
            ),
            "avg_intervention_accuracy": (
                np.mean(self.intervention_accuracy)
                if self.intervention_accuracy
                else 0.0
            ),
            "avg_counterfactual_quality": (
                np.mean(self.counterfactual_quality)
                if self.counterfactual_quality
                else 0.0
            ),
            "avg_causal_reasoning_accuracy": (
                np.mean(self.causal_reasoning_accuracy)
                if self.causal_reasoning_accuracy
                else 0.0
            ),
        }


class IntegratedMetrics:
    """Integrated metrics for causal multi-modal RL systems"""

    def __init__(self):
        self.causal_metrics = CausalDiscoveryMetrics()
        self.multimodal_metrics = MultiModalMetrics()
        self.rl_metrics = CausalRLMetrics()

    def reset(self):
        """Reset all metrics"""
        self.causal_metrics.reset()
        self.multimodal_metrics.reset()
        self.rl_metrics.reset()

    def update_causal(
        self,
        true_graph: nx.DiGraph,
        predicted_graph: nx.DiGraph,
        computation_time: float = 0.0,
    ):
        """Update causal discovery metrics"""
        self.causal_metrics.update(true_graph, predicted_graph, computation_time)

    def update_multimodal(
        self,
        modality_features: Dict[str, np.ndarray],
        predictions: np.ndarray,
        targets: np.ndarray,
        missing_modalities: List[str] = None,
    ):
        """Update multi-modal metrics"""
        self.multimodal_metrics.update(
            modality_features, predictions, targets, missing_modalities
        )

    def update_rl(
        self,
        episode_reward: float,
        steps: int,
        intervention_results: Dict[str, Any] = None,
        counterfactual_results: Dict[str, Any] = None,
    ):
        """Update RL metrics"""
        self.rl_metrics.update(
            episode_reward, steps, intervention_results, counterfactual_results
        )

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all integrated metrics"""
        metrics = {}
        metrics.update(self.causal_metrics.get_metrics())
        metrics.update(self.multimodal_metrics.get_metrics())
        metrics.update(self.rl_metrics.get_metrics())
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            "causal_discovery": self.causal_metrics.get_metrics(),
            "multi_modal": self.multimodal_metrics.get_metrics(),
            "reinforcement_learning": self.rl_metrics.get_metrics(),
            "integrated": self.get_all_metrics(),
        }

