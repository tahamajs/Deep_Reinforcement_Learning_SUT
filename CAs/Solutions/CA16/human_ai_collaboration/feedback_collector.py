"""
Human Feedback Collection and Processing

This module implements tools for collecting and processing human feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from .preference_model import HumanPreference, HumanFeedback, PreferenceDataset


class HumanFeedbackCollector:
    """Collects and manages human feedback for RL training."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feedback_dataset = PreferenceDataset()

        # Feedback collection settings
        self.collection_active = True
        self.feedback_buffer = []

        # Statistics
        self.collection_stats = {
            "total_feedback": 0,
            "preferences": 0,
            "rewards": 0,
            "explanations": 0,
        }

    def collect_preference(
        self,
        state: torch.Tensor,
        action1: torch.Tensor,
        action2: torch.Tensor,
        preference: int,
        confidence: float = 1.0,
        explanation: str = "",
    ) -> HumanPreference:
        """Collect a preference between two actions."""
        pref = HumanPreference(
            state=state.clone(),
            action1=action1.clone(),
            action2=action2.clone(),
            preference=preference,
            confidence=confidence,
            explanation=explanation,
        )

        self.feedback_dataset.add_preference(pref)
        self.collection_stats["preferences"] += 1
        self.collection_stats["total_feedback"] += 1

        if explanation:
            self.collection_stats["explanations"] += 1

        return pref

    def collect_feedback(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        confidence: float = 1.0,
        explanation: str = "",
    ) -> HumanFeedback:
        """Collect feedback on an action."""
        feedback = HumanFeedback(
            state=state.clone(),
            action=action.clone(),
            reward=reward,
            confidence=confidence,
            explanation=explanation,
        )

        self.feedback_dataset.add_feedback(feedback)
        self.collection_stats["rewards"] += 1
        self.collection_stats["total_feedback"] += 1

        if explanation:
            self.collection_stats["explanations"] += 1

        return feedback

    def collect_batch_feedback(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        explanations: Optional[List[str]] = None,
    ) -> List[HumanFeedback]:
        """Collect feedback for a batch of interactions."""
        batch_feedback = []

        for i in range(len(states)):
            explanation = explanations[i] if explanations else ""

            feedback = self.collect_feedback(
                states[i],
                actions[i],
                rewards[i].item(),
                explanation=explanation,
            )
            batch_feedback.append(feedback)

        return batch_feedback

    def get_feedback_dataset(self) -> PreferenceDataset:
        """Get the feedback dataset."""
        return self.feedback_dataset

    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback."""
        stats = self.collection_stats.copy()
        stats.update(self.feedback_dataset.get_statistics())
        return stats

    def clear_feedback(self):
        """Clear all collected feedback."""
        self.feedback_dataset.clear()
        self.feedback_buffer.clear()
        self.collection_stats = {
            "total_feedback": 0,
            "preferences": 0,
            "rewards": 0,
            "explanations": 0,
        }

    def start_collection(self):
        """Start feedback collection."""
        self.collection_active = True

    def stop_collection(self):
        """Stop feedback collection."""
        self.collection_active = False

    def is_collection_active(self) -> bool:
        """Check if feedback collection is active."""
        return self.collection_active


class InteractiveLearner:
    """Learner that interacts with human feedback in real-time."""

    def __init__(
        self,
        agent,
        feedback_collector: HumanFeedbackCollector,
        learning_rate: float = 1e-3,
    ):
        self.agent = agent
        self.feedback_collector = feedback_collector
        self.learning_rate = learning_rate

        # Learning settings
        self.learning_active = True
        self.update_frequency = 10  # Update every N interactions
        self.interaction_count = 0

        # Learning history
        self.learning_history = {
            "updates": [],
            "performance": [],
            "feedback_quality": [],
        }

    def interact(self, state: torch.Tensor) -> Tuple[int, Dict[str, Any]]:
        """Interact with the environment and collect feedback."""
        # Get agent action
        action, action_info = self.agent.select_action(state)

        # Check if feedback is needed
        needs_feedback = self._should_request_feedback(state, action, action_info)

        interaction_info = {
            "action": action,
            "action_info": action_info,
            "needs_feedback": needs_feedback,
            "interaction_count": self.interaction_count,
        }

        self.interaction_count += 1

        # Update agent if needed
        if self.interaction_count % self.update_frequency == 0:
            self._update_agent()

        return action, interaction_info

    def provide_feedback(
        self,
        state: torch.Tensor,
        action: int,
        feedback_type: str,
        feedback_data: Any,
    ):
        """Provide feedback to the learner."""
        if feedback_type == "reward":
            reward = feedback_data["reward"]
            explanation = feedback_data.get("explanation", "")

            action_tensor = torch.tensor([action], dtype=torch.float32)
            self.feedback_collector.collect_feedback(
                state, action_tensor, reward, explanation=explanation
            )

        elif feedback_type == "preference":
            action1 = feedback_data["action1"]
            action2 = feedback_data["action2"]
            preference = feedback_data["preference"]
            explanation = feedback_data.get("explanation", "")

            action1_tensor = torch.tensor([action1], dtype=torch.float32)
            action2_tensor = torch.tensor([action2], dtype=torch.float32)

            self.feedback_collector.collect_preference(
                state,
                action1_tensor,
                action2_tensor,
                preference,
                explanation=explanation,
            )

    def _should_request_feedback(
        self, state: torch.Tensor, action: int, action_info: Dict[str, Any]
    ) -> bool:
        """Determine if feedback should be requested."""
        # Request feedback based on confidence
        confidence = action_info.get("confidence", 0.5)
        return confidence < 0.7

    def _update_agent(self):
        """Update the agent with collected feedback."""
        if not self.learning_active:
            return

        dataset = self.feedback_collector.get_feedback_dataset()

        if len(dataset.feedback) == 0 and len(dataset.preferences) == 0:
            return

        # Update agent with feedback
        update_info = self._perform_update(dataset)

        # Record learning history
        self.learning_history["updates"].append(update_info)

        # Clear processed feedback
        dataset.clear_feedback()

    def _perform_update(self, dataset: PreferenceDataset) -> Dict[str, Any]:
        """Perform agent update with feedback data."""
        update_info = {
            "timestamp": time.time(),
            "feedback_count": len(dataset.feedback),
            "preference_count": len(dataset.preferences),
            "success": False,
        }

        try:
            # Get feedback batch
            if len(dataset.feedback) > 0:
                states, actions, rewards = dataset.get_feedback_batch(
                    min(32, len(dataset.feedback))
                )

                # Update agent
                loss_info = self.agent.update_with_feedback(states, actions, rewards)
                update_info.update(loss_info)
                update_info["success"] = True

        except Exception as e:
            update_info["error"] = str(e)

        return update_info

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        feedback_stats = self.feedback_collector.get_collection_statistics()

        return {
            "interaction_count": self.interaction_count,
            "learning_active": self.learning_active,
            "update_frequency": self.update_frequency,
            "feedback_statistics": feedback_stats,
            "learning_history_size": len(self.learning_history["updates"]),
        }

    def start_learning(self):
        """Start learning from feedback."""
        self.learning_active = True

    def stop_learning(self):
        """Stop learning from feedback."""
        self.learning_active = False


class TrustModel(nn.Module):
    """Model for estimating trust between human and AI."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Trust estimation network
        self.trust_net = nn.Sequential(
            nn.Linear(
                state_dim + action_dim + 2, hidden_dim
            ),  # +2 for performance metrics
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Performance prediction network
        self.performance_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        performance_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate trust level."""
        # Predict performance if not provided
        if performance_metrics is None:
            performance_metrics = self.performance_net(
                torch.cat([states, actions], dim=-1)
            )

        # Combine inputs for trust estimation
        trust_inputs = torch.cat([states, actions, performance_metrics], dim=-1)
        trust_scores = self.trust_net(trust_inputs)

        return trust_scores

    def predict_performance(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Predict performance for given states and actions."""
        return self.performance_net(torch.cat([states, actions], dim=-1))

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        trust_labels: torch.Tensor,
        performance_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        # Trust prediction
        predicted_trust = self.forward(states, actions, performance_labels)
        trust_loss = F.mse_loss(predicted_trust.squeeze(), trust_labels)

        # Performance prediction
        predicted_performance = self.predict_performance(states, actions)
        performance_loss = torch.tensor(0.0)
        if performance_labels is not None:
            performance_loss = F.mse_loss(
                predicted_performance.squeeze(), performance_labels
            )

        # Total loss
        total_loss = trust_loss + 0.1 * performance_loss

        return {
            "trust_loss": trust_loss.item(),
            "performance_loss": performance_loss.item(),
            "total_loss": total_loss.item(),
        }


class TrustManager:
    """Manages trust between human and AI systems."""

    def __init__(self, trust_model: TrustModel):
        self.trust_model = trust_model
        self.trust_history = []
        self.performance_history = []

        # Trust thresholds
        self.trust_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
        }

    def update_trust(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        actual_performance: float,
        human_satisfaction: float,
    ):
        """Update trust based on performance and human satisfaction."""
        # Compute trust score
        performance_tensor = torch.tensor([actual_performance], dtype=torch.float32)
        trust_score = self.trust_model(
            state.unsqueeze(0), action.unsqueeze(0), performance_tensor.unsqueeze(0)
        )
        trust_score = trust_score.item()

        # Record history
        self.trust_history.append(
            {
                "trust_score": trust_score,
                "performance": actual_performance,
                "satisfaction": human_satisfaction,
                "timestamp": time.time(),
            }
        )

        return trust_score

    def get_trust_level(self) -> str:
        """Get current trust level."""
        if not self.trust_history:
            return "unknown"

        recent_trust = np.mean([t["trust_score"] for t in self.trust_history[-10:]])

        if recent_trust >= self.trust_thresholds["high"]:
            return "high"
        elif recent_trust >= self.trust_thresholds["medium"]:
            return "medium"
        elif recent_trust >= self.trust_thresholds["low"]:
            return "low"
        else:
            return "very_low"

    def should_increase_autonomy(self) -> bool:
        """Determine if AI autonomy should be increased."""
        return self.get_trust_level() in ["high", "medium"]

    def should_decrease_autonomy(self) -> bool:
        """Determine if AI autonomy should be decreased."""
        return self.get_trust_level() in ["low", "very_low"]

    def get_trust_statistics(self) -> Dict[str, Any]:
        """Get trust statistics."""
        if not self.trust_history:
            return {"trust_level": "unknown", "avg_trust": 0.0}

        recent_trust = [t["trust_score"] for t in self.trust_history[-10:]]
        recent_performance = [t["performance"] for t in self.trust_history[-10:]]
        recent_satisfaction = [t["satisfaction"] for t in self.trust_history[-10:]]

        return {
            "trust_level": self.get_trust_level(),
            "avg_trust": np.mean(recent_trust),
            "avg_performance": np.mean(recent_performance),
            "avg_satisfaction": np.mean(recent_satisfaction),
            "trust_trend": self._compute_trend(recent_trust),
            "total_interactions": len(self.trust_history),
        }

    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend in values."""
        if len(values) < 2:
            return "stable"

        # Simple trend computation
        first_half = np.mean(values[: len(values) // 2])
        second_half = np.mean(values[len(values) // 2 :])

        if second_half > first_half + 0.1:
            return "increasing"
        elif second_half < first_half - 0.1:
            return "decreasing"
        else:
            return "stable"
