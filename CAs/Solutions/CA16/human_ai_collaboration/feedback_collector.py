"""
Human Feedback Collection

This module provides systems for collecting and managing human feedback:
- Interactive feedback collection
- Feedback aggregation
- Quality assessment
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import threading
import time
from .preference_model import HumanFeedback, HumanPreference

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HumanFeedbackCollector:
    """Base class for collecting human feedback."""

    def __init__(self, feedback_types: List[str] = None):
        self.feedback_types = feedback_types or [
            "preference",
            "correction",
            "demonstration",
        ]
        self.feedback_buffer = defaultdict(list)
        self.feedback_stats = defaultdict(int)

        # Quality assessment
        self.feedback_quality_scores = []

    def collect_feedback(self, feedback: HumanFeedback) -> bool:
        """Collect a piece of human feedback."""
        if feedback.feedback_type not in self.feedback_types:
            print(f"Warning: Unknown feedback type {feedback.feedback_type}")
            return False

        self.feedback_buffer[feedback.feedback_type].append(feedback)
        self.feedback_stats[feedback.feedback_type] += 1

        # Assess feedback quality
        quality_score = self._assess_feedback_quality(feedback)
        self.feedback_quality_scores.append(quality_score)

        return True

    def _assess_feedback_quality(self, feedback: HumanFeedback) -> float:
        """Assess the quality of feedback (to be overridden by subclasses)."""
        # Basic quality assessment based on feedback type and content
        if feedback.feedback_type == "preference":
            # Preferences should have clear A/B distinction
            if hasattr(feedback.content, "preferred"):
                return 1.0 if feedback.content.preferred in ["A", "B"] else 0.5
        elif feedback.feedback_type == "demonstration":
            # Demonstrations should have trajectory data
            if isinstance(feedback.content, dict) and "states" in feedback.content:
                return 1.0
        elif feedback.feedback_type == "correction":
            # Corrections should specify what was wrong
            return 0.8  # Assume reasonable quality

        return 0.5  # Default quality

    def get_feedback_batch(
        self, feedback_type: str, batch_size: int = 32
    ) -> List[HumanFeedback]:
        """Get a batch of feedback of specific type."""
        feedback_list = self.feedback_buffer[feedback_type]
        if len(feedback_list) < batch_size:
            return feedback_list.copy()

        # Return random sample
        indices = np.random.choice(len(feedback_list), batch_size, replace=False)
        return [feedback_list[i] for i in indices]

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected feedback."""
        summary = {
            "total_feedback": sum(self.feedback_stats.values()),
            "feedback_by_type": dict(self.feedback_stats),
            "average_quality": (
                np.mean(self.feedback_quality_scores)
                if self.feedback_quality_scores
                else 0.0
            ),
            "quality_distribution": {
                "high": len([s for s in self.feedback_quality_scores if s > 0.8]),
                "medium": len(
                    [s for s in self.feedback_quality_scores if 0.5 <= s <= 0.8]
                ),
                "low": len([s for s in self.feedback_quality_scores if s < 0.5]),
            },
        }
        return summary

    def clear_feedback(self, feedback_type: Optional[str] = None):
        """Clear collected feedback."""
        if feedback_type:
            self.feedback_buffer[feedback_type].clear()
            self.feedback_stats[feedback_type] = 0
        else:
            self.feedback_buffer.clear()
            self.feedback_stats.clear()
            self.feedback_quality_scores.clear()


class InteractiveFeedbackCollector(HumanFeedbackCollector):
    """Interactive system for real-time feedback collection."""

    def __init__(self, feedback_callback: Optional[Callable] = None, **kwargs):
        super().__init__(**kwargs)
        self.feedback_callback = feedback_callback
        self.is_collecting = False
        self.collection_thread = None

        # Interactive settings
        self.query_frequency = 10  # Query every N steps
        self.max_wait_time = 30  # Maximum wait time for feedback in seconds

    def start_collection(self, environment_interface):
        """Start interactive feedback collection."""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop, args=(environment_interface,)
        )
        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop_collection(self):
        """Stop feedback collection."""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)

    def _collection_loop(self, environment_interface):
        """Main collection loop."""
        step_count = 0

        while self.is_collecting:
            step_count += 1

            # Get current state from environment
            current_state = environment_interface.get_current_state()

            # Decide whether to query for feedback
            if step_count % self.query_frequency == 0:
                self._query_feedback(current_state, environment_interface)

            time.sleep(0.1)  # Small delay to prevent busy waiting

    def _query_feedback(self, state, environment_interface):
        """Query human for feedback on current state."""
        # Generate feedback query
        query = self._generate_feedback_query(state)

        # Send query to human (through callback or interface)
        if self.feedback_callback:
            feedback = self.feedback_callback(query)
            if feedback:
                self.collect_feedback(feedback)
        else:
            # Default behavior: simulate feedback for demonstration
            self._simulate_feedback(state)

    def _generate_feedback_query(self, state) -> Dict[str, Any]:
        """Generate a feedback query for the current state."""
        return {
            "type": "state_evaluation",
            "state": state,
            "timestamp": time.time(),
            "query_text": "How would you rate the current agent behavior?",
            "options": ["good", "neutral", "poor"],
        }

    def _simulate_feedback(self, state):
        """Simulate human feedback for demonstration purposes."""
        # Random feedback simulation
        feedback_types = ["preference", "correction", "demonstration"]
        feedback_type = np.random.choice(feedback_types)

        if feedback_type == "preference":
            # Simulate preference between current and random trajectory
            preference = HumanPreference(
                option_a={"states": [state], "actions": [0], "rewards": [0.0]},
                option_b={
                    "states": [state + np.random.randn(*state.shape) * 0.1],
                    "actions": [1],
                    "rewards": [0.0],
                },
                preferred=np.random.choice(["A", "B", "equal"]),
                confidence=np.random.random(),
            )
            feedback = HumanFeedback("preference", preference)

        elif feedback_type == "correction":
            feedback = HumanFeedback(
                "correction",
                {
                    "incorrect_action": np.random.randint(4),
                    "correct_action": np.random.randint(4),
                    "reason": "Agent made suboptimal choice",
                },
            )

        else:  # demonstration
            feedback = HumanFeedback(
                "demonstration",
                {
                    "states": [state],
                    "actions": [np.random.randint(4)],
                    "rewards": [np.random.random()],
                },
            )

        self.collect_feedback(feedback)


class FeedbackAggregator:
    """Aggregates and processes multiple feedback sources."""

    def __init__(self, aggregation_method: str = "majority_vote"):
        self.aggregation_method = aggregation_method
        self.feedback_sources = {}

        # Aggregation statistics
        self.agreement_scores = []
        self.confidence_scores = []

    def add_feedback_source(self, source_id: str, collector: HumanFeedbackCollector):
        """Add a feedback source."""
        self.feedback_sources[source_id] = collector

    def aggregate_preferences(
        self, trajectory_pairs: List[Tuple[Dict, Dict]]
    ) -> List[HumanPreference]:
        """Aggregate preferences across multiple sources."""
        aggregated_preferences = []

        for traj_a, traj_b in trajectory_pairs:
            preferences = []

            # Collect preferences from all sources
            for source_id, collector in self.feedback_sources.items():
                source_preferences = collector.get_feedback_batch(
                    "preference", batch_size=100
                )

                # Find relevant preferences for this pair
                for pref_feedback in source_preferences:
                    pref = pref_feedback.content
                    if self._trajectories_match(
                        pref.option_a, traj_a
                    ) and self._trajectories_match(pref.option_b, traj_b):
                        preferences.append((pref, pref_feedback.user_id))

            if preferences:
                aggregated_pref = self._aggregate_single_preference(preferences)
                aggregated_preferences.append(aggregated_pref)

        return aggregated_preferences

    def _trajectories_match(
        self, traj1: Dict, traj2: Dict, tolerance: float = 1e-6
    ) -> bool:
        """Check if two trajectories match (approximately)."""
        keys_to_check = ["states", "actions", "rewards"]
        for key in keys_to_check:
            if key in traj1 and key in traj2:
                arr1 = np.array(traj1[key])
                arr2 = np.array(traj2[key])
                if not np.allclose(arr1, arr2, atol=tolerance):
                    return False
        return True

    def _aggregate_single_preference(
        self, preferences: List[Tuple[HumanPreference, str]]
    ) -> HumanPreference:
        """Aggregate multiple preferences for the same pair."""
        if self.aggregation_method == "majority_vote":
            return self._majority_vote_aggregation(preferences)
        elif self.aggregation_method == "confidence_weighted":
            return self._confidence_weighted_aggregation(preferences)
        else:
            # Default to first preference
            return preferences[0][0]

    def _majority_vote_aggregation(
        self, preferences: List[Tuple[HumanPreference, str]]
    ) -> HumanPreference:
        """Aggregate preferences using majority voting."""
        votes = {"A": 0, "B": 0, "equal": 0}

        for pref, _ in preferences:
            votes[pref.preferred] += 1

        # Find majority
        majority_preferred = max(votes, key=votes.get)

        # Average confidence
        avg_confidence = np.mean([pref.confidence for pref, _ in preferences])

        # Create aggregated preference
        aggregated = HumanPreference(
            option_a=preferences[0][0].option_a,
            option_b=preferences[0][0].option_b,
            preferred=majority_preferred,
            confidence=avg_confidence,
            context={"num_sources": len(preferences), "votes": votes},
        )

        # Store agreement score
        total_votes = sum(votes.values())
        agreement = votes[majority_preferred] / total_votes if total_votes > 0 else 0
        self.agreement_scores.append(agreement)

        return aggregated

    def _confidence_weighted_aggregation(
        self, preferences: List[Tuple[HumanPreference, str]]
    ) -> HumanPreference:
        """Aggregate preferences using confidence-weighted voting."""
        weighted_votes = {"A": 0.0, "B": 0.0, "equal": 0.0}
        total_weight = 0.0

        for pref, _ in preferences:
            weight = pref.confidence
            weighted_votes[pref.preferred] += weight
            total_weight += weight

        # Find weighted majority
        if total_weight > 0:
            majority_preferred = max(weighted_votes, key=weighted_votes.get)
        else:
            majority_preferred = "equal"

        # Average confidence
        avg_confidence = np.mean([pref.confidence for pref, _ in preferences])

        aggregated = HumanPreference(
            option_a=preferences[0][0].option_a,
            option_b=preferences[0][0].option_b,
            preferred=majority_preferred,
            confidence=avg_confidence,
            context={
                "aggregation_method": "confidence_weighted",
                "total_weight": total_weight,
            },
        )

        return aggregated

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback aggregation."""
        return {
            "num_sources": len(self.feedback_sources),
            "average_agreement": (
                np.mean(self.agreement_scores) if self.agreement_scores else 0.0
            ),
            "agreement_distribution": {
                "high": len([s for s in self.agreement_scores if s > 0.8]),
                "medium": len([s for s in self.agreement_scores if 0.6 <= s <= 0.8]),
                "low": len([s for s in self.agreement_scores if s < 0.6]),
            },
        }


class FeedbackQualityAssessor:
    """Assesses and improves feedback quality."""

    def __init__(self):
        self.quality_model = nn.Sequential(
            nn.Linear(10, 32),  # Feature vector size
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        ).to(device)

        self.quality_optimizer = torch.optim.Adam(self.quality_model.parameters())
        self.quality_history = []

    def assess_feedback_quality(self, feedback: HumanFeedback) -> float:
        """Assess quality of a feedback instance."""
        features = self._extract_quality_features(feedback)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

        with torch.no_grad():
            quality_score = self.quality_model(features_tensor).item()

        return quality_score

    def _extract_quality_features(self, feedback: HumanFeedback) -> List[float]:
        """Extract features for quality assessment."""
        features = []

        # Feedback type (one-hot encoded)
        type_features = [
            1.0 if feedback.feedback_type == t else 0.0
            for t in ["preference", "correction", "demonstration", "rating", "comment"]
        ]
        features.extend(type_features)

        # Content complexity (length-based)
        if hasattr(feedback.content, "__len__"):
            content_length = (
                len(feedback.content)
                if not isinstance(feedback.content, str)
                else len(feedback.content)
            )
            features.append(min(content_length / 100.0, 1.0))  # Normalize
        else:
            features.append(0.5)

        # Has timestamp
        features.append(1.0 if feedback.timestamp > 0 else 0.0)

        # Has user ID
        features.append(1.0 if feedback.user_id is not None else 0.0)

        return features

    def update_quality_model(
        self, feedback_samples: List[HumanFeedback], quality_labels: List[float]
    ):
        """Update quality assessment model."""
        self.quality_model.train()

        for feedback, label in zip(feedback_samples, quality_labels):
            features = self._extract_quality_features(feedback)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            label_tensor = torch.FloatTensor([label]).to(device)

            self.quality_optimizer.zero_grad()
            pred_quality = self.quality_model(features_tensor)
            loss = nn.MSELoss()(pred_quality, label_tensor)
            loss.backward()
            self.quality_optimizer.step()

            self.quality_history.append(loss.item())
