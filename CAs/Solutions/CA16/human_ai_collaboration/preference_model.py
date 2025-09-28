"""
Preference Reward Models

This module contains models for learning from human preferences:
- Preference reward models
- Human preference representation
- Bradley-Terry preference learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HumanPreference:
    """Represents a human preference between two options."""

    def __init__(
        self,
        option_a: Any,
        option_b: Any,
        preferred: str,
        confidence: float = 1.0,
        context: Optional[Dict] = None,
    ):
        """
        Initialize human preference.

        Args:
            option_a: First option (could be trajectory, state, action, etc.)
            option_b: Second option
            preferred: 'A', 'B', or 'equal'
            confidence: Confidence in preference (0-1)
            context: Additional context information
        """
        self.option_a = option_a
        self.option_b = option_b
        self.preferred = preferred
        self.confidence = confidence
        self.context = context or {}

        if preferred not in ["A", "B", "equal"]:
            raise ValueError("Preferred must be 'A', 'B', or 'equal'")

    def get_preference_vector(self) -> torch.Tensor:
        """Convert preference to vector representation."""
        if self.preferred == "A":
            return torch.tensor([1.0, 0.0, 0.0], device=device)  # A > B
        elif self.preferred == "B":
            return torch.tensor([0.0, 1.0, 0.0], device=device)  # B > A
        else:
            return torch.tensor([0.0, 0.0, 1.0], device=device)  # A == B

    def __str__(self):
        return f"Preference({self.preferred}): A vs B (confidence: {self.confidence})"


class HumanFeedback:
    """Represents various types of human feedback."""

    def __init__(
        self,
        feedback_type: str,
        content: Any,
        timestamp: Optional[float] = None,
        user_id: Optional[str] = None,
    ):
        """
        Initialize human feedback.

        Args:
            feedback_type: Type of feedback ('preference', 'correction', 'demonstration', etc.)
            content: The actual feedback content
            timestamp: When feedback was given
            user_id: Identifier for the human providing feedback
        """
        self.feedback_type = feedback_type
        self.content = content
        self.timestamp = timestamp or torch.tensor(0.0)
        self.user_id = user_id

        valid_types = ["preference", "correction", "demonstration", "rating", "comment"]
        if feedback_type not in valid_types:
            raise ValueError(f"Invalid feedback type. Must be one of {valid_types}")

    def __str__(self):
        return f"Feedback({self.feedback_type}): {type(self.content).__name__}"


class PreferenceRewardModel(nn.Module):
    """Neural network for learning reward functions from human preferences."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        preference_dim: int = 64,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.preference_dim = preference_dim

        self.trajectory_encoder = nn.LSTM(
            state_dim + action_dim + 1,  # state + action + reward
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.preference_network = nn.Sequential(
            nn.Linear(
                hidden_dim * 2 * 2, hidden_dim
            ),  # Concatenated trajectory encodings
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, preference_dim),
        )

        self.bt_model = nn.Sequential(
            nn.Linear(preference_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Logit for A > B
            nn.Sigmoid(),
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(preference_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Positive uncertainty
        )

    def encode_trajectory(self, trajectory: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode a trajectory into a fixed-dimensional representation."""
        states = trajectory["states"]  # (seq_len, state_dim)
        actions = trajectory["actions"]  # (seq_len, action_dim)
        rewards = trajectory["rewards"]  # (seq_len, 1)

        traj_input = torch.cat([states, actions, rewards], dim=-1)

        outputs, (h_n, c_n) = self.trajectory_encoder(traj_input)

        final_hidden = torch.cat(
            [h_n[-2], h_n[-1]], dim=-1
        )  # Last layer, both directions

        return final_hidden

    def compare_trajectories(
        self, traj_a: Dict[str, torch.Tensor], traj_b: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compare two trajectories and predict preference."""
        encoding_a = self.encode_trajectory(traj_a)
        encoding_b = self.encode_trajectory(traj_b)

        combined = torch.cat([encoding_a, encoding_b], dim=-1)

        preference_features = self.preference_network(combined)

        preference_prob = self.bt_model(preference_features)

        return preference_prob.squeeze(), preference_features

    def predict_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward for a state-action pair."""
        sa_input = torch.cat([state, action], dim=-1)
        reward = self.reward_predictor(sa_input)
        return reward.squeeze()

    def estimate_uncertainty(self, preference_features: torch.Tensor) -> torch.Tensor:
        """Estimate uncertainty in preference prediction."""
        uncertainty = self.uncertainty_head(preference_features)
        return uncertainty.squeeze()

    def forward(
        self, batch_trajectories_a: List[Dict], batch_trajectories_b: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for batch of trajectory pairs."""
        batch_size = len(batch_trajectories_a)

        encodings_a = []
        encodings_b = []

        for traj_a, traj_b in zip(batch_trajectories_a, batch_trajectories_b):
            enc_a = self.encode_trajectory(traj_a)
            enc_b = self.encode_trajectory(traj_b)
            encodings_a.append(enc_a)
            encodings_b.append(enc_b)

        encodings_a = torch.stack(encodings_a)
        encodings_b = torch.stack(encodings_b)

        combined = torch.cat([encodings_a, encodings_b], dim=-1)
        preference_features = self.preference_network(combined)
        preference_logits = self.bt_model(preference_features)

        uncertainties = self.uncertainty_head(preference_features)

        return {
            "preference_logits": preference_logits.squeeze(),
            "preference_features": preference_features,
            "uncertainties": uncertainties.squeeze(),
            "trajectory_encodings_a": encodings_a,
            "trajectory_encodings_b": encodings_b,
        }

    def loss_function(
        self, predictions: Dict[str, torch.Tensor], preferences: List[HumanPreference]
    ) -> torch.Tensor:
        """Compute loss for preference learning."""
        preference_logits = predictions["preference_logits"]

        targets = []
        weights = []

        for pref in preferences:
            if pref.preferred == "A":
                targets.append(1.0)  # A preferred over B
            elif pref.preferred == "B":
                targets.append(0.0)  # B preferred over A
            else:  # equal
                targets.append(0.5)  # Neutral

            weights.append(pref.confidence)

        targets = torch.tensor(targets, device=device)
        weights = torch.tensor(weights, device=device)

        bce_loss = F.binary_cross_entropy(preference_logits, targets, reduction="none")
        weighted_loss = (bce_loss * weights).mean()

        uncertainty_loss = predictions["uncertainties"].mean() * 0.1

        total_loss = weighted_loss + uncertainty_loss

        return total_loss


class ActivePreferenceLearner:
    """Active learning for preference collection."""

    def __init__(
        self, reward_model: PreferenceRewardModel, uncertainty_threshold: float = 0.5
    ):
        self.reward_model = reward_model
        self.uncertainty_threshold = uncertainty_threshold

        self.queried_pairs = set()

    def select_query_pair(
        self, candidate_trajectories: List[Dict]
    ) -> Tuple[Dict, Dict]:
        """Select most informative trajectory pair to query."""
        max_uncertainty = -float("inf")
        best_pair = None

        for traj_a, traj_b in itertools.combinations(candidate_trajectories, 2):
            pair_key = (id(traj_a), id(traj_b))

            if pair_key in self.queried_pairs:
                continue

            with torch.no_grad():
                pred_a_b, _ = self.reward_model.compare_trajectories(traj_a, traj_b)
                uncertainty = self._compute_uncertainty(pred_a_b)

            if uncertainty > max_uncertainty:
                max_uncertainty = uncertainty
                best_pair = (traj_a, traj_b)

        if best_pair:
            self.queried_pairs.add((id(best_pair[0]), id(best_pair[1])))

        return best_pair

    def _compute_uncertainty(self, preference_prob: torch.Tensor) -> float:
        """Compute uncertainty from preference probability."""
        if preference_prob.dim() == 0:
            preference_prob = preference_prob.unsqueeze(0)

        entropy = -(
            preference_prob * torch.log(preference_prob + 1e-8)
            + (1 - preference_prob) * torch.log(1 - preference_prob + 1e-8)
        )

        return entropy.item()

    def should_query_human(self, trajectory_pair: Tuple[Dict, Dict]) -> bool:
        """Decide whether to query human for this pair."""
        traj_a, traj_b = trajectory_pair

        with torch.no_grad():
            preference_prob, _ = self.reward_model.compare_trajectories(traj_a, traj_b)
            uncertainty = self._compute_uncertainty(preference_prob)

        return uncertainty > self.uncertainty_threshold


class PreferenceDataset:
    """Dataset for preference learning."""

    def __init__(self, preferences: List[HumanPreference]):
        self.preferences = preferences

    def __len__(self):
        return len(self.preferences)

    def __getitem__(self, idx):
        pref = self.preferences[idx]

        traj_a = self._trajectory_to_tensor(pref.option_a)
        traj_b = self._trajectory_to_tensor(pref.option_b)

        return {
            "trajectory_a": traj_a,
            "trajectory_b": traj_b,
            "preference": pref.get_preference_vector(),
            "confidence": pref.confidence,
        }

    def _trajectory_to_tensor(self, trajectory: Dict) -> Dict[str, torch.Tensor]:
        """Convert trajectory dict to tensor format."""
        tensor_traj = {}
        for key, value in trajectory.items():
            if isinstance(value, np.ndarray):
                tensor_traj[key] = torch.FloatTensor(value)
            elif isinstance(value, list):
                tensor_traj[key] = torch.FloatTensor(value)
            else:
                tensor_traj[key] = value

        return tensor_traj
