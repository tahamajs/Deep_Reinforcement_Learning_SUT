"""
Preference Models for Human-AI Collaboration

This module implements preference learning models including Bradley-Terry models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class HumanPreference:
    """Represents a human preference between two options."""

    state: torch.Tensor
    action1: torch.Tensor
    action2: torch.Tensor
    preference: int  # 0 for action1, 1 for action2
    confidence: float = 1.0
    explanation: str = ""


@dataclass
class HumanFeedback:
    """Represents human feedback on an action."""

    state: torch.Tensor
    action: torch.Tensor
    reward: float
    confidence: float = 1.0
    explanation: str = ""


class PreferenceRewardModel(nn.Module):
    """Learn human preferences using Bradley-Terry model."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Reward prediction networks
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Preference prediction network
        self.preference_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for reward prediction."""
        # Combine state and action
        inputs = torch.cat([states, actions], dim=-1)

        # Predict rewards
        rewards = self.reward_net(inputs)

        # Predict preferences
        if states.shape[0] % 2 == 0:
            # Assuming paired inputs for preference prediction
            batch_size = states.shape[0] // 2
            state1, state2 = states[:batch_size], states[batch_size:]
            action1, action2 = actions[:batch_size], actions[batch_size:]

            reward1 = self.reward_net(torch.cat([state1, action1], dim=-1))
            reward2 = self.reward_net(torch.cat([state2, action2], dim=-1))

            # Predict preference (probability that action1 is preferred)
            preference_input = torch.cat([reward1, reward2], dim=-1)
            preferences = self.preference_net(preference_input)
        else:
            preferences = torch.zeros(states.shape[0], 1)

        return rewards, preferences

    def preference_probability(
        self, state: torch.Tensor, action1: torch.Tensor, action2: torch.Tensor
    ) -> torch.Tensor:
        """Compute probability that action1 is preferred over action2."""
        reward1 = self.reward_net(torch.cat([state, action1], dim=-1))
        reward2 = self.reward_net(torch.cat([state, action2], dim=-1))

        preference_input = torch.cat([reward1, reward2], dim=-1)
        preference_prob = self.preference_net(preference_input)

        return preference_prob

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        preferences: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step."""
        predicted_rewards, predicted_preferences = self.forward(states, actions)

        # Preference loss (Bradley-Terry model)
        preference_loss = F.binary_cross_entropy(
            predicted_preferences.squeeze(), preferences.float()
        )

        # Reward loss (if available)
        reward_loss = torch.tensor(0.0)
        if rewards is not None:
            reward_loss = F.mse_loss(predicted_rewards.squeeze(), rewards)

        # Total loss
        total_loss = preference_loss + 0.1 * reward_loss

        return {
            "preference_loss": preference_loss.item(),
            "reward_loss": reward_loss.item(),
            "total_loss": total_loss.item(),
        }


class BradleyTerryModel(nn.Module):
    """Bradley-Terry model for pairwise preferences."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Preference head
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """Compute preference probability."""
        # Extract features
        feat1 = self.feature_extractor(features1)
        feat2 = self.feature_extractor(features2)

        # Combine features
        combined = torch.cat([feat1, feat2], dim=-1)

        # Predict preference
        preference_logits = self.preference_head(combined)
        preference_prob = torch.sigmoid(preference_logits)

        return preference_prob

    def train_step(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        preferences: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        predicted_preferences = self.forward(features1, features2)
        loss = F.binary_cross_entropy(
            predicted_preferences.squeeze(), preferences.float()
        )

        return {"loss": loss.item()}


class PreferenceDataset:
    """Dataset for storing and managing human preferences."""

    def __init__(self):
        self.preferences = []
        self.feedback = []
        self.metadata = {}

    def add_preference(self, preference: HumanPreference):
        """Add a preference to the dataset."""
        self.preferences.append(preference)

    def add_feedback(self, feedback: HumanFeedback):
        """Add feedback to the dataset."""
        self.feedback.append(feedback)

    def get_preference_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of preferences for training."""
        if len(self.preferences) < batch_size:
            batch_size = len(self.preferences)

        indices = np.random.choice(len(self.preferences), batch_size, replace=False)

        states = []
        actions1 = []
        actions2 = []
        preferences = []

        for idx in indices:
            pref = self.preferences[idx]
            states.append(pref.state)
            actions1.append(pref.action1)
            actions2.append(pref.action2)
            preferences.append(pref.preference)

        return (
            torch.stack(states),
            torch.stack(actions1),
            torch.stack(actions2),
            torch.tensor(preferences),
        )

    def get_feedback_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of feedback for training."""
        if len(self.feedback) < batch_size:
            batch_size = len(self.feedback)

        indices = np.random.choice(len(self.feedback), batch_size, replace=False)

        states = []
        actions = []
        rewards = []

        for idx in indices:
            fb = self.feedback[idx]
            states.append(fb.state)
            actions.append(fb.action)
            rewards.append(fb.reward)

        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.preferences and not self.feedback:
            return {"total_preferences": 0, "total_feedback": 0}

        stats = {
            "total_preferences": len(self.preferences),
            "total_feedback": len(self.feedback),
        }

        if self.preferences:
            preferences = [p.preference for p in self.preferences]
            stats.update(
                {
                    "preference_distribution": {
                        "action1_preferred": sum(1 for p in preferences if p == 0),
                        "action2_preferred": sum(1 for p in preferences if p == 1),
                    },
                    "avg_confidence": np.mean([p.confidence for p in self.preferences]),
                }
            )

        if self.feedback:
            rewards = [f.reward for f in self.feedback]
            stats.update(
                {
                    "feedback_rewards": {
                        "mean": np.mean(rewards),
                        "std": np.std(rewards),
                        "min": np.min(rewards),
                        "max": np.max(rewards),
                    },
                    "avg_feedback_confidence": np.mean(
                        [f.confidence for f in self.feedback]
                    ),
                }
            )

        return stats

    def clear(self):
        """Clear all data."""
        self.preferences.clear()
        self.feedback.clear()
        self.metadata.clear()


class PreferenceTrainer:
    """Trainer for preference models."""

    def __init__(
        self,
        model: PreferenceRewardModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.training_history = {"losses": [], "accuracies": []}

    def train_epoch(
        self, dataset: PreferenceDataset, batch_size: int = 32
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_preference_loss = 0.0
        total_reward_loss = 0.0
        num_batches = 0

        # Train on preferences
        if len(dataset.preferences) > 0:
            pref_states, pref_actions1, pref_actions2, pref_labels = (
                dataset.get_preference_batch(batch_size)
            )

            # Combine actions for forward pass
            combined_states = torch.cat([pref_states, pref_states], dim=0)
            combined_actions = torch.cat([pref_actions1, pref_actions2], dim=0)

            losses = self.model.train_step(
                combined_states, combined_actions, pref_labels
            )

            total_loss += losses["total_loss"]
            total_preference_loss += losses["preference_loss"]
            total_reward_loss += losses["reward_loss"]
            num_batches += 1

        # Train on feedback
        if len(dataset.feedback) > 0:
            fb_states, fb_actions, fb_rewards = dataset.get_feedback_batch(batch_size)

            losses = self.model.train_step(fb_states, fb_actions, None, fb_rewards)

            total_loss += losses["total_loss"]
            total_reward_loss += losses["reward_loss"]
            num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_preference_loss = total_preference_loss / num_batches
            avg_reward_loss = total_reward_loss / num_batches

            self.training_history["losses"].append(avg_loss)

        return {
            "avg_loss": avg_loss,
            "avg_preference_loss": avg_preference_loss,
            "avg_reward_loss": avg_reward_loss,
        }

        return {"avg_loss": 0.0, "avg_preference_loss": 0.0, "avg_reward_loss": 0.0}

    def evaluate(
        self, dataset: PreferenceDataset, batch_size: int = 32
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            # Evaluate on preferences
            if len(dataset.preferences) > 0:
                pref_states, pref_actions1, pref_actions2, pref_labels = (
                    dataset.get_preference_batch(batch_size)
                )

                combined_states = torch.cat([pref_states, pref_states], dim=0)
                combined_actions = torch.cat([pref_actions1, pref_actions2], dim=0)

                losses = self.model.train_step(
                    combined_states, combined_actions, pref_labels
                )
                total_loss += losses["total_loss"]

                # Compute accuracy
                predicted_preferences = self.model.preference_probability(
                    pref_states, pref_actions1, pref_actions2
                )
                predicted_labels = (predicted_preferences > 0.5).long().squeeze()
                correct_predictions += (predicted_labels == pref_labels).sum().item()
                total_predictions += len(pref_labels)

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )

        return {
            "loss": total_loss,
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": total_predictions,
        }
