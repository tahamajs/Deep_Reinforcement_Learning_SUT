"""
Preference Models for Human-AI Collaboration

This module implements preference-based learning and reward modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class HumanPreference:
    """Represents a human preference between two actions or trajectories."""
    state: torch.Tensor
    action1: torch.Tensor
    action2: torch.Tensor
    preference: int  # 0 for action1, 1 for action2
    confidence: float = 1.0
    timestamp: float = 0.0


@dataclass
class HumanFeedback:
    """Represents human feedback on an action or trajectory."""
    state: torch.Tensor
    action: torch.Tensor
    feedback_type: str  # 'positive', 'negative', 'neutral'
    feedback_value: float  # -1 to 1
    explanation: str = ""
    timestamp: float = 0.0


class PreferenceRewardModel(nn.Module):
    """Neural network for modeling human preferences."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to predict rewards and confidence."""
        # Encode states and actions
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        
        # Combine features
        combined_features = torch.cat([state_features, action_features], dim=-1)
        
        # Predict rewards and confidence
        rewards = self.reward_predictor(combined_features)
        confidence = self.confidence_predictor(combined_features)
        
        return rewards.squeeze(-1), confidence.squeeze(-1)
    
    def predict_reward(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Predict reward for a single state-action pair."""
        with torch.no_grad():
            reward, _ = self.forward(state.unsqueeze(0), action.unsqueeze(0))
            return reward.item()


class BradleyTerryModel(nn.Module):
    """Bradley-Terry model for pairwise preference learning."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Preference model
        self.preference_model = PreferenceRewardModel(state_dim, action_dim, hidden_dim)
        
        # Temperature parameter for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, states: torch.Tensor, actions1: torch.Tensor, 
                actions2: torch.Tensor) -> torch.Tensor:
        """Forward pass for pairwise preference prediction."""
        # Get rewards for both actions
        rewards1, _ = self.preference_model(states, actions1)
        rewards2, _ = self.preference_model(states, actions2)
        
        # Compute preference probabilities using Bradley-Terry model
        logits = (rewards1 - rewards2) / self.temperature
        preference_probs = torch.sigmoid(logits)
        
        return preference_probs
    
    def predict_preference(self, state: torch.Tensor, action1: torch.Tensor, 
                          action2: torch.Tensor) -> float:
        """Predict preference probability for two actions."""
        with torch.no_grad():
            prob = self.forward(state.unsqueeze(0), action1.unsqueeze(0), action2.unsqueeze(0))
            return prob.item()
    
    def compute_loss(self, states: torch.Tensor, actions1: torch.Tensor, 
                    actions2: torch.Tensor, preferences: torch.Tensor) -> torch.Tensor:
        """Compute Bradley-Terry loss."""
        preference_probs = self.forward(states, actions1, actions2)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(preference_probs, preferences.float())
        
        return loss


class PreferenceLearner:
    """Learner for human preferences."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Bradley-Terry model
        self.model = BradleyTerryModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'preferences_learned': 0
        }
    
    def learn_preference(self, preference: HumanPreference) -> float:
        """Learn from a single preference."""
        # Prepare data
        state = preference.state.unsqueeze(0)
        action1 = preference.action1.unsqueeze(0)
        action2 = preference.action2.unsqueeze(0)
        target = torch.tensor([preference.preference], dtype=torch.float32)
        
        # Forward pass
        self.model.train()
        preference_prob = self.model(state, action1, action2)
        
        # Compute loss
        loss = F.binary_cross_entropy(preference_prob, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record training history
        self.training_history['losses'].append(loss.item())
        self.training_history['preferences_learned'] += 1
        
        # Compute accuracy
        predicted_preference = 1 if preference_prob.item() > 0.5 else 0
        accuracy = 1.0 if predicted_preference == preference.preference else 0.0
        self.training_history['accuracies'].append(accuracy)
        
        return loss.item()
    
    def learn_preferences(self, preferences: List[HumanPreference]) -> Dict[str, float]:
        """Learn from multiple preferences."""
        total_loss = 0.0
        total_accuracy = 0.0
        
        for preference in preferences:
            loss = self.learn_preference(preference)
            total_loss += loss
        
        # Compute average metrics
        avg_loss = total_loss / len(preferences)
        avg_accuracy = np.mean(self.training_history['accuracies'][-len(preferences):])
        
        return {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'total_preferences': len(preferences)
        }
    
    def predict_preference(self, state: torch.Tensor, action1: torch.Tensor, 
                          action2: torch.Tensor) -> float:
        """Predict preference between two actions."""
        return self.model.predict_preference(state, action1, action2)
    
    def get_reward_model(self) -> PreferenceRewardModel:
        """Get the underlying reward model."""
        return self.model.preference_model
    
    def evaluate(self, test_preferences: List[HumanPreference]) -> Dict[str, float]:
        """Evaluate on test preferences."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        
        with torch.no_grad():
            for preference in test_preferences:
                state = preference.state.unsqueeze(0)
                action1 = preference.action1.unsqueeze(0)
                action2 = preference.action2.unsqueeze(0)
                target = torch.tensor([preference.preference], dtype=torch.float32)
                
                preference_prob = self.model(state, action1, action2)
                loss = F.binary_cross_entropy(preference_prob, target)
                total_loss += loss.item()
                
                predicted_preference = 1 if preference_prob.item() > 0.5 else 0
                if predicted_preference == preference.preference:
                    correct_predictions += 1
        
        accuracy = correct_predictions / len(test_preferences)
        avg_loss = total_loss / len(test_preferences)
        
        return {
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'total_predictions': len(test_preferences)
        }


class ActivePreferenceLearner(PreferenceLearner):
    """Active learning for human preferences."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3):
        super().__init__(state_dim, action_dim, lr)
        self.uncertainty_threshold = 0.1
    
    def select_informative_pairs(self, states: List[torch.Tensor], 
                                actions: List[torch.Tensor], 
                                num_pairs: int = 10) -> List[Tuple[int, int]]:
        """Select most informative action pairs for preference queries."""
        informative_pairs = []
        
        for i, state in enumerate(states):
            for j, action1 in enumerate(actions):
                for k, action2 in enumerate(actions):
                    if j >= k:  # Avoid duplicate pairs
                        continue
                    
                    # Compute uncertainty
                    preference_prob = self.predict_preference(state, action1, action2)
                    uncertainty = abs(preference_prob - 0.5)  # Distance from 0.5
                    
                    if uncertainty < self.uncertainty_threshold:
                        informative_pairs.append((j, k))
        
        # Sort by uncertainty and return top pairs
        informative_pairs.sort(key=lambda x: abs(self.predict_preference(
            states[0], actions[x[0]], actions[x[1]]) - 0.5))
        
        return informative_pairs[:num_pairs]
    
    def update_uncertainty_threshold(self, new_threshold: float):
        """Update uncertainty threshold for active learning."""
        self.uncertainty_threshold = new_threshold