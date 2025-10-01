"""
Communication Protocols for Human-AI Collaboration

This module implements communication systems for human-AI interaction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import time
from .preference_model import HumanFeedback


@dataclass
class CommunicationMessage:
    """Represents a communication message between human and AI."""

    sender: str  # 'human' or 'ai'
    message_type: str  # 'advice', 'question', 'explanation', 'feedback'
    content: str
    timestamp: float
    context: Dict[str, Any] = None


class CommunicationProtocol:
    """Protocol for human-AI communication."""

    def __init__(self, max_message_history: int = 1000):
        self.max_message_history = max_message_history
        self.message_history = deque(maxlen=max_message_history)
        self.active_conversations = {}
        self.communication_statistics = {
            "total_messages": 0,
            "human_messages": 0,
            "ai_messages": 0,
            "avg_response_time": 0.0,
        }

    def send_message(self, message: CommunicationMessage) -> str:
        """Send a message and return message ID."""
        message_id = f"{message.sender}_{int(time.time() * 1000)}"

        # Add to history
        self.message_history.append(message)

        # Update statistics
        self.communication_statistics["total_messages"] += 1
        if message.sender == "human":
            self.communication_statistics["human_messages"] += 1
        else:
            self.communication_statistics["ai_messages"] += 1

        return message_id

    def get_recent_messages(self, limit: int = 10) -> List[CommunicationMessage]:
        """Get recent messages."""
        return list(self.message_history)[-limit:]

    def get_conversation_context(
        self, conversation_id: str
    ) -> List[CommunicationMessage]:
        """Get messages from a specific conversation."""
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        return []

    def start_conversation(self, conversation_id: str) -> str:
        """Start a new conversation."""
        self.active_conversations[conversation_id] = []
        return conversation_id

    def end_conversation(self, conversation_id: str) -> List[CommunicationMessage]:
        """End a conversation and return all messages."""
        if conversation_id in self.active_conversations:
            messages = self.active_conversations[conversation_id]
            del self.active_conversations[conversation_id]
            return messages
        return []

    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return self.communication_statistics.copy()


class AdviceSystem:
    """System for providing and receiving advice."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Advice storage
        self.advice_database = {}
        self.advice_statistics = {
            "total_advice_given": 0,
            "advice_acceptance_rate": 0.0,
            "advice_effectiveness": 0.0,
        }

        # Advice quality assessment
        self.advice_quality_model = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, 64),  # +1 for outcome
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def provide_advice(
        self, state: torch.Tensor, action: torch.Tensor, explanation: str = ""
    ) -> Dict[str, Any]:
        """Provide advice for a state-action pair."""
        advice_id = f"advice_{int(time.time() * 1000)}"

        advice = {
            "id": advice_id,
            "state": state.cpu().numpy(),
            "action": action.cpu().numpy(),
            "explanation": explanation,
            "timestamp": time.time(),
            "quality_score": self._assess_advice_quality(state, action),
            "accepted": False,
            "outcome": None,
        }

        self.advice_database[advice_id] = advice
        self.advice_statistics["total_advice_given"] += 1

        return advice

    def accept_advice(self, advice_id: str, outcome: float = None):
        """Mark advice as accepted and record outcome."""
        if advice_id in self.advice_database:
            advice = self.advice_database[advice_id]
            advice["accepted"] = True
            advice["outcome"] = outcome

            # Update statistics
            self._update_advice_statistics()

    def _assess_advice_quality(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> float:
        """Assess the quality of advice."""
        with torch.no_grad():
            # Simple quality assessment based on state-action features
            input_features = torch.cat(
                [state, action, torch.tensor([0.0])]
            )  # 0.0 for unknown outcome
            quality_score = self.advice_quality_model(input_features.unsqueeze(0))
            return quality_score.item()

    def _update_advice_statistics(self):
        """Update advice statistics."""
        total_advice = len(self.advice_database)
        accepted_advice = sum(
            1 for advice in self.advice_database.values() if advice["accepted"]
        )

        if total_advice > 0:
            self.advice_statistics["advice_acceptance_rate"] = (
                accepted_advice / total_advice
            )

        # Calculate effectiveness (simplified)
        effective_advice = sum(
            1
            for advice in self.advice_database.values()
            if advice["accepted"]
            and advice["outcome"] is not None
            and advice["outcome"] > 0
        )

        if accepted_advice > 0:
            self.advice_statistics["advice_effectiveness"] = (
                effective_advice / accepted_advice
            )

    def get_advice_statistics(self) -> Dict[str, Any]:
        """Get advice system statistics."""
        return self.advice_statistics.copy()

    def get_advice_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """Get advice history."""
        advice_list = list(self.advice_database.values())
        if limit is None:
            return advice_list
        return advice_list[-limit:]


class DemonstrationCollector:
    """Collects and manages human demonstrations."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Demonstration storage
        self.demonstrations = {}
        self.demonstration_statistics = {
            "total_demonstrations": 0,
            "avg_demonstration_length": 0.0,
            "demonstration_quality": 0.0,
        }

        # Quality assessment
        self.quality_assessor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def add_demonstration(
        self,
        states: List[torch.Tensor],
        actions: List[torch.Tensor],
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Add a human demonstration."""
        demo_id = f"demo_{int(time.time() * 1000)}"

        if metadata is None:
            metadata = {}

        # Assess demonstration quality
        quality_score = self._assess_demonstration_quality(states, actions)

        demonstration = {
            "id": demo_id,
            "states": [state.cpu().numpy() for state in states],
            "actions": [action.cpu().numpy() for action in actions],
            "metadata": metadata,
            "timestamp": time.time(),
            "quality_score": quality_score,
            "length": len(states),
        }

        self.demonstrations[demo_id] = demonstration
        self.demonstration_statistics["total_demonstrations"] += 1

        # Update statistics
        self._update_demonstration_statistics()

        return demo_id

    def get_demonstration(self, demo_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific demonstration."""
        return self.demonstrations.get(demo_id)

    def get_all_demonstrations(self) -> List[Dict[str, Any]]:
        """Get all demonstrations."""
        return list(self.demonstrations.values())

    def get_high_quality_demonstrations(
        self, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Get demonstrations above quality threshold."""
        return [
            demo
            for demo in self.demonstrations.values()
            if demo["quality_score"] >= threshold
        ]

    def _assess_demonstration_quality(
        self, states: List[torch.Tensor], actions: List[torch.Tensor]
    ) -> float:
        """Assess the quality of a demonstration."""
        if not states or not actions:
            return 0.0

        # Simple quality assessment based on consistency and length
        consistency_score = self._compute_consistency(states, actions)
        length_score = min(len(states) / 100.0, 1.0)  # Normalize length

        # Combine scores
        quality_score = 0.7 * consistency_score + 0.3 * length_score

        return quality_score

    def _compute_consistency(
        self, states: List[torch.Tensor], actions: List[torch.Tensor]
    ) -> float:
        """Compute consistency score for demonstration."""
        if len(states) < 2:
            return 1.0

        # Compute state transitions
        state_changes = []
        for i in range(1, len(states)):
            change = torch.norm(states[i] - states[i - 1]).item()
            state_changes.append(change)

        # Consistency based on smoothness of transitions
        if state_changes:
            avg_change = np.mean(state_changes)
            consistency = 1.0 / (
                1.0 + avg_change
            )  # Higher consistency for smoother transitions
        else:
            consistency = 1.0

        return consistency

    def _update_demonstration_statistics(self):
        """Update demonstration statistics."""
        if not self.demonstrations:
            return

        # Update average length
        total_length = sum(demo["length"] for demo in self.demonstrations.values())
        self.demonstration_statistics["avg_demonstration_length"] = total_length / len(
            self.demonstrations
        )

        # Update average quality
        total_quality = sum(
            demo["quality_score"] for demo in self.demonstrations.values()
        )
        self.demonstration_statistics["demonstration_quality"] = total_quality / len(
            self.demonstrations
        )

    def get_demonstration_statistics(self) -> Dict[str, Any]:
        """Get demonstration statistics."""
        return self.demonstration_statistics.copy()

    def export_demonstrations(self, format: str = "numpy") -> Dict[str, Any]:
        """Export demonstrations in specified format."""
        if format == "numpy":
            return {
                "demonstrations": self.demonstrations,
                "statistics": self.demonstration_statistics,
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def import_demonstrations(self, data: Dict[str, Any]):
        """Import demonstrations from data."""
        if "demonstrations" in data:
            self.demonstrations.update(data["demonstrations"])

        if "statistics" in data:
            self.demonstration_statistics.update(data["statistics"])

        # Recompute statistics
        self._update_demonstration_statistics()
