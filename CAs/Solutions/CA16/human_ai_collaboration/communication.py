"""
Communication Protocols for Human-AI Collaboration

This module implements communication protocols and advice systems for human-AI interaction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import time
from dataclasses import dataclass


@dataclass
class CommunicationMessage:
    """Represents a communication message between human and AI."""

    sender: str  # "human" or "ai"
    message_type: str
    content: Any
    timestamp: float
    priority: int = 1  # 1=low, 2=medium, 3=high
    explanation: str = ""


@dataclass
class Advice:
    """Represents advice from human to AI or vice versa."""

    advisor: str  # "human" or "ai"
    advice_type: str
    content: str
    confidence: float
    context: Dict[str, Any]
    timestamp: float


class CommunicationProtocol:
    """Protocol for managing communication between human and AI."""

    def __init__(self):
        self.message_queue = []
        self.communication_history = []
        self.protocol_rules = {
            "max_queue_size": 100,
            "message_timeout": 30.0,  # seconds
            "priority_weights": {1: 0.1, 2: 0.5, 3: 1.0},
        }

    def send_message(self, message: CommunicationMessage) -> bool:
        """Send a message through the protocol."""
        # Check queue size
        if len(self.message_queue) >= self.protocol_rules["max_queue_size"]:
            # Remove oldest low-priority message
            self._cleanup_queue()

        # Add message to queue
        self.message_queue.append(message)
        self.communication_history.append(message)

        return True

    def receive_message(self, sender: str) -> Optional[CommunicationMessage]:
        """Receive a message from a specific sender."""
        # Find highest priority message from sender
        for message in sorted(
            self.message_queue, key=lambda m: m.priority, reverse=True
        ):
            if message.sender == sender:
                self.message_queue.remove(message)
                return message

        return None

    def get_all_messages(
        self, message_type: Optional[str] = None
    ) -> List[CommunicationMessage]:
        """Get all messages, optionally filtered by type."""
        if message_type is None:
            return self.message_queue.copy()
        else:
            return [
                msg for msg in self.message_queue if msg.message_type == message_type
            ]

    def _cleanup_queue(self):
        """Clean up old and low-priority messages."""
        current_time = time.time()

        # Remove timed out messages
        self.message_queue = [
            msg
            for msg in self.message_queue
            if current_time - msg.timestamp < self.protocol_rules["message_timeout"]
        ]

        # Remove lowest priority messages if still over limit
        if len(self.message_queue) >= self.protocol_rules["max_queue_size"]:
            self.message_queue.sort(key=lambda m: m.priority)
            excess = len(self.message_queue) - self.protocol_rules["max_queue_size"] + 1
            self.message_queue = self.message_queue[excess:]

    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        if not self.communication_history:
            return {"total_messages": 0}

        sender_counts = {}
        type_counts = {}
        priority_counts = {1: 0, 2: 0, 3: 0}

        for message in self.communication_history:
            sender_counts[message.sender] = sender_counts.get(message.sender, 0) + 1
            type_counts[message.message_type] = (
                type_counts.get(message.message_type, 0) + 1
            )
            priority_counts[message.priority] += 1

        return {
            "total_messages": len(self.communication_history),
            "queue_size": len(self.message_queue),
            "sender_distribution": sender_counts,
            "type_distribution": type_counts,
            "priority_distribution": priority_counts,
        }


class AdviceSystem:
    """System for managing advice between human and AI."""

    def __init__(self):
        self.advice_history = []
        self.advice_rules = {
            "max_advice_age": 300.0,  # 5 minutes
            "confidence_threshold": 0.7,
            "advice_weight": 0.3,
        }

    def provide_advice(
        self,
        advisor: str,
        advice_type: str,
        content: str,
        confidence: float,
        context: Dict[str, Any],
    ) -> Advice:
        """Provide advice from advisor to recipient."""
        advice = Advice(
            advisor=advisor,
            advice_type=advice_type,
            content=content,
            confidence=confidence,
            context=context,
            timestamp=time.time(),
        )

        self.advice_history.append(advice)
        return advice

    def get_relevant_advice(
        self,
        current_context: Dict[str, Any],
        advice_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Advice]:
        """Get relevant advice for current context."""
        current_time = time.time()
        relevant_advice = []

        for advice in self.advice_history:
            # Check age
            if current_time - advice.timestamp > self.advice_rules["max_advice_age"]:
                continue

            # Check type filter
            if advice_type is not None and advice.advice_type != advice_type:
                continue

            # Check confidence filter
            if min_confidence is not None and advice.confidence < min_confidence:
                continue

            # Check context relevance (simplified)
            if self._is_context_relevant(advice.context, current_context):
                relevant_advice.append(advice)

        # Sort by confidence and recency
        relevant_advice.sort(
            key=lambda a: (a.confidence, current_time - a.timestamp), reverse=True
        )

        return relevant_advice

    def _is_context_relevant(
        self, advice_context: Dict[str, Any], current_context: Dict[str, Any]
    ) -> bool:
        """Check if advice context is relevant to current context."""
        # Simple relevance check based on shared keys
        shared_keys = set(advice_context.keys()) & set(current_context.keys())
        if len(shared_keys) == 0:
            return False

        # Check if values are similar for shared keys
        relevance_score = 0
        for key in shared_keys:
            if advice_context[key] == current_context[key]:
                relevance_score += 1

        return relevance_score > 0

    def get_advice_statistics(self) -> Dict[str, Any]:
        """Get advice system statistics."""
        if not self.advice_history:
            return {"total_advice": 0}

        advisor_counts = {}
        type_counts = {}
        confidence_scores = []

        for advice in self.advice_history:
            advisor_counts[advice.advisor] = advisor_counts.get(advice.advisor, 0) + 1
            type_counts[advice.advice_type] = type_counts.get(advice.advice_type, 0) + 1
            confidence_scores.append(advice.confidence)

        return {
            "total_advice": len(self.advice_history),
            "advisor_distribution": advisor_counts,
            "type_distribution": type_counts,
            "avg_confidence": np.mean(confidence_scores),
            "confidence_std": np.std(confidence_scores),
        }


class DemonstrationCollector:
    """Collects and manages demonstrations for learning."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.demonstrations = []
        self.collection_active = True

    def start_demonstration(self, demo_id: str) -> str:
        """Start collecting a demonstration."""
        if not self.collection_active:
            raise ValueError("Demonstration collection is not active")

        demo = {
            "demo_id": demo_id,
            "trajectory": [],
            "start_time": time.time(),
            "completed": False,
        }

        self.demonstrations.append(demo)
        return demo_id

    def add_demonstration_step(
        self, demo_id: str, state: torch.Tensor, action: torch.Tensor, reward: float
    ):
        """Add a step to an ongoing demonstration."""
        demo = self._find_demonstration(demo_id)
        if demo is None:
            raise ValueError(f"Demonstration {demo_id} not found")

        step = {
            "state": state.clone(),
            "action": action.clone(),
            "reward": reward,
            "timestamp": time.time(),
        }

        demo["trajectory"].append(step)

    def end_demonstration(self, demo_id: str) -> Dict[str, Any]:
        """End a demonstration and return summary."""
        demo = self._find_demonstration(demo_id)
        if demo is None:
            raise ValueError(f"Demonstration {demo_id} not found")

        demo["completed"] = True
        demo["end_time"] = time.time()
        demo["duration"] = demo["end_time"] - demo["start_time"]

        # Compute statistics
        rewards = [step["reward"] for step in demo["trajectory"]]
        demo["statistics"] = {
            "total_steps": len(demo["trajectory"]),
            "total_reward": sum(rewards),
            "avg_reward": np.mean(rewards),
            "duration": demo["duration"],
        }

        return demo["statistics"]

    def _find_demonstration(self, demo_id: str) -> Optional[Dict[str, Any]]:
        """Find a demonstration by ID."""
        for demo in self.demonstrations:
            if demo["demo_id"] == demo_id:
                return demo
        return None

    def get_demonstration_trajectory(
        self, demo_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get the trajectory of a specific demonstration."""
        demo = self._find_demonstration(demo_id)
        if demo is None:
            return None
        return demo["trajectory"]

    def get_all_demonstrations(self) -> List[Dict[str, Any]]:
        """Get all demonstrations."""
        return self.demonstrations.copy()

    def get_completed_demonstrations(self) -> List[Dict[str, Any]]:
        """Get only completed demonstrations."""
        return [demo for demo in self.demonstrations if demo["completed"]]

    def get_demonstration_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected demonstrations."""
        if not self.demonstrations:
            return {"total_demonstrations": 0}

        completed_demos = self.get_completed_demonstrations()
        total_steps = sum(len(demo["trajectory"]) for demo in completed_demos)
        total_rewards = [demo["statistics"]["total_reward"] for demo in completed_demos]

        return {
            "total_demonstrations": len(self.demonstrations),
            "completed_demonstrations": len(completed_demos),
            "total_steps": total_steps,
            "avg_steps_per_demo": (
                total_steps / len(completed_demos) if completed_demos else 0
            ),
            "avg_total_reward": np.mean(total_rewards) if total_rewards else 0,
            "reward_std": np.std(total_rewards) if total_rewards else 0,
        }

    def clear_demonstrations(self):
        """Clear all demonstrations."""
        self.demonstrations.clear()

    def export_demonstrations(self) -> Dict[str, Any]:
        """Export demonstrations in a format suitable for training."""
        completed_demos = self.get_completed_demonstrations()

        if not completed_demos:
            return {"states": [], "actions": [], "rewards": []}

        all_states = []
        all_actions = []
        all_rewards = []

        for demo in completed_demos:
            for step in demo["trajectory"]:
                all_states.append(step["state"])
                all_actions.append(step["action"])
                all_rewards.append(step["reward"])

        return {
            "states": torch.stack(all_states) if all_states else torch.tensor([]),
            "actions": torch.stack(all_actions) if all_actions else torch.tensor([]),
            "rewards": torch.tensor(all_rewards) if all_rewards else torch.tensor([]),
        }
