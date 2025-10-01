"""
Collaborative Agent for Human-AI Interaction

This module implements collaborative agents that work with humans.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from .preference_model import HumanPreference, PreferenceLearner
from .feedback_collector import HumanFeedbackCollector, TrustModel


class CollaborativeAgent:
    """Agent that collaborates with humans."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Human collaboration components
        self.preference_learner = PreferenceLearner(state_dim, action_dim, lr)
        self.feedback_collector = HumanFeedbackCollector()
        self.trust_model = TrustModel(state_dim, action_dim)

        # Collaboration parameters
        self.collaboration_threshold = 0.3  # When to ask for human help
        self.confidence_history = []

        # Training history
        self.training_history = {
            "losses": [],
            "rewards": [],
            "collaboration_events": [],
        }

    def select_action(
        self, state: torch.Tensor, human_available: bool = False
    ) -> Tuple[int, float]:
        """Select action with optional human collaboration."""
        with torch.no_grad():
            action_logits = self.policy(state.unsqueeze(0))
            action_probs = F.softmax(action_logits, dim=-1)

            # Calculate confidence
            confidence = torch.max(action_probs).item()
            self.confidence_history.append(confidence)

            # Decide whether to collaborate with human
            if human_available and confidence < self.collaboration_threshold:
                # Request human input
                return self._request_human_input(state, action_probs)
            else:
                # Use AI decision
                action = torch.multinomial(action_probs, 1)
                return action.item(), confidence

    def _request_human_input(
        self, state: torch.Tensor, action_probs: torch.Tensor
    ) -> Tuple[int, float]:
        """Request human input for action selection."""
        # Simulate human providing input
        human_action = torch.argmax(action_probs).item()
        human_confidence = 0.8

        # Record collaboration event
        self.training_history["collaboration_events"].append(
            {
                "state": state.cpu().numpy(),
                "ai_action": torch.argmax(action_probs).item(),
                "human_action": human_action,
                "human_confidence": human_confidence,
            }
        )

        return human_action, human_confidence

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        human_feedbacks: List = None,
    ) -> Dict[str, float]:
        """Update policy using rewards and human feedback."""
        # Standard policy gradient update
        action_logits = self.policy(states)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        policy_loss = -(log_probs * rewards.unsqueeze(1)).mean()

        # Human feedback update
        feedback_loss = 0.0
        if human_feedbacks:
            feedback_loss = self.preference_learner.learn_preferences(human_feedbacks)[
                "avg_loss"
            ]

        # Total loss
        total_loss = policy_loss + 0.1 * feedback_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Record training history
        self.training_history["losses"].append(total_loss.item())
        self.training_history["rewards"].append(rewards.mean().item())

        return {
            "policy_loss": policy_loss.item(),
            "feedback_loss": feedback_loss,
            "total_loss": total_loss.item(),
        }

    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Get collaboration statistics."""
        return {
            "collaboration_events": len(self.training_history["collaboration_events"]),
            "avg_confidence": (
                np.mean(self.confidence_history) if self.confidence_history else 0.0
            ),
            "feedback_statistics": self.feedback_collector.get_feedback_statistics(),
            "trust_metrics": self.trust_model.compute_trust_metrics(),
        }


class SharedAutonomyController:
    """Controller for shared autonomy between human and AI."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Arbitration parameters
        self.human_authority_threshold = 0.7
        self.ai_authority_threshold = 0.3
        self.blending_weight = 0.5

        # Authority history
        self.authority_history = []

    def arbitrate(
        self,
        ai_action: int,
        human_action: int,
        ai_confidence: float,
        human_confidence: float,
    ) -> int:
        """Arbitrate between AI and human actions."""
        # Determine authority levels
        ai_authority = self._compute_ai_authority(ai_confidence)
        human_authority = self._compute_human_authority(human_confidence)

        # Record authority decision
        self.authority_history.append(
            {
                "ai_action": ai_action,
                "human_action": human_action,
                "ai_authority": ai_authority,
                "human_authority": human_authority,
                "final_action": None,
            }
        )

        # Make arbitration decision
        if human_authority > self.human_authority_threshold:
            final_action = human_action
        elif ai_authority > self.ai_authority_threshold:
            final_action = ai_action
        else:
            # Blend actions
            final_action = self._blend_actions(
                ai_action, human_action, ai_authority, human_authority
            )

        # Update history
        self.authority_history[-1]["final_action"] = final_action

        return final_action

    def _compute_ai_authority(self, ai_confidence: float) -> float:
        """Compute AI authority level."""
        return ai_confidence

    def _compute_human_authority(self, human_confidence: float) -> float:
        """Compute human authority level."""
        return human_confidence

    def _blend_actions(
        self,
        ai_action: int,
        human_action: int,
        ai_authority: float,
        human_authority: float,
    ) -> int:
        """Blend AI and human actions."""
        # Simple blending - choose action with higher authority
        if ai_authority > human_authority:
            return ai_action
        else:
            return human_action

    def get_authority_statistics(self) -> Dict[str, Any]:
        """Get authority arbitration statistics."""
        if not self.authority_history:
            return {"total_decisions": 0, "ai_wins": 0, "human_wins": 0, "blended": 0}

        ai_wins = sum(
            1
            for entry in self.authority_history
            if entry["final_action"] == entry["ai_action"]
        )
        human_wins = sum(
            1
            for entry in self.authority_history
            if entry["final_action"] == entry["human_action"]
        )
        blended = len(self.authority_history) - ai_wins - human_wins

        return {
            "total_decisions": len(self.authority_history),
            "ai_wins": ai_wins,
            "human_wins": human_wins,
            "blended": blended,
            "ai_win_rate": ai_wins / len(self.authority_history),
            "human_win_rate": human_wins / len(self.authority_history),
        }


class HumanAICoordinator:
    """Coordinates between human and AI agents."""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Coordination components
        self.collaborative_agent = CollaborativeAgent(state_dim, action_dim)
        self.shared_autonomy = SharedAutonomyController(state_dim, action_dim)

        # Coordination parameters
        self.coordination_mode = (
            "collaborative"  # 'collaborative', 'autonomous', 'supervised'
        )
        self.human_availability = True

        # Coordination history
        self.coordination_history = []

    def coordinate_action(
        self,
        state: torch.Tensor,
        human_action: int = None,
        human_confidence: float = 0.5,
    ) -> Tuple[int, Dict[str, Any]]:
        """Coordinate action selection between human and AI."""
        # Get AI action
        ai_action, ai_confidence = self.collaborative_agent.select_action(
            state, human_available=self.human_availability
        )

        # Coordinate based on mode
        if self.coordination_mode == "collaborative":
            if human_action is not None:
                final_action = self.shared_autonomy.arbitrate(
                    ai_action, human_action, ai_confidence, human_confidence
                )
            else:
                final_action = ai_action
        elif self.coordination_mode == "autonomous":
            final_action = ai_action
        else:  # supervised
            final_action = human_action if human_action is not None else ai_action

        # Record coordination event
        coordination_info = {
            "state": state.cpu().numpy(),
            "ai_action": ai_action,
            "human_action": human_action,
            "final_action": final_action,
            "ai_confidence": ai_confidence,
            "human_confidence": human_confidence,
            "coordination_mode": self.coordination_mode,
        }

        self.coordination_history.append(coordination_info)

        return final_action, coordination_info

    def update_coordination(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        human_feedbacks: List = None,
    ):
        """Update coordination based on experience."""
        # Update collaborative agent
        update_info = self.collaborative_agent.update(
            states, actions, rewards, human_feedbacks
        )

        return update_info

    def set_coordination_mode(self, mode: str):
        """Set coordination mode."""
        if mode in ["collaborative", "autonomous", "supervised"]:
            self.coordination_mode = mode
        else:
            raise ValueError(f"Invalid coordination mode: {mode}")

    def set_human_availability(self, available: bool):
        """Set human availability."""
        self.human_availability = available

    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            "coordination_mode": self.coordination_mode,
            "human_availability": self.human_availability,
            "total_coordinations": len(self.coordination_history),
            "collaboration_stats": self.collaborative_agent.get_collaboration_statistics(),
            "authority_stats": self.shared_autonomy.get_authority_statistics(),
        }
