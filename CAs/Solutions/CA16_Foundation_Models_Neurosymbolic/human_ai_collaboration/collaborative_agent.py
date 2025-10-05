"""
Collaborative Agent for Human-AI Interaction

This module implements collaborative agents that work with human feedback and preferences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import random


class CollaborativeAgent:
    """RL agent that collaborates with human feedback and preferences."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        collaboration_threshold: float = 0.7,
        lr: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.collaboration_threshold = collaboration_threshold

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters())
            + list(self.value_net.parameters())
            + list(self.confidence_net.parameters()),
            lr=lr,
        )

        # Human feedback storage
        self.human_feedback = []
        self.preference_data = []

        # Trust and collaboration metrics
        self.trust_score = 0.5
        self.collaboration_history = []

    def select_action(self, state: torch.Tensor) -> Tuple[int, Dict[str, Any]]:
        """Select action with confidence estimation."""
        with torch.no_grad():
            # Get action probabilities
            action_logits = self.policy_net(state)
            action_probs = F.softmax(action_logits, dim=-1)

            # Get confidence
            confidence = self.confidence_net(state).item()

            # Select action
            action = torch.multinomial(action_probs, 1).item()

            # Determine if human collaboration is needed
            needs_collaboration = confidence < self.collaboration_threshold

            collaboration_info = {
                "action": action,
                "confidence": confidence,
                "action_probs": action_probs.cpu().numpy(),
                "needs_collaboration": needs_collaboration,
                "trust_score": self.trust_score,
            }

            return action, collaboration_info

    def update_with_feedback(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        human_feedback: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Update agent with human feedback."""
        # Standard policy gradient update
        action_logits = self.policy_net(states)
        action_probs = F.softmax(action_logits, dim=-1)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))

        # Value function update
        values = self.value_net(states).squeeze(-1)
        value_loss = F.mse_loss(values, rewards)

        # Policy loss with human feedback
        if human_feedback is not None:
            # Weight policy loss by human feedback
            policy_loss = -(log_probs.squeeze() * human_feedback).mean()
        else:
            # Standard policy gradient
            advantages = rewards - values.detach()
            policy_loss = -(log_probs.squeeze() * advantages).mean()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters())
            + list(self.value_net.parameters())
            + list(self.confidence_net.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    def add_human_feedback(
        self, state: torch.Tensor, action: int, feedback: float, explanation: str = ""
    ):
        """Add human feedback to the agent's memory."""
        self.human_feedback.append(
            {
                "state": state.cpu().numpy(),
                "action": action,
                "feedback": feedback,
                "explanation": explanation,
                "timestamp": len(self.human_feedback),
            }
        )

    def update_trust(self, predicted_outcome: float, actual_outcome: float):
        """Update trust score based on prediction accuracy."""
        prediction_error = abs(predicted_outcome - actual_outcome)
        
        # Update trust score (higher error = lower trust)
        trust_update = 0.1 * (1.0 - prediction_error)
        self.trust_score = np.clip(self.trust_score + trust_update, 0.0, 1.0)

    def get_collaboration_recommendation(self, state: torch.Tensor) -> Dict[str, Any]:
        """Get recommendation for human collaboration."""
        with torch.no_grad():
            confidence = self.confidence_net(state).item()
            
            recommendation = {
                "should_collaborate": confidence < self.collaboration_threshold,
                "confidence": confidence,
                "trust_level": self.trust_score,
                "collaboration_type": self._determine_collaboration_type(confidence),
                "explanation": self._generate_explanation(confidence),
            }
            
            return recommendation

    def _determine_collaboration_type(self, confidence: float) -> str:
        """Determine the type of collaboration needed."""
        if confidence < 0.3:
            return "full_guidance"
        elif confidence < 0.5:
            return "partial_guidance"
        elif confidence < 0.7:
            return "confirmation"
        else:
            return "monitoring"

    def _generate_explanation(self, confidence: float) -> str:
        """Generate explanation for collaboration recommendation."""
        if confidence < 0.3:
            return "Low confidence - full human guidance recommended"
        elif confidence < 0.5:
            return "Moderate confidence - partial human guidance helpful"
        elif confidence < 0.7:
            return "Good confidence - human confirmation suggested"
        else:
            return "High confidence - minimal human intervention needed"

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about human feedback."""
        if not self.human_feedback:
            return {"total_feedback": 0, "avg_feedback": 0.0}

        feedbacks = [f["feedback"] for f in self.human_feedback]
        
        return {
            "total_feedback": len(self.human_feedback),
            "avg_feedback": np.mean(feedbacks),
            "std_feedback": np.std(feedbacks),
            "positive_feedback": sum(1 for f in feedbacks if f > 0),
            "negative_feedback": sum(1 for f in feedbacks if f < 0),
            "neutral_feedback": sum(1 for f in feedbacks if f == 0),
        }


class SharedAutonomyController:
    """Controller for shared autonomy between human and AI."""

    def __init__(self, agent: CollaborativeAgent, human_weight: float = 0.5):
        self.agent = agent
        self.human_weight = human_weight
        self.ai_weight = 1.0 - human_weight
        
        # Autonomy levels
        self.autonomy_levels = {
            "full_ai": 0.0,
            "mostly_ai": 0.2,
            "balanced": 0.5,
            "mostly_human": 0.8,
            "full_human": 1.0,
        }
        
        self.current_autonomy = "balanced"

    def get_combined_action(
        self, state: torch.Tensor, human_action: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Get combined action from AI and human."""
        ai_action, ai_info = self.agent.select_action(state)
        
        if human_action is None:
            # No human input, use AI action
            return ai_action, {
                "action": ai_action,
                "source": "ai",
                "autonomy_level": self.current_autonomy,
                "confidence": ai_info["confidence"],
            }
        
        # Combine AI and human actions
        if self.current_autonomy == "full_ai":
            final_action = ai_action
            source = "ai"
        elif self.current_autonomy == "full_human":
            final_action = human_action
            source = "human"
        else:
            # Weighted combination
            human_weight = self.autonomy_levels[self.current_autonomy]
            ai_weight = 1.0 - human_weight
            
            # Simple weighted selection (in practice, this could be more sophisticated)
            if random.random() < human_weight:
                final_action = human_action
                source = "human"
            else:
                final_action = ai_action
                source = "ai"
        
        return final_action, {
            "action": final_action,
            "source": source,
            "ai_action": ai_action,
            "human_action": human_action,
            "autonomy_level": self.current_autonomy,
            "ai_confidence": ai_info["confidence"],
            "human_weight": self.autonomy_levels[self.current_autonomy],
            "ai_weight": 1.0 - self.autonomy_levels[self.current_autonomy],
        }

    def adjust_autonomy(self, performance_metric: float, human_satisfaction: float):
        """Adjust autonomy level based on performance and human satisfaction."""
        # Simple autonomy adjustment logic
        if performance_metric > 0.8 and human_satisfaction > 0.7:
            # High performance and satisfaction - increase AI autonomy
            if self.current_autonomy == "full_human":
                self.current_autonomy = "mostly_human"
            elif self.current_autonomy == "mostly_human":
                self.current_autonomy = "balanced"
            elif self.current_autonomy == "balanced":
                self.current_autonomy = "mostly_ai"
        elif performance_metric < 0.5 or human_satisfaction < 0.5:
            # Low performance or satisfaction - increase human autonomy
            if self.current_autonomy == "full_ai":
                self.current_autonomy = "mostly_ai"
            elif self.current_autonomy == "mostly_ai":
                self.current_autonomy = "balanced"
            elif self.current_autonomy == "balanced":
                self.current_autonomy = "mostly_human"

    def get_autonomy_info(self) -> Dict[str, Any]:
        """Get information about current autonomy level."""
        return {
            "current_autonomy": self.current_autonomy,
            "human_weight": self.autonomy_levels[self.current_autonomy],
            "ai_weight": 1.0 - self.autonomy_levels[self.current_autonomy],
            "available_levels": list(self.autonomy_levels.keys()),
        }


class HumanAICoordinator:
    """Coordinates interaction between human and AI agents."""

    def __init__(self, agent: CollaborativeAgent, controller: SharedAutonomyController):
        self.agent = agent
        self.controller = controller
        
        # Communication protocols
        self.communication_log = []
        self.interaction_history = []

    def coordinate_interaction(
        self, state: torch.Tensor, human_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Coordinate interaction between human and AI."""
        # Get AI recommendation
        ai_action, ai_info = self.agent.select_action(state)
        
        # Get collaboration recommendation
        collaboration_rec = self.agent.get_collaboration_recommendation(state)
        
        # Determine if human input is needed
        needs_human_input = (
            collaboration_rec["should_collaborate"]
            or self.controller.current_autonomy in ["mostly_human", "full_human"]
        )
        
        # Process human input if provided
        human_action = None
        if human_input is not None:
            human_action = human_input.get("action", None)
            if "feedback" in human_input:
                self.agent.add_human_feedback(
                    state,
                    human_action or ai_action,
                    human_input["feedback"],
                    human_input.get("explanation", ""),
                )
        
        # Get combined action
        final_action, action_info = self.controller.get_combined_action(
            state, human_action
        )
        
        # Log interaction
        interaction = {
            "state": state.cpu().numpy(),
            "ai_action": ai_action,
            "human_action": human_action,
            "final_action": final_action,
            "needs_human_input": needs_human_input,
            "collaboration_recommendation": collaboration_rec,
            "action_info": action_info,
            "timestamp": len(self.interaction_history),
        }
        
        self.interaction_history.append(interaction)
        
        return {
            "action": final_action,
            "needs_human_input": needs_human_input,
            "ai_recommendation": ai_action,
            "human_input": human_action,
            "collaboration_info": collaboration_rec,
            "autonomy_info": self.controller.get_autonomy_info(),
            "interaction_log": interaction,
        }

    def update_performance(
        self, performance_metric: float, human_satisfaction: float
    ):
        """Update system performance and adjust autonomy."""
        self.controller.adjust_autonomy(performance_metric, human_satisfaction)
        
        # Log performance update
        self.communication_log.append({
            "type": "performance_update",
            "performance": performance_metric,
            "satisfaction": human_satisfaction,
            "new_autonomy": self.controller.current_autonomy,
            "timestamp": len(self.communication_log),
        })

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "agent_trust": self.agent.trust_score,
            "autonomy_level": self.controller.current_autonomy,
            "total_interactions": len(self.interaction_history),
            "feedback_statistics": self.agent.get_feedback_statistics(),
            "communication_log_size": len(self.communication_log),
            "system_health": self._assess_system_health(),
        }

    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        if self.agent.trust_score > 0.8 and len(self.interaction_history) > 10:
            return "excellent"
        elif self.agent.trust_score > 0.6 and len(self.interaction_history) > 5:
            return "good"
        elif self.agent.trust_score > 0.4:
            return "fair"
        else:
            return "needs_attention"