"""
Collaborative Agents

This module provides agents that collaborate with humans:
- Human-AI partnerships
- Collaborative decision making
- Interactive learning agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict, deque
import threading
import time

from .preference_model import PreferenceRewardModel, HumanPreference
from .feedback_collector import HumanFeedbackCollector

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CollaborativeAgent:
    """Agent that collaborates with humans for decision making."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        reward_model: PreferenceRewardModel,
        feedback_collector: HumanFeedbackCollector,
        exploration_rate: float = 0.1,
        collaboration_threshold: float = 0.7,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_model = reward_model
        self.feedback_collector = feedback_collector
        self.exploration_rate = exploration_rate
        self.collaboration_threshold = collaboration_threshold

        # Decision making components
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        ).to(device)

        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        ).to(device)

        # Collaboration state
        self.human_trust_level = 0.5
        self.collaboration_history = deque(maxlen=100)
        self.pending_feedback_requests = []

        # Learning
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters())
            + list(self.value_network.parameters()),
            lr=1e-3,
        )

        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)

    def act(
        self, state: np.ndarray, request_human_input: bool = True
    ) -> Tuple[int, Dict]:
        """Select action with potential human collaboration."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Get AI action
        ai_action, ai_info = self._get_ai_action(state_tensor)

        # Decide whether to collaborate
        should_collaborate = self._should_collaborate(state, ai_info)

        if should_collaborate and request_human_input:
            human_action = self._request_human_action(state, ai_action)
            if human_action is not None:
                action = human_action
                collaboration_info = {
                    "collaborated": True,
                    "ai_suggestion": ai_action,
                    "human_override": True,
                    "trust_level": self.human_trust_level,
                }
            else:
                action = ai_action
                collaboration_info = {
                    "collaborated": True,
                    "ai_suggestion": ai_action,
                    "human_override": False,
                    "trust_level": self.human_trust_level,
                }
        else:
            action = ai_action
            collaboration_info = {
                "collaborated": False,
                "ai_suggestion": ai_action,
                "trust_level": self.human_trust_level,
            }

        # Store collaboration info
        self.collaboration_history.append(collaboration_info)

        return action, {**ai_info, **collaboration_info}

    def _get_ai_action(self, state_tensor: torch.Tensor) -> Tuple[int, Dict]:
        """Get action from AI policy."""
        self.policy_network.eval()
        self.value_network.eval()

        with torch.no_grad():
            action_logits = self.policy_network(state_tensor)
            value = self.value_network(state_tensor)

            # Exploration
            if np.random.random() < self.exploration_rate:
                action = np.random.randint(self.action_dim)
            else:
                action = torch.argmax(action_logits, dim=-1).item()

            action_probs = F.softmax(action_logits, dim=-1).squeeze().cpu().numpy()

        info = {
            "action_logits": action_logits.squeeze().cpu().numpy(),
            "action_probs": action_probs,
            "value": value.item(),
            "entropy": -np.sum(action_probs * np.log(action_probs + 1e-8)),
        }

        return action, info

    def _should_collaborate(self, state: np.ndarray, ai_info: Dict) -> bool:
        """Decide whether to request human collaboration."""
        # Factors for collaboration decision:
        # 1. Uncertainty in AI decision
        entropy = ai_info["entropy"]
        high_uncertainty = entropy > 1.0  # Threshold for high uncertainty

        # 2. Novel state (could be detected by reward model uncertainty)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            # Estimate novelty by reward prediction uncertainty
            reward_pred = self.reward_model.predict_reward(
                state_tensor, torch.zeros(1, self.action_dim).to(device)
            )
            novelty_score = 1.0 / (
                1.0 + abs(reward_pred.item())
            )  # Higher when reward is close to 0

        # 3. Human trust level
        trust_factor = (
            1.0 - self.human_trust_level
        )  # More collaboration when trust is low

        # 4. Recent collaboration success
        recent_success_rate = self._get_recent_collaboration_success()

        # Combine factors
        collaboration_score = (
            0.4 * (entropy / 2.0)  # Normalized entropy
            + 0.3 * novelty_score
            + 0.2 * trust_factor
            + 0.1 * (1.0 - recent_success_rate)
        )

        return collaboration_score > self.collaboration_threshold

    def _request_human_action(
        self, state: np.ndarray, ai_suggestion: int
    ) -> Optional[int]:
        """Request human action input."""
        # This would typically interface with a human input system
        # For simulation, we'll use a placeholder
        print(
            f"Requesting human input for state: {state[:5]}... AI suggests action {ai_suggestion}"
        )

        # Simulate human response (in real implementation, this would wait for actual input)
        # For now, return None to indicate no human input available
        return None

    def _get_recent_collaboration_success(self) -> float:
        """Get success rate of recent collaborations."""
        if len(self.collaboration_history) == 0:
            return 0.5

        recent = list(self.collaboration_history)[-10:]  # Last 10 collaborations
        successful = sum(1 for c in recent if c.get("human_override", False))
        return successful / len(recent) if recent else 0.5

    def learn_from_feedback(self, feedback_batch: List[HumanFeedback]):
        """Learn from human feedback."""
        for feedback in feedback_batch:
            if feedback.feedback_type == "preference":
                self._learn_from_preference(feedback.content)
            elif feedback.feedback_type == "correction":
                self._learn_from_correction(feedback.content)
            elif feedback.feedback_type == "demonstration":
                self._learn_from_demonstration(feedback.content)

    def _learn_from_preference(self, preference: HumanPreference):
        """Learn from preference feedback."""
        # Update reward model with preference
        self.reward_model.loss_function(
            self.reward_model.forward([preference.option_a], [preference.option_b]),
            [preference],
        )

        # Update trust level based on preference agreement
        # This is a simplified implementation
        self.human_trust_level = min(1.0, self.human_trust_level + 0.01)

    def _learn_from_correction(self, correction: Dict):
        """Learn from correction feedback."""
        # Update policy based on correction
        incorrect_action = correction.get("incorrect_action")
        correct_action = correction.get("correct_action")

        if incorrect_action is not None and correct_action is not None:
            # This would require storing the state where correction occurred
            # For now, just update trust
            self.human_trust_level = min(1.0, self.human_trust_level + 0.05)

    def _learn_from_demonstration(self, demonstration: Dict):
        """Learn from human demonstration."""
        # Add demonstration to experience buffer
        states = demonstration.get("states", [])
        actions = demonstration.get("actions", [])
        rewards = demonstration.get("rewards", [])

        for s, a, r in zip(states, actions, rewards):
            self.experience_buffer.append(
                {"state": s, "action": a, "reward": r, "demonstration": True}
            )

        self.human_trust_level = min(1.0, self.human_trust_level + 0.02)

    def update_policy(self, batch_size: int = 32, gamma: float = 0.99):
        """Update policy using experience replay."""
        if len(self.experience_buffer) < batch_size:
            return

        # Sample batch
        batch_indices = np.random.choice(
            len(self.experience_buffer), batch_size, replace=False
        )
        batch = [self.experience_buffer[i] for i in batch_indices]

        # Prepare batch data
        states = torch.FloatTensor([exp["state"] for exp in batch]).to(device)
        actions = torch.LongTensor([exp["action"] for exp in batch]).to(device)
        rewards = torch.FloatTensor([exp["reward"] for exp in batch]).to(device)

        # Compute targets
        with torch.no_grad():
            next_values = self.value_network(states).squeeze()
            targets = rewards + gamma * next_values

        # Update value network
        current_values = self.value_network(states).squeeze()
        value_loss = F.mse_loss(current_values, targets)

        # Update policy network
        action_logits = self.policy_network(states)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()

        # Policy gradient with advantage
        advantages = targets - current_values.detach()
        policy_loss = -(selected_log_probs * advantages).mean()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }


class HumanAIPartnership:
    """Manages long-term human-AI collaboration."""

    def __init__(
        self,
        agent: CollaborativeAgent,
        feedback_collector: HumanFeedbackCollector,
        partnership_goals: List[str] = None,
    ):
        self.agent = agent
        self.feedback_collector = feedback_collector
        self.partnership_goals = partnership_goals or [
            "improve_performance",
            "build_trust",
            "learn_preferences",
        ]

        # Partnership state
        self.partnership_metrics = {
            "total_interactions": 0,
            "successful_collaborations": 0,
            "human_satisfaction": 0.5,
            "ai_performance": 0.5,
            "trust_level": 0.5,
        }

        self.interaction_history = deque(maxlen=1000)
        self.goal_progress = defaultdict(float)

    def interact(
        self, state: np.ndarray, human_available: bool = True
    ) -> Dict[str, Any]:
        """Handle a single interaction in the partnership."""
        self.partnership_metrics["total_interactions"] += 1

        # Agent action
        action, action_info = self.agent.act(state, request_human_input=human_available)

        # Simulate environment step (placeholder)
        next_state = state + np.random.randn(*state.shape) * 0.1
        reward = np.random.random() - 0.5  # Random reward
        done = np.random.random() < 0.01  # Small chance of episode end

        # Store interaction
        interaction = {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "action_info": action_info,
            "timestamp": time.time(),
        }
        self.interaction_history.append(interaction)

        # Update partnership metrics
        self._update_metrics(interaction)

        # Collect feedback if available
        if human_available and np.random.random() < 0.3:  # 30% chance of feedback
            feedback = self._simulate_human_feedback(interaction)
            self.feedback_collector.collect_feedback(feedback)
            self.agent.learn_from_feedback([feedback])

        return {
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "collaboration_info": action_info,
        }

    def _update_metrics(self, interaction: Dict):
        """Update partnership metrics based on interaction."""
        action_info = interaction["action_info"]

        # Update successful collaborations
        if action_info.get("collaborated", False):
            self.partnership_metrics["successful_collaborations"] += 1

        # Update trust level
        self.partnership_metrics["trust_level"] = self.agent.human_trust_level

        # Update AI performance (simplified)
        reward = interaction["reward"]
        self.partnership_metrics["ai_performance"] = 0.9 * self.partnership_metrics[
            "ai_performance"
        ] + 0.1 * max(0, reward + 0.5)

    def _simulate_human_feedback(self, interaction: Dict) -> Any:
        """Simulate human feedback for demonstration."""
        from .feedback_collector import HumanFeedback

        feedback_types = ["preference", "correction", "demonstration"]
        feedback_type = np.random.choice(feedback_types, p=[0.5, 0.3, 0.2])

        if feedback_type == "preference":
            # Create preference between current action and alternative
            current_traj = {
                "states": [interaction["state"]],
                "actions": [interaction["action"]],
                "rewards": [interaction["reward"]],
            }
            alt_action = (interaction["action"] + 1) % self.agent.action_dim
            alt_traj = {
                "states": [interaction["state"]],
                "actions": [alt_action],
                "rewards": [interaction["reward"] + np.random.randn() * 0.1],
            }

            preference = HumanPreference(
                option_a=current_traj,
                option_b=alt_traj,
                preferred=(
                    "A" if interaction["reward"] > alt_traj["rewards"][0] else "B"
                ),
                confidence=np.random.random() * 0.5 + 0.5,
            )
            feedback = HumanFeedback("preference", preference)

        elif feedback_type == "correction":
            feedback = HumanFeedback(
                "correction",
                {
                    "state": interaction["state"],
                    "incorrect_action": interaction["action"],
                    "correct_action": (interaction["action"] + 1)
                    % self.agent.action_dim,
                    "reason": "Better alternative available",
                },
            )

        else:  # demonstration
            feedback = HumanFeedback(
                "demonstration",
                {
                    "states": [interaction["state"]],
                    "actions": [interaction["action"]],
                    "rewards": [interaction["reward"]],
                },
            )

        return feedback

    def get_partnership_status(self) -> Dict[str, Any]:
        """Get current status of the human-AI partnership."""
        collaboration_rate = self.partnership_metrics[
            "successful_collaborations"
        ] / max(1, self.partnership_metrics["total_interactions"])

        return {
            "metrics": self.partnership_metrics,
            "collaboration_rate": collaboration_rate,
            "goal_progress": dict(self.goal_progress),
            "recent_interactions": (
                list(self.interaction_history)[-5:] if self.interaction_history else []
            ),
        }

    def update_goals(self, new_goals: List[str]):
        """Update partnership goals."""
        self.partnership_goals = new_goals
        self.goal_progress.clear()

    def optimize_partnership(self):
        """Optimize the partnership based on current metrics."""
        # Adjust collaboration threshold based on trust
        trust = self.partnership_metrics["trust_level"]
        if trust > 0.8:
            self.agent.collaboration_threshold = 0.8  # Less collaboration needed
        elif trust < 0.3:
            self.agent.collaboration_threshold = 0.5  # More collaboration needed
        else:
            self.agent.collaboration_threshold = 0.7  # Default

        # Adjust exploration based on performance
        performance = self.partnership_metrics["ai_performance"]
        self.agent.exploration_rate = max(0.05, 0.2 - performance * 0.15)
