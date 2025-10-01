"""
Human-AI Collaboration for Deep Reinforcement Learning

This module contains implementations of human-AI collaboration components including:
- Preference-based learning
- Interactive imitation learning
- Shared autonomy and control
- Trust and calibration
- Human feedback integration
"""

from .preference_model import (
    HumanPreference,
    HumanFeedback,
    PreferenceRewardModel,
    BradleyTerryModel,
)

from .feedback_collector import HumanFeedbackCollector, InteractiveLearner, TrustModel

from .collaborative_agent import (
    CollaborativeAgent,
    SharedAutonomyController,
    HumanAICoordinator,
)

from .communication import CommunicationProtocol, AdviceSystem, DemonstrationCollector

__all__ = [
    "HumanPreference",
    "HumanFeedback",
    "PreferenceRewardModel",
    "BradleyTerryModel",
    "HumanFeedbackCollector",
    "InteractiveLearner",
    "TrustModel",
    "CollaborativeAgent",
    "SharedAutonomyController",
    "HumanAICoordinator",
    "CommunicationProtocol",
    "AdviceSystem",
    "DemonstrationCollector",
]
