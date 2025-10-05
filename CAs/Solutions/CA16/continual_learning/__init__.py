"""
Continual and Lifelong Learning for Deep Reinforcement Learning

This module contains implementations of continual learning components including:
- Elastic Weight Consolidation (EWC)
- Progressive Networks
- Experience Replay
- Meta-Learning approaches
- Dynamic Architectures
"""

from .ewc import ElasticWeightConsolidation, OnlineEWC, EWCWrapper

from .progressive_networks import (
    ProgressiveNetwork,
    DynamicProgressiveNetwork,
    ProgressiveNetworkTrainer,
)

from .experience_replay import (
    Experience,
    ExperienceReplay,
    PrioritizedExperienceReplay,
    TaskBalancedReplay,
    ContinualReplay,
)

from .meta_learning import MAML, Reptile, MetaLearningTrainer

from .dynamic_architectures import (
    DynamicNetwork,
    AdaptiveNetwork,
    ModularNetwork,
)

from .continual_agent import ContinualLearningAgent

__all__ = [
    "ElasticWeightConsolidation",
    "OnlineEWC",
    "EWCWrapper",
    "ProgressiveNetwork",
    "DynamicProgressiveNetwork",
    "ProgressiveNetworkTrainer",
    "Experience",
    "ExperienceReplay",
    "PrioritizedExperienceReplay",
    "TaskBalancedReplay",
    "ContinualReplay",
    "MAML",
    "Reptile",
    "MetaLearningTrainer",
    "DynamicNetwork",
    "AdaptiveNetwork",
    "ModularNetwork",
    "ContinualLearningAgent",
]
