"""
Continual and Lifelong Learning for Deep Reinforcement Learning

This module contains implementations of continual learning components including:
- Elastic Weight Consolidation (EWC)
- Progressive Networks
- Experience Replay
- Meta-Learning approaches
- Dynamic Architectures
"""

from .ewc import ElasticWeightConsolidation, EWCNetwork

from .progressive_networks import (
    ProgressiveNetwork,
    ProgressiveColumn,
    LateralConnection,
)

from .experience_replay import (
    ExperienceReplay,
    PrioritizedExperienceReplay,
    ContinualExperienceReplay,
)

from .meta_learning import MAML, Reptile, MetaLearner

from .dynamic_architectures import (
    DynamicNetwork,
    TaskSpecificHead,
    KnowledgeDistillation,
)

from .continual_agent import (
    ContinualLearningAgent,
    LifelongLearner,
    TransferLearningAgent,
)

__all__ = [
    "ElasticWeightConsolidation",
    "EWCNetwork",
    "ProgressiveNetwork",
    "ProgressiveColumn",
    "LateralConnection",
    "ExperienceReplay",
    "PrioritizedExperienceReplay",
    "ContinualExperienceReplay",
    "MAML",
    "Reptile",
    "MetaLearner",
    "DynamicNetwork",
    "TaskSpecificHead",
    "KnowledgeDistillation",
    "ContinualLearningAgent",
    "LifelongLearner",
    "TransferLearningAgent",
]
