"""
Continual Learning Package

This package provides continual learning algorithms for reinforcement learning,
enabling agents to learn new tasks while retaining knowledge from previous tasks.
"""

from .elastic_weight_consolidation import ElasticWeightConsolidation, EWCTrainer

from .progressive_networks import ProgressiveNetwork, ProgressiveNetworkTrainer

from .meta_learning import MAMLAgent, MetaLearner

__all__ = [
    # Elastic Weight Consolidation
    "ElasticWeightConsolidation",
    "EWCTrainer",
    # Progressive Networks
    "ProgressiveNetwork",
    "ProgressiveNetworkTrainer",
    # Meta Learning
    "MAMLAgent",
    "MetaLearner",
]
