"""
Meta-Learning Module
====================

This module implements advanced meta-learning algorithms for reinforcement learning:
- Model-Agnostic Meta-Learning (MAML)
- Reptile
- Memory-Augmented Networks (MANN)
- Meta-Gradient Reinforcement Learning
- Probabilistic Meta-Learning
- Few-Shot RL
"""

from .meta_learning import (
    MAML,
    Reptile,
    MemoryAugmentedNetwork,
    MetaGradientRL,
    ProbabilisticMetaLearning,
    FewShotRL,
    MetaLearningTrainer,
    demonstrate_meta_learning,
    create_meta_learning_visualizations
)

__all__ = [
    'MAML',
    'Reptile',
    'MemoryAugmentedNetwork',
    'MetaGradientRL',
    'ProbabilisticMetaLearning',
    'FewShotRL',
    'MetaLearningTrainer',
    'demonstrate_meta_learning',
    'create_meta_learning_visualizations'
]

