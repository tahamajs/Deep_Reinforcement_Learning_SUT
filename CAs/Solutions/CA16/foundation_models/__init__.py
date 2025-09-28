"""
Foundation Models for RL

This module provides foundation model implementations for reinforcement learning:
- Decision Transformers for sequence-based decision making
- Multi-task foundation models
- In-context learning capabilities
- Training and evaluation utilities
"""

from .algorithms import (
    DecisionTransformer,
    MultiTaskRLFoundationModel,
    InContextLearningRL,
    FoundationModelTrainer as BaseFoundationModelTrainer,
)

from .training import (
    TrajectoryDataset,
    MultiTaskTrajectoryDataset,
    FoundationModelEvaluator,
    FoundationModelTrainer,
    create_trajectory_dataset_from_env,
    plot_training_curves,
)

__all__ = [
    # Core algorithms
    "DecisionTransformer",
    "MultiTaskRLFoundationModel",
    "InContextLearningRL",
    "BaseFoundationModelTrainer",
    # Training utilities
    "TrajectoryDataset",
    "MultiTaskTrajectoryDataset",
    "FoundationModelEvaluator",
    "FoundationModelTrainer",
    "create_trajectory_dataset_from_env",
    "plot_training_curves",
]

from .algorithms import (
    DecisionTransformer,
    MultiTaskRLFoundationModel,
    InContextLearningRL,
    FoundationModelTrainer,
)

__all__ = [
    "DecisionTransformer",
    "MultiTaskRLFoundationModel",
    "InContextLearningRL",
    "FoundationModelTrainer",
]
